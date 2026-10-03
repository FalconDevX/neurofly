"""Panel BANC na żywo: oczy drona → FlyVis → BANC v888 → komenda. Rysuje go ``BrainView``:

- ``python -m sim.viewer --brain``: sieć tylko obserwuje lot (sterujesz klawiszami), ``BrainPanel`` liczy ją w osobnym wątku,
- ``python -m sim.run_env --banc <wagi>``: sieć steruje (``sim/banc_pilot.py``), panel pokazuje też, skąd jest kurs,
- ``scripts/fly_banc.py --local --brain --video``: panel w wideo.

Układ panelu (po angielsku, jak eksplorator i prezentacja):

1. Mapa neuronów (widok z boku na somy z meta ``position``: mózg z lewej, VNC z prawej, prawa strona muchy u góry).
   Jasność = siła sygnału: o ile aktywność neuronu różni się od jego aktywności sprzed ~2 s (jedna barwa, bez
   mieszania kolorów). Neurony grup lotu jako kropki w kolorach eksploratora (DN bursztynowe, MN różowe, haltery niebieskie).
2. Siła sygnału lewo / prawo dla grup na drodze wzrok → DN → MN: zmiana średniej grupy w % względem stanu na
   początku epizodu (stałe odniesienie, więc trwała różnica L/P — to, co skręca dron — nie znika).
3. Sterowanie: skąd jest kurs (BANC / GPS i w jakim udziale), głosy neuronów za skrętem w lewo i w prawo
   (wagi dekodera × aktywność), yaw z BANC, z GPS i wynikowy, oraz thrust / roll / pitch ze źródłem.

Na starcie ``BrainPanel`` kalibruje kontroler na scenach ``DroneEnv`` (jak ``fly_banc.py``, ~20 s), opcjonalnie wagi
dekodera (``--brain-decoder``). Wymaga środowiska wzroku (.venv312: torch, flyvis) i danych ``data/banc_888``.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

VOXEL_NM = np.array([4.0, 4.0, 45.0])
W = 400            # szerokość panelu [px] (viewer skaluje go do okna)
H = 720            # wysokość panelu [px]
MAP_H = 300        # wysokość mapy neuronów
EMA = 0.015        # średnia odniesienia mapy (~2 s przy 30 Hz)
NORM_EMA = 0.1     # wygładzanie skali mapy (99. percentyl sygnału) — mapa nie mruga
BAR_FULL = 0.10    # względna zmiana średniej grupy dająca pełny pasek (10 %)

# paleta jak w eksploratorze (zinc + kolory klas)
BG = (9, 9, 11)
CARD = (24, 24, 27)
LINE = (39, 39, 42)
TEXT = (250, 250, 250)
DIM = (161, 161, 170)
FAINT = (113, 113, 122)
TEAL = (47, 211, 196)
AMBER = (255, 181, 71)
PINK = (255, 79, 176)
BLUE = (79, 140, 255)
RED = (255, 45, 111)
GROUP_COLOR = {"visual": TEAL, "dn_flight_steering": AMBER, "dn_flight_power": AMBER, "wing_steering": PINK,
               "wing_power": PINK, "wing_tension": PINK, "haltere_aff": BLUE}
SIGNAL_ROWS = (  # (etykieta, grupa bez _L/_R) — kolejność drogi sygnału
    ("Vision (VPN)", "visual"),
    ("DN steering", "dn_flight_steering"),
    ("DN power", "dn_flight_power"),
    ("MN steering", "wing_steering"),
    ("MN power", "wing_power"),
    ("Halteres", "haltere_aff"),
)


def _font_path(names: tuple[str, ...]) -> str | None:
    """Roboto (jak w prezentacji) z czcionek użytkownika albo systemu; potem Segoe UI / DejaVu."""
    dirs = [os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts"), r"C:\Windows\Fonts",
            "/usr/share/fonts/truetype/roboto", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]
    for n in names:
        for d in dirs:
            f = os.path.join(d, n)
            if os.path.exists(f):
                return f
    return None


_FONTS: dict = {}


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    key = (size, bold)
    if key not in _FONTS:
        names = (("Roboto-Medium.ttf", "seguisb.ttf", "DejaVuSans-Bold.ttf") if bold
                 else ("Roboto-Regular.ttf", "segoeui.ttf", "DejaVuSans.ttf"))
        path = _font_path(names)
        _FONTS[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    return _FONTS[key]


def steer_from_decoder(w_yaw: np.ndarray, x: np.ndarray, yaw_banc: float, bias: bool, yaw_gps: float | None = None,
                       w_vis: float = 1.0, distance: float | None = None, mode: str = "flying") -> dict:
    """Informacje do sekcji sterowania. ``w_yaw``, ``x``: wiersz yaw dekodera i znormalizowane cechy (z wyrazem wolnym
    na końcu, gdy ``bias``). Głosy = wkłady neuronów ``w·x`` za skrętem w prawo (> 0) i w lewo (< 0), bez wyrazu
    wolnego. Konwencja BANC: yaw + = w prawo."""
    n = min(len(w_yaw), len(x))
    contrib = np.asarray(w_yaw[:n], float) * np.nan_to_num(np.asarray(x[:n], float))
    neurons = contrib[:-1] if bias else contrib
    return {"mode": mode, "yaw_banc": float(yaw_banc), "yaw_gps": yaw_gps, "w_vis": float(w_vis), "distance": distance,
            "push_right": float(neurons[neurons > 0].sum()), "push_left": float(-neurons[neurons < 0].sum()),
            "n_votes": int(np.count_nonzero(np.abs(neurons) > 1e-12))}


class BrainPanel:
    def __init__(self, decoder_path: Path | None = None, readout: str | None = None) -> None:
        self.decoder_path, self.readout = decoder_path, readout
        self.status = "loading BANC v888…"
        self.image: np.ndarray | None = None  # ostatni gotowy panel (RGB uint8), czytany przez Overlays
        self._frames = None
        self._gyro = (0.0, 0.0, 0.0)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self.image = self._placeholder()
        threading.Thread(target=self._run, daemon=True).start()

    # --- wywoływane z innych wątków ---
    def submit(self, frames) -> None:
        """Nowe klatki oczu (lewa, prawa); starsze nieprzetworzone są pomijane."""
        with self._lock:
            self._frames = frames
        self._wake.set()

    def set_gyro(self, gyro) -> None:
        """Żyroskop w konwencji ``DroneEnv.imu`` (yaw + = w prawo)."""
        self._gyro = tuple(float(g) for g in gyro)

    def close(self) -> None:
        self._stop = True
        self._wake.set()

    # --- wątek sieci ---
    def _setup(self) -> None:
        from banc_control import BancController, Connectome
        from banc_control.readout import LinearDecoder
        from sim.env import DroneEnv
        from visual_pipeline import VisionBridge
        from visual_pipeline.server import ControlServer, LocalClient

        t0 = time.perf_counter()
        self.c = c = Connectome.from_banc()
        readout = self.readout
        if readout is None and self.decoder_path is not None:
            readout = "dn" if np.load(self.decoder_path)["M"].shape[1] > 6 else "mn"
        self.ctrl = BancController(c, decoder=LinearDecoder(), readout=readout or "mn")
        self.bridge = VisionBridge(fps=30, fisheye=True)

        self.status = "calibrating on DroneEnv scenes…"
        self.image = self._placeholder()
        env = DroneEnv()  # osobny model tylko do scen kalibracyjnych (jak fly_banc.py)
        LocalClient(ControlServer(self.bridge, self.ctrl)).calibrate(env.calibration_render)
        env.close()
        if self.decoder_path is not None:
            self.ctrl.decoder.load_weights(self.decoder_path)

        self.view = BrainView(c)
        self.status = f"BANC v888: {c.n:,} neurons, readout {self.ctrl.readout}, ready after {time.perf_counter() - t0:.0f} s"
        print("panel mózgu:", self.status)

    def _run(self) -> None:
        try:
            self._setup()
        except Exception as e:  # panel nie może wywrócić podglądu
            self.status = f"panel error: {type(e).__name__}: {e}"
            print(self.status)
            self.image = self._placeholder()
            return
        from banc_control import ImuState

        started = False
        while not self._stop:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                frames, self._frames = self._frames, None
            if frames is None:
                continue
            t = time.perf_counter()
            if not started:
                self.ctrl.dyn.reset()
                self.ctrl.warm_start(self.bridge.settle(*frames))
                started = True
            visual = self.bridge.step_batch(*frames)
            cmd = self.ctrl.step(visual, ImuState(gyro=self._gyro))
            rates = self.ctrl.dyn.rates_at(np.arange(self.c.n))
            steer = {"mode": "observing"}
            feats = (getattr(cmd, "debug", None) or {}).get("motor_features")
            if self.decoder_path is not None and feats is not None:
                dec = self.ctrl.decoder
                steer = steer_from_decoder(dec.M[3], dec.normalized(feats), cmd.yaw, dec.bias, mode="observing")
            self.image = self.view.render(rates, cmd, (time.perf_counter() - t) * 1e3, steer)

    def _placeholder(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((14, 10), "BANC v888", font=_font(15, True), fill=TEXT)
        d.text((14, 36), self.status, font=_font(12), fill=DIM)
        return np.asarray(im)


class BrainView:
    """Rysowanie panelu dla connectomu ``c`` (bez wątków): ``render(rates, cmd, ms, steer)`` → obraz RGB.
    Używane przez ``BrainPanel`` (podgląd), ``sim/banc_pilot.py`` (pilot) i ``scripts/fly_banc.py --brain`` (wideo)."""

    def __init__(self, c) -> None:
        import pandas as pd

        from banc_control.connectome import DEFAULT_DATA_DIR, META_FILE

        self.n = c.n
        # Pozycje som w kolejności neuronów connectomu → piksele: oś ciała (y BANC) w poziomie, mózg z lewej.
        meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE, columns=["banc_888_id", "position"])
        pos = meta.set_index(meta.banc_888_id.astype(np.int64)).position.reindex(c.root_ids)  # w pliku ID to tekst
        xyz = np.array([[float(v) for v in p.split(",")] if isinstance(p, str) else [np.nan] * 3
                        for p in pos]) * VOXEL_NM / 1000.0
        ok = np.isfinite(xyz).all(axis=1)
        lo, hi = np.percentile(xyz[ok], 0.5, axis=0), np.percentile(xyz[ok], 99.5, axis=0)
        top, mw, mh = 34, W - 28, MAP_H - 34 - 30
        s = min(mw / (hi[1] - lo[1]), mh / (hi[0] - lo[0]))
        px = (xyz[:, 1] - lo[1]) * s + (W - (hi[1] - lo[1]) * s) / 2
        py = (xyz[:, 0] - lo[0]) * s + top + (mh - (hi[0] - lo[0]) * s) / 2  # w BANC prawa strona muchy ma mniejsze x → u góry
        ok &= (px >= 0) & (px < W) & (py >= top) & (py < MAP_H - 30)
        self._ok = np.flatnonzero(ok)
        self._pix = py[ok].astype(int) * W + px[ok].astype(int)
        density = np.bincount(self._pix, minlength=W * MAP_H).astype(np.float32)
        self._count = np.maximum(density, 1.0)
        dens = (np.log1p(density) / np.log1p(density.max()))[:, None]
        self._base = (np.array(BG, np.float32) + dens * np.array([52, 52, 60], np.float32)).astype(np.float32)  # anatomia
        self._occ = np.flatnonzero(density > 0)  # tylko te piksele zmieniają kolor
        self._base_img = np.clip(self._base, 0, 255).astype(np.uint8).reshape(MAP_H, W, 3)
        self._side_y = (top, MAP_H - 30)
        groups = np.asarray(c.groups)
        flight = np.flatnonzero((groups != "") & ok)
        self._flight = flight
        self._flight_xy = np.stack([px[flight], py[flight]], 1).astype(int)
        self._flight_col = np.array([GROUP_COLOR.get(g.rsplit("_", 1)[0], DIM) for g in groups[flight]], np.float32)
        self._group_idx = {g: c.group_indices(g) for _, base in SIGNAL_ROWS for g in (f"{base}_L", f"{base}_R")}
        self._static = self._static_layer()  # napisy, legenda i tła pasków rysowane raz (tekst PIL jest drogi)
        self.reset()

    def _static_layer(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        img = np.empty((H, W, 3), np.uint8)
        img[:] = BG
        img[:MAP_H] = self._base_img
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        self._header(d, None, None, static=True)
        self._signal(d, None, MAP_H + 12, static=True)
        out = np.asarray(im).copy()
        self._base_img = out[:MAP_H].copy()  # mapa bez sygnału, z napisami
        return out

    def reset(self) -> None:
        """Nowe odniesienia (na początku epizodu)."""
        self._ref = None
        self._norm = None
        self._group_ref = None
        self._group_peak = {}

    # --- rysowanie ---
    def render(self, rates: np.ndarray, cmd, ms: float | None = None, steer: dict | None = None) -> np.ndarray:
        """``rates``: aktywność wszystkich neuronów (kolejność connectomu), ``cmd``: FlightCommand (konwencja BANC),
        ``steer``: opcjonalnie ``steer_from_decoder(...)`` — skąd jest kurs."""
        from PIL import Image, ImageDraw

        img = self._static.copy()
        img[:MAP_H] = self._map(rates)
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        self._header(d, ms, steer)
        y = self._signal(d, rates, MAP_H + 12)
        self._steering(d, cmd, steer or {}, y + 10)
        return np.asarray(im)

    def _map(self, rates: np.ndarray) -> np.ndarray:
        rates = rates.astype(np.float32, copy=False)
        if self._ref is None:
            self._ref = rates.copy()
        # siła sygnału neuronu: względna zmiana aktywności wobec średniej z ~2 s; skala mapy = bieżący 99. percentyl
        # (wygładzony), więc najjaśniejsze miejsca to zawsze najsilniejsze zmiany w tej chwili
        rel = np.abs(rates - self._ref) / (np.abs(self._ref) + 1e-9)
        self._ref += EMA * (rates - self._ref)
        val = np.bincount(self._pix, weights=rel[self._ok], minlength=W * MAP_H)[self._occ] / self._count[self._occ]
        k = int(0.99 * (len(val) - 1))
        p99 = max(float(np.partition(val, k)[k]), 1e-6)
        self._norm = p99 if self._norm is None else self._norm + NORM_EMA * (p99 - self._norm)
        out = self._base_img.copy()
        flat = out.reshape(-1, 3)
        flat[self._occ] = np.clip(_ramp(self._base[self._occ], np.minimum(val / self._norm, 1.0)[:, None]), 0, 255)
        a = np.minimum(rel / self._norm, 1.0)
        # neurony lotu: kropki 2×2 w kolorze grupy, jaśniejsze przy silnym sygnale
        col = np.clip(self._flight_col * (0.45 + 0.55 * a[self._flight][:, None]), 0, 255).astype(np.uint8)
        x, y = self._flight_xy[:, 0], self._flight_xy[:, 1]
        for dx in (0, 1):
            for dy in (0, 1):
                out[np.clip(y + dy, 0, MAP_H - 1), np.clip(x + dx, 0, W - 1)] = col
        return out

    def _header(self, d, ms: float | None, steer: dict | None, static: bool = False) -> None:
        info = f"{self.n:,} neurons"
        if not static:
            if ms is not None:
                d.text((106 + d.textlength(info, font=_font(11)), 12), f"  ·  {ms:.0f} ms/frame", font=_font(11), fill=DIM)
        else:
            d.text((14, 9), "BANC v888", font=_font(15, True), fill=TEXT)
            d.text((106, 12), info, font=_font(11), fill=DIM)
        mode = (steer or {}).get("mode", "")
        if mode:
            label = "BANC FLIES" if mode == "flying" else "OBSERVING"
            col = RED if mode == "flying" else DIM
            tw = d.textlength(label, font=_font(10, True))
            d.rounded_rectangle([W - 24 - tw, 8, W - 10, 26], radius=9, outline=col, width=1)
            d.text((W - 17 - tw, 11), label, font=_font(10, True), fill=col)
        if not static:
            return
        f = _font(10)
        d.text((6, self._side_y[0]), "R", font=f, fill=FAINT)
        d.text((6, self._side_y[1] - 12), "L", font=f, fill=FAINT)
        d.text((22, self._side_y[1] - 12), "brain", font=f, fill=FAINT)
        d.text((W - 34, self._side_y[1] - 12), "VNC", font=f, fill=FAINT)
        # legenda
        y = MAP_H - 20
        d.text((14, y), "Signal strength", font=f, fill=DIM)
        ramp = _ramp(np.array([[[30.0, 30, 34]]]), np.linspace(0, 1, 80)[None, :, None])[0]
        for k, c in enumerate(ramp):
            d.line([(100 + k, y + 3), (100 + k, y + 11)], fill=tuple(int(v) for v in c))
        d.text((186, y), "now vs. 2 s ago", font=f, fill=FAINT)
        x = 290
        for lab, col in (("DN", AMBER), ("MN", PINK), ("halt.", BLUE)):
            d.ellipse([x, y + 4, x + 7, y + 11], fill=col)
            d.text((x + 10, y), lab, font=f, fill=FAINT)
            x += 36
        d.line([(0, MAP_H - 1), (W, MAP_H - 1)], fill=LINE)

    def _signal(self, d, rates: np.ndarray | None, y: int, static: bool = False) -> int:
        """``static``: tylko napisy i tła (raz), inaczej tylko paski i liczby."""
        f = _font(11)
        cx, half = 262, 118  # oś: lewa strona muchy | prawa
        if static:
            d.text((14, y), "SIGNAL STRENGTH", font=_font(10, True), fill=RED)
            d.text((cx - 4 - d.textlength("fly's left", font=_font(10)), y), "fly's left", font=_font(10), fill=FAINT)
            d.text((cx + 4, y), "fly's right", font=_font(10), fill=FAINT)
            d.text((14, y + 16), "change since episode start", font=_font(10), fill=FAINT)
            y0 = y = y + 32
            for label, _ in SIGNAL_ROWS:
                d.text((14, y), label, font=f, fill=TEXT)
                d.rectangle([cx - 2 - half, y + 2, cx - 2, y + 13], fill=CARD)
                d.rectangle([cx + 2, y + 2, cx + 2 + half, y + 13], fill=CARD)
                y += 19
            d.line([(cx, y0 - 2), (cx, y)], fill=FAINT)
            d.text((14, y + 1), "bright bar = more active, dark bar = less active; full bar = 10 %", font=_font(10), fill=FAINT)
            return y + 18
        means = {g: float(rates[i].mean()) if len(i) else 0.0 for g, i in self._group_idx.items()}
        if self._group_ref is None:
            self._group_ref = dict(means)
        for g, m in means.items():  # szczyt grupy: mianownik, gdy na starcie grupa milczy (np. haltery bez obrotu)
            base = g.rsplit("_", 1)[0]
            self._group_peak[base] = max(self._group_peak.get(base, 0.0), abs(m))
        y += 32
        for label, base in SIGNAL_ROWS:
            col = GROUP_COLOR[base]
            for side, sgn in (("L", -1), ("R", 1)):
                g = f"{base}_{side}"
                ref = self._group_ref[g]
                rel = (means[g] - ref) / max(abs(ref), 0.1 * self._group_peak[base], 1e-12)
                v = float(np.clip(abs(rel) / BAR_FULL, 0, 1))
                x0, x1 = (cx - 2 - half, cx - 2) if sgn < 0 else (cx + 2, cx + 2 + half)
                n = int(v * half)
                if n:
                    bar = [cx - 2 - n, y + 2, cx - 2, y + 13] if sgn < 0 else [cx + 2, y + 2, cx + 2 + n, y + 13]
                    d.rectangle(bar, fill=col if rel > 0 else tuple(int(c * 0.4) for c in col))
                txt = f"{100 * rel:+.1f}%" if abs(rel) < 9.995 else (">+999%" if rel > 0 else "<−999%")
                tw = d.textlength(txt, font=_font(10))
                tx = x0 + 4 if sgn < 0 else x1 - tw - 4
                d.text((tx, y + 1), txt, font=_font(10), fill=TEXT if v > 0.2 else DIM)
            y += 19
        d.line([(cx, y - 19 * len(SIGNAL_ROWS) - 2), (cx, y)], fill=FAINT)
        return y + 18

    def _steering(self, d, cmd, steer: dict, y: int) -> None:
        f = _font(11)
        x0, x1 = 120, W - 14
        d.line([(0, y - 4), (W, y - 4)], fill=LINE)
        y += 4
        d.text((14, y), "HOW THE DRONE IS STEERED", font=_font(10, True), fill=RED)
        y += 20
        yaw = float(np.clip(cmd.yaw, -1, 1))
        mode = steer.get("mode", "")
        if mode == "observing":
            d.text((14, y), "Keyboard flies; the network only proposes a heading.", font=f, fill=DIM)
            y += 22
        elif "w_vis" in steer and steer.get("yaw_gps") is not None:
            w_vis = float(steer["w_vis"])
            d.text((14, y), "Heading from", font=f, fill=TEXT)
            split = int(x0 + (x1 - x0) * w_vis)
            d.rectangle([x0, y + 1, x1, y + 16], fill=CARD)
            if split > x0:
                d.rectangle([x0, y + 1, split, y + 16], fill=RED)
            if split < x1:
                d.rectangle([split, y + 1, x1, y + 16], fill=BLUE)
            fb = _font(10, True)
            d.text((x0 + 5, y + 2), f"BANC {100 * w_vis:.0f}%", font=fb, fill=TEXT)
            lg = f"GPS {100 * (1 - w_vis):.0f}%"
            d.text((x1 - 5 - d.textlength(lg, font=fb), y + 2), lg, font=fb, fill=TEXT)
            y += 20
            dist = steer.get("distance")
            if dist is not None and np.isfinite(dist):
                d.text((x0, y), f"target {dist:.1f} m away (GPS) · near the target BANC decides", font=_font(10), fill=FAINT)
                y += 18
        if "push_left" in steer:
            pl, pr = steer["push_left"], steer["push_right"]
            tot = pl + pr
            d.text((14, y), "Neuron votes", font=f, fill=TEXT)
            d.rectangle([x0, y + 1, x1, y + 16], fill=CARD)
            if tot > 1e-9:
                mid = int(x0 + (x1 - x0) * pl / tot)
                d.rectangle([x0, y + 1, mid, y + 16], fill=(110, 70, 22))
                d.rectangle([mid, y + 1, x1, y + 16], fill=AMBER)
            fb = _font(10, True)
            d.text((x0 + 5, y + 2), f"left {pl:.2f}", font=fb, fill=TEXT)
            rt = f"right {pr:.2f}"
            d.text((x1 - 5 - d.textlength(rt, font=fb), y + 2), rt, font=fb, fill=BG)
            y += 20
            d.text((x0, y), f"{steer.get('n_votes', 0)} flight neurons × decoder weights", font=_font(10), fill=FAINT)
            y += 18
        rows = []
        if "yaw_banc" in steer:
            rows.append(("Yaw from BANC", steer["yaw_banc"], AMBER))
        if steer.get("yaw_gps") is not None and mode != "observing":
            rows.append(("Yaw from GPS", steer["yaw_gps"], BLUE))
        rows.append(("BANC proposes" if mode == "observing" else "Yaw command", yaw, RED))
        for label, v, col in rows:
            d.text((14, y), label, font=f, fill=TEXT)
            _cbar(d, x0, y + 1, x1 - x0 - 50, 15, float(np.clip(v, -1, 1)), col)
            d.text((x1 - 40, y), f"{v:+.2f}", font=f, fill=TEXT)
            y += 20
        turn = "TURN RIGHT" if yaw > 0.15 else "TURN LEFT" if yaw < -0.15 else "STRAIGHT"
        d.text((x0, y), turn, font=_font(10, True), fill=TEXT)
        d.text((x0 + 80, y), "−1 = full left, +1 = full right", font=_font(10), fill=FAINT)
        y += 22
        d.text((14, y), "Thrust", font=f, fill=TEXT)
        _pbar(d, x0, y + 1, 110, 15, float(np.clip(cmd.thrust, 0, 1)), PINK)
        d.text((x0 + 118, y), f"{cmd.thrust:.2f}", font=f, fill=TEXT)
        d.text((x0 + 160, y + 1), f"roll {cmd.roll:+.2f}   pitch {cmd.pitch:+.2f}", font=_font(10), fill=DIM)
        y += 20
        src = "BANC + drone sensors" if steer.get("yaw_gps") is not None else "the decoder"
        d.text((x0, y), f"from {src} · 0.5 thrust ≈ hover", font=_font(10), fill=FAINT)


def _ramp(base: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Jedna barwa: tło → teal → (przy najsilniejszym sygnale) jaśniej, bez mieszania dwóch kolorów."""
    v = np.asarray(v, np.float32)
    out = base * (1 - v) + np.array(TEAL, np.float32) * v
    return out + (255 - out) * np.clip((v - 0.7) / 0.3, 0, 1) * 0.55


def _cbar(d, x: int, y: int, w: int, h: int, v: float, col) -> None:
    """Pasek od środka: w prawo dla v > 0, w lewo dla v < 0 (jeden kolor)."""
    d.rectangle([x, y, x + w, y + h], fill=CARD)
    mid = x + w // 2
    end = int(mid + v * (w // 2))
    if end != mid:
        d.rectangle([min(mid, end), y + 1, max(mid, end), y + h - 1], fill=col)
    d.line([(mid, y - 2), (mid, y + h + 2)], fill=FAINT)


def _pbar(d, x: int, y: int, w: int, h: int, v: float, col) -> None:
    """Pasek od zera do v (0..1) ze znacznikiem zawisu (0.5)."""
    d.rectangle([x, y, x + w, y + h], fill=CARD)
    if v > 0:
        d.rectangle([x, y + 1, x + int(v * w), y + h - 1], fill=col)
    d.line([(x + w // 2, y - 2), (x + w // 2, y + h + 2)], fill=FAINT)
