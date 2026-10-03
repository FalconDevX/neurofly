"""Panel „mózg” w podglądzie (``python -m sim.viewer --brain``): oczy drona → FlyVis → BANC v888 na żywo.

Sieć tylko obserwuje lot (sterujesz klawiszami), nie steruje dronem. W osobnym wątku:
klatki oczu z ``EyesWorker`` → ``VisionBridge`` → ``BancController.step`` (+ żyroskop drona na
aferenty halter) → obraz panelu:

- góra: widok z przodu na wszystkie neurony BANC (somy z meta ``position``; mózg u góry, VNC niżej);
  szare tło = gęstość neuronów, czerwony / niebieski = wzrost / spadek aktywności względem
  powolnej średniej (kilka sekund), neurony grup lotu narysowane grubiej,
- dół: grupy lotu L / P (oficjalne adnotacje BANC, jak w ``banc_control``) — zmiana średniej
  aktywności w %, oraz komendy z dekodera (yaw, thrust) — „co BANC kazałby zrobić”.

Na starcie kalibracja kontrolera na scenach ``DroneEnv`` (jak w ``fly_banc.py``, ~20 s), opcjonalnie
wagi dekodera (``--brain-decoder``, np. z ``train_decoder.py --readout dn``). Wymaga środowiska
wzroku (.venv312: torch, flyvis) i danych ``data/banc_888``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

VOXEL_NM = np.array([4.0, 4.0, 45.0])
BG = np.array([9, 9, 11], np.float32)  # tło panelu (RGB), zinc-950 jak w eksploratorze
UP = np.array([251, 113, 133], np.float32)    # rose-400: wzrost aktywności
DOWN = np.array([96, 165, 250], np.float32)   # blue-400: spadek
BARS = (  # (etykieta, grupa bez _L/_R)
    ("wzrok (VPN)", "visual"),
    ("DN sterujace", "dn_flight_steering"),
    ("DN mocy", "dn_flight_power"),
    ("MN mocy", "wing_power"),
    ("MN sterujace", "wing_steering"),
    ("MN napiecia", "wing_tension"),
    ("haltery", "haltere_aff"),
)
SCATTER_W, SCATTER_H = 360, 430  # rozdzielczość widoku neuronów (potem skalowany do panelu)
BARS_H = 230
EMA = 0.03       # średnia odniesienia aktywności (~1 s przy 30 Hz)
REL_FULL = 0.05  # względna zmiana aktywności neuronu dająca pełny kolor
BAR_FULL = 0.05  # względna zmiana średniej grupy dająca pełny pasek


class BrainPanel:
    def __init__(self, decoder_path: Path | None = None, readout: str | None = None) -> None:
        self.decoder_path, self.readout = decoder_path, readout
        self.status = "ładowanie BANC v888…"
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

        self.status = "kalibracja na scenach DroneEnv…"
        env = DroneEnv()  # osobny model tylko do scen kalibracyjnych (jak fly_banc.py)
        LocalClient(ControlServer(self.bridge, self.ctrl)).calibrate(env.calibration_render)
        env.close()
        if self.decoder_path is not None:
            self.ctrl.decoder.load_weights(self.decoder_path)

        self.view = BrainView(c)
        self.status = f"BANC v888: {c.n:,} neuronów, odczyt {self.ctrl.readout}, gotowe po {time.perf_counter() - t0:.0f} s"
        print("panel mózgu:", self.status)

    def _run(self) -> None:
        try:
            self._setup()
        except Exception as e:  # panel nie może wywrócić podglądu
            self.status = f"błąd panelu: {type(e).__name__}: {e}"
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
            self.image = self.view.render(rates, cmd, (time.perf_counter() - t) * 1e3)

    def _placeholder(self) -> np.ndarray:
        img = np.empty((SCATTER_H + BARS_H, SCATTER_W, 3), np.uint8)
        img[:] = BG.astype(np.uint8)
        text = self.status.encode("ascii", "replace").decode()
        cv2.putText(img, text[:48], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (250, 250, 250), 1, cv2.LINE_AA)
        return img


class BrainView:
    """Rysowanie panelu dla connectomu ``c`` (bez wątków): ``render(rates, cmd)`` → obraz RGB.
    Używane przez ``BrainPanel`` (podgląd) i ``scripts/fly_banc.py --brain`` (wideo)."""

    def __init__(self, c) -> None:
        import pandas as pd

        from banc_control.connectome import DEFAULT_DATA_DIR, META_FILE

        # Pozycje som w kolejności neuronów connectomu → piksele widoku z przodu (x w poprzek, y w dół).
        meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE, columns=["banc_888_id", "position"])
        pos = meta.set_index(meta.banc_888_id.astype(np.int64)).position.reindex(c.root_ids)  # w pliku ID to tekst
        xyz = np.array([[float(v) for v in p.split(",")] if isinstance(p, str) else [np.nan] * 3
                        for p in pos]) * VOXEL_NM / 1000.0
        ok = np.isfinite(xyz).all(axis=1)
        lo, hi = np.percentile(xyz[ok], 0.5, axis=0), np.percentile(xyz[ok], 99.5, axis=0)
        s = min((SCATTER_W - 16) / (hi[0] - lo[0]), (SCATTER_H - 16) / (hi[1] - lo[1]))
        px = ((xyz[:, 0] - lo[0]) * s + (SCATTER_W - (hi[0] - lo[0]) * s) / 2)
        py = ((xyz[:, 1] - lo[1]) * s + 8)
        ok &= (px >= 0) & (px < SCATTER_W) & (py >= 0) & (py < SCATTER_H)
        self._ok = np.flatnonzero(ok)
        self._pix = (py[ok].astype(int) * SCATTER_W + px[ok].astype(int))
        density = np.bincount(self._pix, minlength=SCATTER_W * SCATTER_H).astype(np.float32)
        self._count = np.maximum(density, 1.0)
        gray = np.log1p(density) / np.log1p(density.max())
        self._base = (BG + gray[:, None] * np.array([82, 82, 91], np.float32)).reshape(SCATTER_H, SCATTER_W, 3)  # zinc
        flight = np.flatnonzero((c.groups != "") & ok)
        self._flight = flight
        fx, fy = px[flight].astype(int), py[flight].astype(int)
        d = np.arange(-1, 2)  # kropka 3×3 px
        self._flight_yy = np.clip(fy[:, None, None] + d[None, :, None], 0, SCATTER_H - 1)
        self._flight_xx = np.clip(fx[:, None, None] + d[None, None, :], 0, SCATTER_W - 1)
        self._group_idx = {g: c.group_indices(g) for _, base in BARS for g in (f"{base}_L", f"{base}_R")}
        self._ref = None
        self._group_ref = None

    def reset(self) -> None:
        """Nowa średnia odniesienia (np. na początku epizodu)."""
        self._ref = self._group_ref = None

    def render(self, rates: np.ndarray, cmd, ms: float | None = None) -> np.ndarray:
        """``rates``: aktywność wszystkich neuronów (kolejność connectomu), ``cmd``: FlightCommand."""
        if self._ref is None:
            self._ref = rates.copy()
        rel = np.clip((rates - self._ref) / (np.abs(self._ref) + 1e-9), -REL_FULL, REL_FULL) / REL_FULL
        self._ref += EMA * (rates - self._ref)

        val = (np.bincount(self._pix, weights=rel[self._ok], minlength=SCATTER_W * SCATTER_H) / self._count)
        img = _tint(self._base, val.reshape(SCATTER_H, SCATTER_W), strength=4.0)
        img[self._flight_yy, self._flight_xx] = _colors(rel[self._flight])[:, None, None]  # grupy lotu: większe kropki
        scatter = np.clip(img, 0, 255).astype(np.uint8)
        scatter[:24] = BG.astype(np.uint8)  # paski pod napisami, żeby nie nachodziły na neurony
        scatter[SCATTER_H - 22:] = BG.astype(np.uint8)
        scatter[24] = scatter[SCATTER_H - 23] = (39, 39, 42)  # linia zinc-800
        cv2.putText(scatter, "BANC v888 - widok z przodu", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (250, 250, 250), 1, cv2.LINE_AA)
        cv2.putText(scatter, "rozowy = wzrost, niebieski = spadek", (8, SCATTER_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (161, 161, 170), 1, cv2.LINE_AA)
        return np.vstack([scatter, self._bars(rates, cmd, ms)])

    def _bars(self, rates: np.ndarray, cmd, ms: float | None) -> np.ndarray:
        img = np.empty((BARS_H, SCATTER_W, 3), np.uint8)
        img[:] = BG.astype(np.uint8)
        means = {g: float(rates[i].mean()) if len(i) else 0.0 for g, i in self._group_idx.items()}
        if self._group_ref is None:
            self._group_ref = dict(means)
        font, white, dim = cv2.FONT_HERSHEY_SIMPLEX, (250, 250, 250), (161, 161, 170)
        cv2.putText(img, "L", (150, 14), font, 0.4, dim, 1, cv2.LINE_AA)
        cv2.putText(img, "P", (268, 14), font, 0.4, dim, 1, cv2.LINE_AA)
        y = 22
        for label, base in BARS:
            cv2.putText(img, label, (6, y + 10), font, 0.38, white, 1, cv2.LINE_AA)
            for k, side in enumerate("LR"):
                g = f"{base}_{side}"
                ref = self._group_ref[g]
                rel = (means[g] - ref) / (abs(ref) + 1e-12)
                self._group_ref[g] += EMA * (means[g] - ref)
                _hbar(img, 112 + k * 118, y, 110, 12, np.clip(rel / BAR_FULL, -1, 1))
            y += 18
        y += 8
        cv2.putText(img, "dekoder:", (6, y + 10), font, 0.4, white, 1, cv2.LINE_AA)
        cv2.putText(img, "yaw", (80, y + 10), font, 0.38, dim, 1, cv2.LINE_AA)
        _hbar(img, 112, y, 228, 12, float(np.clip(cmd.yaw, -1, 1)))
        y += 18
        cv2.putText(img, "thrust", (68, y + 10), font, 0.38, dim, 1, cv2.LINE_AA)
        _hbar(img, 112, y, 228, 12, float(np.clip(2 * cmd.thrust - 1, -1, 1)))
        turn = "skret w PRAWO" if cmd.yaw > 0.15 else "skret w LEWO" if cmd.yaw < -0.15 else "prosto"
        timing = f"   ({ms:.0f} ms/klatke)" if ms is not None else ""
        cv2.putText(img, f"BANC: {turn}{timing}", (6, BARS_H - 8), font, 0.4, white, 1, cv2.LINE_AA)
        return img


def _colors(v: np.ndarray) -> np.ndarray:
    """Kolory neuronów grup lotu: szary → czerwony (wzrost) / niebieski (spadek)."""
    a = np.minimum(np.abs(v), 1.0)[:, None]
    hot = np.where(v[:, None] > 0, UP, DOWN)
    return np.array([161.0, 161, 170]) * (1 - a) + hot * a


def _tint(base: np.ndarray, val: np.ndarray, strength: float) -> np.ndarray:
    a = np.clip(np.abs(val) * strength, 0, 1)[..., None]
    hot = np.where(val[..., None] > 0, UP, DOWN)
    return base * (1 - a) + hot * a


def _hbar(img: np.ndarray, x: int, y: int, w: int, h: int, v: float) -> None:
    """Pasek od środka: w prawo dla v > 0 (czerwony), w lewo dla v < 0 (niebieski)."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (24, 24, 27), -1)  # zinc-900
    cv2.rectangle(img, (x, y), (x + w, y + h), (39, 39, 42), 1)   # ramka zinc-800
    mid = x + w // 2
    end = int(mid + v * (w // 2))
    if end != mid:
        cv2.rectangle(img, (min(mid, end), y + 1), (max(mid, end), y + h - 1),
                      tuple(int(c) for c in (UP if v > 0 else DOWN)), -1)
    cv2.line(img, (mid, y), (mid, y + h), (82, 82, 91), 1)
