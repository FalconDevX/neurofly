"""Eksplorator BANC v888 w przeglądarce, serwowany z Pythona (bez npm i builda).

    python scripts/download_banc.py
    python scripts/export_viz_data.py
    python scripts/export_anatomy.py      # wymaga szkieletów SWC w data/banc_888/swc
    python scripts/explorer.py            # http://localhost:8000

Strona to `explorer/` (HTML + JS, three.js z CDN). Dane idą prosto z `data/viz/`:
`/data/banc_anatomy.json` bez zmian, `/data/cmds.json` liczone tu z siatki w `banc_viz.json`
(komendy Planu C dla kierunków beacona z anatomii, bez obrotu: roll_rate = 0).

Model na żywo (domyślnie, gdy jest `data/banc_888/`): pełny BANC v888 w `BancController` na CUDA
(torch; bez CUDA na CPU) liczy się w osobnym wątku, 50 kroków/s (4 podkroki × 5 ms = czas rzeczywisty).
Strona odpytuje `/api/frame?bearing=<deg>&yaw_rate=<rad/s>`: ustawia wejście (FakeVision + IMU) i dostaje
aktywność wszystkich som (uint8, skala log 10⁻⁶..1, kwantyzowana na GPU), neuronów lotu (float32) i komendy.
Wejście wzroku to nadal `FakeVision` (lewa/prawa strona `visual_projection`) — nasze założenie, nie FlyVis.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "explorer"
VIZ = ROOT / "data" / "viz"
VOXEL_NM = np.array([4.0, 4.0, 45.0])  # jak w scripts/export_anatomy.py
CALIB_STEPS = 30  # jak w scripts/export_viz_data.py


def load_data(viz: Path) -> tuple[dict[str, bytes], dict]:
    anatomy, grid_file = viz / "banc_anatomy.json", viz / "banc_viz.json"
    if not anatomy.exists() or not grid_file.exists():
        raise SystemExit(
            f"Brak danych w {viz}. W katalogu repo uruchom:\n"
            "  python scripts/download_banc.py\n  python scripts/export_viz_data.py\n  python scripts/export_anatomy.py"
        )
    raw = anatomy.read_bytes()
    parsed = json.loads(raw)
    bearings = parsed["bearings"]
    grid = json.loads(grid_file.read_text(encoding="utf-8"))["grid"]
    cmds = [next(g["cmd"] for g in grid if g["bearing"] == b and g["roll_rate"] == 0) for b in bearings]
    print(f"dane z {viz} (beacon {', '.join(map(str, bearings))}°)")
    return {"/data/banc_anatomy.json": raw, "/data/cmds.json": json.dumps(cmds).encode()}, parsed


class LiveModel:
    """BancController na pełnym v888 w osobnym wątku; ostatnia klatka gotowa do wysłania."""

    def __init__(self, anatomy: dict, device: str | None, hz: float) -> None:
        self.anatomy, self.device_req, self.hz = anatomy, device, hz
        self.lock = threading.Lock()
        self.bearing, self.yaw_rate = 0.0, 0.0
        self.frame: bytes | None = None
        self.state: dict = {}
        self.info: dict = {"ready": False, "status": "wczytywanie BANC v888…"}

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _setup(self) -> None:
        import pandas as pd

        from banc_control import BancController, Connectome
        from banc_control.connectome import DEFAULT_DATA_DIR, META_FILE, NON_NEURONS
        from banc_control.contracts import VisualBatch
        from banc_control.stubs import FakeVision

        meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE)
        meta = meta[~meta["super_class"].isin(NON_NEURONS)].reset_index(drop=True)
        c = Connectome.from_banc()
        assert len(meta) == c.n
        # te same somy i w tej samej kolejności co w eksporcie anatomii
        xyz = np.array([[float(v) for v in p.split(",")] for p in meta["position"]]) * VOXEL_NM / 1000.0
        lo, hi = np.percentile(xyz, 0.05, axis=0), np.percentile(xyz, 99.95, axis=0)
        self.somas = np.flatnonzero(np.all((xyz >= lo) & (xyz <= hi), axis=1))
        n_somas = len(self.anatomy["somas"]["super_class"])
        if len(self.somas) != n_somas:
            raise RuntimeError(f"somy: {len(self.somas)} w meta vs {n_somas} w banc_anatomy.json — przelicz eksport")
        row = {str(r): i for i, r in enumerate(meta["banc_888_id"])}
        self.flight = np.array([row[str(n["id"])] for n in self.anatomy["neurons"]])

        self.info["status"] = "kalibracja…"
        ctrl = BancController(c, device=self.device_req)
        vision = FakeVision(c)
        ctrl.calibrate_rest(CALIB_STEPS, visual=vision(0.0))
        ctrl.calibrate_scale([vision(np.deg2rad(-60)), vision(np.deg2rad(60))], steps=CALIB_STEPS)
        ctrl.calibrate_haltere_sign(vision(0.0), steps=CALIB_STEPS)
        ctrl.dyn.reset()

        first = vision(0.0)
        ids = np.fromiter((a.banc_root_id for a in first), np.int64, len(first))
        cache: dict[int, VisualBatch] = {}

        def visual(deg: float) -> VisualBatch:  # wejście co 1°; ids to ten sam obiekt (kontroler cache'uje indeksy)
            k = int(round(deg))
            if k not in cache:
                cache[k] = VisualBatch(ids, np.array([a.activity for a in vision(np.deg2rad(k))]))
            return cache[k]

        self.ctrl, self.visual = ctrl, visual
        dev = ctrl.dyn.device
        name = "CPU"
        if dev != "cpu":
            import torch

            name = torch.cuda.get_device_name(torch.device(dev))
        self.info = {"ready": True, "device": dev, "gpu": name, "n": int(c.n), "edges": int(c.W.nnz),
                     "hz": self.hz, "status": "na żywo"}
        print(f"model na żywo: {c.n} neuronów, {c.W.nnz} krawędzi na {dev} ({name})")

    def _run(self) -> None:
        from banc_control.contracts import ImuState

        try:
            self._setup()
        except Exception as e:  # strona dalej działa na wyeksportowanych danych
            self.info = {"ready": False, "status": f"błąd: {e}"}
            print("model na żywo wyłączony:", e)
            return
        period, t_model, ms = 1.0 / self.hz, 0.0, None
        dt_model = self.ctrl.substeps * 0.005  # dt 5 ms na podkrok
        while True:
            t0 = time.perf_counter()
            with self.lock:
                b, yr = self.bearing, self.yaw_rate
            cmd = self.ctrl.step(self.visual(b), ImuState(gyro=(0.0, 0.0, yr)))
            somas = self.ctrl.dyn.levels_at(self.somas)  # kwantyzacja na GPU, 1 B/neuron
            flight = self.ctrl.dyn.rates_at(self.flight).astype(np.float32)
            took = (time.perf_counter() - t0) * 1000.0
            ms = took if ms is None else 0.9 * ms + 0.1 * took
            t_model += dt_model
            state = {"cmd": {k: round(getattr(cmd, k), 4) for k in ("thrust", "roll", "pitch", "yaw")},
                     "bearing": b, "yaw_rate": yr, "ms": round(ms, 2), "t": round(t_model, 2)}
            body = somas.tobytes() + b"\0" * (-len(somas) % 4) + flight.tobytes()
            with self.lock:
                self.frame, self.state = body, state
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def request(self, bearing: float | None, yaw_rate: float | None) -> tuple[bytes, dict] | None:
        with self.lock:
            if bearing is not None:
                self.bearing = float(np.clip(bearing, -90, 90))
            if yaw_rate is not None:
                self.yaw_rate = float(np.clip(yaw_rate, -5, 5))
            return (self.frame, self.state) if self.frame is not None else None


class Handler(SimpleHTTPRequestHandler):
    data: dict[str, bytes] = {}
    live: LiveModel | None = None

    def _send(self, body: bytes, ctype: str, extra: dict[str, str] | None = None, status: int = 200) -> None:
        gz = len(body) > 1024 and "gzip" in self.headers.get("Accept-Encoding", "")
        if gz:
            body = gzip.compress(body, compresslevel=1 if ctype == "application/octet-stream" else 5)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path in self.data:
            return self._send(self.data[url.path], "application/json")
        if url.path == "/api/live":
            info = self.live.info if self.live else {"ready": False, "status": "wyłączony (--no-live)"}
            return self._send(json.dumps(info).encode(), "application/json")
        if url.path == "/api/frame":
            q = {k: float(v[0]) for k, v in parse_qs(url.query).items() if k in ("bearing", "yaw_rate")}
            got = self.live.request(q.get("bearing"), q.get("yaw_rate")) if self.live else None
            if got is None:
                return self._send(b"{}", "application/json", status=503)
            body, state = got
            return self._send(body, "application/octet-stream", {"X-State": json.dumps(state)})
        return super().do_GET()

    def log_message(self, fmt: str, *args) -> None:  # bez logu każdego pliku
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--viz", type=Path, default=VIZ, help="katalog z banc_anatomy.json i banc_viz.json")
    p.add_argument("--no-live", action="store_true", help="bez modelu na żywo, tylko wyeksportowane dane")
    p.add_argument("--device", default=None, help="cuda / cpu (domyślnie cuda, gdy dostępna)")
    p.add_argument("--hz", type=float, default=50.0, help="kroki modelu na sekundę (50 = czas rzeczywisty)")
    p.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    a = p.parse_args()

    Handler.data, anatomy = load_data(a.viz)
    if not a.no_live:
        Handler.live = LiveModel(anatomy, a.device, a.hz)
        Handler.live.start()
    srv = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(STATIC)))
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}/"
    print(f"eksplorator: {url}  (Ctrl+C kończy)")
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
