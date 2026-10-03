"""Serwer wzroku ZMQ dla symulatora Osoby 3 (protokół: visual_pipeline/zmq_protocol.py).

    python scripts/vision_server.py                      # tryb command: wzrok + BancController
    python scripts/vision_server.py --mode activity      # sama aktywność BANC (dla Osoby 2)
    python scripts/vision_server.py --no-fisheye         # klatki już równokątne (FakeStereoCamera)
    python scripts/vision_server.py --decoder data/decoders/planB.npz   # wagi z train_decoder.py

Tryb command na starcie kalibruje kontroler na syntetycznych scenach (FakeStereoCamera: cel na
wprost i ±60°, obrót przy pasach). To tylko punkt startowy: symulator powinien od razu wysłać
sceny ze swojego renderera przez ``VisionClient.calibrate`` (jasność sceny MuJoCo jest inna,
bez tego thrust stał na 0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visual_pipeline import VisionBridge  # noqa: E402
from visual_pipeline.server import ControlServer, calibrate_controller  # noqa: E402
from visual_pipeline.zmq_protocol import DEFAULT_ADDRESS, decode  # noqa: E402


def synthetic_scenes(bridge: VisionBridge, fps: float, turn_deg_s: float = 90.0, frames: int = 45) -> dict:
    from visual_pipeline.fake_camera import FakeStereoCamera

    cam = FakeStereoCamera(beacon_azimuth=0.0)
    bars = [FakeStereoCamera(np.deg2rad(a)) for a in range(0, 360, 30)]

    def bars_at(yaw):
        f = [c.render(yaw) for c in bars]
        return np.minimum.reduce([x[0] for x in f]), np.minimum.reduce([x[1] for x in f])

    def turn(rate_deg):
        bridge.settle(*bars_at(0.0))
        return [bridge.step_batch(*bars_at(np.deg2rad(rate_deg) / fps * k)) for k in range(frames)]

    scenes = {name: [bridge.settle(*cam.render(yaw=-np.deg2rad(b)))]
              for name, b in (("neutral", 0), ("left", -60), ("right", 60))}
    scenes["turn_right"], scenes["turn_left"] = turn(turn_deg_s), turn(-turn_deg_s)
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDRESS.replace("127.0.0.1", "*"))
    ap.add_argument("--mode", choices=("command", "activity"), default="command")
    ap.add_argument("--no-fisheye", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--decoder", type=Path, help="wagi dekodera z train_decoder.py (po każdej kalibracji)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    bridge = VisionBridge(fps=args.fps, fisheye=not args.no_fisheye)
    ctrl = None
    if args.mode == "command":
        from banc_control import BancController, Connectome

        ctrl = BancController(Connectome.from_banc())
        fisheye, bridge.fisheye = bridge.fisheye, False  # syntetyczne klatki są już równokątne
        try:
            print("kalibracja syntetyczna:", calibrate_controller(ctrl, synthetic_scenes(bridge, args.fps)),
                  flush=True)
        finally:
            bridge.fisheye = fisheye
    if args.decoder and ctrl is not None:
        ctrl.decoder.load_weights(args.decoder)
    server = ControlServer(bridge, ctrl, decoder_weights=args.decoder)
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(args.address)
    print(f"serwer wzroku ({args.mode}) gotowy na {args.address} po {time.perf_counter() - t0:.0f} s, "
          f"{len(bridge.root_ids)} neuronów BANC", flush=True)

    bridge.reset()
    n = 0
    while True:
        header, left, right = decode(sock.recv_multipart())
        try:
            reply, extra = server.handle(header, left, right)
        except Exception as e:  # klient REQ czeka na odpowiedź; błąd zamiast zawieszenia
            reply, extra = {"error": f"{type(e).__name__}: {e}"}, []
        sock.send_multipart([json.dumps(reply).encode(), *extra])
        if header.get("calib") == "finish":
            print("kalibracja na scenach symulatora:", reply, flush=True)
        n += 1
        if n % 300 == 0 and "timing_ms" in reply:
            print(f"{n} klatek, ostatnia {reply['timing_ms']['total']} ms", flush=True)


if __name__ == "__main__":
    main()
