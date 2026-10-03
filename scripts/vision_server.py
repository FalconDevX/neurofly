"""Serwer wzroku ZMQ dla symulatora Osoby 3 (protokół: visual_pipeline/zmq_protocol.py).

    python scripts/vision_server.py                      # tryb command: wzrok + BancController
    python scripts/vision_server.py --mode activity      # sama aktywność BANC (dla Osoby 2)
    python scripts/vision_server.py --no-fisheye         # klatki już równokątne (FakeStereoCamera)

Tryb command kalibruje kontroler na syntetycznych scenach (FakeStereoCamera: cel na wprost
i ±60°), bo serwer nie ma dostępu do renderera symulatora. To rozwiązanie tymczasowe.
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
from banc_control import ImuState  # noqa: E402
from visual_pipeline import VisionBridge  # noqa: E402
from visual_pipeline.zmq_protocol import DEFAULT_ADDRESS, decode_request  # noqa: E402


def calibrated_controller(bridge: VisionBridge):
    from banc_control import BancController, Connectome
    from visual_pipeline.fake_camera import FakeStereoCamera

    ctrl = BancController(Connectome.from_banc())
    cam = FakeStereoCamera(beacon_azimuth=0.0)
    fisheye, bridge.fisheye = bridge.fisheye, False  # syntetyczne klatki są już równokątne
    try:
        def scene(deg):
            return bridge.settle(*cam.render(yaw=-np.deg2rad(deg)))

        neutral = scene(0)
        ctrl.calibrate_rest(30, visual=neutral)
        ctrl.calibrate_scale([scene(-60), scene(60)])
        ctrl.calibrate_haltere_sign(neutral)
    finally:
        bridge.fisheye = fisheye
    return ctrl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDRESS.replace("127.0.0.1", "*"))
    ap.add_argument("--mode", choices=("command", "activity"), default="command")
    ap.add_argument("--no-fisheye", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    bridge = VisionBridge(fps=args.fps, fisheye=not args.no_fisheye)
    ctrl = calibrated_controller(bridge) if args.mode == "command" else None
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(args.address)
    print(f"serwer wzroku ({args.mode}) gotowy na {args.address} po {time.perf_counter() - t0:.0f} s, "
          f"{len(bridge.root_ids)} neuronów BANC", flush=True)

    bridge.reset()
    n = 0
    while True:
        left, right, imu, reset = decode_request(sock.recv_multipart())
        t = time.perf_counter()
        if reset:
            bridge.reset()
            if ctrl is not None:
                ctrl.dyn.reset()
        batch = bridge.step_batch(left, right)
        timing = {k: round(v, 2) for k, v in bridge.last_timings.items()}
        if ctrl is None:
            timing["total"] = round((time.perf_counter() - t) * 1e3, 2)
            sock.send_multipart([json.dumps({"n": len(batch.root_ids), "timing_ms": timing}).encode(),
                                 batch.root_ids.astype(np.int64).tobytes(),
                                 batch.activity.astype(np.float32).tobytes()])
        else:
            state = ImuState(**{k: tuple(v) for k, v in imu.items()}) if imu else None
            cmd = ctrl.step(batch, state).clipped()
            timing["total"] = round((time.perf_counter() - t) * 1e3, 2)
            sock.send_string(json.dumps({"thrust": cmd.thrust, "roll": cmd.roll, "pitch": cmd.pitch,
                                         "yaw": cmd.yaw, "unmatched_ids": ctrl.unmatched_ids,
                                         "timing_ms": timing}))
        n += 1
        if n % 300 == 0:
            print(f"{n} klatek, ostatnia {timing['total']} ms", flush=True)


if __name__ == "__main__":
    main()
