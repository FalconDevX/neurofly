"""Serwer wzroku ZMQ dla symulatora Osoby 3 (protokół: visual_pipeline/zmq_protocol.py).

    python scripts/vision_server.py                      # tryb command: wzrok + BancController
    python scripts/vision_server.py --mode activity      # sama aktywność BANC (dla Osoby 2)
    python scripts/vision_server.py --no-fisheye         # klatki już równokątne (FakeStereoCamera)

Tryb command na starcie kalibruje kontroler na syntetycznych scenach (FakeStereoCamera: cel
na wprost i ±60°). Symulator powinien zaraz potem wysłać żądanie ``calibrate`` ze scenami
z MuJoCo (``VisionClient.calibrate``): jasność i tekstury sceny przesuwają punkt odniesienia
dekodera, więc kalibracja syntetyczna nie pasuje do symulatora (np. thrust stoi na 0).
Każda odpowiedź ma pole ``calibration``: "synthetic" albo "sim".
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
from visual_pipeline.zmq_protocol import (  # noqa: E402
    CALIBRATION_BEARING_DEG,
    CALIBRATION_SCENES,
    DEFAULT_ADDRESS,
    decode_message,
)


def calibrate(ctrl, bridge: VisionBridge, scenes: dict) -> float:
    """Kalibracja jak w docs/osoba2-plan.md: zawis na scenie neutralnej, skala na celu ±60°.

    ``scenes``: {"rest" | "left" | "right": (lewa, prawa)}. Zwraca znak sprzężenia halter.
    """
    responses = {name: bridge.settle(*frames) for name, frames in scenes.items()}
    ctrl.calibrate_rest(30, visual=responses["rest"])
    ctrl.calibrate_scale([responses["left"], responses["right"]])
    sign = ctrl.calibrate_haltere_sign(responses["rest"])
    bridge.reset()
    return sign


def synthetic_scenes() -> dict:
    from visual_pipeline.fake_camera import FakeStereoCamera

    cam = FakeStereoCamera(beacon_azimuth=0.0)
    return {name: cam.render(yaw=-np.deg2rad(CALIBRATION_BEARING_DEG[name])) for name in CALIBRATION_SCENES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDRESS.replace("127.0.0.1", "*"))
    ap.add_argument("--mode", choices=("command", "activity"), default="command")
    ap.add_argument("--no-fisheye", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    bridge = VisionBridge(fps=args.fps, fisheye=not args.no_fisheye)
    ctrl, calibration = None, None
    if args.mode == "command":
        from banc_control import BancController, Connectome

        ctrl = BancController(Connectome.from_banc())
        fisheye, bridge.fisheye = bridge.fisheye, False  # syntetyczne klatki są już równokątne
        try:
            calibrate(ctrl, bridge, synthetic_scenes())
        finally:
            bridge.fisheye = fisheye
        calibration = "synthetic"
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(args.address)
    print(f"serwer wzroku ({args.mode}) gotowy na {args.address} po {time.perf_counter() - t0:.0f} s, "
          f"{len(bridge.root_ids)} neuronów BANC", flush=True)

    bridge.reset()
    n = 0
    while True:
        header, frames = decode_message(sock.recv_multipart())
        if "calibrate" in header:
            if ctrl is None:
                sock.send_string(json.dumps({"error": "kalibracja tylko w trybie command"}))
                continue
            t = time.perf_counter()
            scenes = {name: (frames[2 * i], frames[2 * i + 1]) for i, name in enumerate(header["calibrate"])}
            sign = calibrate(ctrl, bridge, scenes)
            calibration = "sim"
            print(f"kalibracja na scenach z symulatora: {time.perf_counter() - t:.1f} s, "
                  f"znak halter {sign:+.0f}", flush=True)
            sock.send_string(json.dumps({"calibration": calibration, "haltere_sign": sign}))
            continue
        left, right = frames
        imu, reset = header.get("imu"), bool(header.get("reset", False))
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
                                         "calibration": calibration, "timing_ms": timing}))
        n += 1
        if n % 300 == 0:
            print(f"{n} klatek, ostatnia {timing['total']} ms", flush=True)


if __name__ == "__main__":
    main()
