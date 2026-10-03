"""Pełna pętla: kamera stereo → VisionBridge (Osoba 1) → BancController (Osoba 2) → ToyDrone.

    python scripts/run_closed_loop.py [--frames 150] [--bearing-deg 30]

Gazebo (Osoba 3) zastępuje FakeStereoCamera + ToyDrone. Kalibracja jak w docs/osoba2-plan.md,
ale na prawdziwych odpowiedziach FlyVis: scena neutralna (cel na wprost) i cel ±60°.
Wypisuje kurs względem celu w czasie oraz czas każdego kroku pętli.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banc_control import BancController, Connectome  # noqa: E402
from banc_control.stubs import ToyDrone  # noqa: E402
from visual_pipeline import VisionBridge  # noqa: E402
from visual_pipeline.fake_camera import FakeStereoCamera  # noqa: E402

FPS = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--bearing-deg", type=float, default=30.0)
    ap.add_argument("--yaw-only", action="store_true",
                    help="dron trzyma poziom (roll = 0), jak quadcopter z własną stabilizacją")
    ap.add_argument("--no-imu", action="store_true",
                    help="bez sprzężenia gyro → haltery (izoluje wpływ samego wzroku)")
    ap.add_argument("--records", action="store_true",
                    help="list[BancActivation] zamiast VisualBatch (wolniej, do porównania)")
    args = ap.parse_args()

    t = time.perf_counter()
    c = Connectome.from_banc()
    ctrl = BancController(c)
    bridge = VisionBridge(fps=FPS)
    cam = FakeStereoCamera(beacon_azimuth=0.0)
    print(f"start: {time.perf_counter() - t:.0f} s, BANC {c.n} neuronów, "
          f"wzrok → {len(bridge.root_ids)} neuronów BANC", flush=True)

    def scene(bearing_deg: float):
        return bridge.settle(*cam.render(yaw=-np.deg2rad(bearing_deg)))

    t = time.perf_counter()
    neutral = scene(0)
    ctrl.calibrate_rest(30, visual=neutral)
    ctrl.calibrate_scale([scene(-60), scene(60)])
    sign = ctrl.calibrate_haltere_sign(neutral)
    print(f"kalibracja: {time.perf_counter() - t:.0f} s, znak halter {sign:+.0f}", flush=True)

    # Kontrola w otwartej pętli: czy cel z lewej i z prawej daje przeciwne komendy?
    for b in (-60, 60):
        ctrl.dyn.reset()
        vis = scene(b)
        for _ in range(30):
            cmd = ctrl.step(vis)
        print(f"cel {b:+d}°: roll {cmd.roll:+.2f}  yaw {cmd.yaw:+.2f}  thrust {cmd.thrust:.2f}  "
              f"unmatched_ids {cmd.debug['unmatched_ids']}", flush=True)

    drone = ToyDrone(dt=1 / FPS)
    drone.yaw = -np.deg2rad(args.bearing_deg)  # cel po prawej
    ctrl.dyn.reset()
    bridge.settle(*cam.render(yaw=drone.yaw))  # bez skoku odpowiedzi szare tło → scena
    step = bridge.step if args.records else bridge.step_batch
    imu, log, times = None, [], []
    for k in range(args.frames):
        t0 = time.perf_counter()
        left, right = cam.render(yaw=drone.yaw, roll=drone.roll)
        t1 = time.perf_counter()
        visual = step(left, right)
        t2 = time.perf_counter()
        cmd = ctrl.step(visual, imu).clipped()
        t3 = time.perf_counter()
        if args.yaw_only:
            cmd.roll = 0.0
        imu = drone.step(cmd)
        if args.no_imu:
            imu = None
        times.append({"camera": (t1 - t0) * 1e3, "records": 0.0, **bridge.last_timings,
                      "controller": (t3 - t2) * 1e3,
                      "total": (time.perf_counter() - t0) * 1e3})
        log.append((k / FPS, np.rad2deg(cam.bearing(drone.yaw)), np.rad2deg(drone.roll), cmd.yaw, cmd.roll))
        if k % 15 == 0:
            print(f"t={k / FPS:4.1f} s  cel {log[-1][1]:+6.1f}°  roll {log[-1][2]:+6.1f}°  "
                  f"cmd yaw {cmd.yaw:+.2f} roll {cmd.roll:+.2f}", flush=True)

    keys = list(times[0])
    mean = {k: np.mean([x[k] for x in times[5:]]) for k in keys}  # bez rozgrzewki
    print("\nśredni czas kroku [ms]: " + "  ".join(f"{k} {v:.1f}" for k, v in mean.items()))
    print(f"budżet {1e3 / FPS:.0f} ms → {'OK' if mean['total'] <= 1e3 / FPS else 'ZA WOLNO'} "
          f"({1e3 / mean['total']:.1f} FPS)")
    print(f"kąt do celu: start {log[0][1]:+.1f}°, koniec {log[-1][1]:+.1f}°")


if __name__ == "__main__":
    main()
