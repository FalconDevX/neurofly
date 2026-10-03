"""Szablon dla Osoby 3: X2 z kamerami-oczami w MuJoCo ↔ serwer wzroku przez ZMQ.

    python scripts/vision_server.py            # osobny terminal / środowisko Osoby 1
    python scripts/example_sim_client.py       # środowisko symulatora (mujoco, numpy, pyzmq)

Wymaga tylko mujoco, numpy i pyzmq (bez torch/FlyVis). Najpierw kalibracja kontrolera na
scenach z MuJoCo (``VisionClient.calibrate``), potem dron wisi w miejscu i obraca się wolno
w prawo; pętla wypisuje komendę i opóźnienie. Mixer FlightCommand → silniki X2 to
zakres Osoby 3, tu go nie ma.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from visual_pipeline.drone_eyes import MujocoEyes, x2_with_eyes  # noqa: E402
from visual_pipeline.zmq_protocol import DEFAULT_ADDRESS, VisionClient  # noqa: E402

MENAGERIE = ROOT / "third_party" / "mujoco_menagerie" / "skydio_x2"
BEACON_XY = (4.0, -1.5)
BEACON = (f'    <geom type="cylinder" size="0.15 2" pos="{BEACON_XY[0]} {BEACON_XY[1]} 2" '
          'rgba="0.05 0.05 0.05 1"/>')
BEACON_HEADING = float(np.arctan2(-BEACON_XY[1], BEACON_XY[0]))  # kurs na cel, + = w prawo


def set_pose(data, heading: float) -> None:
    """Zawis na 1 m, kurs ``heading`` [rad], + = w prawo (MuJoCo: obrót wokół +z, + = w lewo)."""
    data.qpos[:3] = (0, 0, 1.0)
    data.qpos[3:7] = (np.cos(-heading / 2), 0, 0, np.sin(-heading / 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDRESS)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--no-calib", action="store_true", help="zostaw kalibrację syntetyczną serwera")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(x2_with_eyes(MENAGERIE, BEACON)))
    data = mujoco.MjData(model)
    eyes = MujocoEyes(model)
    client = VisionClient(args.address)
    gyro = model.sensor("body_gyro").adr[0]
    accel = model.sensor("body_linacc").adr[0]

    def render(bearing: float):
        """Klatki, gdy cel jest pod kątem ``bearing`` (+ = w prawo) od kierunku lotu."""
        set_pose(data, BEACON_HEADING - bearing)
        mujoco.mj_forward(model, data)
        return eyes.render(data)

    if not args.no_calib:
        t = time.perf_counter()
        print("kalibracja:", client.calibrate(render), f"({time.perf_counter() - t:.0f} s)", flush=True)

    yaw = 0.0
    for k in range(args.frames):
        yaw += np.deg2rad(30) / 30  # 30°/s w prawo
        set_pose(data, yaw)
        mujoco.mj_forward(model, data)
        left, right = eyes.render(data)
        gx, gy, gz = data.sensordata[gyro:gyro + 3]
        # banc_control: yaw + = w prawo; MuJoCo: obrót wokół +z (w górę) + = w lewo.
        # roll się zgadza: obrót wokół +x (przód) opuszcza prawe skrzydło.
        imu = {"gyro": [float(gx), float(gy), float(-gz)],
               "accel": data.sensordata[accel:accel + 3].tolist()}
        t = time.perf_counter()
        reply = client.step(left, right, imu, reset=(k == 0))
        rtt = (time.perf_counter() - t) * 1e3
        if k % 15 == 0:
            print(f"klatka {k:3d}: thrust {reply['thrust']:.2f} roll {reply['roll']:+.2f} "
                  f"yaw {reply['yaw']:+.2f} | serwer {reply['timing_ms']['total']:.1f} ms, "
                  f"z przesyłem {rtt:.1f} ms, unmatched {reply['unmatched_ids']}", flush=True)


if __name__ == "__main__":
    main()
