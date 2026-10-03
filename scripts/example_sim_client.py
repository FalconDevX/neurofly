"""Szablon dla Osoby 3: X2 z kamerami-oczami w MuJoCo ↔ serwer wzroku przez ZMQ.

    python scripts/vision_server.py            # osobny terminal / środowisko Osoby 1
    python scripts/example_sim_client.py       # środowisko symulatora (mujoco, numpy, pyzmq)

Wymaga tylko mujoco, numpy i pyzmq (bez torch/FlyVis). Najpierw kalibracja kontrolera na
scenach z MuJoCo (cel na wprost i ±60°), potem dron wisi w miejscu i obraca się wolno
w prawo; pętla wypisuje komendę i opóźnienie. Mixer FlightCommand → silniki X2 to zakres
Osoby 3, tu go nie ma. ``--no-calibrate``: zostaje kalibracja syntetyczna serwera.
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
from visual_pipeline.zmq_protocol import (  # noqa: E402
    CALIBRATION_BEARING_DEG,
    DEFAULT_ADDRESS,
    VisionClient,
)

MENAGERIE = ROOT / "third_party" / "mujoco_menagerie" / "skydio_x2"
BEACON = '    <geom type="cylinder" size="0.15 2" pos="4 0 2" rgba="0.05 0.05 0.05 1"/>'  # azymut 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDRESS)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(x2_with_eyes(MENAGERIE, BEACON)))
    data = mujoco.MjData(model)
    eyes = MujocoEyes(model)
    client = VisionClient(args.address)
    gyro = model.sensor("body_gyro").adr[0]
    accel = model.sensor("body_linacc").adr[0]

    def render(yaw: float):
        """Klatki dla kursu ``yaw`` (rad, + = w prawo); cel stoi na azymucie 0."""
        data.qpos[:3] = (0, 0, 1.0)
        data.qpos[3:7] = (np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2))
        mujoco.mj_forward(model, data)
        return eyes.render(data)

    if not args.no_calibrate:
        t = time.perf_counter()
        # cel pod kątem b względem drona ⇔ kurs drona −b
        scenes = {name: render(-np.deg2rad(b)) for name, b in CALIBRATION_BEARING_DEG.items()}
        reply = client.calibrate(scenes)
        print(f"kalibracja: {reply} ({time.perf_counter() - t:.1f} s)", flush=True)

    yaw = -np.deg2rad(30)  # cel 30° w prawo
    for k in range(args.frames):
        yaw += np.deg2rad(30) / 30  # obrót 30°/s w prawo
        left, right = render(yaw)
        gx, gy, gz = data.sensordata[gyro:gyro + 3]
        # banc_control: yaw + = w prawo; MuJoCo: obrót wokół +z (w górę) + = w lewo.
        # roll się zgadza: obrót wokół +x (przód) opuszcza prawe skrzydło.
        imu = {"gyro": [float(gx), float(gy), float(-gz)],
               "accel": data.sensordata[accel:accel + 3].tolist()}
        t = time.perf_counter()
        reply = client.step(left, right, imu, reset=(k == 0))
        rtt = (time.perf_counter() - t) * 1e3
        if k % 15 == 0:
            print(f"klatka {k:3d} (cel {np.rad2deg(-yaw):+5.0f}°): thrust {reply['thrust']:.2f} "
                  f"roll {reply['roll']:+.2f} yaw {reply['yaw']:+.2f} | kalibracja {reply['calibration']}, "
                  f"serwer {reply['timing_ms']['total']:.1f} ms, z przesyłem {rtt:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
