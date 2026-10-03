"""Odruch optomotoryczny całej pętli wzrok → BANC → komenda: wymuszony obrót drona przy
scenie z 12 pionowymi pasami. Stabilizacja = komenda yaw przeciwna do obrotu.

    python scripts/check_optomotor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banc_control import BancController, Connectome  # noqa: E402
from visual_pipeline import VisionBridge  # noqa: E402
from visual_pipeline.fake_camera import FakeStereoCamera  # noqa: E402

FPS = 30


def main() -> None:
    ctrl, bridge = BancController(Connectome.from_banc()), VisionBridge(fps=FPS)
    target = FakeStereoCamera(0.0)
    ctrl.calibrate_rest(30, visual=bridge.settle(*target.render(0)))
    ctrl.calibrate_scale([bridge.settle(*target.render(np.deg2rad(60))),
                          bridge.settle(*target.render(-np.deg2rad(60)))])

    bars = [FakeStereoCamera(np.deg2rad(a)) for a in range(0, 360, 30)]

    def render(yaw):
        frames = [c.render(yaw) for c in bars]
        return np.minimum.reduce([f[0] for f in frames]), np.minimum.reduce([f[1] for f in frames])

    for rate_deg in (-90, 90):
        bridge.settle(*render(0.0))
        ctrl.dyn.reset()
        yaw, cmds = 0.0, []
        for k in range(60):
            yaw += np.deg2rad(rate_deg) / FPS
            cmd = ctrl.step(bridge.step_batch(*render(yaw)))
            if k >= 20:
                cmds.append(cmd.yaw)
        m = np.mean(cmds)
        verdict = "hamuje obrót (stabilizuje)" if np.sign(m) == -np.sign(rate_deg) else "wzmacnia obrót"
        print(f"wymuszony obrót {rate_deg:+d}°/s (+ = w prawo): średni cmd.yaw {m:+.2f} → {verdict}", flush=True)


if __name__ == "__main__":
    main()
