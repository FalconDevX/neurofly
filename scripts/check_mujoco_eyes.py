"""Test specyfikacji kamer-oczu na prawdziwym renderze MuJoCo (Skydio X2 z mujoco_menagerie).

    python scripts/fetch_menagerie.py
    python scripts/check_mujoco_eyes.py [--preview eyes.png]

Wokół drona stoi 12 ciemnych słupów. Skrypt zapisuje podgląd (surowy kadr, po korekcji
„rybiego oka", widok muchy) i wymusza obrót drona ±90°/s: w BANC prawe oko powinno
aktywować T4b/T5b przy skręcie w prawo, lewe T4a/T5a (jak scripts/check_rotation_banc.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from check_rotation_banc import rotation_check  # noqa: E402
from visual_pipeline import VisionBridge  # noqa: E402
from visual_pipeline.drone_eyes import MujocoEyes, x2_with_eyes  # noqa: E402
from visual_pipeline.frames import to_luminance  # noqa: E402

MENAGERIE = ROOT / "third_party" / "mujoco_menagerie" / "skydio_x2"
POLES = "\n".join(
    f'    <geom type="cylinder" size="0.15 2" pos="{3 * np.cos(a):.3f} {3 * np.sin(a):.3f} 2" rgba="0.05 0.05 0.05 1"/>'
    for a in np.deg2rad(np.arange(0, 360, 30))
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=Path, help="zapisz podgląd oczu do PNG")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(x2_with_eyes(MENAGERIE, POLES)))
    data = mujoco.MjData(model)
    eyes = MujocoEyes(model)
    bridge = VisionBridge(fps=30, fisheye=True)

    def render(yaw: float):
        data.qpos[:3] = (0, 0, 1.0)
        # yaw + = w prawo = obrót wokół +z o −yaw (MuJoCo: +z w górę, +y w lewo)
        data.qpos[3:7] = (np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2))
        mujoco.mj_forward(model, data)
        return eyes.render(data)

    if args.preview:
        import imageio.v3 as iio

        left, right = render(0.0)
        rows = []
        for img in (left, right):
            fish = to_luminance(bridge.retina.correct_fisheye(img))
            fly = bridge.retina.hex_pxls_to_human_readable(
                bridge.retina.raw_image_to_hex_pxls(fish).sum(1, keepdims=True), color_8bit=True)
            rows.append(np.concatenate([img, fish, np.repeat(fly, 3, axis=2)], axis=1))
        iio.imwrite(args.preview, np.concatenate(rows, axis=0))
        print(f"podgląd: {args.preview} (wiersze: lewe, prawe oko; kolumny: surowy, rybie oko, widok muchy)")

    sys.exit(0 if rotation_check(bridge, render) else 1)


if __name__ == "__main__":
    main()
