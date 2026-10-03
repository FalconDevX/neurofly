"""Obrazek demo potoku Osoby 1 na renderze MuJoCo: kamera → oko muchy → FlyVis → BANC.

    python scripts/fetch_menagerie.py
    python scripts/demo_figure.py [--out docs/img/osoba1_pipeline.png]

Dron X2 obraca się w prawo (60°/s) wśród ciemnych słupów; po 20 klatkach rysujemy:
kadr prawego oka, to co widzi Retina, odpowiedź FlyVis T4a/T4b na siatce kolumn i aktywność
przypisanych neuronów BANC na pozycjach ich ciał komórek (widok z przodu mózgu).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from banc_control.connectome import DEFAULT_DATA_DIR, META_FILE  # noqa: E402
from check_mujoco_eyes import MENAGERIE, POLES  # noqa: E402
from flyvis.utils.hex_utils import hex_to_pixel  # noqa: E402
from visual_pipeline import VisionBridge  # noqa: E402
from visual_pipeline.drone_eyes import MujocoEyes, x2_with_eyes  # noqa: E402
from visual_pipeline.frames import to_luminance  # noqa: E402

RATE_DEG, FRAMES = 60, 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img" / "osoba1_pipeline.png")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(x2_with_eyes(MENAGERIE, POLES)))
    data = mujoco.MjData(model)
    eyes, bridge = MujocoEyes(model), VisionBridge(fps=30, fisheye=True)

    def render(yaw):
        data.qpos[:3] = (0, 0, 1.0)
        data.qpos[3:7] = (np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2))
        mujoco.mj_forward(model, data)
        return eyes.render(data)

    bridge.settle(*render(0.0))
    yaw = 0.0
    for _ in range(FRAMES):
        yaw += np.deg2rad(RATE_DEG) / 30
        left, right = render(yaw)
        banc_act = bridge.step_arrays(left, right)
    flyvis_act = bridge.stepper.state.nodes.activity.cpu().numpy()[1]  # prawe oko
    types, u, v = bridge.stepper.cell_index()

    fish = to_luminance(bridge.retina.correct_fisheye(right))
    fly = bridge.retina.hex_pxls_to_human_readable(
        bridge.retina.raw_image_to_hex_pxls(fish).sum(1, keepdims=True), color_8bit=True)[..., 0]

    meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE, columns=["banc_888_id", "root_position_nm"])
    meta = meta[meta.root_position_nm.notna() & meta.banc_888_id.notna()]
    meta = meta.assign(banc_888_id=meta.banc_888_id.astype("int64")).drop_duplicates("banc_888_id").set_index("banc_888_id")
    xyz = meta.root_position_nm.reindex(bridge.root_ids).str.split(",", expand=True).astype(float).to_numpy()

    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.6]})
    ax[0].imshow(right)
    ax[0].set_title("1. Kamera MuJoCo (prawe oko)")
    ax[1].imshow(fly, cmap="gray")
    ax[1].set_title("2. Retina: 721 omatidiów")
    for k, t in enumerate(("T4a", "T4b")):
        m = types == t
        x, y = hex_to_pixel(u[m], v[m])
        lim = np.abs(flyvis_act[m]).max()
        ax[2 + k].scatter(x, y, c=flyvis_act[m], cmap="RdBu_r", vmin=-lim, vmax=lim, s=14, marker="h")
        ax[2 + k].set_aspect("equal")
        ax[2 + k].set_title(f"3. FlyVis {t} ({'przód→tył' if t == 'T4a' else 'tył→przód'})")
    ok = np.isfinite(xyz).all(1)
    lim = np.percentile(np.abs(banc_act[ok]), 98)
    sc = ax[4].scatter(xyz[ok, 0] / 1000, -xyz[ok, 1] / 1000, c=banc_act[ok], cmap="RdBu_r",
                       vmin=-lim, vmax=lim, s=2)
    ax[4].set_aspect("equal")
    ax[4].set_title(f"4. BANC: {ok.sum()} neuronów (pozycje ciał, µm)")
    fig.colorbar(sc, ax=ax[4], fraction=0.04, label="aktywność")
    for a in ax[:4]:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Osoba 1: dron obraca się w prawo {RATE_DEG}°/s; prawe oko widzi ruch tył→przód (T4b). "
                 "Siatka FlyVis jest odbita względem kadru (przód po prawej); skrajne kolumny to efekt brzegu modelu.")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"zapisano {args.out}")


if __name__ == "__main__":
    main()
