"""Check mapper orientation against FlyVis' training renderer (BoxEye).

The same smooth random grey images go through (a) flygym Retina + our RetinaMapper and
(b) FlyVis BoxEye. Our hexal vector is then re-indexed by each of the 12 symmetries of
the hex lattice (6 rotations x optional mirror); the correct orientation should give
by far the highest correlation with BoxEye, and it should be the identity.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flygym.vision.retina import Retina
from flyvis.datasets.rendering import BoxEye
from flyvis.utils.hex_utils import get_hex_coords

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visual_pipeline.retina_mapper import EXTENT, RetinaMapper  # noqa: E402
from visual_pipeline.retina_mapper import ommatidia_centroids as _ommatidia_centroids  # noqa: E402

rng = np.random.default_rng(0)
retina, mapper, box = Retina(), RetinaMapper(), BoxEye(extent=EXTENT, kernel_size=13)
H, W = retina.nrows, retina.ncols

# Region of the raw image covered by flygym's ommatidia, to crop for BoxEye.
col, row = _ommatidia_centroids()
r0, r1, c0, c1 = int(row.min()), int(row.max()) + 1, int(col.min()), int(col.max()) + 1
box_h, box_w = (box.min_frame_size + box.kernel_size - 1).cpu().tolist()


def smooth_image():
    low = rng.normal(size=(1, 1, 6, 6))
    img = F.interpolate(torch.tensor(low, dtype=torch.float32), size=(H, W), mode="bicubic")[0, 0].cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return img


def symmetries():
    u, v = get_hex_coords(EXTENT)
    index = {(a, b): i for i, (a, b) in enumerate(zip(u, v))}
    q, r, s = u, v, -u - v
    out = []
    for mirror in (False, True):
        a, b, c = (q, s, r) if mirror else (q, r, s)
        for k in range(6):
            out.append((f"{'mirror+' if mirror else ''}rot{60 * k}", np.array([index[(x, y)] for x, y in zip(a, b)])))
            a, b, c = -b, -c, -a
    return out


syms = symmetries()
corr = {name: [] for name, _ in syms}
for _ in range(20):
    grey = smooth_image()
    ours = mapper.to_flyvis(retina.raw_image_to_hex_pxls(np.repeat((grey * 255).astype(np.uint8)[..., None], 3, 2)), "right")
    crop = torch.tensor(grey[r0:r1, c0:c1], dtype=torch.float32)[None, None]
    crop = F.interpolate(crop, size=(box_h, box_w), mode="bilinear")
    ref = box(crop.to(box.conv.weight.device))[0, 0, 0].cpu().numpy()
    for name, idx in syms:
        moved = np.empty_like(ours)
        moved[idx] = ours
        corr[name].append(np.corrcoef(moved, ref)[0, 1])

for name, c in sorted(corr.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"{name:14s} mean r = {np.mean(c):+.3f}  (min {np.min(c):+.3f})")
