"""Map flygym Retina output (721 ommatidia, flygym ID order) onto FlyVis hexals (u, v order).

flygym's compound-eye map is a flat-top hex lattice drawn in image pixels; FlyVis uses
axial hex coordinates (u, v) with the same flat-top orientation (hex_to_pixel "default").
We take each ommatidium's pixel centroid, rescale to FlyVis units and round to (u, v).

Orientation is verified against FlyVis' training renderer (BoxEye) in
scripts/check_retina_orientation.py: r = 0.996 for this mapping vs <= 0.16 for the other
11 lattice symmetries. The left eye is the horizontal mirror of the right (bilateral symmetry).
"""
import numpy as np
from flygym import assets_dir
from flyvis.utils.hex_utils import get_hex_coords, pixel_to_hex

EXTENT = 15


def ommatidia_centroids():
    id_map = np.load(assets_dir / "model/neuromechfly/compound_eye.npz")["ommatidia_id_map"]
    rows, cols = np.nonzero(id_map)
    ids = id_map[rows, cols]
    counts = np.bincount(ids)[1:]
    row_c = np.bincount(ids, weights=rows)[1:] / counts
    col_c = np.bincount(ids, weights=cols)[1:] / counts
    return col_c, row_c


class RetinaMapper:
    def __init__(self):
        col, row = ommatidia_centroids()
        x, y = col - col.mean(), -(row - row.mean())  # image rows grow downward

        # Nearest-neighbour spacing in FlyVis "default" units is sqrt(3).
        pts = np.stack([x, y], 1)
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        np.fill_diagonal(d, np.inf)
        scale = np.sqrt(3) / np.median(d.min(1))

        self.flygym_to_flyvis_idx = {}
        flyvis_u, flyvis_v = get_hex_coords(EXTENT)
        flyvis_index = {(a, b): i for i, (a, b) in enumerate(zip(flyvis_u, flyvis_v))}

        for eye, sign in (("right", 1.0), ("left", -1.0)):
            u, v = pixel_to_hex(sign * x * scale, y * scale)
            u, v = np.rint(u).astype(int), np.rint(v).astype(int)
            try:
                perm = np.array([flyvis_index[(a, b)] for a, b in zip(u, v)])
            except KeyError as e:
                raise RuntimeError(f"{eye}: ommatidium maps outside FlyVis lattice: {e}")
            if len(set(perm)) != len(perm):
                raise RuntimeError(f"{eye}: two ommatidia map to the same hexal")
            # order[i] = flygym ommatidium index that feeds FlyVis hexal i
            order = np.empty_like(perm)
            order[perm] = np.arange(len(perm))
            self.flygym_to_flyvis_idx[eye] = order

    def to_flyvis(self, hex_pxls: np.ndarray, eye: str) -> np.ndarray:
        """(721, 2) flygym Retina output -> (721,) luminance in FlyVis hexal order.

        Each ommatidium has only one active receptor channel (pale or yellow), so the
        channels are summed to get a single luminance value in [0, 1].
        """
        return hex_pxls.sum(axis=1)[self.flygym_to_flyvis_idx[eye]]
