"""Column position of a BANC neuron from its skeleton: mean of the distal arbor.

Soma location comes from the metadata; nodes farther from the soma than half the maximum
distance are taken as the arbor (for medulla columnar cells: the part inside the column).
Somata are stacked several deep in the cortex, so they place cells ~2.7x worse than this.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def arbor_point(path: Path, soma_um: np.ndarray):
    if not path.exists():
        return None
    xyz = np.loadtxt(path, usecols=(2, 3, 4), ndmin=2)
    d = np.linalg.norm(xyz - soma_um, axis=1)
    return xyz[d >= 0.5 * d.max()].mean(axis=0)


def arbor_positions(meta: pd.DataFrame, skel_dir: Path) -> pd.DataFrame:
    """meta: rows with root_id and x, y, z soma position in nm -> DataFrame ax, ay, az in nm."""
    rows = {}
    for r in meta.itertuples():
        p = arbor_point(skel_dir / f"{r.root_id}.swc", np.array([r.x, r.y, r.z]) / 1000)
        if p is not None:
            rows[r.root_id] = p * 1000
    return pd.DataFrame.from_dict(rows, orient="index", columns=["ax", "ay", "az"])
