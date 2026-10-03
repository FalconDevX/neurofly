"""Validate the FlyVis->BANC map: strongly connected columnar partners should land in the
same or a neighbouring hexal. Reports the hex distance between mapped partners, compared
with a shuffled baseline.

Caveat: positions of non-Mi1 types were derived from their partners, so this checks the
map's internal consistency, not global rotation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banc_control.connectome import DEFAULT_DATA_DIR  # noqa: E402

MAP_FILE = Path(__file__).resolve().parent.parent / "visual_pipeline" / "flyvis_banc_map.csv"

PAIRS = [("Mi1", "T4a"), ("Mi1", "T4c"), ("Tm3", "T4b"), ("Mi9", "T4d"), ("Tm1", "T5a"),
         ("Tm9", "T5c"), ("L1", "Mi1"), ("L2", "Tm1"), ("Mi4", "T4a"), ("C3", "Mi1")]

mp = pd.read_csv(MAP_FILE)
mp = mp[~mp.flyvis_type.str.startswith("CT1")]
ed = pd.read_feather(DEFAULT_DATA_DIR / "banc_888_edgelist_simple_v3.feather", columns=["pre", "post", "count"])
ed = ed.astype({"pre": "int64", "post": "int64"})
ed = ed[ed.pre.isin(mp.banc_888_id) & ed.post.isin(mp.banc_888_id) & (ed["count"] >= 5)]


def hexdist(du, dv):
    return (np.abs(du) + np.abs(du + dv) + np.abs(dv)) / 2


rng = np.random.default_rng(0)
for eye in ("right", "left"):
    m = mp[mp.eye == eye].set_index("banc_888_id")[["flyvis_type", "u", "v"]]
    m = m[~m.index.duplicated()]
    print(f"--- {eye} eye: hex distance between mapped synaptic partners (count >= 5) ---")
    for a, b in PAIRS:
        j = ed.join(m.add_prefix("a_"), on="pre", how="inner").join(m.add_prefix("b_"), on="post", how="inner")
        j = j[(j.a_flyvis_type == a) & (j.b_flyvis_type == b)]
        if len(j) < 10:
            print(f"{a:>4s} -> {b:4s}  too few pairs ({len(j)})")
            continue
        d = hexdist(j.a_u - j.b_u, j.a_v - j.b_v)
        perm = rng.permutation(len(j))
        d0 = hexdist(j.a_u.values[perm] - j.b_u.values, j.a_v.values[perm] - j.b_v.values)
        print(f"{a:>4s} -> {b:4s}  n={len(j):4d}  median {np.median(d):.1f}  "
              f"<=1 col: {np.mean(d <= 1):4.0%}   (shuffled median {np.median(d0):.1f})")
