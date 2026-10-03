"""Dlaczego dane T4 po lewej stronie BANC wskazują orientację odwróconą o 180°?

1. Zgodność kierunku dla każdej pary (wejście → podtyp T4) w ostatecznej orientacji mapy,
   po obu stronach. Odwrócone tylko Mi4/Mi9 → etykiety Mi4/Mi9; odwrócone całe podtypy →
   etykiety T4.
2. Warstwy płytki lobuli: aksony T4a/b/c/d kończą się w warstwach 1/2/3/4, ułożonych po kolei.
   Kolejność zakończeń aksonów wzdłuż osi warstw musi być taka sama po obu stronach.
   To sprawdza etykiety T4 bez przesunięć i bez orientacji.

    python scripts/investigate_left_t4.py
"""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flyvis_banc_map as b  # noqa: E402

T4 = ["T4a", "T4b", "T4c", "T4d"]


def pair_cosines(diag):
    rows = []
    for (tgt, src), r in diag["cells"].groupby(["tgt", "src"])[["wx", "wy", "w"]].sum().iterrows():
        fv = b.FV_OFFSETS[tgt].get(src)
        if fv is None or r.w < 100 or np.linalg.norm(fv[0]) < 0.3 * b.COLUMN:
            continue
        a = diag["Q"] @ np.array([r.wx / r.w, r.wy / r.w])
        rows.append((tgt, src, a @ fv[0] / (np.linalg.norm(a) * np.linalg.norm(fv[0])), int(r.w)))
    return rows


def axon_terminals(side, diag):
    """Mean of skeleton nodes far from the medulla (> 20 µm from any Mi1 node) per T4 cell."""
    m = diag["positions"]
    mi1 = m[m.flyvis_type == "Mi1"]
    pts = []
    for _, r in mi1.iterrows():
        xyz = b._skeleton_nodes(r.root_id_meta)
        if xyz is not None:
            pts.append(xyz)
    tree = cKDTree(np.vstack(pts))
    out = {}
    for t in T4:
        cells = []
        for _, r in m[m.flyvis_type == t].iterrows():
            xyz = b._skeleton_nodes(r.root_id_meta)
            if xyz is None:
                continue
            far = tree.query(xyz)[0] > 20
            if far.sum() >= 3:
                cells.append(xyz[far].mean(0))
        out[t] = np.array(cells)
    return out


def layer_order(terms):
    """Order of subtype means along the axis that best separates them (first PC of means)."""
    means = np.array([terms[t].mean(0) for t in T4])
    c = means - means.mean(0)
    axis = np.linalg.svd(c)[2][0]
    proj = c @ axis
    order = [T4[i] for i in np.argsort(proj)]
    return order, proj


def main() -> None:
    right_frame, _, rdiag = b.map_side("right")
    _, _, ldiag = b.map_side("left", partner=right_frame)

    print("\n=== 1. Zgodność kierunku (cosinus) wejście → T4, ostateczna orientacja ===")
    for side, diag in (("right", rdiag), ("left", ldiag)):
        print(f"--- {side} ---")
        for tgt, src, cos, w in pair_cosines(diag):
            print(f"  {src:>8s} → {tgt}: {cos:+.2f}  ({w} synaps)")

    print("\n=== 2. Kolejność warstw płytki lobuli (zakończenia aksonów T4) ===")
    orders = {}
    for side, diag in (("right", rdiag), ("left", ldiag)):
        terms = axon_terminals(side, diag)
        order, proj = layer_order(terms)
        orders[side] = order
        spread = {t: np.linalg.norm(terms[t].std(0)) for t in T4}
        print(f"{side:5s}: {' < '.join(order)}  (n = {', '.join(f'{t} {len(terms[t])}' for t in T4)}; "
              f"rozrzut µm {', '.join(f'{spread[t]:.0f}' for t in T4)})")
    seq = ["T4a", "T4b", "T4c", "T4d"]
    for side, order in orders.items():
        ok = order in (seq, seq[::-1])
        print(f"{side}: kolejność warstw {'zgodna' if ok else 'NIEZGODNA'} z a-b-c-d")
    if orders["left"] != orders["right"] and orders["left"] != orders["right"][::-1]:
        swaps = [p for p in permutations(T4) if [dict(zip(T4, p))[t] for t in orders["right"]] in
                 (orders["left"], orders["left"][::-1])]
        print("zamiany etykiet zgodne z lewą stroną:", [dict((a, c) for a, c in zip(T4, p) if a != c) for p in swaps][:4])


if __name__ == "__main__":
    main()
