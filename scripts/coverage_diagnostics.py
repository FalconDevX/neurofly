"""Co ogranicza pokrycie mapy FlyVis → BANC? (prawa strona, typy z dużą liczbą komórek)

Dla każdego typu: ile komórek BANC jest w ogóle, ile z nich leży w polu FlyVis (promień 15
kolumn), ile kolumn FlyVis ma najbliższy neuron BANC w ≤ 1 / ≤ 1,5 kolumny, ile przypisano.

    python scripts/coverage_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flyvis_banc_map as b  # noqa: E402
from flyvis.utils.hex_utils import hex_to_pixel, pixel_to_hex  # noqa: E402

TYPES = ["Mi1", "Tm3", "T4a", "T4c", "Mi4", "Mi9", "Tm1", "T5a", "C3", "L2"]


def main() -> None:
    _, mapped, diag = b.map_side("right")
    m = diag["positions"]
    rows = []
    for t in TYPES:
        bc = m[m.flyvis_type == t]
        fv = b.nodes[b.nodes.flyvis_type == t]
        F = np.stack(hex_to_pixel(fv.u.values, fv.v.values), 1)
        H = bc[["hx", "hy"]].values
        u, v = pixel_to_hex(H[:, 0], H[:, 1])
        radius = (np.abs(u) + np.abs(u + v) + np.abs(v)) / 2
        d = cKDTree(H).query(F)[0] / b.COLUMN
        rows.append({
            "typ": t, "BANC": len(bc), "z_kości": (bc.placed_by == "skeleton").mean(),
            "BANC_w_polu_FlyVis": int((radius <= 15.5).sum()),
            "kolumny_≤1": (d <= 1).mean(), "kolumny_≤1.5": (d <= 1.5).mean(),
            "przypisane": int((mapped.flyvis_type == t).sum()),
        })
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.0%}"))
    all_r = np.concatenate([
        (lambda uv: (np.abs(uv[0]) + np.abs(uv[0] + uv[1]) + np.abs(uv[1])) / 2)(pixel_to_hex(*m[["hx", "hy"]].values.T))
    ])
    print(f"\npromień arkusza BANC w kolumnach (wszystkie typy): mediana {np.median(all_r):.1f}, "
          f"90% {np.percentile(all_r, 90):.1f}, max {all_r.max():.1f} (FlyVis: 15)")
    mi1 = m[m.flyvis_type == "Mi1"][["hx", "hy"]].values
    print(f"środek Mi1 względem środka FlyVis: {mi1.mean(0) / b.COLUMN} kolumn")


if __name__ == "__main__":
    main()
