"""Liczy na pełnym BANC v888 dane do wizualizacji: połączenia między grupami i odpowiedzi
grup / komend na siatkę (kierunek beacona × prędkość przechyłu z IMU).

    python scripts/export_viz_data.py  →  data/viz/banc_viz.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from banc_control import BancController, Connectome, ImuState  # noqa: E402
from banc_control.connectome import DEFAULT_DATA_DIR, META_FILE, NON_NEURONS, SIDE, flight_groups  # noqa: E402
from banc_control.readout import motor_features  # noqa: E402
from banc_control.stubs import FakeVision  # noqa: E402

BEARINGS_DEG = list(range(-60, 61, 15))
ROLL_RATES = [-3.0, -1.5, 0.0, 1.5, 3.0]
STEPS = 30


def display_groups(meta: pd.DataFrame) -> np.ndarray:
    """Grupy do rysunku: grupy lotu + szersze tło anatomiczne (wszystko z oficjalnych kolumn)."""
    g = flight_groups(meta)
    side = meta["side"].map(SIDE).fillna("")
    sc = meta["super_class"].fillna("")
    for name, mask in [
        ("central", sc == "central_brain_intrinsic"),
        ("dn_other", sc == "descending"),
        ("vnc", sc == "ventral_nerve_cord_intrinsic"),
    ]:
        m = mask & (g == "") & (side != "")
        g[m] = name + "_" + side[m]
    return g.to_numpy().astype(str)


def main() -> None:
    meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE)
    meta = meta[~meta["super_class"].isin(NON_NEURONS)].reset_index(drop=True)
    c = Connectome.from_banc()
    assert len(meta) == c.n
    groups = display_groups(meta)
    names = sorted(set(groups) - {""})
    idx = {n: np.flatnonzero(groups == n) for n in names}

    # średnia znakowana frakcja wejścia: ile wejścia grupy B pochodzi od grupy A
    W = c.W.tocsc()
    conn = {}
    for a in names:
        Wa = W[:, idx[a]].tocsr()
        for b in names:
            w = float(Wa[idx[b]].sum() / len(idx[b]))
            if abs(w) >= 0.005:
                conn.setdefault(a, {})[b] = round(w, 4)

    ctrl = BancController(c)
    vision = FakeVision(c)
    ctrl.calibrate_rest(STEPS, visual=vision(0.0))
    ctrl.calibrate_scale([vision(np.deg2rad(-60)), vision(np.deg2rad(60))], steps=STEPS)
    sign = ctrl.calibrate_haltere_sign(vision(0.0), steps=STEPS)
    print("haltere_sign =", sign)

    grid = []
    t0 = time.time()
    for b in BEARINGS_DEG:
        for rr in ROLL_RATES:
            ctrl.dyn.reset()
            for _ in range(STEPS):
                cmd = ctrl.step(vision(np.deg2rad(b)), ImuState(gyro=(rr, 0.0, 0.0)))
            r = ctrl.dyn.r
            grid.append({
                "bearing": b, "roll_rate": rr,
                "act": {n: round(float(r[idx[n]].mean()), 6) for n in names},
                "mn": [round(float(x), 6) for x in motor_features(c, r)],
                "cmd": {k: round(getattr(cmd, k), 4) for k in ("thrust", "roll", "pitch", "yaw")},
            })
        print(f"bearing {b:+d}° gotowe ({time.time() - t0:.0f} s)")

    out = {
        "source": "BANC v888, banc_888_meta.feather + banc_888_edgelist_simple_v2.feather (count >= 5)",
        "haltere_sign": sign,
        "n_neurons": int(c.n), "n_edges": int(c.W.nnz),
        "sizes": {n: int(len(idx[n])) for n in names},
        "conn": conn, "grid": grid,
        "bearings": BEARINGS_DEG, "roll_rates": ROLL_RATES,
    }
    dest = ROOT / "data" / "viz" / "banc_viz.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out), encoding="utf-8")
    print("zapisano", dest, f"{dest.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
