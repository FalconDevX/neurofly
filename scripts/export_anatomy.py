"""Eksport prawdziwej anatomii BANC v888 do wizualizacji:
  * pozycje som wszystkich neuronów (meta `position`, voxel 4×4×45 nm → µm),
  * uproszczone szkielety SWC neuronów lotu (DN flight, MN skrzydeł, aferenty halter),
  * aktywność każdego neuronu z symulacji na pełnym grafie dla beacona lewo / prosto / prawo.

    python scripts/export_anatomy.py  →  data/viz/banc_anatomy.json
Szkielety: data/banc_888/swc/<root_id>_{skeleton,l2}.swc (oficjalny bucket, banc_banc_space_swc/).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from banc_control import BancController, Connectome  # noqa: E402
from banc_control.connectome import DEFAULT_DATA_DIR, META_FILE, NON_NEURONS, flight_groups  # noqa: E402
from banc_control.stubs import FakeVision  # noqa: E402

SWC_DIR = DEFAULT_DATA_DIR / "swc"
VOXEL_NM = np.array([4.0, 4.0, 45.0])
BEARINGS = (-45, 0, 45)
STEPS = 30
NODE_SPACING_UM = 8.0
MIN_BRANCH_UM = 20.0  # krótsze gałązki pomijamy — przy tej skali są niewidoczne


def load_swc(path: Path) -> list[list[tuple[float, float, float]]]:
    """Polilinie szkieletu (µm). Usuwa końcowe gałązki krótsze niż MIN_BRANCH_UM i rozrzedza
    węzły do ~NODE_SPACING_UM; pień i rozgałęzienia zostają."""
    rows = [ln.split() for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    a = np.array([r for r in rows if len(r) == 7], dtype=float)
    ids, xyz, parent_id = a[:, 0].astype(int), a[:, 2:5] / 1000.0, a[:, 6].astype(int)
    row = {i: k for k, i in enumerate(ids)}
    parent = np.array([row.get(p, -1) for p in parent_id])
    children: dict[int, list[int]] = {}
    for k, p in enumerate(parent):
        children.setdefault(p, []).append(k)

    order, stack = [], list(children.get(-1, []))  # rodzic przed dziećmi
    while stack:
        k = stack.pop()
        order.append(k)
        stack += children.get(k, [])
    depth = np.zeros(len(ids))  # najdłuższa droga od węzła do liścia w jego poddrzewie
    for k in reversed(order):
        p = parent[k]
        if p >= 0:
            depth[p] = max(depth[p], depth[k] + np.linalg.norm(xyz[k] - xyz[p]))
    keep = depth >= MIN_BRANCH_UM
    keep[children.get(-1, [])] = True

    lines = []
    for k in order:
        p = parent[k]
        if not keep[k] or p < 0:
            continue
        kept_kids = [c for c in children.get(p, []) if keep[c]]
        starts_line = not lines or len(kept_kids) > 1 or lines[-1][-1] is not None and lines[-1][1] != p
        if starts_line:
            lines.append([[xyz[p]], k])
        line, _ = lines[-1]
        if np.linalg.norm(xyz[k] - line[-1]) >= NODE_SPACING_UM or not any(keep[c] for c in children.get(k, [])):
            line.append(xyz[k])
        lines[-1][1] = k
    return [[tuple(np.round(pt, 1)) for pt in line] for line, _ in lines if len(line) > 1]


def main() -> None:
    meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE)
    meta = meta[~meta["super_class"].isin(NON_NEURONS)].reset_index(drop=True)
    c = Connectome.from_banc()
    assert len(meta) == c.n
    groups = flight_groups(meta).to_numpy().astype(str)

    xyz = np.array([[float(v) for v in p.split(",")] for p in meta["position"]]) * VOXEL_NM / 1000.0
    lo, hi = np.percentile(xyz, 0.05, axis=0), np.percentile(xyz, 99.95, axis=0)
    inside = np.all((xyz >= lo) & (xyz <= hi), axis=1)

    ctrl = BancController(c)
    vision = FakeVision(c)
    act = []
    for b in BEARINGS:
        ctrl.dyn.reset()
        for _ in range(STEPS):
            ctrl.step(vision(np.deg2rad(b)))
        act.append(ctrl.dyn.r.copy())
        print(f"aktywność dla beacona {b:+d}° policzona")
    act = np.array(act)  # (3, N)

    # kwantyzacja logarytmiczna 0..35 → jeden znak base36 na neuron i warunek
    def q(v):
        lv = np.log10(np.clip(v, 1e-6, 1.0))  # −6..0
        return np.clip(np.round((lv + 6) / 6 * 35), 0, 35).astype(int)

    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    keep = np.flatnonzero(inside)
    sc = meta["super_class"].fillna("unknown").to_numpy()
    classes = sorted(set(sc))
    cls_idx = {s: i for i, s in enumerate(classes)}
    somas = {
        "xyz": np.round(xyz[keep]).astype(int).ravel().tolist(),
        "act": ["".join(alphabet[k] for k in q(a[keep])) for a in act],
        "super_class": [cls_idx[s] for s in sc[keep]],
    }

    neurons = []
    flight = np.array([g.startswith(("dn_flight", "wing_", "haltere")) for g in groups])
    for i in np.flatnonzero(flight):
        rid = meta.at[i, "banc_888_id"]
        path = next((p for p in (SWC_DIR / f"{rid}_skeleton.swc", SWC_DIR / f"{rid}_l2.swc") if p.exists()), None)
        if path is None:
            continue
        neurons.append({
            "id": rid, "group": groups[i], "cell_type": meta.at[i, "cell_type"] or "",
            "act": [round(float(a[i]), 6) for a in act],
            "lines": load_swc(path),
        })

    out = {
        "source": "BANC v888: banc_888_meta.feather (position), banc_banc_space_swc/*.swc, symulacja na pełnym grafie",
        "bearings": list(BEARINGS), "classes": classes, "somas": somas, "neurons": neurons,
        "n_somas": int(len(keep)), "n_total": int(c.n),
    }
    dest = ROOT / "data" / "viz" / "banc_anatomy.json"
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    pts = sum(len(l) for n in neurons for l in n["lines"])
    print(f"zapisano {dest} {dest.stat().st_size / 1e6:.1f} MB · somy {len(keep)} · szkielety {len(neurons)} · węzły {pts}")


if __name__ == "__main__":
    main()
