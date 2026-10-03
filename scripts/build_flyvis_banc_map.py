"""Etap 2: przypisuje komórki FlyVis (typ, u, v) neuronom BANC, osobno dla każdej strony.

    python scripts/download_banc.py           # meta BANC v888
    python scripts/fetch_skeletons.py Mi1 T4a T4b T4c T4d Mi4 Mi9 Tm3   # ~8 tys. plików
    python scripts/build_flyvis_banc_map.py   # sam dociąga edgelist v3 (~340 MB)


1. Reference sheet: Mi1 cell positions (nm) flattened with Isomap, scaled so the sheet
   area per Mi1 cell equals one FlyVis column (hexagon with spacing sqrt(3)).
2. Other types: position = synapse-weighted mean of already-placed partners, iterated.
   Following connectivity (not soma location) handles the optic chiasm.
3. Orientation: orthogonal Procrustes (rotation or reflection) between FlyVis T4 input
   offset vectors and the same offsets measured in BANC, with T4/Mi4/Mi9/Tm3 positions
   taken from their own skeletons (projected onto the Mi1 sheet), not from step 2.
   Bootstrap over T4 cells gives the angle's 95% CI. Right side: +94° (CI ±2°). Left side
   (6x fewer typed T4): mirrored right-side map across the midline; it agrees with the dorsal
   rim area (DRA) landmark, while the left T4 data fit points the opposite way (180°).
4. Per type: optimal 1:1 assignment of BANC neurons to FlyVis hexals, max distance 1 column.

Output: visual_pipeline/flyvis_banc_map.csv (one row per FlyVis cell and side with a BANC match).
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, cKDTree
from sklearn.manifold import Isomap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banc_control.connectome import DEFAULT_DATA_DIR, META_FILE  # noqa: E402
from visual_pipeline import datamate_win_fix  # noqa: E402, F401
from visual_pipeline.arbor_positions import arbor_positions  # noqa: E402
from visual_pipeline.bridge import MAP_FILE  # noqa: E402
from flyvis import NetworkView  # noqa: E402
from flyvis.utils.hex_utils import hex_to_pixel  # noqa: E402

EDGES_V3 = "banc_888_edgelist_simple_v3.feather"
BUCKET = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome"
SKEL_DIR = DEFAULT_DATA_DIR / "skeletons"

COLUMN = np.sqrt(3)  # nearest-neighbour hexal distance in hex_to_pixel "default" units
MAX_DIST = 1.0 * COLUMN
# FlyVis type -> BANC cell_type(s). Types absent here are matched by identical name.
ALIASES = {"TmY9": ["TmY9q", "TmY9q__perp"], "CT1(M10)": ["CT1"], "CT1(Lo1)": ["CT1"]}
MANY_TO_ONE = {"CT1(M10)", "CT1(Lo1)"}  # one giant BANC neuron covers all FlyVis compartments
DIRECTIONAL = [f"T{n}{s}" for n in (4, 5) for s in "abcd"]

# ---------- FlyVis ----------
conn = NetworkView("flow/0000/000").init_network().connectome
nodes = pd.DataFrame({
    "flyvis_index": np.arange(len(conn.nodes.type)),
    "flyvis_type": np.array(conn.nodes.type[:]).astype(str),
    "u": np.array(conn.nodes.u[:]), "v": np.array(conn.nodes.v[:]),
})
fv_types = sorted(nodes.flyvis_type.unique())
e = conn.edges
fv_edges = pd.DataFrame({
    "src": np.array(e.source_type[:]).astype(str), "tgt": np.array(e.target_type[:]).astype(str),
    "du": np.array(e.du[:]), "dv": np.array(e.dv[:]), "n": np.array(e.n_syn[:]),
    "tu": np.array(e.target_u[:]), "tv": np.array(e.target_v[:]),
})
fv_edges = fv_edges[(fv_edges.tu == 0) & (fv_edges.tv == 0)]  # one central target per type


def flyvis_offsets(target):
    """Synapse-weighted mean source position relative to target, in hex pixel units."""
    out = {}
    for src, g in fv_edges[fv_edges.tgt == target].groupby("src"):
        # du = target_u - source_u, so the source sits at -du relative to the target.
        x, y = hex_to_pixel(-g.du.values, -g.dv.values)
        out[src] = (np.average(np.stack([x, y], 1), axis=0, weights=g.n.values), g.n.sum())
    return out


# ---------- BANC ----------
meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE,
                       columns=["root_id", "banc_888_id", "cell_type", "side", "root_position_nm"])
banc_type_of = {}
for t in fv_types:
    for bt in ALIASES.get(t, [t]):
        banc_type_of.setdefault(bt, t)
meta = meta[meta.cell_type.isin(banc_type_of) & meta.side.isin(["left", "right"])
            & meta.root_position_nm.notna()].copy()
meta["flyvis_type"] = meta.cell_type.map(banc_type_of)
xyz = meta.root_position_nm.astype(str).str.split(",", expand=True)
meta[["x", "y", "z"]] = xyz.astype(float).values

if not (DEFAULT_DATA_DIR / EDGES_V3).exists():
    print(f"pobieram {EDGES_V3} …", flush=True)
    urllib.request.urlretrieve(f"{BUCKET}/compiled_data/banc_888/{EDGES_V3}", DEFAULT_DATA_DIR / EDGES_V3)
edges = pd.read_feather(DEFAULT_DATA_DIR / EDGES_V3, columns=["pre", "post", "count"])
edges = edges.astype({"pre": "int64", "post": "int64"}).rename(columns={"count": "w"})
# The edgelist may use either ID column; keep the one that overlaps it, as "root_id".
pre_ids = set(edges.pre.unique())
overlap = {c: meta[c].astype("int64").isin(pre_ids).sum() for c in ("root_id", "banc_888_id")}
id_col = max(overlap, key=overlap.get)
print(f"edgelist ID overlap: {overlap} -> using {id_col}", flush=True)
meta["root_id_meta"] = meta.root_id.astype("int64")
meta["banc_888_id"] = meta.banc_888_id.astype("int64")
meta["root_id"] = meta[id_col].astype("int64")
ids = set(meta.root_id)
edges = edges[edges.pre.isin(ids) & edges.post.isin(ids)]
print(f"BANC: {len(meta)} typed neurons of FlyVis types, {len(edges)} edges among them", flush=True)


# Types placed by their own skeleton (arbor inside the medulla). T5 dendrites are in the
# lobula and CT1 is one giant neuron, so those are placed through their partners instead.
SKEL_TYPES = set(fv_types) - {"T5a", "T5b", "T5c", "T5d", "CT1(M10)", "CT1(Lo1)", "Mi1"}
MEDULLA_RADIUS_UM = 4.0  # węzeł szkieletu "w medulli" = blisko węzła kolumny Mi1
# Inputs whose offset defines the T4 direction. Mi1 sits almost on the T4 column (cosine
# +0.20 even on the well-annotated right side) and minor inputs (TmY15, T4→T4) are noisy;
# with them the left fit was dominated by Mi1→T4a and the right angle drifted by ~14°.
ORIENTATION_SOURCES = {"Tm3", "Mi4", "Mi9", "C3"}
FV_OFFSETS = {t: flyvis_offsets(t) for t in DIRECTIONAL}


def _skeleton_nodes(root_id_meta):
    path = SKEL_DIR / f"{root_id_meta}.swc"
    return np.loadtxt(path, usecols=(2, 3, 4), ndmin=2) if path.exists() else None


def skeleton_sheet_positions(m, mi1, mi1_pos):
    """Sheet (px, py) of SKEL_TYPES neurons from their own skeleton nodes inside the medulla.

    Each distal Mi1 node is labelled with its Mi1's sheet position; a node of another neuron
    takes the labels of its nearest Mi1 nodes. T4 dendrites (M10) and Mi4/Mi9/Tm3 arbors are
    placed by their own anatomy, not by their partners.
    """
    pts, lab = [], []
    for rid, r in mi1.iterrows():
        xyz = _skeleton_nodes(r.root_id_meta)
        d = np.linalg.norm(xyz - np.array([r.x, r.y, r.z]) / 1000, axis=1)
        xyz = xyz[d >= 0.5 * d.max()]
        pts.append(xyz)
        lab.append(np.repeat(mi1_pos.loc[[rid]].values, len(xyz), axis=0))
    pts, lab = np.vstack(pts), np.vstack(lab)
    tree = cKDTree(pts)
    out = {rid: p for rid, p in zip(mi1_pos.index, mi1_pos.values)}
    for rid, r in m[m.flyvis_type.isin(SKEL_TYPES)].iterrows():
        xyz = _skeleton_nodes(r.root_id_meta)
        if xyz is None:
            continue
        d, idx = tree.query(xyz, k=3)
        inside = d[:, 0] < MEDULLA_RADIUS_UM
        if inside.sum() >= 2:
            out[rid] = lab[idx[inside]].mean(axis=(0, 1))
    df = pd.DataFrame.from_dict(out, orient="index", columns=["px", "py"])
    df["flyvis_type"] = m.flyvis_type.reindex(df.index)
    return df


def offset_sums(posdf, ed, targets):
    """Per target cell and source type: synapse-weighted sums of (source - target) position."""
    tgt_ids = posdf.index[posdf.flyvis_type.isin(targets)]
    j = ed[ed.post.isin(tgt_ids) & ed.pre.isin(posdf.index)]
    j = j.join(posdf, on="pre").join(posdf, on="post", rsuffix="_t")
    j["wx"], j["wy"] = j.w * (j.px - j.px_t), j.w * (j.py - j.py_t)
    g = j.groupby(["post", "flyvis_type_t", "flyvis_type"])[["wx", "wy", "w"]].sum().reset_index()
    return g.rename(columns={"post": "target", "flyvis_type_t": "tgt", "flyvis_type": "src"})


def angle_of(Q):
    return float(np.rad2deg(np.arctan2(Q[1, 0], Q[0, 0])))


def sheet_axes(P, flat):
    """(2, 3): sheet axes mapped into 3D by a linear fit (average tangent directions)."""
    J, *_ = np.linalg.lstsq(flat - flat.mean(0), P - P.mean(0), rcond=None)
    return J


def sheet_handedness(J, outward):
    """Sign of det[a, b, n]: sheet axes a, b in 3D vs the outward normal n."""
    return float(np.sign(np.linalg.det(np.stack([J[0], J[1], outward]))))


def mirrored_orientation(Q_other, J_other, J_this, lateral):
    """This side's Q predicted by mirroring the other side's lattice→3D map across the midline.

    lattice dir e → 3D: J_other.T @ Q_other.T @ e → mirror → this sheet via pinv(J_this.T).
    """
    u = lateral / np.linalg.norm(lateral)
    M = np.eye(3) - 2 * np.outer(u, u)
    Qt = np.linalg.pinv(J_this.T) @ M @ J_other.T @ Q_other.T  # sheet ← lattice
    U, _, Vt = np.linalg.svd(Qt.T)
    return U @ Vt  # nearest orthogonal matrix


def _procrustes(cells, force_det=None):
    A, B, W = [], [], []
    for (tgt, src), r in cells.groupby(["tgt", "src"])[["wx", "wy", "w"]].sum().iterrows():
        fv = FV_OFFSETS[tgt].get(src)
        if src not in ORIENTATION_SOURCES or fv is None or r.w < 100 or np.linalg.norm(fv[0]) < 0.3 * COLUMN:
            continue
        A.append([r.wx / r.w, r.wy / r.w])
        B.append(fv[0])
        W.append(fv[1])
    if len(A) < 2:
        raise ValueError(f"za mało par wejście → T4 do orientacji ({len(A)})")
    A, B, W = np.array(A), np.array(B), np.array(W)
    U, _, Vt = np.linalg.svd((B * W[:, None]).T @ A)
    best = {}
    for det in (1, -1):
        Q = U @ np.diag([1, det]) @ Vt
        pred = A @ Q.T
        cos = np.sum(pred * B, 1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(B, axis=1))
        best[det] = (np.average(cos, weights=W), Q)
    det = force_det or max(best, key=lambda d: best[d][0])
    return det, best[det][1], best[det][0]


def cosine_at(cells, Q):
    """Mean cosine between FlyVis offsets and BANC offsets rotated by a given Q."""
    num = den = 0.0
    for (tgt, src), r in cells.groupby(["tgt", "src"])[["wx", "wy", "w"]].sum().iterrows():
        fv = FV_OFFSETS[tgt].get(src)
        if src not in ORIENTATION_SOURCES or fv is None or r.w < 100 or np.linalg.norm(fv[0]) < 0.3 * COLUMN:
            continue
        a = Q @ np.array([r.wx / r.w, r.wy / r.w])
        num += fv[1] * a @ fv[0] / (np.linalg.norm(a) * np.linalg.norm(fv[0]))
        den += fv[1]
    return num / den


def fit_orientation(cells, n_boot=200, force_det=None):
    """Procrustes FlyVis ↔ BANC offsets; bootstrap over target cells for the angle's 95% CI.

    Returns det, Q, mean cosine, (ci_low, ci_high, fraction of resamples choosing the other det).
    With ``force_det`` the handedness is fixed and only the angle is fitted.
    """
    det, Q, cos = _procrustes(cells, force_det)
    a0, devs, flips = angle_of(Q), [], 0
    ids = cells.target.unique()
    by_target = cells.set_index("target")
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        try:
            d2, Q2, _ = _procrustes(by_target.loc[rng.choice(ids, len(ids))].reset_index(), force_det)
        except ValueError:
            continue
        if d2 != det:
            flips += 1
            continue
        devs.append((angle_of(Q2) - a0 + 180) % 360 - 180)
    ci = (a0 + np.percentile(devs, 2.5), a0 + np.percentile(devs, 97.5), flips / max(n_boot, 1)) if devs else (np.nan, np.nan, 0.0)
    return det, Q, cos, ci


MI1_SIDE_CENTERS = meta[meta.flyvis_type == "Mi1"].groupby("side")[["x", "y", "z"]].mean()
MIDLINE = MI1_SIDE_CENTERS.mean().values

# Dorsal rim area (DRA) columns sit at the dorsal edge of the medulla: an anatomical "up".
_dra = pd.read_feather(DEFAULT_DATA_DIR / META_FILE, columns=["cell_type", "cell_sub_class", "side", "root_position_nm"])
_dra = _dra[(_dra.cell_type.isin(["Mi1_DRA", "DmDRA1", "DmDRA2"]) | (_dra.cell_sub_class == "dorsal_rim"))
            & _dra.root_position_nm.notna()].copy()
_dra[["x", "y", "z"]] = _dra.root_position_nm.str.split(",", expand=True).astype(float).values
DRA_CENTERS = _dra.groupby("side")[["x", "y", "z"]].mean()
DRA_COUNTS = _dra.side.value_counts()


def dorsal_agreement(Q, J, side):
    """Cosine between the lattice's dorsal direction (hex +y, from the T4c motion test)
    mapped into 3D, and the direction from the medulla centre to the DRA neurons."""
    up_3d = J.T @ Q.T @ np.array([0.0, 1.0])
    dra = DRA_CENTERS.loc[side].values - MI1_SIDE_CENTERS.loc[side].values
    return float(up_3d @ dra / (np.linalg.norm(up_3d) * np.linalg.norm(dra)))


def map_side(side, partner=None):
    """``partner``: (Q, J, handedness) of the other, reliable side. The optic lobes are mirror
    images, so this side's lattice→3D map has the opposite handedness, and its angle can be
    predicted by mirroring the partner's map across the midline."""
    m = meta[meta.side == side].set_index("root_id")
    ed = edges[edges.pre.isin(m.index) & edges.post.isin(m.index)]

    # 1. Reference sheet from Mi1.
    # Skeleton files are keyed by the metadata root_id; prefer arbor over soma position.
    mi1 = m[m.flyvis_type == "Mi1"]
    arb = arbor_positions(mi1.reset_index().assign(root_id=lambda d: d.root_id_meta), SKEL_DIR)
    arb = arb.reindex(mi1.root_id_meta.values)
    mi1 = mi1[arb.notna().all(axis=1).values]
    P = arb.dropna().values
    print(f"[{side}] reference: {len(mi1)} Mi1 with arbor positions", flush=True)
    flat = Isomap(n_neighbors=12, n_components=2).fit_transform(P)
    # Somata are stacked several deep, so nearest-neighbour distance underestimates the
    # column spacing (~2.7x). Use area per Mi1 (one per column) = area of one hexagon.
    spacing = np.sqrt(ConvexHull(flat).volume / len(flat) / (np.sqrt(3) / 2))
    flat = (flat - np.median(flat, 0)) * (COLUMN / spacing)
    pos = pd.DataFrame(flat, index=mi1.index, columns=["px", "py"])

    # 2. Neurons with a skeleton in the medulla: own arbor projected onto the Mi1 sheet.
    #    The rest (no skeleton, T5, CT1): synapse-weighted mean of placed partners, iterated.
    sk = skeleton_sheet_positions(m, mi1, pos)
    anchors = sk[["px", "py"]]
    und = pd.concat([ed.rename(columns={"pre": "a", "post": "b"}), ed.rename(columns={"pre": "b", "post": "a"})])
    pos = anchors
    for _ in range(6):
        j = und[und.b.isin(pos.index) & ~und.a.isin(anchors.index)].join(pos, on="b")
        j[["px", "py"]] = j[["px", "py"]].mul(j.w, axis=0)
        agg = j.groupby("a")[["px", "py", "w"]].sum()
        pos = pd.concat([anchors, agg[["px", "py"]].div(agg.w, axis=0)])
    m = m.join(pos, how="inner")
    m["placed_by"] = np.where(m.index.isin(anchors.index), "skeleton", "partners")
    print(f"[{side}] positions: {(m.placed_by == 'skeleton').sum()} from skeletons, "
          f"{(m.placed_by == 'partners').sum()} from partners", flush=True)

    # 3. Orientation from T4 input offsets, using skeleton positions only.
    cells = offset_sums(sk, ed, [t for t in DIRECTIONAL if t.startswith("T4")])
    lateral = MI1_SIDE_CENTERS.loc[side].values - MIDLINE
    J = sheet_axes(P, flat)
    h = sheet_handedness(J, lateral)
    force = None if partner is None else int(-partner[2] * h)
    det, Q, cos, (lo, hi, flips) = fit_orientation(cells, force_det=force)
    if partner is not None:
        free = fit_orientation(cells, n_boot=0)
        Q_sym = mirrored_orientation(partner[0], partner[1], J, lateral)
        print(f"[{side}] handedness fixed by mirror symmetry (free fit would choose "
              f"{'reflection' if free[0] == -1 else 'rotation'}); mirrored partner predicts "
              f"{'reflection' if np.linalg.det(Q_sym) < 0 else 'rotation'} {angle_of(Q_sym):+.0f}° "
              f"(data cosine there {cosine_at(cells, Q_sym):+.2f}, rotated 180°: "
              f"{cosine_at(cells, -Q_sym):+.2f})", flush=True)
        print(f"[{side}] data fit below; DRA check: data fit {dorsal_agreement(Q, J, side):+.2f}, "
              f"mirrored partner {dorsal_agreement(Q_sym, J, side):+.2f}", flush=True)
        # Left BANC has ~6x fewer typed T4 and its T4 offsets give the right axis but the
        # opposite sense (180°). Mirror symmetry and the DRA landmark agree with each other,
        # so the mirrored partner orientation is used.
    Q_data = Q
    print(f"[{side}] orientation from skeletons ({len(sk)} neurons, {cells.target.nunique()} T4): "
          f"{'reflection' if det == -1 else 'rotation'} {angle_of(Q):+.0f}° "
          f"(bootstrap 95%: {lo:+.0f}°..{hi:+.0f}°, other handedness in {flips:.0%}), "
          f"mean cosine {cos:+.2f}", flush=True)
    if partner is not None:
        Q = Q_sym
        print(f"[{side}] using mirrored partner orientation {angle_of(Q):+.0f}° "
              f"(data fit {angle_of(Q_data):+.0f}°)", flush=True)
    print(f"[{side}] dorsal check vs {DRA_COUNTS.get(side, 0)} DRA neurons: lattice 'up' "
          f"cosine {dorsal_agreement(Q, J, side):+.2f} (rotated 180°: {dorsal_agreement(-Q, J, side):+.2f})",
          flush=True)
    m[["hx", "hy"]] = m[["px", "py"]].values @ Q.T

    # 4. Per-type assignment.
    rows = []
    for t in fv_types:
        fv = nodes[nodes.flyvis_type == t]
        bc = m[m.flyvis_type == t]
        if not len(bc):
            continue
        fx, fy = hex_to_pixel(fv.u.values, fv.v.values)
        F = np.stack([fx, fy], 1)
        if t in MANY_TO_ONE:
            rid = bc.index[0]
            for i in fv.flyvis_index:
                rows.append((i, rid, np.nan))
            continue
        D = np.linalg.norm(F[:, None] - bc[["hx", "hy"]].values[None], axis=2)
        # Pairs beyond MAX_DIST get a prohibitive cost, so the assignment first maximises the
        # number of pairs within 1 column and only then minimises distance. Plain distance
        # cost shifted whole chains just past the threshold (Mi1: 95% of columns had a cell
        # within 1 column but only 70% were kept).
        r, c = linear_sum_assignment(np.where(D <= MAX_DIST, D, 1e6))
        keep = D[r, c] <= MAX_DIST
        for i, rid, d in zip(fv.flyvis_index.values[r[keep]], bc.index[c[keep]], D[r, c][keep]):
            rows.append((i, rid, d / COLUMN))
    out = pd.DataFrame(rows, columns=["flyvis_index", "root_id", "dist_columns"])
    out["eye"] = side
    out = out.merge(m[["placed_by"]], left_on="root_id", right_index=True, how="left")
    diag = {"cells": cells, "Q_data": Q_data, "Q": Q, "J": J, "positions": m}
    return (Q, J, det * h), out.merge(nodes, on="flyvis_index").merge(
        meta[["root_id", "root_id_meta", "banc_888_id", "cell_type"]].rename(
            columns={"cell_type": "banc_cell_type"}), on="root_id"), diag


def main() -> None:
    # Right side first: ~6x more typed T4 than left, its handedness is unambiguous (bootstrap).
    right_frame, right, _ = map_side("right")
    _, left, _ = map_side("left", partner=right_frame)
    result = pd.concat([right, left], ignore_index=True)
    result = result.drop(columns="root_id").rename(columns={"root_id_meta": "root_id"})
    result = result[["eye", "flyvis_index", "flyvis_type", "u", "v", "root_id", "banc_888_id",
                     "banc_cell_type", "dist_columns", "placed_by"]]
    result.to_csv(MAP_FILE, index=False)

    summary = result.groupby(["flyvis_type", "eye"]).size().unstack(fill_value=0)
    summary["flyvis_cells"] = nodes.flyvis_type.value_counts()
    print(summary.to_string())
    for side in ("right", "left"):
        r = result[result.eye == side]
        print(f"[{side}] mapped {len(r)} / {len(nodes)} FlyVis cells, median dist "
              f"{r.dist_columns.median():.2f} columns, placed by skeleton "
              f"{(r.placed_by == 'skeleton').mean():.0%}")


if __name__ == "__main__":
    main()
