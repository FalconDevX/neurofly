"""Etap 2: przypisuje komórki FlyVis (typ, u, v) neuronom BANC, osobno dla każdej strony.

    python scripts/download_banc.py           # meta BANC v888
    python scripts/fetch_skeletons.py Mi1     # szkielety arkusza referencyjnego
    python scripts/build_flyvis_banc_map.py   # sam dociąga edgelist v3 (~340 MB)


1. Reference sheet: Mi1 cell positions (nm) flattened with Isomap, scaled so the sheet
   area per Mi1 cell equals one FlyVis column (hexagon with spacing sqrt(3)).
2. Other types: position = synapse-weighted mean of already-placed partners, iterated.
   Following connectivity (not soma location) handles the optic chiasm.
3. Orientation: orthogonal Procrustes (rotation or reflection) between FlyVis T4/T5 input
   offset vectors and the same offsets measured in BANC.
4. Per type: optimal 1:1 assignment of BANC neurons to FlyVis hexals, max distance 1 column.

Output: visual_pipeline/flyvis_banc_map.csv (one row per FlyVis cell and side with a BANC match).
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
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


def map_side(side):
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

    # 2. Propagate positions along synapses (undirected), Mi1 fixed.
    und = pd.concat([ed.rename(columns={"pre": "a", "post": "b"}), ed.rename(columns={"pre": "b", "post": "a"})])
    for _ in range(6):
        j = und[und.b.isin(pos.index) & ~und.a.isin(mi1.index)].join(pos, on="b")
        j[["px", "py"]] = j[["px", "py"]].mul(j.w, axis=0)
        agg = j.groupby("a")[["px", "py", "w"]].sum()
        new = agg[["px", "py"]].div(agg.w, axis=0)
        pos = pd.concat([pos.loc[mi1.index], new])
    m = m.join(pos, how="inner")

    # 3. Orientation from T4/T5 input offsets.
    A, B, W = [], [], []
    for t in DIRECTIONAL:
        tgt = m[m.flyvis_type == t]
        j = ed[ed.post.isin(tgt.index)].join(m[["flyvis_type", "px", "py"]], on="pre").join(
            m[["px", "py"]], on="post", rsuffix="_t")
        j["dx"], j["dy"] = j.px - j.px_t, j.py - j.py_t
        for src, (fv_vec, fv_n) in flyvis_offsets(t).items():
            g = j[j.flyvis_type == src]
            if len(g) < 20 or np.linalg.norm(fv_vec) < 0.3 * COLUMN:
                continue
            A.append(np.average(g[["dx", "dy"]].values, axis=0, weights=g.w.values))
            B.append(fv_vec)
            W.append(fv_n)
    A, B, W = np.array(A), np.array(B), np.array(W)
    U, _, Vt = np.linalg.svd((B * W[:, None]).T @ A)
    best = {}
    for det in (1, -1):
        Q = U @ np.diag([1, det]) @ Vt
        pred = A @ Q.T
        cos = np.sum(pred * B, 1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(B, axis=1))
        best[det] = (np.average(cos, weights=W), Q)
    det = max(best, key=lambda d: best[d][0])
    Q = best[det][1]
    print(f"[{side}] orientation from {len(A)} offset pairs: mean cosine "
          f"rotation {best[1][0]:+.2f}, reflection {best[-1][0]:+.2f} -> using "
          f"{'reflection' if det == -1 else 'rotation'}", flush=True)
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
        r, c = linear_sum_assignment(D)
        keep = D[r, c] <= MAX_DIST
        for i, rid, d in zip(fv.flyvis_index.values[r[keep]], bc.index[c[keep]], D[r, c][keep]):
            rows.append((i, rid, d / COLUMN))
    out = pd.DataFrame(rows, columns=["flyvis_index", "root_id", "dist_columns"])
    out["eye"] = side
    return out.merge(nodes, on="flyvis_index").merge(
        meta[["root_id", "root_id_meta", "banc_888_id", "cell_type"]].rename(
            columns={"cell_type": "banc_cell_type"}), on="root_id")


result = pd.concat([map_side(s) for s in ("right", "left")], ignore_index=True)
result = result.drop(columns="root_id").rename(columns={"root_id_meta": "root_id"})
result = result[["eye", "flyvis_index", "flyvis_type", "u", "v", "root_id", "banc_888_id",
                 "banc_cell_type", "dist_columns"]]
result.to_csv(MAP_FILE, index=False)

summary = result.groupby(["flyvis_type", "eye"]).size().unstack(fill_value=0)
summary["flyvis_cells"] = nodes.flyvis_type.value_counts()
print(summary.to_string())
for side in ("right", "left"):
    r = result[result.eye == side]
    print(f"[{side}] mapped {len(r)} / {len(nodes)} FlyVis cells, median dist "
          f"{r.dist_columns.median():.2f} columns")
