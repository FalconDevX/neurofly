"""Oficjalny connectome BANC (Brain And Nerve Cord), materializacja v888.

Źródło (publiczne, bez logowania):
  gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/
    banc_888_meta.feather                 — 188k neuronów, adnotacje
    banc_888_edgelist_simple_v2.feather   — krawędzie pre → post (synapsy v2, size ≥ 5)
DOI: https://doi.org/10.7910/DVN/7WTH1N  ·  pobieranie: scripts/download_banc.py

Wagi: W[post, pre] = znak(NT pre) * count / suma |count| wejść neuronu post.
Normalizacja wejść ogranicza |suma wiersza| ≤ 1, więc dynamika z gain < 1 jest stabilna.

Grupy funkcjonalne biorę WYŁĄCZNIE z oficjalnych kolumn meta
(super_class, side, super_cluster, cell_function, body_part_sensory, flow).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BANC_VERSION = 888
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / f"banc_{BANC_VERSION}"
META_FILE = f"banc_{BANC_VERSION}_meta.feather"
EDGES_FILE = f"banc_{BANC_VERSION}_edgelist_simple_v2.feather"

# Konwencja jak w modelach całego mózgu muszki: ACh pobudza, GABA i glutaminian hamują.
# Histamina (fotoreceptory) też hamuje. Modulatory i brak predykcji → traktujemy jako +.
INHIBITORY_NT = {"gaba", "glutamate", "histamine"}
NON_NEURONS = {"glia", "trachea", "not_a_neuron"}

MOTOR_GROUPS = (
    "wing_power_L", "wing_power_R",        # DLM / DVM — mięśnie mocy
    "wing_steering_L", "wing_steering_R",  # b1, b2, i1, iii1, … — mięśnie sterujące
    "wing_tension_L", "wing_tension_R",    # tp, ps — napięcie skrzydła
)
HALTERE_GROUPS = ("haltere_aff_L", "haltere_aff_R")
SIDE = {"left": "L", "right": "R"}


def flight_groups(meta: pd.DataFrame) -> pd.Series:
    """Przypisuje grupę funkcjonalną z oficjalnych adnotacji BANC; "" = brak grupy."""
    col = lambda c: meta[c].fillna("").astype(str) if c in meta else pd.Series("", index=meta.index)  # noqa: E731
    sc, side = col("super_class"), col("side").map(SIDE).fillna("")
    func, cluster = col("cell_function"), col("super_cluster")
    groups = pd.Series("", index=meta.index, dtype=object)
    rules = [
        ("visual", sc == "visual_projection"),
        ("dn_flight_power", (sc == "descending") & (cluster == "flight power")),
        ("dn_flight_steering", (sc == "descending") & cluster.str.startswith("flight steering")),
        ("haltere_aff", (col("flow") == "afferent") & col("body_part_sensory").str.contains("haltere")),
        ("wing_power", (sc == "motor") & (func == "wing_power")),
        ("wing_steering", (sc == "motor") & (func == "wing_steering")),
        ("wing_tension", (sc == "motor") & (func == "wing_tension")),
    ]
    for name, mask in rules:
        mask = mask & (side != "")
        groups[mask] = name + "_" + side[mask]
    return groups


@dataclass
class Connectome:
    root_ids: np.ndarray        # (N,) int64 — BANC v888 root IDs
    cell_types: np.ndarray      # (N,) str
    super_class: np.ndarray     # (N,) str
    groups: np.ndarray          # (N,) str — grupa funkcjonalna lub ""
    W: sp.csr_matrix            # (N, N) — wiersz = post, kolumna = pre

    def __post_init__(self) -> None:
        self._index = pd.Index(self.root_ids)

    @property
    def n(self) -> int:
        return len(self.root_ids)

    def index_of(self, root_id: int) -> int | None:
        i = self.indices_of(np.array([root_id]))[0]
        return None if i < 0 else int(i)

    def indices_of(self, root_ids: np.ndarray) -> np.ndarray:
        """Wektorowo: root ID → indeks w grafie, -1 gdy brak."""
        return self._index.get_indexer(np.asarray(root_ids, dtype=np.int64))

    def group_indices(self, group: str) -> np.ndarray:
        return np.flatnonzero(self.groups == group)

    @classmethod
    def from_banc(cls, data_dir: str | Path = DEFAULT_DATA_DIR, min_count: int = 5) -> Connectome:
        """Wczytuje oficjalne pliki BANC v888 z ``data_dir``."""
        data_dir = Path(data_dir)
        missing = [f for f in (META_FILE, EDGES_FILE) if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Brak {missing} w {data_dir}. Uruchom: python scripts/download_banc.py")
        meta = pd.read_feather(data_dir / META_FILE)
        edges = pd.read_feather(data_dir / EDGES_FILE, columns=["pre", "post", "count"])
        return cls.from_banc_tables(meta, edges, min_count=min_count)

    @classmethod
    def from_banc_tables(cls, meta: pd.DataFrame, edges: pd.DataFrame, min_count: int = 5) -> Connectome:
        """Buduje connectome z tabel w schemacie BANC (meta: ``banc_888_id``…, edges: ``pre``, ``post``, ``count``)."""
        meta = meta[~meta["super_class"].isin(NON_NEURONS)].reset_index(drop=True)
        root_ids = meta["banc_888_id"].astype(np.int64).to_numpy()
        index = pd.Series(np.arange(len(root_ids)), index=root_ids)

        # autapsy: dokumentacja v888 mówi, że są usunięte, ale część została w pliku — paper ich nie liczy
        edges = edges[(edges["count"] >= min_count) & (edges["pre"] != edges["post"])]
        pre_ids, post_ids = edges["pre"].astype(np.int64), edges["post"].astype(np.int64)
        keep = (pre_ids.isin(index.index) & post_ids.isin(index.index)).to_numpy()
        pre = index.loc[pre_ids[keep]].to_numpy()
        post = index.loc[post_ids[keep]].to_numpy()

        nt = meta["neurotransmitter_predicted"].fillna("").str.lower()
        sign = np.where(nt.isin(INHIBITORY_NT).to_numpy(), -1.0, 1.0)
        W = _normalized_weights(pre, post, edges["count"].to_numpy(float)[keep], sign, len(root_ids))

        return cls(
            root_ids,
            meta["cell_type"].fillna("").astype(str).to_numpy(),
            meta["super_class"].fillna("").astype(str).to_numpy(),
            flight_groups(meta).astype(str).to_numpy(),
            W,
        )


def _normalized_weights(pre, post, count, sign, n) -> sp.csr_matrix:
    W = sp.coo_matrix((count * sign[pre], (post, pre)), shape=(n, n)).tocsr()
    W.sum_duplicates()
    in_total = np.asarray(abs(W).sum(axis=1)).ravel()
    in_total[in_total == 0] = 1.0
    return (sp.diags(1.0 / in_total) @ W).tocsr()
