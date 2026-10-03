"""Warianty kontrolne connectome: czy zachowanie wynika z połączeń BANC, czy z dekodera?

Każdy wariant zwraca nowy ``Connectome`` z tymi samymi neuronami (te same indeksy, grupy, odczyt),
zmienia się tylko macierz ``W``:

* ``shuffle_targets`` — każda krawędź dostaje losowy neuron docelowy (permutacja kolumny post).
  Zachowane: liczba wyjść i wejść każdego neuronu, znak NT neuronu pre, rozkład wag.
  Zniszczone: KTO z kim jest połączony. Wiersze normalizowane jak w oryginale (stabilność).
* ``shuffle_signs`` — znaki NT (pobudzający / hamujący) przetasowane między neuronami.
* ``lesion`` — usunięcie grup (zerowanie wejść i wyjść), jak lezja w eksperymencie na muszce.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from .connectome import Connectome


def _row_normalize(W: sp.csr_matrix) -> sp.csr_matrix:
    in_total = np.asarray(abs(W).sum(axis=1)).ravel()
    in_total[in_total == 0] = 1.0
    return (sp.diags(1.0 / in_total) @ W).tocsr()


def shuffle_targets(c: Connectome, seed: int = 0) -> Connectome:
    coo = c.W.tocoo()
    post = np.random.default_rng(seed).permutation(coo.row)
    keep = post != coo.col  # bez autapsów, jak w oryginale
    W = sp.coo_matrix((coo.data[keep], (post[keep], coo.col[keep])), shape=c.W.shape).tocsr()
    W.sum_duplicates()
    return replace(c, W=_row_normalize(W))


def shuffle_signs(c: Connectome, seed: int = 0) -> Connectome:
    W = c.W.tocsc()
    sign = np.sign(np.asarray(W.max(axis=0).todense()).ravel() + np.asarray(W.min(axis=0).todense()).ravel())
    sign[sign == 0] = 1.0
    new = np.random.default_rng(seed).permutation(sign)
    return replace(c, W=(W @ sp.diags(new * sign)).tocsr())  # sign² = 1: kolumna pre dostaje nowy znak


def lesion(c: Connectome, groups: list[str]) -> Connectome:
    """Usuwa neurony z grup (prefiksy, np. "visual" → visual_L i visual_R): zero wejść i wyjść."""
    hit = np.zeros(c.n, bool)
    for g in groups:
        hit |= np.char.startswith(c.groups.astype(str), g)
    if not hit.any():
        raise ValueError(f"lezja {groups}: brak neuronów w tych grupach")
    keep = sp.diags((~hit).astype(float))
    return replace(c, W=(keep @ c.W @ keep).tocsr())
