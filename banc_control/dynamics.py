"""Dynamika szybkości odpalania (rate model) na grafie BANC.

    tau * dr/dt = -r + relu_tanh(gain * W r + I_ext + bias)

Jeden podkrok = jedno mnożenie macierzy rzadkiej. Dwa backendy z tym samym API:
  * ``device="cuda"`` — torch, macierz CSR float32 na GPU (domyślnie, gdy CUDA jest dostępna),
  * ``device="cpu"``  — scipy, float64.
"""

from __future__ import annotations

import warnings

import numpy as np

from .connectome import Connectome

try:
    import torch
except ImportError:  # backend CPU działa bez torcha
    torch = None


def default_device() -> str:
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


class RateDynamics:
    def __init__(
        self,
        connectome: Connectome,
        tau_ms: float = 20.0,
        dt_ms: float = 5.0,
        gain: float = 0.9,
        bias: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.c = connectome
        self.alpha = dt_ms / tau_ms
        self.gain = gain
        self.bias = bias
        self.device = device or default_device()
        if self.device == "cpu":
            self._W = connectome.W
            self._r = np.zeros(connectome.n)
        else:
            W = connectome.W.tocsr().astype(np.float32)
            with warnings.catch_warnings():  # torch oznacza CSR jako "beta"; matmul CSR × wektor jest stabilny
                warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
                self._W = torch.sparse_csr_tensor(
                    torch.from_numpy(W.indptr.astype(np.int64)),
                    torch.from_numpy(W.indices.astype(np.int64)),
                    torch.from_numpy(W.data),
                    size=W.shape, device=self.device,
                )
            self._r = torch.zeros(connectome.n, device=self.device)

    @property
    def r(self) -> np.ndarray:
        """Aktualne aktywności (N,) jako numpy."""
        return self._r if self.device == "cpu" else self._r.cpu().numpy()

    def _dev_index(self, idx: np.ndarray):
        """Indeksy na GPU; ta sama tablica z klatki na klatkę jest kopiowana tylko raz."""
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get(id(idx))
        if hit is None or hit[0] is not idx:
            hit = cache[id(idx)] = (idx, torch.as_tensor(idx, device=self.device))
        return hit[1]

    def rates_at(self, idx: np.ndarray) -> np.ndarray:
        """Aktywności wybranych neuronów bez kopiowania całego wektora z GPU."""
        if self.device == "cpu":
            return self._r[idx]
        return self._r[self._dev_index(idx)].cpu().numpy()

    def levels_at(self, idx: np.ndarray, floor: float = 1e-6) -> np.ndarray:
        """Aktywności wybranych neuronów w skali log ``floor``..1 → uint8 0..255, liczone na urządzeniu
        (z GPU wraca 1 bajt na neuron zamiast 4)."""
        decades = -np.log10(floor)
        if self.device == "cpu":
            lv = (np.log10(np.clip(self._r[idx], floor, 1.0)) + decades) / decades
            return np.round(lv * 255).astype(np.uint8)
        r = self._r[self._dev_index(idx)]
        lv = (torch.log10(r.clamp(floor, 1.0)) + decades) / decades
        return (lv * 255).round().to(torch.uint8).cpu().numpy()

    def reset(self) -> None:
        self._r[:] = 0.0

    def step(self, external: np.ndarray, substeps: int = 1) -> None:
        """Wykonuje ``substeps`` kroków całkowania z wejściem ``external`` (N,)."""
        if self.device == "cpu":
            for _ in range(substeps):
                drive = self.gain * (self._W @ self._r) + external + self.bias
                self._r += self.alpha * (-self._r + np.tanh(np.maximum(drive, 0.0)))
            return
        ext = torch.as_tensor(external, dtype=torch.float32, device=self.device) + self.bias
        for _ in range(substeps):
            drive = self.gain * (self._W @ self._r) + ext
            self._r += self.alpha * (-self._r + torch.tanh(torch.relu(drive)))
