"""Dynamika szybkości odpalania (rate model) na grafie BANC.

    tau * dr/dt = -r + relu_tanh(gain * W r + I_ext + bias)

Model jest celowo prosty: jeden krok = jedno mnożenie macierzy rzadkiej, więc działa
w czasie rzeczywistym także dla pełnego BANC (~160k neuronów, kilka mln krawędzi).
"""

from __future__ import annotations

import numpy as np

from .connectome import Connectome


class RateDynamics:
    def __init__(
        self,
        connectome: Connectome,
        tau_ms: float = 20.0,
        dt_ms: float = 5.0,
        gain: float = 0.9,
        bias: float = 0.0,
    ) -> None:
        self.c = connectome
        self.alpha = dt_ms / tau_ms
        self.gain = gain
        self.bias = bias
        self.r = np.zeros(connectome.n)

    def reset(self) -> None:
        self.r[:] = 0.0

    def step(self, external: np.ndarray, substeps: int = 1) -> np.ndarray:
        """Wykonuje ``substeps`` kroków całkowania z wejściem ``external`` (N,)."""
        for _ in range(substeps):
            drive = self.gain * (self.c.W @ self.r) + external + self.bias
            self.r += self.alpha * (-self.r + np.tanh(np.maximum(drive, 0.0)))
        return self.r
