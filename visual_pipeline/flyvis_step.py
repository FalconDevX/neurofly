"""Stateful, frame-by-frame FlyVis simulation."""
import numpy as np
import torch

from . import datamate_win_fix  # noqa: F401  (must precede FlyVis cache writes on Windows)
from flyvis import NetworkView

MODEL = "flow/0000/000"


class FlyVisStepper:
    def __init__(self, fps: float = 30.0, dt: float = 1 / 90, model: str = MODEL):
        self.network = NetworkView(model).init_network()
        self.dt = dt
        # Integration substeps per camera frame (FlyVis warns above dt = 1/50).
        self.substeps = max(1, round(1 / (fps * dt)))
        self.state = None

    def reset(self, batch_size: int = 1):
        """Start from the steady state after grey input, like FlyVis' simulate()."""
        with torch.no_grad():
            self.state = self.network.steady_state(1.0, self.dt, batch_size)

    @torch.no_grad()
    def step(self, luminance: np.ndarray) -> np.ndarray:
        """luminance: (batch, 721) in [0, 1], FlyVis hexal order -> (batch, n_cells) activity."""
        if self.state is None:
            self.reset(luminance.shape[0])
        x = torch.as_tensor(luminance, dtype=torch.float32)
        movie = x[:, None, None, :].repeat(1, self.substeps, 1, 1)
        states = self.network.simulate(movie, self.dt, initial_state=self.state, as_states=True)
        self.state = states[-1]
        return self.state.nodes.activity.cpu().numpy()

    def cell_index(self):
        """Per-cell metadata (type, u, v) aligned with the activity vector."""
        nodes = self.network.connectome.nodes
        return (
            np.array(nodes.type[:]).astype(str),
            np.array(nodes.u[:]),
            np.array(nodes.v[:]),
        )
