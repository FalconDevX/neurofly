"""Stuby na okno integracji 4–6 h: udają Osobę 1 (wzrok → BANC) i Osobę 3 (dron)."""

from __future__ import annotations

import numpy as np

from .connectome import Connectome
from .contracts import BancActivation, FlightCommand, ImuState


class FakeVision:
    """Udaje Osobę 1: beacon pod kątem ``bearing`` (rad, + = w prawo) pobudza neurony
    BANC ``visual_projection`` po tej stronie. Prawdziwe wejście przyjdzie z FlyVis."""

    def __init__(self, connectome: Connectome, seed: int = 0) -> None:
        self.c = connectome
        self.rng = np.random.default_rng(seed)
        self.left = connectome.group_indices("visual_L")
        self.right = connectome.group_indices("visual_R")

    def __call__(self, bearing: float, brightness: float = 1.0) -> list[BancActivation]:
        out = []
        for idx, side in ((self.left, -1.0), (self.right, 1.0)):
            level = brightness * 0.5 * (1.0 + np.tanh(3.0 * side * bearing))
            act = np.clip(level + self.rng.normal(0, 0.05, len(idx)), 0, None)
            out += [BancActivation(int(self.c.root_ids[i]), str(self.c.cell_types[i]), float(a))
                    for i, a in zip(idx, act)]
        return out


class ToyDrone:
    """Minimalna fizyka 1-osiowa (wysokość + roll + yaw) zamiast Gazebo, do testów pętli."""

    def __init__(self, dt: float = 0.02) -> None:
        self.dt = dt
        self.z = self.vz = 0.0
        self.roll = self.roll_rate = 0.0
        self.yaw = self.yaw_rate = 0.0

    def step(self, cmd: FlightCommand, wind_torque: float = 0.0) -> ImuState:
        self.vz += (9.81 * (cmd.thrust / 0.5) - 9.81) * self.dt
        self.z = max(0.0, self.z + self.vz * self.dt)
        self.roll_rate += (8.0 * cmd.roll + wind_torque - 0.5 * self.roll_rate) * self.dt
        self.roll += self.roll_rate * self.dt
        self.yaw_rate += (4.0 * cmd.yaw - 0.5 * self.yaw_rate) * self.dt
        self.yaw += self.yaw_rate * self.dt
        return ImuState(gyro=(self.roll_rate, 0.0, self.yaw_rate))
