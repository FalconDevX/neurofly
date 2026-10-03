"""BancController — pełny krok Osoby 2: aktywność BANC od Osoby 1 + IMU → FlightCommand."""

from __future__ import annotations

import numpy as np

from .connectome import HALTERE_GROUPS, MOTOR_GROUPS, Connectome
from .contracts import BancActivation, FlightCommand, ImuState, VisualBatch
from .dynamics import RateDynamics
from .readout import ManualDecoder


class BancController:
    def __init__(
        self,
        connectome: Connectome,
        decoder: ManualDecoder | None = None,
        visual_gain: float = 1.0,
        haltere_gain: float = 0.5,
        substeps: int = 4,
        **dynamics_kw,
    ) -> None:
        self.c = connectome
        self.dyn = RateDynamics(connectome, **dynamics_kw)
        self.decoder = decoder or ManualDecoder()
        self.visual_gain = visual_gain
        self.haltere_gain = haltere_gain
        self.substeps = substeps
        self.haltere_sign = 1.0  # ustalany przez calibrate_haltere_sign()
        self._haltere = {g: connectome.group_indices(g) for g in HALTERE_GROUPS}
        motor = [connectome.group_indices(g) for g in MOTOR_GROUPS]
        self._motor_idx = np.concatenate(motor)
        self._motor_split = np.cumsum([len(m) for m in motor])[:-1]
        self.unmatched_ids = 0
        self._batch_ids: np.ndarray | None = None
        self._batch_idx = np.empty(0, dtype=np.int64)

    def _batch_index(self, root_ids: np.ndarray) -> np.ndarray:
        """Indeksy neuronów dla ``VisualBatch.root_ids`` (-1 = brak), liczone raz na tablicę."""
        if self._batch_ids is not root_ids:
            self._batch_idx = self.c.indices_of(np.asarray(root_ids, dtype=np.int64))
            self._batch_ids = root_ids
        return self._batch_idx

    def external_input(self, visual: list[BancActivation] | VisualBatch, imu: ImuState | None) -> np.ndarray:
        ext = np.zeros(self.c.n)
        if isinstance(visual, VisualBatch):
            idx = self._batch_index(visual.root_ids)
            act = np.asarray(visual.activity, dtype=float)
        else:
            idx = self.c.indices_of(np.fromiter((a.banc_root_id for a in visual), np.int64, len(visual)))
            act = np.fromiter((a.activity for a in visual), float, len(visual))
        hit = idx >= 0
        self.unmatched_ids = int((~hit).sum())
        np.add.at(ext, idx[hit], self.visual_gain * act[hit])
        if imu is not None:
            # Haltery mierzą prędkość kątową. Kodowanie gyro → aferenty halter L/R (oficjalna
            # grupa BANC body_part_sensory=haltere) to nasza abstrakcja, nie model czucia halter.
            roll_rate, _, yaw_rate = imu.gyro
            drive = self.haltere_sign * (roll_rate + yaw_rate)
            ext[self._haltere["haltere_aff_R"]] += self.haltere_gain * max(drive, 0.0)
            ext[self._haltere["haltere_aff_L"]] += self.haltere_gain * max(-drive, 0.0)
        return ext

    def motor_features(self) -> np.ndarray:
        """Średnia aktywność grup MOTOR_GROUPS (6,); z GPU kopiuje tylko motoneurony."""
        parts = np.split(self.dyn.rates_at(self._motor_idx), self._motor_split)
        return np.array([p.mean() if len(p) else 0.0 for p in parts])

    def _settle(self, visual: list[BancActivation], imu: ImuState | None, steps: int) -> np.ndarray:
        self.dyn.reset()
        for _ in range(steps):
            self.dyn.step(self.external_input(visual, imu), self.substeps)
        return self.motor_features()

    def calibrate_rest(self, steps: int = 50, visual: list[BancActivation] | None = None) -> None:
        """Baseline dekodera = odczyt MN dla sceny neutralnej (``visual``, np. beacon na wprost).
        Komendy liczymy jako odchylenie od tego stanu, więc scena neutralna → zawis."""
        self.decoder.calibrate(self._settle(visual or [], None, steps))
        self.dyn.reset()

    def calibrate_haltere_sign(self, visual: list[BancActivation] | None = None,
                               rate: float = 1.5, steps: int = 30) -> float:
        """Kodowanie gyro → aferenty halter L/R jest naszą abstrakcją, więc znak dobieramy
        empirycznie: ten, przy którym BANC odpowiada na przechył komendą roll przeciwną."""
        self.haltere_sign = 1.0
        roll = self.decoder.decode(self._settle(visual or [], ImuState(gyro=(rate, 0.0, 0.0)), steps)).roll
        self.haltere_sign = -1.0 if roll > 0 else 1.0
        self.dyn.reset()
        return self.haltere_sign

    def calibrate_scale(self, stimuli: list[list[BancActivation]], steps: int = 30) -> None:
        """Plan C: odpala bodźce referencyjne (np. beacon lewo/prawo) i skaluje odczyt MN."""
        self.decoder.calibrate_scale([self._settle(v, None, steps) for v in stimuli])
        self.dyn.reset()

    def step(self, visual: list[BancActivation] | VisualBatch, imu: ImuState | None = None) -> FlightCommand:
        self.dyn.step(self.external_input(visual, imu), self.substeps)
        feats = self.motor_features()
        cmd = self.decoder.decode(feats)
        cmd.debug = {"motor_features": feats, "unmatched_ids": self.unmatched_ids}
        return cmd
