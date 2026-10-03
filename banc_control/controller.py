"""BancController — pełny krok Osoby 2: aktywność BANC od Osoby 1 + IMU → FlightCommand."""

from __future__ import annotations

import numpy as np

from .connectome import HALTERE_GROUPS, MOTOR_GROUPS, Connectome

DN_GROUPS = ("dn_flight_power_L", "dn_flight_power_R", "dn_flight_steering_L", "dn_flight_steering_R")
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
        haltere_roll_weight: float = 0.0,
        haltere_yaw_weight: float = 0.5,
        substeps: int = 4,
        readout: str = "mn",
        **dynamics_kw,
    ) -> None:
        """``readout``: "mn" — 6 średnich grup MN skrzydeł; "dn" — do tego każdy neuron DN lotu osobno
        (w średnich grup informacja o stronie celu się znosi, patrz ``scripts/check_side_decoding.py``)."""
        self.c = connectome
        self.dyn = RateDynamics(connectome, **dynamics_kw)
        self.decoder = decoder or ManualDecoder()
        self.visual_gain = visual_gain
        self.haltere_gain = haltere_gain
        # Wagi roll_rate / yaw_rate w napędzie halter. Aferenty halter L/R w v888 ruszają głównie
        # MN sterujące (yaw ±1), roll słabo (~−0.15), więc jeden kanał L/R nie stabilizuje obu osi:
        # z roll w pętli dron wpadał w ciągły obrót. Domyślnie tylko yaw, poziom trzyma symulator.
        self.haltere_roll_weight = haltere_roll_weight
        self.haltere_yaw_weight = haltere_yaw_weight
        self.substeps = substeps
        self.haltere_sign = 1.0  # roll_rate → haltery; ustalany przez calibrate_haltere_sign()
        self.yaw_axis_sign = 1.0  # odwracany przez calibrate_yaw_sign()
        self.haltere_yaw_sign = 1.0  # yaw_rate → haltery; osobny znak, ta sama kalibracja
        self._haltere = {g: connectome.group_indices(g) for g in HALTERE_GROUPS}
        motor = [connectome.group_indices(g) for g in MOTOR_GROUPS]
        self._motor_idx = np.concatenate(motor)
        self._motor_split = np.cumsum([len(m) for m in motor])[:-1]
        if readout not in ("mn", "dn"):
            raise ValueError(f"readout {readout!r}: 'mn' albo 'dn'")
        self.readout = readout
        self._extra_idx = (np.concatenate([connectome.group_indices(g) for g in DN_GROUPS]) if readout == "dn"
                           else np.empty(0, dtype=np.int64))
        self._read_idx = np.r_[self._motor_idx, self._extra_idx]
        self.decoder.ensure_features(len(MOTOR_GROUPS) + len(self._extra_idx))
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
            drive = (self.haltere_roll_weight * self.haltere_sign * roll_rate
                     + self.haltere_yaw_weight * self.haltere_yaw_sign * yaw_rate)
            ext[self._haltere["haltere_aff_R"]] += self.haltere_gain * max(drive, 0.0)
            ext[self._haltere["haltere_aff_L"]] += self.haltere_gain * max(-drive, 0.0)
        return ext

    def motor_features(self) -> np.ndarray:
        """Średnia aktywność grup MOTOR_GROUPS (6,), przy ``readout="dn"`` + pojedyncze DN lotu;
        z GPU kopiuje tylko te neurony."""
        rates = self.dyn.rates_at(self._read_idx)
        parts = np.split(rates[:len(self._motor_idx)], self._motor_split)
        return np.r_[[p.mean() if len(p) else 0.0 for p in parts], rates[len(self._motor_idx):]]

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
        """Kodowanie gyro → aferenty halter L/R jest naszą abstrakcją, więc znaki dobieramy
        empirycznie: te, przy których BANC odpowiada na obrót komendą przeciwną. Osobno dla
        roll_rate (komenda roll) i yaw_rate (komenda yaw), przy wagach 1. Zwraca znak roll.
        Po ``calibrate_yaw_sign`` — ona może odwrócić oś yaw dekodera."""
        weights = self.haltere_roll_weight, self.haltere_yaw_weight
        self.haltere_sign = self.haltere_yaw_sign = 1.0
        self.haltere_roll_weight, self.haltere_yaw_weight = 1.0, 0.0
        roll = self.decoder.decode(self._settle(visual or [], ImuState(gyro=(rate, 0.0, 0.0)), steps)).roll
        self.haltere_sign = -1.0 if roll > 0 else 1.0
        self.haltere_roll_weight, self.haltere_yaw_weight = 0.0, 1.0
        yaw = self.decoder.decode(self._settle(visual or [], ImuState(gyro=(0.0, 0.0, rate)), steps)).yaw
        self.haltere_yaw_sign = -1.0 if yaw > 0 else 1.0
        self.haltere_roll_weight, self.haltere_yaw_weight = weights
        self.dyn.reset()
        return self.haltere_sign

    def warm_start(self, visual, steps: int = 30) -> None:
        """Po ``dyn.reset()`` stan startuje od zera, daleko od spoczynku z kalibracji: pierwsze klatki
        dawały thrust 0 / yaw ±1. Dochodzi do stanu ustalonego dla bieżącej sceny (bez IMU)."""
        ext = self.external_input(visual, None)
        for _ in range(steps):
            self.dyn.step(ext, self.substeps)

    def _raw_command(self, visual, imu: ImuState | None, steps: int) -> np.ndarray:
        """thrust/roll/pitch/yaw dekodera przed przycięciem do [-1, 1]."""
        f = self._settle(visual or [], imu, steps)
        self.dyn.reset()
        return self.decoder.M @ self.decoder.normalized(f)

    def calibrate_haltere_gain(self, visual: list[BancActivation] | None = None, rate: float = 1.0,
                               target: float = 0.5, steps: int = 30, iters: int = 4) -> float:
        """``haltere_gain`` tak, żeby obrót ``rate`` rad/s dawał |yaw| ≈ ``target`` (przed przycięciem).

        Skala odczytu MN pochodzi z bodźców wzrokowych, a w scenie MuJoCo wzrok zmienia MN o ~1%
        spoczynku: przy stałym gain 0.5 haltery dawały surowy yaw ~6000 na 1 rad/s, pętla gyro → yaw
        oscylowała ±1 co klatkę, a thrust stał na 0. Odpowiedź jest prawie liniowa w gain, więc
        kilka kroków skalowania wystarcza. Po ``calibrate_haltere_sign``."""
        if self.haltere_yaw_weight == 0:
            return self.haltere_gain
        base = self._raw_command(visual, None, steps)[3]
        for _ in range(iters):
            resp = abs(self._raw_command(visual, ImuState(gyro=(0.0, 0.0, rate)), steps)[3] - base)
            if resp < 1e-12:
                break
            self.haltere_gain *= target / resp
        return self.haltere_gain

    def _mean_yaw(self, frames: list, skip: float = 0.5) -> float:
        self.dyn.reset()
        yaws = [self.decoder.decode(self._step_features(v, None)).yaw for v in frames]
        return float(np.mean(yaws[int(len(yaws) * skip):]))

    def calibrate_yaw_sign(self, turn_right: list, turn_left: list) -> float:
        """Znak osi yaw z odruchu optomotorycznego, nie ze statycznego celu (ten po poprawce
        orientacji oczu nie rozróżnia stron). ``turn_right`` / ``turn_left``: kolejne klatki
        wzroku przy wymuszonym obrocie drona w prawo / w lewo. Stabilizacja = komenda yaw
        przeciwna do obrotu; jeśli wychodzi zgodna, odwracamy oś yaw dekodera (przełożenie
        MN sterujących L−R → yaw to nasze założenie). Zwraca różnicę yaw(prawo) − yaw(lewo)
        po kalibracji (< 0 = stabilizuje)."""
        diff = self._mean_yaw(turn_right) - self._mean_yaw(turn_left)
        if diff > 0:
            self.decoder.M[3] *= -1
            self.yaw_axis_sign *= -1
            diff = -diff
        self.dyn.reset()
        return diff

    def calibrate_scale(self, stimuli: list[list[BancActivation]], steps: int = 30) -> None:
        """Plan C: odpala bodźce referencyjne (np. beacon lewo/prawo) i skaluje odczyt MN."""
        self.decoder.calibrate_scale([self._settle(v, None, steps) for v in stimuli])
        self.dyn.reset()

    def _step_features(self, visual: list[BancActivation] | VisualBatch, imu: ImuState | None) -> np.ndarray:
        self.dyn.step(self.external_input(visual, imu), self.substeps)
        return self.motor_features()

    def step(self, visual: list[BancActivation] | VisualBatch, imu: ImuState | None = None) -> FlightCommand:
        feats = self._step_features(visual, imu)
        cmd = self.decoder.decode(feats)
        cmd.debug = {"motor_features": feats, "unmatched_ids": self.unmatched_ids}
        return cmd
