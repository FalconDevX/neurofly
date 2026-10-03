"""Odczyt motoryczny: aktywność oficjalnych grup motoneuronów skrzydeł BANC → thrust / roll / pitch / yaw.

Grupy MN pochodzą z adnotacji BANC (cell_function: wing_power / wing_steering / wing_tension).
Przełożenie na osie quadcoptera to NASZA decyzja projektowa (nie wynik z BANC):
  thrust ← średnia aktywność MN mocy (DLM/DVM) obu stron
  roll   ← asymetria MN mocy L − R      (mocniejsze lewe skrzydło → przechył w prawo)
  yaw    ← asymetria MN sterujących L − R
  pitch  ← brak ręcznego mapowania (w Planie C stały trim); uczony w Planach A/B

Trzy dekodery = Plany z tablicy Miro:
  Plan C  ManualDecoder   — ręcznie skalibrowane wzmocnienia, zero uczenia (gwarantowane demo)
  Plan B  LinearDecoder   — BANC/VNC stałe, uczy się tylko mała warstwa liniowa (LMS)
  Plan A  AdaptiveDecoder — Plan B + uczenie z nagrody (stabilność / beacon) metodą perturbacji wag
"""

from __future__ import annotations

import numpy as np

from .connectome import MOTOR_GROUPS, Connectome
from .contracts import FlightCommand

AXES = ("thrust", "roll", "pitch", "yaw")


def motor_features(connectome: Connectome, rates: np.ndarray) -> np.ndarray:
    """Średnia aktywność każdej grupy z MOTOR_GROUPS → wektor (6,)."""
    out = np.zeros(len(MOTOR_GROUPS))
    for k, g in enumerate(MOTOR_GROUPS):
        idx = connectome.group_indices(g)
        if len(idx):
            out[k] = rates[idx].mean()
    return out


def _manual_matrix() -> np.ndarray:
    # kolumny: power_L, power_R, steering_L, steering_R, tension_L, tension_R
    return np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],     # thrust
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],    # roll
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # pitch — tylko trim / uczenie
        [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],    # yaw
    ])


class ManualDecoder:
    """Plan C. ``baseline`` odejmuje aktywność spoczynkową, ``hover_thrust`` / ``pitch_trim`` = trymy."""

    def __init__(self, gains=(0.2, 1.0, 1.0, 1.0), hover_thrust: float = 0.5, pitch_trim: float = 0.0) -> None:
        self.M = np.diag(gains) @ _manual_matrix()
        self.hover_thrust = hover_thrust
        self.pitch_trim = pitch_trim
        self.baseline = np.zeros(len(MOTOR_GROUPS))
        self.scale = np.ones(len(MOTOR_GROUPS))

    def calibrate(self, resting_features: np.ndarray) -> None:
        self.baseline = resting_features.copy()

    def calibrate_scale(self, stimulus_features: list[np.ndarray]) -> None:
        """Normalizuje każdą grupę MN do max |odchylenia od baseline| na bodźcach referencyjnych.
        Na pełnym BANC aktywność MN jest rzędu 1e-3, więc bez tego komendy byłyby ~0."""
        dev = np.abs(np.array(stimulus_features) - self.baseline).max(axis=0)
        self.scale = np.where(dev > 1e-9, dev, 1.0)

    def normalized(self, features: np.ndarray) -> np.ndarray:
        return (features - self.baseline) / self.scale

    def decode(self, features: np.ndarray) -> FlightCommand:
        thrust, roll, pitch, yaw = self.M @ self.normalized(features)
        return FlightCommand(self.hover_thrust + thrust, roll, self.pitch_trim + pitch, yaw).clipped()

    def save(self, path, **meta) -> None:
        """Zapisuje wagi (``M``) i trymy; ``baseline``/``scale`` nie — pochodzą z kalibracji sceny."""
        np.savez(path, M=self.M, hover_thrust=self.hover_thrust, pitch_trim=self.pitch_trim,
                 meta=np.array(repr(meta)))

    def load_weights(self, path) -> None:
        """Wczytuje ``M`` i trymy z ``save``. Wywoływać PO kalibracji: ``calibrate_yaw_sign`` odwraca
        ``M[3]``, a wyuczona macierz ma już znak yaw z kalibracji, przy której była uczona."""
        d = np.load(path)
        self.M = d["M"].copy()
        self.hover_thrust, self.pitch_trim = float(d["hover_thrust"]), float(d["pitch_trim"])


class LinearDecoder(ManualDecoder):
    """Plan B. Startuje z wag ręcznych i dostraja je LMS-em do komend docelowych."""

    def __init__(self, lr: float = 0.05, **kw) -> None:
        super().__init__(**kw)
        self.lr = lr

    def fit_step(self, features: np.ndarray, target: FlightCommand) -> float:
        x = self.normalized(features)
        pred = self.M @ x
        tgt = np.array([target.thrust - self.hover_thrust, target.roll, target.pitch - self.pitch_trim, target.yaw])
        err = tgt - pred
        # Znormalizowany LMS: w locie cechy potrafią wyjść daleko poza skalę z kalibracji,
        # a zwykły LMS przy dużym |x| się rozbiega.
        self.M += self.lr * np.outer(err, x) / (1.0 + x @ x)
        return float((err ** 2).mean())


class AdaptiveDecoder(LinearDecoder):
    """Plan A. Node-perturbation: szum na wyjściu, aktualizacja proporcjonalna do (R − R̄)."""

    def __init__(self, noise: float = 0.05, reward_lr: float = 0.1, **kw) -> None:
        super().__init__(**kw)
        self.noise = noise
        self.reward_lr = reward_lr
        self.reward_avg = 0.0
        self._last: tuple[np.ndarray, np.ndarray] | None = None
        self._rng = np.random.default_rng(0)

    def decode(self, features: np.ndarray) -> FlightCommand:
        x = self.normalized(features)
        eps = self._rng.normal(0.0, self.noise, size=4)
        self._last = (x, eps)
        thrust, roll, pitch, yaw = self.M @ x + eps
        return FlightCommand(self.hover_thrust + thrust, roll, self.pitch_trim + pitch, yaw).clipped()

    def reward(self, r: float) -> None:
        if self._last is None:
            return
        x, eps = self._last
        self.M += self.reward_lr * (r - self.reward_avg) * np.outer(eps, x)
        self.reward_avg += 0.05 * (r - self.reward_avg)
