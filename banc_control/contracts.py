"""Kontrakty danych między Osobą 1 (wzrok → BANC), Osobą 2 (BANC → sterowanie) i Osobą 3 (dron).

Format wejścia od Osoby 1 jest zgodny z dokumentem "Drosophila Vision → BANC — plan integracji":
    [{"banc_root_id": int, "cell_type": str, "activity": float}, ...]
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class BancActivation:
    """Pojedynczy rekord aktywności przekazany przez Osobę 1."""

    banc_root_id: int
    cell_type: str
    activity: float


@dataclass(frozen=True)
class VisualBatch:
    """Szybka ścieżka w tym samym procesie: te same dane co ``list[BancActivation]``, jako tablice.

    Tworzenie ~18k obiektów ``BancActivation`` co klatkę kosztuje ~25 ms; tablice ~0 ms.
    ``root_ids`` ma być tym samym obiektem z klatki na klatkę (kontroler cache'uje indeksy).
    """

    root_ids: np.ndarray  # (N,) int64, banc_888_id
    activity: np.ndarray  # (N,) float


def parse_visual_activity(payload: str | list[dict]) -> list[BancActivation]:
    """Parsuje JSON (lub listę słowników) od Osoby 1."""
    records = json.loads(payload) if isinstance(payload, str) else payload
    return [
        BancActivation(int(r["banc_root_id"]), str(r.get("cell_type", "")), float(r["activity"]))
        for r in records
    ]


@dataclass
class ImuState:
    """Odczyt IMU drona od Osoby 3 — u muszki odpowiada to sprzężeniu z halter."""

    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad/s: roll, pitch, yaw rate
    accel: tuple[float, float, float] = (0.0, 0.0, -9.81)  # m/s^2


@dataclass
class FlightCommand:
    """Wyjście Osoby 2 → wejście Osoby 3 (mixer quadcoptera).

    thrust w [0, 1] (0.5 ≈ zawis po kalibracji), roll/pitch/yaw w [-1, 1].
    """

    thrust: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    debug: dict = field(default_factory=dict)

    def clipped(self) -> FlightCommand:
        clip = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731
        return FlightCommand(
            thrust=clip(self.thrust, 0.0, 1.0),
            roll=clip(self.roll, -1.0, 1.0),
            pitch=clip(self.pitch, -1.0, 1.0),
            yaw=clip(self.yaw, -1.0, 1.0),
            debug=self.debug,
        )

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("debug")
        return json.dumps(d)
