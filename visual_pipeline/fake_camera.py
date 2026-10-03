"""Syntetyczne kamery stereo zamiast Gazebo: niebo, ziemia i ciemny pionowy pas (cel).

Konwencje jak w ``banc_control.stubs``: azymut i kurs + = w prawo, roll + = prawe skrzydło
w dół. Każde oko to rzut równoprostokątny (azymut × elewacja) o rozmiarze, jakiego oczekuje
flygym Retina. Kamery mają naturalny układ obrazu: prawa patrzy w prawo, więc przód widzi
po lewej stronie kadru, lewa odwrotnie; ``RetinaMapper`` odbija lewe oko, co daje spójny obraz.
"""

from __future__ import annotations

import numpy as np

H, W = 512, 450        # flygym Retina raw image (rows, cols)
SPAN_AZ = np.deg2rad(160)  # poziome pole widzenia jednego oka
SPAN_EL = np.deg2rad(150)
EYE_CENTER = np.deg2rad(70)  # oś oka od kierunku lotu → 20° obuocznego nakładania z przodu

SKY, GROUND, BAR = 200, 70, 15
BAR_WIDTH = np.deg2rad(15)
# Retina uśrednia ~300 pikseli na omatidium, więc liczymy co 4. piksel i powielamy (16× szybciej).
STEP = 4


def _eye_directions(sign: float) -> np.ndarray:
    """(H/STEP, W/STEP, 3) kierunki pikseli w układzie drona: x przód, y prawo, z góra."""
    col = ((np.arange(0, W, STEP) + STEP / 2) / W - 0.5) * SPAN_AZ
    row = (0.5 - (np.arange(0, H, STEP) + STEP / 2) / H) * SPAN_EL
    az = sign * EYE_CENTER + col[None, :]
    el = np.repeat(row[:, None], len(col), axis=1)
    az = np.broadcast_to(az, el.shape)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


class FakeStereoCamera:
    def __init__(self, beacon_azimuth: float = 0.0) -> None:
        self.beacon_azimuth = beacon_azimuth  # rad, w układzie świata
        self._dirs = {"left": _eye_directions(-1.0), "right": _eye_directions(1.0)}

    def render(self, yaw: float = 0.0, roll: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Zwraca (lewa, prawa) klatkę uint8 (H, W, 3) dla danej orientacji drona."""
        c, s = np.cos(roll), np.sin(roll)
        rot = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])  # roll + → prawa strona w dół
        frames = []
        for eye in ("left", "right"):
            d = self._dirs[eye] @ rot.T
            el = np.arcsin(np.clip(d[..., 2], -1, 1))
            az = np.arctan2(d[..., 1], d[..., 0]) + yaw
            img = np.where(el > 0, SKY, GROUND).astype(np.uint8)
            rel = np.angle(np.exp(1j * (az - self.beacon_azimuth)))
            img[np.abs(rel) < BAR_WIDTH / 2] = BAR
            img = np.repeat(np.repeat(img, STEP, axis=0), STEP, axis=1)[:H, :W]
            frames.append(np.repeat(img[..., None], 3, axis=2))
        return frames[0], frames[1]

    def bearing(self, yaw: float) -> float:
        """Kąt do celu względem kursu drona (rad, + = cel po prawej)."""
        return float(np.angle(np.exp(1j * (self.beacon_azimuth - yaw))))
