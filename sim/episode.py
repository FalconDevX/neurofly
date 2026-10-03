"""Warunki końca epizodu: wywrotka drona = nieudana próba.

Quadcopter odwrócony śmigłami w dół albo leżący na boku na ziemi nie ma już jak wrócić do lotu,
więc dla uczenia to porażka i reset. Krótki przewrót w powietrzu (manewr) nie kończy próby —
stąd progi czasowe.

Używane w podglądzie (sim/viewer.py), i w WorldEnv (sim/world_env.py) jako terminated z wynikiem porażki.
"""

import numpy as np

UPSIDE_DOWN_TIME = 1.0          # s do góry nogami (przechył > 90°), w powietrzu albo na ziemi
ON_SIDE_TIME = 1.5              # s leżenia na ziemi z dużym przechyłem, prawie bez ruchu
ON_SIDE_TILT = np.deg2rad(60)
REST_SPEED = 0.3                # m/s
REST_SPIN = 1.0                 # rad/s


class CrashDetector:
    def __init__(self, model, body="x2"):
        self.body_id = model.body(body).id
        self.drone_geoms = np.flatnonzero(model.geom_bodyid == self.body_id)
        self.reset()

    def reset(self):
        self.upside_down_for = 0.0
        self.on_side_for = 0.0

    def tilt(self, data):
        """Kąt między osią z drona a pionem [rad]: 0 = poziomo, pi = do góry nogami."""
        return float(np.arccos(np.clip(data.xmat[self.body_id][8], -1, 1)))

    def touching_ground(self, data):
        """Czy dron dotyka czegokolwiek (terenu, bloku)."""
        contacts = data.contact[:data.ncon]
        g1, g2 = contacts.geom1, contacts.geom2
        return bool(np.any(np.isin(g1, self.drone_geoms) ^ np.isin(g2, self.drone_geoms)))

    def update(self, data, dt):
        """Wywołać po krokach symulacji. Zwraca powód wywrotki albo None."""
        tilt = self.tilt(data)
        self.upside_down_for = self.upside_down_for + dt if tilt > np.pi / 2 else 0.0

        resting = (tilt > ON_SIDE_TILT and self.touching_ground(data)
                   and np.linalg.norm(data.qvel[0:3]) < REST_SPEED
                   and np.linalg.norm(data.qvel[3:6]) < REST_SPIN)
        self.on_side_for = self.on_side_for + dt if resting else 0.0

        if self.upside_down_for >= UPSIDE_DOWN_TIME:
            return "do góry nogami"
        if self.on_side_for >= ON_SIDE_TIME:
            return "leży na boku"
        return None
