"""Ślad lotu: linia za dronem pokazująca, którędy leciał (tylko wizualizacja — nie wpływa na fizykę).

Punkt dopisywany co MIN_STEP ruchu, pamiętane ostatnie MAX_POINTS (~30 m trasy). Starsze odcinki bledną,
więc widać kierunek ruchu. Rysowany jak łopaty: dopisany do MjvScene (viewer.user_scn albo renderer.scene).
"""

from collections import deque

import mujoco
import numpy as np

MAX_POINTS = 300
MIN_STEP = 0.10                  # m (co 10 cm: ten sam wygląd, połowa kosztu renderu)
WIDTH = 0.014                    # m, promień kapsuły odcinka (widoczny z daleka, nie za gruby z bliska)
NEW_RGBA = np.array([0.15, 0.85, 1.0, 0.95], np.float32)   # najnowszy odcinek
OLD_RGBA = np.array([0.15, 0.45, 1.0, 0.15], np.float32)   # najstarszy (prawie przezroczysty)
OFFSET = np.array([0.0, 0.0, 0.03])  # trochę ponad środkiem drona, żeby nie znikał w kadłubie


class Trail:
    def __init__(self, max_points=MAX_POINTS, min_step=MIN_STEP):
        self.points = deque(maxlen=max_points)
        self.min_step = min_step
        self.visible = True

    def reset(self):
        self.points.clear()

    def add(self, position):
        """Dopisuje pozycję drona, jeśli przesunął się co najmniej o min_step od ostatniego punktu."""
        p = np.asarray(position, float) + OFFSET
        if not self.points or np.linalg.norm(p - self.points[-1]) >= self.min_step:
            self.points.append(p)

    def draw(self, scn):
        """Dopisuje odcinki śladu do sceny (od najstarszego, bladego, do najnowszego)."""
        if not self.visible or len(self.points) < 2:
            return
        n = len(self.points) - 1
        pts = list(self.points)
        for i in range(n):
            if scn.ngeom >= scn.maxgeom:
                return
            t = (i + 1) / n  # 0 = najstarszy, 1 = najnowszy
            rgba = (OLD_RGBA + t * (NEW_RGBA - OLD_RGBA)).astype(np.float32)
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, WIDTH, pts[i], pts[i + 1])
            scn.ngeom += 1
