"""Obracające się łopaty śmigieł drona X2 — wyłącznie wizualizacja.

Łopaty nie istnieją w modelu fizycznym (ciąg i moment dają aktuatory thrust1..4),
tylko są dorysowywane do sceny (MjvScene) w każdej klatce: w passive viewerze
przez viewer.user_scn, przy renderowaniu offscreen przez renderer.scene.

Prędkość obrotu ∝ sqrt(ciąg), bo ciąg śmigła ∝ omega². Prawdziwe ~1000 rad/s przy
60 FPS dałyby efekt stroboskopowy, więc prędkość wizualna jest skalowana do
MAX_VISUAL_OMEGA. (Bez tarczy rozmycia: dorysowane geometrie zawsze rzucają pełny cień.)
"""

import mujoco
import numpy as np

BLADES_PER_PROP = 2
BLADE_RADIUS = 0.125          # m, zgodnie z kolizyjną elipsoidą rotora (.13)
BLADE_CHORD = 0.012           # m, połowa szerokości łopaty
BLADE_THICKNESS = 0.0018      # m, połowa grubości łopaty
BLADE_PITCH = np.deg2rad(12)  # skręcenie łopaty wokół jej osi
HUB_RADIUS = 0.012
HUB_HALF_HEIGHT = 0.006
LIFT = 0.006                  # łopaty nad górną krawędzią silnika z siatki
MAX_VISUAL_OMEGA = 40.0       # rad/s, < pi/2 na klatkę przy 60 FPS dla 2 łopat

BLADE_RGBA = np.array([0.12, 0.12, 0.13, 1.0], dtype=np.float32)
HUB_RGBA = np.array([0.35, 0.35, 0.37, 1.0], dtype=np.float32)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class PropellerVisuals:
    """Stan kątów łopat i dorysowywanie ich do MjvScene."""

    def __init__(self, model, body="x2", n_props=4):
        self.body_id = model.body(body).id
        self.site_ids = [model.site(f"prop{i + 1}").id for i in range(n_props)]
        self.actuator_ids = [model.actuator(f"thrust{i + 1}").id for i in range(n_props)]
        # Moment reakcji na korpus (gear[5]) jest przeciwny do kierunku obrotu śmigła.
        self.spin_dir = np.array([-np.sign(model.actuator_gear[a, 5]) for a in self.actuator_ids])
        self.ctrl_max = np.array([model.actuator_ctrlrange[a, 1] for a in self.actuator_ids])
        self.angles = np.random.default_rng(0).uniform(0, np.pi, n_props)

    def reset(self):
        self.angles[:] = np.random.default_rng(0).uniform(0, np.pi, len(self.angles))

    def speed_fraction(self, data):
        ctrl = np.clip(data.ctrl[self.actuator_ids], 0, self.ctrl_max)
        return np.sqrt(ctrl / self.ctrl_max)

    def advance(self, data, dt):
        omega = MAX_VISUAL_OMEGA * self.speed_fraction(data) * self.spin_dir
        self.angles = (self.angles + omega * dt) % (2 * np.pi)

    def draw(self, scn, data):
        """Dopisuje geometrie łopat na końcu scn.geoms (wywołać po update_scene / pod viewer.lock())."""
        body_rot = data.xmat[self.body_id].reshape(3, 3)
        for i, site in enumerate(self.site_ids):
            hub = data.site_xpos[site] + body_rot[:, 2] * LIFT
            for k in range(BLADES_PER_PROP):
                azimuth = _rot_z(self.angles[i] + 2 * np.pi * k / BLADES_PER_PROP)
                rot = body_rot @ azimuth @ _rot_x(BLADE_PITCH * self.spin_dir[i])
                pos = hub + body_rot @ azimuth @ np.array([BLADE_RADIUS / 2, 0, 0])
                self._add(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          [BLADE_RADIUS / 2, BLADE_CHORD, BLADE_THICKNESS], pos, rot, BLADE_RGBA)
            self._add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER,
                      [HUB_RADIUS, HUB_HALF_HEIGHT, 0], hub, body_rot, HUB_RGBA)

    @staticmethod
    def _add(scn, geom_type, size, pos, rot, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], geom_type, np.asarray(size, dtype=float),
                            np.asarray(pos, dtype=float), np.asarray(rot, dtype=float).ravel(), rgba)
        scn.ngeom += 1
