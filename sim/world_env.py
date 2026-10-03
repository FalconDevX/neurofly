"""WorldEnv — środowisko treningowe (Gymnasium) drona X2: lot do beacona na losowym świecie.

(Inne niż sim/env.py: DroneEnv tam to pętla FlyVis + BANC na płaskiej scenie ze słupem; WorldEnv to
pełny świat Osoby 3 — teren, bloki, cel z flagą, wiatr, wywrotka — z interfejsem Gymnasium.)

    from sim.world_env import WorldEnv
    env = WorldEnv()                      # scena beacon, tryb acro (Plan A), oczy włączone
    obs, info = env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step([0.5, 0, 0, 0])  # zawis

Uruchamianie z gotowym sterownikiem zastępczym: python -m sim.run_env (patrz tam).
Ręczne testowanie dalej przez python -m sim.viewer.

Akcja = FlightCommand Osoby 2 (banc_control/contracts.py), 4 liczby:
    thrust [0, 1]  — 0.5 = ciąg zawisu, 1 = 2x zawis
    roll, pitch, yaw [-1, 1]
      control="acro" (domyślnie, Plan A): zadane prędkości kątowe (x MAX_RATE) — bez autostabilizacji,
                                           stabilizacji ma się nauczyć model
      control="angle" (Plan C, zapasowy): roll/pitch = zadany przechył (x MAX_TILT), symulator trzyma kąt;
                                           yaw dalej jako prędkość
Obserwacja (tylko to, co ma muszka):
    eyes: (2, 512, 450, 3) uint8 — lewe i prawe oko (MujocoEyes Osoby 1), gdy eyes=True
    imu:  (6,) float32 — żyroskop [rad/s] i akcelerometr [m/s^2] w układzie drona (ImuState)
Prawdziwy stan (pozycja, prędkość, cel, odległość) jest tylko w info — do nagrody, metryk i testów.

Koniec epizodu (info["outcome"]): "cel" (dron nad polem lądowania), "wywrotka: ..." (sim/episode.py),
"poza planszą" (ściana albo > 8 m) — terminated; "limit czasu" — truncated.
Nagroda: postęp w stronę celu [m] - koszt czasu, +/- premia/kara na końcu (stałe REWARD_*).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from sim.control import RateCommand, RateController, euler_zyx
from sim.episode import CrashDetector
from sim.propellers import PropellerVisuals
from sim.target import Target
from sim.terrain import load_scene, outside_arena, randomize, upload_terrain
from sim.wind import Wind
from visual_pipeline.drone_eyes import EYE_H, EYE_W, MujocoEyes

BEACON_SCENE = Path(__file__).parent / "assets" / "scene_beacon.xml"

MAX_RATE = np.array([3.0, 3.0, 1.5])   # rad/s dla roll, pitch, yaw = 1 (acro)
MAX_TILT = np.deg2rad(25)              # rad dla roll, pitch = 1 (angle)
ANGLE_GAIN = 7.0                       # 1/s, kąt -> prędkość kątowa w trybie angle

REWARD_PROGRESS = 1.0      # za każdy metr zbliżenia do celu
REWARD_TIME = -0.01        # za krok (żeby nie wisiał w miejscu)
REWARD_GOAL = 10.0
REWARD_CRASH = -10.0       # wywrotka albo wyjście poza planszę

START_TILT = np.deg2rad(5)  # losowe zaburzenie startu (start_noise=True)
START_VEL = 0.3             # m/s
START_SPIN = 0.3            # rad/s


def to_action(cmd, hover_thrust):
    """RateCommand (sim/control.py) -> akcja WorldEnv [thrust 0..1, roll, pitch, yaw -1..1] w trybie acro."""
    rates = np.array([cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate]) / MAX_RATE
    return np.concatenate([[cmd.thrust / (2 * hover_thrust)], np.clip(rates, -1, 1)])


class WorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, scene=BEACON_SCENE, control="acro", eyes=True, substeps=3, max_time=60.0,
                 start_noise=True, wind=False, wind_speed=8.0, render_mode=None):
        assert control in ("acro", "angle")
        self.model = load_scene(scene)
        self.data = mujoco.MjData(self.model)
        self.control = control
        self.substeps = substeps                     # kroków fizyki (10 ms) na krok środowiska / klatkę oczu
        self.dt = substeps * self.model.opt.timestep
        self.max_time = max_time
        self.start_noise = start_noise
        self.wind = Wind(mean_speed=wind_speed, gust_speed=0.375 * wind_speed)  # 8 m/s -> podmuchy ±3
        self.wind_on = wind
        self.render_mode = render_mode

        self.rate_ctrl = RateController(self.model)
        self.crash = CrashDetector(self.model)
        self.target = Target.from_model(self.model)
        self.has_terrain = self.model.nhfield > 0
        self.drone_id = self.model.body("x2").id
        self.key_id = self.model.key("hover").id
        self.gyro_adr = self.model.sensor_adr[self.model.sensor("body_gyro").id]
        self.accel_adr = self.model.sensor_adr[self.model.sensor("body_linacc").id]
        self.eyes = MujocoEyes(self.model) if eyes else None
        self._renderer = None
        self._props = None

        self.action_space = spaces.Box(np.array([0, -1, -1, -1], np.float32), np.ones(4, np.float32))
        obs = {"imu": spaces.Box(-np.inf, np.inf, (6,), np.float32)}
        if eyes:
            obs["eyes"] = spaces.Box(0, 255, (2, EYE_H, EYE_W, 3), np.uint8)
        self.observation_space = spaces.Dict(obs)
        self.world_seed = None

    # --- API Gymnasium ---------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.world_seed = options.get("world_seed", int(self.np_random.integers(1_000_000)))
        if self.has_terrain:
            randomize(self.model, self.data, self.world_seed)
            for renderer in (self.eyes.renderer if self.eyes else None, self._renderer):
                if renderer is not None:
                    upload_terrain(renderer, self.model)  # inaczej kamery widzą poprzedni teren
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        if self.start_noise:
            self._disturb_start()
        mujoco.mj_forward(self.model, self.data)
        self.rate_ctrl.reset()
        self.crash.reset()
        self.wind.reset(self.world_seed)
        self.wind.enabled = options.get("wind", self.wind_on)
        self.wind.step(self.model, 0.0)
        self.prev_distance = self._distance()
        return self._observation(), self._info(outcome=None)

    def step(self, action):
        cmd = self.command(action)
        for _ in range(self.substeps):
            self.wind.step(self.model, self.model.opt.timestep)
            self.rate_ctrl.apply(self._rate_command(cmd), self.model, self.data)
            mujoco.mj_step(self.model, self.data)

        distance = self._distance()
        reward = REWARD_PROGRESS * (self.prev_distance - distance) + REWARD_TIME
        self.prev_distance = distance

        outcome, terminated = None, True
        crashed = self.crash.update(self.data, self.dt)
        if self.target is not None and self.target.reached(self.data, self.drone_id):
            outcome, reward = "cel", reward + REWARD_GOAL
        elif crashed:
            outcome, reward = f"wywrotka: {crashed}", reward + REWARD_CRASH
        elif self.has_terrain and outside_arena(self.model, self.data, self.drone_id):
            outcome, reward = "poza planszą", reward + REWARD_CRASH
        else:
            terminated = False
        truncated = not terminated and self.data.time >= self.max_time
        if truncated:
            outcome = "limit czasu"
        return self._observation(), float(reward), terminated, truncated, self._info(outcome)

    def render(self):
        """Klatka z kamery za dronem (rgb_array) — do wideo i podglądu."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._props = PropellerVisuals(self.model)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._cam.trackbodyid = self.drone_id
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.5, 0, -15
        self._props.advance(self.data, self.dt)
        self._renderer.update_scene(self.data, self._cam)
        self._props.draw(self._renderer.scene, self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- akcja ---------------------------------------------------------------------------------

    @staticmethod
    def command(action):
        """Akcja jako ndarray [thrust, roll, pitch, yaw] albo obiekt z polami FlightCommand Osoby 2."""
        if hasattr(action, "thrust"):
            action = [action.thrust, action.roll, action.pitch, action.yaw]
        return np.clip(np.asarray(action, dtype=float), [0, -1, -1, -1], 1)

    def _rate_command(self, cmd):
        thrust = 2 * cmd[0] * self.rate_ctrl.hover_thrust
        if self.control == "acro":
            roll_rate, pitch_rate, yaw_rate = cmd[1:] * MAX_RATE
        else:  # angle: symulator trzyma zadany przechył (Plan C)
            roll, pitch, _ = euler_zyx(self.data)
            roll_rate = ANGLE_GAIN * (cmd[1] * MAX_TILT - roll)
            pitch_rate = ANGLE_GAIN * (cmd[2] * MAX_TILT - pitch)
            yaw_rate = cmd[3] * MAX_RATE[2]
        return RateCommand(thrust, roll_rate, pitch_rate, yaw_rate)

    # --- obserwacja i info ------------------------------------------------------------------------

    def imu(self):
        """(gyro[3] rad/s, accel[3] m/s^2) w układzie drona — jak ImuState w banc_control."""
        s = self.data.sensordata
        return s[self.gyro_adr:self.gyro_adr + 3].copy(), s[self.accel_adr:self.accel_adr + 3].copy()

    def _observation(self):
        gyro, accel = self.imu()
        obs = {"imu": np.concatenate([gyro, accel]).astype(np.float32)}
        if self.eyes is not None:
            obs["eyes"] = np.stack(self.eyes.render(self.data))
        return obs

    def _distance(self):
        return self.target.distance(self.data, self.drone_id) if self.target is not None else 0.0

    def _info(self, outcome):
        roll, pitch, yaw = euler_zyx(self.data)
        return {
            "outcome": outcome,
            "time": self.data.time,
            "world_seed": self.world_seed,
            "position": self.data.qpos[0:3].copy(),
            "velocity": self.data.qvel[0:3].copy(),
            "attitude": np.array([roll, pitch, yaw]),
            "target": self.target.position(self.data) if self.target is not None else None,
            "distance": self.prev_distance,
            "wind": self.wind.velocity,
        }

    def _disturb_start(self):
        rng = self.np_random
        quat = np.zeros(4)
        roll, pitch = rng.uniform(-START_TILT, START_TILT, 2)
        mujoco.mju_euler2Quat(quat, np.array([roll, pitch, 0.0]), "xyz")
        self.data.qpos[3:7] = quat
        self.data.qvel[0:3] = rng.uniform(-START_VEL, START_VEL, 3)
        self.data.qvel[3:6] = rng.uniform(-START_SPIN, START_SPIN, 3)
