"""DroneEnv: X2 z oczami w MuJoCo jako środowisko dla pętli wzrok → BANC → sterowanie.

    env = DroneEnv()
    obs, info = env.reset(bearing_deg=60)          # cel 60° w prawo
    obs, reward, terminated, truncated, info = env.step(FlightCommand(...))

``obs``: ``Observation(left, right, imu)`` — surowe klatki oczu uint8 (512, 450, 3) i IMU w konwencji
``banc_control.ImuState`` jako słownik (do ``VisionClient.step``). ``info``: stan z symulatora
(kąt do celu, wysokość, pozycja) — tylko do nagrody i metryk, nie jako wejście sieci.

Jeden krok = jedna klatka kamer (domyślnie 30 FPS, tyle co FlyVis w serwerze) = ``substeps`` kroków
fizyki; regulator działa w każdym kroku fizyki.

Przełożenie ``FlightCommand`` → ``RateCommand`` to NASZE założenie (``Pilot``):
  - ``mode="angle"`` (domyślnie, wariant demo Planu C): symulator trzyma poziom i hamuje dryf
    (``VelocityController`` Osoby 3, używa prawdziwej prędkości z symulatora — jak dron z GPS/optical flow);
    BANC daje yaw (±1 → ±``max_yaw_rate``) i thrust jako prędkość wznoszenia
    ((thrust − 0.5) · ``climb_gain``, martwa strefa ``thrust_deadband``); ``thrust_mode="hold"``:
    wysokość trzyma symulator, thrust z BANC tylko w logu (w v888 thrust w locie jest stale < 0.5
    i dron opada); roll z BANC tylko gdy
    ``use_roll`` (wtedy ±1 → ±``max_speed`` w bok), pitch ignorowany;
  - ``mode="acro"``: komendy jako znormalizowane prędkości kątowe (plan Osoby 3), thrust 0.5 = zawis.
Znaki: banc yaw + = w prawo = −ω_z MuJoCo; roll + = prawe skrzydło w dół = +ω_x; pitch + = nos w dół = +ω_y.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from banc_control.contracts import FlightCommand
from sim.control import RateCommand, RateController, VelocityController, euler_zyx
from sim.world import build_scene


@dataclass
class Observation:
    left: np.ndarray | None
    right: np.ndarray | None
    imu: dict


@dataclass
class Gust:
    """Zakłócenie w oknie czasu: siła [N] w układzie świata i moment [N·m] (z = odchylenie, + = w lewo)."""

    start: float
    duration: float
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Scenario:
    name: str
    bearing_deg: float = 0.0       # cel względem kierunku lotu na starcie, + = w prawo
    duration: float = 8.0          # s
    gusts: list[Gust] = field(default_factory=list)


SCENARIOS = {
    "hover": Scenario("hover", 0.0, 6.0),
    "turn_right": Scenario("turn_right", 60.0, 8.0),
    "turn_left": Scenario("turn_left", -60.0, 8.0),
    # Podmuch: moment odchylenia (obraca w lewo) + boczna siła, cel na wprost.
    "gust": Scenario("gust", 0.0, 8.0, [Gust(2.0, 0.4, force=(0.0, 2.0, 0.0), torque=(0.0, 0.0, 0.4))]),
}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Pilot:
    """FlightCommand (BANC) → RateCommand (regulator Osoby 3), patrz docstring modułu."""

    def __init__(self, rate_ctrl: RateController, mode: str = "angle", max_yaw_rate: float = 1.0,
                 climb_gain: float = 1.0, thrust_deadband: float = 0.05, use_roll: bool = False,
                 max_speed: float = 1.0, max_rate: float = 2.0, thrust_mode: str = "banc") -> None:
        if mode not in ("angle", "acro"):
            raise ValueError(f"mode: 'angle' albo 'acro', jest {mode!r}")
        if thrust_mode not in ("banc", "hold"):
            raise ValueError(f"thrust_mode: 'banc' albo 'hold', jest {thrust_mode!r}")
        self.thrust_mode = thrust_mode
        self.rate_ctrl = rate_ctrl
        self.velocity = VelocityController(rate_ctrl)
        self.mode = mode
        self.max_yaw_rate, self.climb_gain, self.thrust_deadband = max_yaw_rate, climb_gain, thrust_deadband
        self.use_roll, self.max_speed, self.max_rate = use_roll, max_speed, max_rate

    def reset(self) -> None:
        self.velocity.reset()

    def rate_command(self, cmd: FlightCommand, data) -> RateCommand:
        if self.mode == "acro":
            return RateCommand(thrust=self.rate_ctrl.hover_thrust * 2.0 * cmd.thrust,
                               roll_rate=self.max_rate * cmd.roll, pitch_rate=self.max_rate * cmd.pitch,
                               yaw_rate=-self.max_rate * cmd.yaw)
        climb = 0.0 if self.thrust_mode == "hold" else cmd.thrust - 0.5
        climb = 0.0 if abs(climb) < self.thrust_deadband else self.climb_gain * climb
        left = -self.max_speed * cmd.roll if self.use_roll else 0.0
        return self.velocity.compute(data, forward=0.0, left=left, up=climb,
                                     yaw_rate=-self.max_yaw_rate * cmd.yaw)


class DroneEnv:
    def __init__(self, eyes: bool = True, fps: float = 30.0, substeps: int = 5, start_z: float = 1.0,
                 beacon_distance: float = 4.0, alt_weight: float = 0.5, min_z: float = 0.15,
                 **pilot_kw) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(build_scene()))
        self.model.opt.timestep = 1.0 / (fps * substeps)
        self.data = mujoco.MjData(self.model)
        self.fps, self.substeps = fps, substeps
        self.start_z, self.beacon_distance = start_z, beacon_distance
        self.alt_weight, self.min_z = alt_weight, min_z
        self.rate_ctrl = RateController(self.model)
        self.pilot = Pilot(self.rate_ctrl, **pilot_kw)
        self.body = self.model.body("x2").id
        self.beacon = self.model.body("beacon").mocapid[0]
        self.gyro = self.model.sensor("body_gyro").adr[0]
        self.accel = self.model.sensor("body_linacc").adr[0]
        self.eyes = None
        if eyes:
            from visual_pipeline.drone_eyes import MujocoEyes
            self.eyes = MujocoEyes(self.model)
            # Mapa cieni reflektora śledzącego drona daje przy horyzoncie ciemną plamę, która jeździ
            # razem z dronem (fałszywy ruch dla FlyVis) — oczy renderują bez cieni.
            self.eyes.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.chase = None
        self.scenario = SCENARIOS["hover"]
        self.target_heading = 0.0

    # --- stan ---------------------------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(self.data.time)

    def heading(self) -> float:
        """Kurs drona, + = w prawo (MuJoCo yaw + = w lewo)."""
        return -euler_zyx(self.data)[2]

    def bearing(self) -> float:
        """Kąt do celu względem kierunku lotu [rad], + = cel w prawo. Z pozycji, nie z kursu startowego."""
        d = self.data.mocap_pos[self.beacon][:2] - self.data.qpos[:2]
        return float(wrap(-np.arctan2(d[1], d[0]) - self.heading()))

    def imu(self) -> dict:
        gx, gy, gz = self.data.sensordata[self.gyro:self.gyro + 3]
        return {"gyro": [float(gx), float(gy), float(-gz)],
                "accel": self.data.sensordata[self.accel:self.accel + 3].tolist()}

    def observe(self) -> Observation:
        left, right = self.eyes.render(self.data) if self.eyes else (None, None)
        return Observation(left, right, self.imu())

    def info(self) -> dict:
        roll, pitch, _ = euler_zyx(self.data)
        return {"t": self.time, "bearing": self.bearing(), "heading": self.heading(),
                "z": float(self.data.qpos[2]), "xy": self.data.qpos[:2].tolist(),
                "roll": float(roll), "pitch": float(pitch)}

    # --- epizod ---------------------------------------------------------------------------------
    def _set_pose(self, heading: float) -> None:
        self.data.qpos[:3] = (0.0, 0.0, self.start_z)
        self.data.qpos[3:7] = (np.cos(-heading / 2), 0.0, 0.0, np.sin(-heading / 2))
        self.data.qvel[:] = 0.0

    def _place_beacon(self, heading: float) -> None:
        """Cel w odległości ``beacon_distance`` w kierunku ``heading`` (+ = w prawo) od startu."""
        self.data.mocap_pos[self.beacon] = (self.beacon_distance * np.cos(heading),
                                            -self.beacon_distance * np.sin(heading), 0.0)

    def reset(self, scenario: str | Scenario | None = None, bearing_deg: float | None = None,
              rng: np.random.Generator | None = None) -> tuple[Observation, dict]:
        """Zawis na ``start_z``, cel pod kątem ``bearing_deg`` (domyślnie z scenariusza)."""
        if isinstance(scenario, str):
            scenario = SCENARIOS[scenario]
        self.scenario = scenario or Scenario("custom")
        bearing = np.deg2rad(self.scenario.bearing_deg if bearing_deg is None else bearing_deg)
        mujoco.mj_resetData(self.model, self.data)
        heading0 = rng.uniform(-np.pi, np.pi) if rng is not None else 0.0
        self.target_heading = heading0 + bearing
        self._set_pose(heading0)
        self._place_beacon(self.target_heading)
        self.data.ctrl[:] = self.rate_ctrl.hover_thrust / self.model.nu
        mujoco.mj_forward(self.model, self.data)
        self.pilot.reset()
        return self.observe(), self.info()

    def calibration_render(self, bearing: float) -> tuple[np.ndarray, np.ndarray]:
        """``render(bearing)`` dla ``VisionClient.calibrate``: dron w zawisie, cel pod ``bearing`` [rad].

        Ustawia pozę bez fizyki; po kalibracji trzeba zrobić ``reset``.
        """
        self._place_beacon(0.0)
        self._set_pose(-bearing)
        mujoco.mj_forward(self.model, self.data)
        return self.eyes.render(self.data)

    def reward(self, info: dict) -> float:
        return -abs(info["bearing"]) / np.pi - self.alt_weight * abs(info["z"] - self.start_z)

    def _apply_gusts(self) -> None:
        self.data.xfrc_applied[self.body] = 0.0
        for g in self.scenario.gusts:
            if g.start <= self.time < g.start + g.duration:
                self.data.xfrc_applied[self.body] += (*g.force, *g.torque)

    def step(self, cmd: FlightCommand | dict) -> tuple[Observation, float, bool, bool, dict]:
        if isinstance(cmd, dict):
            cmd = FlightCommand(**{k: cmd[k] for k in ("thrust", "roll", "pitch", "yaw")})
        cmd = cmd.clipped()
        dt = self.model.opt.timestep
        for _ in range(self.substeps):
            self._apply_gusts()
            self.data.ctrl[:] = self.rate_ctrl.compute(self.pilot.rate_command(cmd, self.data), self.data, dt)
            mujoco.mj_step(self.model, self.data)
        info = self.info()
        info["cmd"] = {"thrust": cmd.thrust, "roll": cmd.roll, "pitch": cmd.pitch, "yaw": cmd.yaw}
        tilt = np.arccos(np.clip(np.cos(info["roll"]) * np.cos(info["pitch"]), -1, 1))
        terminated = info["z"] < self.min_z or tilt > np.deg2rad(60)
        truncated = self.time >= self.scenario.duration - 1e-9
        return self.observe(), self.reward(info), bool(terminated), bool(truncated), info

    def render_chase(self) -> np.ndarray:
        """Widok z kamery za dronem (``track`` z modelu X2), do wideo."""
        if self.chase is None:
            self.chase = mujoco.Renderer(self.model, 360, 480)
        self.chase.update_scene(self.data, camera="track")
        return self.chase.render().copy()

    def close(self) -> None:
        for r in (self.chase, getattr(self.eyes, "renderer", None)):
            if r is not None:
                r.close()



def summarize(log: list[dict], fps: float = 30.0, settle_deg: float = 10.0) -> dict:
    """Metryki epizodu z listy ``info``: błąd kursu w czasie, wysokość, dryf."""
    b = np.rad2deg(np.abs([i["bearing"] for i in log]))
    z = np.array([i["z"] for i in log])
    tail = b[-int(fps):]
    outside = np.nonzero(b > settle_deg)[0]  # czas ustalenia: od kiedy |błąd| ≤ settle_deg do końca
    if not len(outside):
        settled = 0.0
    elif outside[-1] == len(b) - 1:
        settled = None
    else:
        settled = (outside[-1] + 1) / fps
    return {"start_deg": float(b[0]), "final_deg": float(b[-1]), "mean_last_1s_deg": float(tail.mean()),
            "max_deg": float(b.max()), "settle_time_s": settled,
            "z_min": float(z.min()), "z_max": float(z.max()),
            "drift_m": float(np.hypot(*log[-1]["xy"])), "duration_s": float(log[-1]["t"])}
