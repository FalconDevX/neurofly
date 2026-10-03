"""Sterowanie quadcopterem X2.

Interfejs zespołu (ustalony z Osobą 2): RateCommand = ciąg zbiorczy + zadane prędkości kątowe
w układzie drona. RateController zamienia je na siły 4 silników (PID prędkości + mixer).

RateController nie stabilizuje: tylko realizuje zadane prędkości kątowe (przy zerze dron przestaje
się obracać, ale zostaje w bieżącym przechyle) — stabilizacji ma się nauczyć model.
VelocityController (autostabilizacja: poziomowanie, hamowanie, trzymanie wysokości) służy tylko
do ręcznych testów w podglądzie i jest włączany Altem.

Układ drona: x do przodu, y w lewo, z w górę. Dodatni pitch = nos w dół (lot do przodu),
dodatni roll = przechył w prawo (lot w prawo, czyli w -y).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

GRAVITY = 9.81


@dataclass
class RateCommand:
    thrust: float          # N, ciąg zbiorczy wszystkich silników
    roll_rate: float = 0.0   # rad/s wokół x drona
    pitch_rate: float = 0.0  # rad/s wokół y drona
    yaw_rate: float = 0.0    # rad/s wokół z drona


def body_rates(data):
    """Prędkość kątowa w układzie drona (dla freejoint qvel[3:6] jest lokalna)."""
    return data.qvel[3:6].copy()


def euler_zyx(data):
    """(roll, pitch, yaw) z kwaternionu freejointa."""
    w, x, y, z = data.qpos[3:7]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


MOTOR_TAU = 0.04  # s, typowa stała czasowa rozpędzania śmigła drona tej wielkości (do włączenia: motor_tau)


class RateController:
    """PID prędkości kątowych + mixer: RateCommand -> siły silników (data.ctrl).

    motor_tau > 0: opóźnienie silników — siła każdego silnika dochodzi do zadanej z tą stałą czasową
    (filtr pierwszego rzędu), jak prawdziwe śmigło, które musi się rozpędzić. 0 = natychmiast (domyślnie,
    zgodność z dotychczasowym zachowaniem; realizm pod Plan A: MOTOR_TAU).
    """

    def __init__(self, model, body="x2", kp=(18.0, 18.0, 10.0), ki=(4.0, 4.0, 2.0), kd=(0.3, 0.3, 0.0),
                 motor_tau=0.0):
        self.motor_tau = motor_tau
        bid = model.body(body).id
        self.mass = model.body_subtreemass[bid]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_iquat[bid])
        rot = rot.reshape(3, 3)
        self.inertia = rot @ np.diag(model.body_inertia[bid]) @ rot.T
        self.kp, self.ki, self.kd = map(np.array, (kp, ki, kd))
        self.ctrl_lo, self.ctrl_hi = model.actuator_ctrlrange.T.copy()

        # Macierz alokacji: [Fz, tx, ty, tz] = A @ f dla sił silników f (wzdłuż z drona).
        com = model.body_ipos[bid]
        columns = []
        for a in range(model.nu):
            r = model.site_pos[model.actuator_trnid[a, 0]] - com
            gear = model.actuator_gear[a]
            columns.append([gear[2], r[1] * gear[2], -r[0] * gear[2], gear[5]])
        self.mix = np.linalg.inv(np.array(columns).T)
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.motor_force = None  # stan opóźnienia silników; startuje od bieżącego data.ctrl

    @property
    def hover_thrust(self):
        return self.mass * GRAVITY

    def compute(self, cmd: RateCommand, data, dt):
        rate_des = np.array([cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate])
        error = rate_des - body_rates(data)
        self.integral = np.clip(self.integral + error * dt, -0.5, 0.5)
        deriv = (error - self.prev_error) / dt
        self.prev_error = error
        accel = self.kp * error + self.ki * self.integral + self.kd * deriv
        torque = self.inertia @ accel + np.cross(body_rates(data), self.inertia @ body_rates(data))
        # Priorytet: ciąg i roll/pitch, yaw tylko tyle, ile zmieści się w zakresie silników.
        # Samo obcięcie sił przy dużym yaw zerowało jedną parę silników i zawyżało ciąg zbiorczy
        # (yaw ±1 co klatkę → wznoszenie ~2 m/s).
        base = self.mix @ np.array([cmd.thrust, torque[0], torque[1], 0.0])
        yaw = self.mix @ np.array([0.0, 0.0, 0.0, torque[2]])
        with np.errstate(divide="ignore", invalid="ignore"):
            room = np.where(yaw > 0, (self.ctrl_hi - base) / yaw, (self.ctrl_lo - base) / yaw)
        scale = float(np.clip(np.nanmin(np.where(yaw != 0, room, np.inf)), 0.0, 1.0))
        return np.clip(base + scale * yaw, self.ctrl_lo, self.ctrl_hi)

    def apply(self, cmd: RateCommand, model, data):
        forces = self.compute(cmd, data, model.opt.timestep)
        if self.motor_tau > 0:
            if self.motor_force is None:
                self.motor_force = data.ctrl.copy()
            self.motor_force += (1 - np.exp(-model.opt.timestep / self.motor_tau)) * (forces - self.motor_force)
            forces = self.motor_force
        data.ctrl[:] = forces


class VelocityController:
    """Autostabilizacja do ręcznych testów (Alt w podglądzie): prędkość zadana -> RateCommand.

    Po puszczeniu klawiszy (prędkości 0) dron hamuje i trzyma wysokość. Używa prawdziwego stanu
    z symulatora, więc nie jest częścią pętli modelu — tam idzie sam RateController.
    """

    def __init__(self, rate_ctrl: RateController, max_tilt=np.deg2rad(25),
                 k_vel=2.0, k_att=7.0, k_alt=3.0, k_vz=4.0):
        self.rate_ctrl = rate_ctrl
        self.max_tilt = max_tilt
        self.k_vel, self.k_att, self.k_alt, self.k_vz = k_vel, k_att, k_alt, k_vz
        self.z_ref = None

    def reset(self):
        self.z_ref = None
        self.rate_ctrl.reset()

    def compute(self, data, forward=0.0, left=0.0, up=0.0, yaw_rate=0.0):
        """forward/left/up w m/s, yaw_rate w rad/s (dodatni = w lewo)."""
        roll, pitch, yaw = euler_zyx(data)
        vel_world = data.qvel[0:3]
        c, s = np.cos(yaw), np.sin(yaw)
        v_fwd = c * vel_world[0] + s * vel_world[1]
        v_left = -s * vel_world[0] + c * vel_world[1]

        # Prędkość pozioma -> przyspieszenie -> przechylenie.
        a_fwd = self.k_vel * (forward - v_fwd)
        a_left = self.k_vel * (left - v_left)
        pitch_des = np.clip(a_fwd / GRAVITY, -self.max_tilt, self.max_tilt)
        roll_des = np.clip(-a_left / GRAVITY, -self.max_tilt, self.max_tilt)

        # Wysokość: przy wznoszeniu/opadaniu śledzimy prędkość, po puszczeniu trzymamy bieżącą wysokość.
        z = data.qpos[2]
        if self.z_ref is None or up:
            self.z_ref = z
        a_z = self.k_alt * (self.z_ref - z) + self.k_vz * (up - vel_world[2])
        tilt = max(np.cos(roll) * np.cos(pitch), 0.5)
        thrust = self.rate_ctrl.mass * (GRAVITY + a_z) / tilt

        return RateCommand(
            thrust=thrust,
            roll_rate=self.k_att * (roll_des - roll),
            pitch_rate=self.k_att * (pitch_des - pitch),
            yaw_rate=yaw_rate,
        )

    def apply(self, model, data, **setpoints):
        cmd = self.compute(data, **setpoints)
        self.rate_ctrl.apply(cmd, model, data)
        return cmd
