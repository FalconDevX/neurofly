"""Model w dronie: BANC v888 + wytrenowany dekoder jako pilot ``WorldEnv`` (świat Osoby 3).

    python -m sim.run_env --banc data/decoders/planB_dn.npz          # okno: dron leci sam, panel BANC
    python -m sim.banc_pilot data/decoders/planB_dn.npz --seeds 1 2 3  # bez okna: wyniki lotów

Co klatkę: oczy z obserwacji ``WorldEnv`` → FlyVis (``VisionBridge``) → BANC (``BancController``,
żyroskop na aferenty halter) → dekoder → komenda. Wariant demo (Plan C, jak ``DroneEnv`` w trybie angle
z ``--thrust hold``) — NASZE ZAŁOŻENIE, nie wynik BANC:
  - yaw z BANC (±1 → ±``max_yaw_rate`` rad/s),
  - poziom, wysokość ``cruise_height`` nad terenem (promień w dół) i stała prędkość do przodu
    (``forward_speed``) trzyma ``VelocityController`` symulatora (prawdziwy stan), bo thrust z BANC jest
    za słaby, a roll/pitch nie są uczone. Bez trzymania wysokości nad terenem dron ze startu na 0.3 m
    wjeżdżał w pierwszy pagórek. 1 m nad ziemią: przelot nad polem liczy się jako „cel” (≤ 1.5 m).
Model widzi tylko oczy i żyroskop; pozycji celu nie dostaje.

Kalibracja (jak w ``fly_banc.py``): ``calib="drone"`` — sceny ``DroneEnv`` (te, na których dekoder był
uczony), ``calib="world"`` — te same ujęcia w bieżącym świecie (dron na starcie obracany względem celu).
``beacon_scale``: mnożnik średnicy masztu celu w modelu (zmiana w pamięci, scena na dysku bez zmian);
maszt Osoby 3 ma 8 cm, z 15–23 m to ~0.25° — poniżej rozdzielczości oka muszki (~5°).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from sim.control import VelocityController
from sim.world_env import MAX_RATE


def to_action(cmd, hover_thrust):
    """RateCommand → akcja WorldEnv (jak ``sim.run_env.to_action``)."""
    rates = np.array([cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate]) / MAX_RATE
    return np.concatenate([[cmd.thrust / (2 * hover_thrust)], np.clip(rates, -1, 1)])


class BancPilot:
    def __init__(self, decoder_path: Path, forward_speed: float = 1.0, max_yaw_rate: float = 1.0,
                 calib: str = "drone", beacon_scale: float = 1.0, brain: bool = True,
                 cruise_height: float = 1.0) -> None:
        from banc_control import BancController, Connectome
        from banc_control.readout import LinearDecoder
        from visual_pipeline import VisionBridge

        t0 = time.perf_counter()
        self.decoder_path = Path(decoder_path)
        readout = "dn" if np.load(self.decoder_path)["M"].shape[1] > 6 else "mn"
        self.ctrl = BancController(Connectome.from_banc(), decoder=LinearDecoder(), readout=readout)
        self.bridge = VisionBridge(fps=30, fisheye=True)
        self.forward_speed, self.max_yaw_rate, self.cruise_height = forward_speed, max_yaw_rate, cruise_height
        self.calib, self.beacon_scale = calib, beacon_scale
        self.view = None
        if brain:
            from sim.brain_panel import BrainView

            self.view = BrainView(self.ctrl.c)
        self.image = None  # panel BANC (dla sim.viewer.Overlays)
        self.cmd = None
        self.env = None
        print(f"pilot BANC: odczyt {readout}, {self.ctrl.c.n:,} neuronów, {time.perf_counter() - t0:.0f} s", flush=True)

    # --- przygotowanie ---
    def bind(self, env) -> dict:
        """Podpina środowisko, kalibruje kontroler. Zmienia stan ``env.data`` — potem ``env.reset``."""
        from visual_pipeline.server import ControlServer, LocalClient

        self.env = env
        self.stab = VelocityController(env.rate_ctrl)
        if self.beacon_scale != 1.0:
            for name in ("beacon_pole", "beacon_top"):
                env.model.geom_size[env.model.geom(name).id, 0] *= self.beacon_scale
        if self.calib == "world":
            render = self._world_render
        else:
            from sim.env import DroneEnv

            self._drone_env = DroneEnv()
            render = self._drone_env.calibration_render
        t = time.perf_counter()
        calib = LocalClient(ControlServer(self.bridge, self.ctrl)).calibrate(render)
        self.ctrl.decoder.load_weights(self.decoder_path)
        print(f"kalibracja ({self.calib}) {time.perf_counter() - t:.0f} s: yaw_axis_sign {calib['yaw_axis_sign']:+.0f}",
              flush=True)
        return calib

    def _world_render(self, bearing: float):
        """Dron na starcie, obrócony tak, żeby cel był pod kątem ``bearing`` (+ = w prawo)."""
        env = self.env
        mujoco.mj_resetDataKeyframe(env.model, env.data, env.key_id)
        mujoco.mj_forward(env.model, env.data)
        d = env.target.position(env.data) - env.data.xpos[env.drone_id]
        yaw = np.arctan2(d[1], d[0]) + bearing  # MuJoCo: yaw + = w lewo, więc cel wypada w prawo
        env.data.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(env.model, env.data)
        return env.eyes.render(env.data)

    def reset(self, obs) -> None:
        """Po ``env.reset``: stan sieci jak w ustalonym stanie dla pierwszej klatki."""
        left, right = obs["eyes"]
        self.bridge.reset()
        self.ctrl.dyn.reset()
        self.ctrl.warm_start(self.bridge.settle(left, right))
        self.stab.reset()
        if self.view is not None:
            self.view.reset()

    # --- krok ---
    def __call__(self, obs) -> np.ndarray:
        from banc_control import ImuState

        left, right = obs["eyes"]
        gx, gy, gz = (float(v) for v in obs["imu"][:3])
        cmd = self.ctrl.step(self.bridge.step_batch(left, right), ImuState(gyro=(gx, gy, -gz)))  # yaw + = w prawo
        self.cmd = cmd
        if self.view is not None:
            self.image = self.view.render(self.ctrl.dyn.rates_at(np.arange(self.ctrl.c.n)), cmd)
        self.stab.z_ref = self._ground_z() + self.cruise_height
        rc = self.stab.compute(self.env.data, forward=self.forward_speed,
                               yaw_rate=-self.max_yaw_rate * float(np.clip(cmd.yaw, -1, 1)))  # MuJoCo + = w lewo
        return to_action(rc, self.env.rate_ctrl.hover_thrust)


    def _ground_z(self) -> float:
        """Wysokość terenu pod dronem (promień w dół, bez drona i bez celu)."""
        env = self.env
        pos = env.data.xpos[env.drone_id]
        geom = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(env.model, env.data, pos, np.array([0.0, 0.0, -1.0]), None, 1, env.drone_id, geom)
        return float(pos[2] - dist) if dist >= 0 else float(pos[2] - self.cruise_height)


def bearing_to_target(env) -> float:
    """Kąt do celu w stopniach (+ = w prawo) — tylko do metryk, pilot go nie zna."""
    d = env.target.position(env.data) - env.data.xpos[env.drone_id]
    _, _, yaw = env._info(None)["attitude"]
    return float(np.rad2deg((yaw - np.arctan2(d[1], d[0]) + np.pi) % (2 * np.pi) - np.pi))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decoder", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--forward", type=float, default=1.0, help="m/s do przodu (nasze założenie)")
    ap.add_argument("--height", type=float, default=1.0, help="m nad terenem")
    ap.add_argument("--calib", choices=("drone", "world"), default="drone")
    ap.add_argument("--beacon-scale", type=float, default=1.0)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--no-yaw", action="store_true", help="kontrola: BANC wyłączony, dron leci prosto")
    args = ap.parse_args()

    from sim.world_env import WorldEnv

    env = WorldEnv(control="acro", start_noise=False, max_time=args.max_time)
    pilot = BancPilot(args.decoder, args.forward, calib=args.calib, beacon_scale=args.beacon_scale, brain=False,
                      cruise_height=args.height)
    if args.no_yaw:
        pilot.max_yaw_rate = 0.0
    obs, _ = env.reset(seed=0)
    pilot.bind(env)
    wins = 0
    for seed in args.seeds:
        obs, info = env.reset(options={"world_seed": seed})
        pilot.reset(obs)
        b0, d0, dmin, done, k = bearing_to_target(env), info["distance"], info["distance"], False, 0
        while not done:
            obs, _, term, trunc, info = env.step(pilot(obs))
            dmin = min(dmin, info["distance"])
            done = term or trunc
            if k % 150 == 0:
                print(f"  t {info['time']:4.1f} s  kąt do celu {bearing_to_target(env):+6.1f}°  odl. {info['distance']:5.1f} m  "
                      f"yaw {pilot.cmd.yaw:+.2f}", flush=True)
            k += 1
        wins += info["outcome"] == "cel"
        print(f"świat {seed}: start kąt {b0:+.0f}°, {d0:.1f} m → {info['outcome']} po {info['time']:.1f} s, "
              f"najbliżej {dmin:.1f} m", flush=True)
    print(f"cel osiągnięty {wins}/{len(args.seeds)}")


if __name__ == "__main__":
    main()
