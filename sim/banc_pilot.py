"""Model w dronie: BANC v888 + wytrenowany dekoder jako pilot ``WorldEnv`` (świat Osoby 3).

    python -m sim.run_env --banc data/decoders/world.npz              # okno: dron leci sam, panel BANC
    python -m sim.banc_pilot data/decoders/world.npz --seeds 1 2 3     # bez okna: wyniki lotów
    python scripts/train_world.py                                      # trening (1 GPU); 2 GPU: train_distributed --world

Co klatkę: oczy z obserwacji ``WorldEnv`` → FlyVis (``VisionBridge``) → BANC (``BancController``,
żyroskop na aferenty halter) → dekoder → komenda. Model widzi tylko oczy i żyroskop; pozycji celu,
wysokości ani prędkości nie dostaje.

Dwa tryby (podział sterowania to NASZE ZAŁOŻENIE, nie wynik BANC):
  - ``assist=False`` (domyślnie): BANC daje thrust (wysokość), pitch (prędkość do przodu), roll (kasowanie
    dryfu w bok) i yaw (kierunek); ``WorldEnv(control="angle")`` tylko utrzymuje zadany przechył. Dekoder z wyrazem wolnym
    (``bias``), uczony w tym świecie przez ``train_world.py`` / ``train_distributed.py --world``.
  - ``assist=True``: tylko yaw z BANC; wysokość ``cruise_height`` nad terenem i stałą prędkość
    ``forward_speed`` trzyma ``VelocityController`` (prawdziwy stan). Dla dekoderów z ``train_decoder.py``.

Kalibracja (jak w ``fly_banc.py``): ``calib="drone"`` — sceny ``DroneEnv``, ``calib="world"`` — te same ujęcia
w bieżącym świecie. ``beacon_scale``: mnożnik średnicy masztu celu (zmiana w pamięci, scena na dysku bez zmian);
maszt Osoby 3 ma 8 cm, z 15–23 m to ~0.25° — poniżej rozdzielczości oka muszki (~5°).

Nauczyciel (``teacher``, tylko do treningu): z prawdziwego stanu — yaw ∝ kąt do celu, pitch do prędkości
``speed``·max(cos kąta, 0) (najpierw obrót, potem lot), thrust do wysokości ``height`` nad terenem.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from sim.control import VelocityController, euler_zyx
from sim.world_env import MAX_TILT, to_action

GRAVITY = 9.81


def ground_z(env) -> float:
    """Wysokość terenu pod dronem (promień w dół tylko w heightfield; bez terenu: 0)."""
    pos = env.data.xpos[env.drone_id]
    if not env.has_terrain:
        return 0.0
    gid = env.model.geom("terrain").id
    dist = mujoco.mj_rayHfield(env.model, env.data, gid, pos, np.array([0.0, 0.0, -1.0]))
    return float(pos[2] - dist) if dist >= 0 else 0.0


def clear_corridor(env, width: float = 3.0) -> int:
    """Chowa pod teren bloki w pasie ±``width`` m (+ zasięg bloku) wokół odcinka start–cel.

    Na planszy stoi 40–60 bloków, na 1 m wysokości prosta droga do celu zwykle jest zablokowana,
    a omijania przeszkód nie uczymy (NASZE ZAŁOŻENIE: czysty korytarz). Zmiana tylko w pamięci modelu;
    reszta bloków zostaje i jest widoczna dla oczu. Wywoływać po ``env.reset``. Zwraca liczbę schowanych.
    """
    from sim import blocks

    if not blocks.has_pool(env.model):
        return 0
    start = env.data.xpos[env.drone_id][:2].copy()
    seg = env.target.position(env.data)[:2] - start
    hidden = 0
    for i in range(blocks.POOL_SIZE):
        body = env.model.body(f"block{i}").id
        pos = env.model.body_pos[body]
        if pos[2] <= blocks.HIDDEN_Z + 1:
            continue
        t = np.clip(np.dot(pos[:2] - start, seg) / max(np.dot(seg, seg), 1e-9), 0.0, 1.0)
        if np.linalg.norm(pos[:2] - start - t * seg) < width + blocks.ENVELOPE_RADIUS:
            env.model.body_pos[body] = (0, 0, blocks.HIDDEN_Z)
            hidden += 1
    mujoco.mj_forward(env.model, env.data)
    return hidden


def bearing(env) -> float:
    """Kąt do celu [rad], + = w prawo (tylko nauczyciel i metryki)."""
    d = env.target.position(env.data) - env.data.xpos[env.drone_id]
    _, _, yaw = euler_zyx(env.data)
    return float((yaw - np.arctan2(d[1], d[0]) + np.pi) % (2 * np.pi) - np.pi)


def teacher(env, height: float = 1.0, speed: float = 1.0):
    """Komenda wzorcowa (FlightCommand, konwencja BANC: yaw + = w prawo, pitch + = do przodu)."""
    from banc_control import FlightCommand

    roll, pitch, yaw = euler_zyx(env.data)
    b = bearing(env)
    v = env.data.qvel[0:3]
    v_fwd = np.cos(yaw) * v[0] + np.sin(yaw) * v[1]
    v_left = -np.sin(yaw) * v[0] + np.cos(yaw) * v[1]
    v_des = speed * max(np.cos(b), 0.0)
    tilt = np.clip(2.0 * (v_des - v_fwd) / GRAVITY, -MAX_TILT, MAX_TILT)
    # roll kasuje dryf w bok: bez tego po skręcie dron krążył wokół celu w odległości ~5 m
    side = np.clip(2.0 * v_left / GRAVITY, -MAX_TILT, MAX_TILT)  # jak VelocityController: roll_des = -a_left/g
    h = env.data.xpos[env.drone_id][2] - ground_z(env)
    a_z = 3.0 * (height - h) - 4.0 * v[2]
    thrust = 0.5 * (1.0 + a_z / GRAVITY) / max(np.cos(roll) * np.cos(pitch), 0.5)  # 0.5 = zawis w WorldEnv
    return FlightCommand(float(np.clip(thrust, 0, 1)), float(side / MAX_TILT), float(tilt / MAX_TILT),
                         float(np.clip(1.5 * b, -1, 1)))


class BancPilot:
    def __init__(self, decoder_path: Path | None, forward_speed: float = 1.0, max_yaw_rate: float = 1.0,
                 calib: str = "drone", beacon_scale: float = 1.0, brain: bool = True,
                 cruise_height: float = 1.0, assist: bool | None = None, lr: float = 0.5,
                 readout: str | None = None) -> None:
        from banc_control import BancController, Connectome
        from banc_control.readout import LinearDecoder
        from visual_pipeline import VisionBridge

        t0 = time.perf_counter()
        self.decoder_path = Path(decoder_path) if decoder_path else None
        M = np.load(self.decoder_path)["M"] if self.decoder_path else None
        if readout is None:
            readout = "dn" if M is None or M.shape[1] > 7 else "mn"
        if assist is None:  # wagi bez wyrazu wolnego (train_decoder.py) → tryb ze wspomaganiem
            n = 6 + (375 if readout == "dn" else 0)
            assist = M is not None and M.shape[1] == n
        self.assist = assist
        self.ctrl = BancController(Connectome.from_banc(), decoder=LinearDecoder(lr=lr, bias=not assist),
                                   readout=readout)
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
        print(f"pilot BANC: odczyt {readout}, {'yaw (wspomaganie)' if assist else 'thrust/pitch/yaw'}, "
              f"{self.ctrl.c.n:,} neuronów, {time.perf_counter() - t0:.0f} s", flush=True)

    # --- przygotowanie ---
    def bind(self, env) -> dict:
        """Podpina środowisko, kalibruje kontroler. Zmienia stan ``env.data`` — potem ``env.reset``."""
        from visual_pipeline.server import ControlServer, LocalClient

        if not self.assist and env.control != "angle":
            raise ValueError("pilot bez wspomagania wymaga WorldEnv(control='angle')")
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
        if self.decoder_path:
            self.ctrl.decoder.load_weights(self.decoder_path)
        print(f"kalibracja ({self.calib}) {time.perf_counter() - t:.0f} s: yaw_axis_sign {calib['yaw_axis_sign']:+.0f}",
              flush=True)
        return calib

    def _world_render(self, bearing_rad: float):
        """Dron na starcie, obrócony tak, żeby cel był pod kątem ``bearing_rad`` (+ = w prawo)."""
        env = self.env
        mujoco.mj_resetDataKeyframe(env.model, env.data, env.key_id)
        mujoco.mj_forward(env.model, env.data)
        d = env.target.position(env.data) - env.data.xpos[env.drone_id]
        yaw = np.arctan2(d[1], d[0]) + bearing_rad  # MuJoCo: yaw + = w lewo, więc cel wypada w prawo
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
    def decide(self, obs):
        """Obserwacja → FlightCommand z BANC (konwencja BANC); ``cmd.debug["motor_features"]`` do uczenia."""
        from banc_control import ImuState

        left, right = obs["eyes"]
        gx, gy, gz = (float(v) for v in obs["imu"][:3])
        cmd = self.ctrl.step(self.bridge.step_batch(left, right), ImuState(gyro=(gx, gy, -gz)))  # yaw + = w prawo
        if self.assist:
            cmd.roll = 0.0  # ze wspomaganiem roll z BANC nie idzie do drona
        self.cmd = cmd
        if self.view is not None:
            self.image = self.view.render(self.ctrl.dyn.rates_at(np.arange(self.ctrl.c.n)), cmd)
        return cmd

    def action(self, cmd) -> np.ndarray:
        """FlightCommand (konwencja BANC) → akcja WorldEnv."""
        if self.assist:
            self.stab.z_ref = ground_z(self.env) + self.cruise_height
            rc = self.stab.compute(self.env.data, forward=self.forward_speed,
                                   yaw_rate=-self.max_yaw_rate * float(np.clip(cmd.yaw, -1, 1)))  # MuJoCo + = w lewo
            return to_action(rc, self.env.rate_ctrl.hover_thrust)
        # angle: [thrust, roll, pitch, yaw] — zadany przechył; WorldEnv yaw + = w lewo
        return np.array([cmd.thrust, cmd.roll, cmd.pitch, -cmd.yaw])

    def __call__(self, obs) -> np.ndarray:
        return self.action(self.decide(obs))


class WorldRunner:
    """Epizody lotu do celu w ``WorldEnv``: ewaluacja i trening Planu B (DAgger z ``teacher``)."""

    def __init__(self, env, pilot: BancPilot, height: float = 1.0, speed: float = 1.0, clear_path: bool = True) -> None:
        self.env, self.pilot, self.height, self.speed, self.clear_path = env, pilot, height, speed, clear_path

    def reset(self, world_seed: int, start_noise: bool = False):
        """Nowy epizod w świecie ``world_seed`` (z czystym korytarzem do celu) → (obs, info)."""
        env = self.env
        env.start_noise = start_noise
        obs, info = env.reset(options={"world_seed": int(world_seed)})
        if self.clear_path and clear_corridor(env):
            obs["eyes"] = np.stack(env.eyes.render(env.data))  # oczy już bez schowanych bloków
        self.pilot.reset(obs)
        return obs, info

    def episode(self, world_seed: int, learn: bool = False, beta: float = 0.0,
                rng: np.random.Generator | None = None, start_noise: bool = False) -> dict:
        env, pilot, dec = self.env, self.pilot, self.pilot.ctrl.decoder
        obs, info = self.reset(world_seed, start_noise)
        b0, d0 = np.rad2deg(bearing(env)), info["distance"]
        dmin, losses, heights, done = d0, [], [], False
        while not done:
            cmd = pilot.decide(obs)
            act = cmd
            if learn:
                target = teacher(env, self.height, self.speed)
                losses.append(dec.fit_step(cmd.debug["motor_features"], target, apply=False))
                if rng.random() < beta:
                    act = target
            obs, _, term, trunc, info = env.step(pilot.action(act))
            dmin = min(dmin, info["distance"])
            heights.append(env.data.xpos[env.drone_id][2] - ground_z(env))
            done = term or trunc
        if learn:
            dec.apply_pending()  # wagi stałe w trakcie lotu, krok po epizodzie
        return {"world": int(world_seed), "outcome": info["outcome"], "reached": info["outcome"] == "cel",
                "time": float(info["time"]), "start_dist": float(d0), "min_dist": float(dmin),
                "bearing": float(b0), "final_deg": float(abs(np.rad2deg(bearing(env)))),
                "mean_height": float(np.mean(heights)), "loss": float(np.mean(losses)) if losses else float("nan")}

    def evaluate(self, seeds) -> dict:
        eps = [self.episode(s) for s in seeds]
        return {"episodes": eps, "reached": int(sum(e["reached"] for e in eps)), "n": len(eps),
                "mean_min_dist": float(np.mean([e["min_dist"] for e in eps])),
                "mean_final_deg": float(np.mean([e["final_deg"] for e in eps]))}


def show_world(tag: str, ev: dict) -> None:
    per = "  ".join(f"{e['world']}:{'CEL' if e['reached'] else e['outcome'].split(':')[0]}"
                    f"({e['min_dist']:.0f}m)" for e in ev["episodes"])
    print(f"{tag}: cel {ev['reached']}/{ev['n']}, średnio najbliżej {ev['mean_min_dist']:.1f} m  ({per})", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decoder", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105, 106])
    ap.add_argument("--calib", choices=("drone", "world"), default="drone")
    ap.add_argument("--beacon-scale", type=float, default=4.0)
    ap.add_argument("--max-time", type=float, default=40.0)
    args = ap.parse_args()

    from sim.world_env import WorldEnv

    pilot = BancPilot(args.decoder, calib=args.calib, beacon_scale=args.beacon_scale, brain=False)
    env = WorldEnv(control="acro" if pilot.assist else "angle", start_noise=False, max_time=args.max_time)
    env.reset(seed=0)
    pilot.bind(env)
    show_world("wynik", WorldRunner(env, pilot).evaluate(args.seeds))


if __name__ == "__main__":
    main()
