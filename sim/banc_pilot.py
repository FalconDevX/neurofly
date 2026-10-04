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
w bieżącym świecie. ``beacon_scale`` / ``beacon_alpha``: szerokość i przezroczystość celu — prostopadłościanu
``target_box`` Osoby 3 (zmiana w pamięci, scena na dysku bez zmian). Cel ma 30 cm szerokości: z 15–23 m to ~1°,
poniżej rozdzielczości oka muszki (~5°) — stąd pomiar widoczności (scripts/check_beacon_visibility.py).

Nauczyciel (``teacher``, tylko do treningu): z prawdziwego stanu — yaw ∝ kąt do celu, pitch do prędkości
``speed``·max(cos kąta, 0) (najpierw obrót, potem lot), thrust do wysokości ``height`` nad terenem.
Z ``avoid=True`` omija bloki: promienie MuJoCo w wachlarzu ±90° na wysokości drona (zasięg ``AVOID_RANGE``),
kierunek = najbliższa celowi wolna luka (poszerzona o margines na rozmiar drona), przed blokiem zwalnia.
Nauczyciel zna geometrię świata — BANC jej nie dostaje, ma się tego nauczyć z oczu (NASZE ZAŁOŻENIE: uczenie
z nauczycielem, nie wrodzony odruch).

Przeszkody (``WorldRunner(obstacles=…)``): ``clear`` — czysty korytarz (jak dotąd), ``path`` — korytarz,
ale ``path_blocks`` bloków w połowie trasy zostaje (losowane z ziarna świata), ``keep`` — wszystkie bloki.
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


BEACON_DARK_RED = (0.30, 0.02, 0.02)  # FlyVis widzi tylko jasność: ciemny cel odcina się od nieba, pomarańczowy nie


def set_beacon(model, scale: float = 1.0, alpha: float | None = None, color=None) -> None:
    """Cel = prostopadłościan ``target_box`` ze sceny Osoby 3: ``scale`` mnoży szerokość (x, y), ``alpha`` =
    przezroczystość (1 = pełny). Zmiana w pamięci modelu; ``sim.target.Target.reached`` czyta rozmiar z modelu."""
    gid = model.geom("target_box").id
    if scale != 1.0:
        model.geom_size[gid, :2] *= scale  # szerszy w x i y, wysokość bez zmian (stoi na ziemi)
    if color is not None:
        model.geom_rgba[gid, :3] = color
    if alpha is not None:
        model.geom_rgba[gid, 3] = alpha  # target_box ma kolor w rgba (bez materiału)


def clear_corridor(env, width: float = 3.0, keep: int = 0, rng: np.random.Generator | None = None) -> int:
    """Chowa pod teren bloki w pasie ±``width`` m (+ zasięg bloku) wokół odcinka start–cel.

    Na planszy stoi 60–85 bloków, na 1 m wysokości prosta droga do celu zwykle jest zablokowana.
    ``keep`` = 0: czysty korytarz (NASZE ZAŁOŻENIE w treningu bez omijania). ``keep`` > 0: tyle bloków
    stojących blisko linii (≤ 2.5 m od środka) w połowie trasy (20–80 %) zostaje — przeszkody do ominięcia,
    wybór z ``rng``. Zmiana tylko w pamięci modelu; reszta bloków zostaje i jest widoczna dla oczu.
    Wywoływać po ``env.reset``. Zwraca liczbę schowanych.
    """
    from sim import blocks

    if not blocks.has_pool(env.model):
        return 0
    start = env.data.xpos[env.drone_id][:2].copy()
    seg = env.target.position(env.data)[:2] - start
    inside, on_path = [], []
    for i in range(blocks.POOL_SIZE):
        body = env.model.body(f"block{i}").id
        pos = env.model.body_pos[body]
        if pos[2] <= blocks.HIDDEN_Z + 1:
            continue
        t = np.clip(np.dot(pos[:2] - start, seg) / max(np.dot(seg, seg), 1e-9), 0.0, 1.0)
        off = np.linalg.norm(pos[:2] - start - t * seg)
        if off < width + blocks.ENVELOPE_RADIUS:
            inside.append(body)
            if off < 2.5 and 0.2 <= t <= 0.8:
                on_path.append(body)
    kept = set()
    if keep and on_path:
        rng = rng or np.random.default_rng()
        kept = set(rng.choice(on_path, size=min(keep, len(on_path)), replace=False).tolist())
    hidden = 0
    for body in inside:
        if body not in kept:
            blocks.hide(env.model, body)  # pod ziemię, maleńki i niewidoczny
            hidden += 1
    mujoco.mj_forward(env.model, env.data)
    return hidden


AVOID_RANGE = 4.0                                 # m, zasięg promieni nauczyciela
AVOID_FAN = np.deg2rad(np.arange(-90, 91, 10))    # kąty promieni względem nosa, + = w prawo
AVOID_MARGIN = 2                                  # sąsiednie promienie (po 10°) też muszą być wolne


def block_geoms(model) -> set:
    """Geomy bloków (sim/blocks.py); pusty zbiór, gdy scena nie ma bloków."""
    from sim import blocks

    if not blocks.has_pool(model):
        return set()
    bodies = {model.body(f"block{i}").id for i in range(blocks.POOL_SIZE)}
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}


def scan_blocks(env, angles=AVOID_FAN, reach: float = AVOID_RANGE) -> np.ndarray:
    """Odległość do bloku wzdłuż poziomych promieni z drona (kąty jak ``bearing``: + = w prawo); inf = wolne."""
    if not hasattr(env, "_block_geoms"):
        env._block_geoms = block_geoms(env.model)
    pos = env.data.xpos[env.drone_id].copy()
    _, _, yaw = euler_zyx(env.data)
    gid = np.zeros(1, np.int32)
    out = np.full(len(angles), np.inf)
    for i, a in enumerate(angles):
        h = yaw - a  # MuJoCo: yaw + = w lewo
        d = mujoco.mj_ray(env.model, env.data, pos, np.array([np.cos(h), np.sin(h), 0.0]), None, 1, env.drone_id, gid)
        if 0 <= d < reach and int(gid[0]) in env._block_geoms:
            out[i] = d
    return out


def avoid_heading(env, b: float) -> tuple[float, float]:
    """Kierunek omijania [rad, + = w prawo] i prześwit na wprost [m] (inf = wolne). ``env._avoid_side``:
    strona ostatniego objazdu (histereza, żeby nie przeskakiwać między lukami po obu stronach bloku)."""
    d = scan_blocks(env)
    blocked = np.isfinite(d)
    front = float(d[np.abs(AVOID_FAN) <= np.deg2rad(20)].min())
    if not blocked.any() or abs(b) > AVOID_FAN[-1]:
        env._avoid_side = 0
        return b, front
    wide = blocked.copy()
    for k in range(1, AVOID_MARGIN + 1):
        wide[k:] |= blocked[:-k]
        wide[:-k] |= blocked[k:]
    i_b = int(np.argmin(np.abs(AVOID_FAN - b)))
    if not wide[i_b]:
        env._avoid_side = 0
        return b, front
    free = np.flatnonzero(~wide)
    if len(free) == 0:  # wszędzie blisko: w stronę największego prześwitu
        return float(AVOID_FAN[int(np.argmax(np.where(blocked, d, AVOID_RANGE)))]), front
    side = getattr(env, "_avoid_side", 0)
    cost = np.abs(AVOID_FAN[free] - b) + 0.5 * (side != 0) * (np.sign(AVOID_FAN[free] - b) != side)
    theta = float(AVOID_FAN[free[int(np.argmin(cost))]])
    env._avoid_side = int(np.sign(theta - b))
    return theta, front


def bearing(env) -> float:
    """Kąt do celu [rad], + = w prawo (tylko nauczyciel i metryki)."""
    d = env.target.position(env.data) - env.data.xpos[env.drone_id]
    _, _, yaw = euler_zyx(env.data)
    return float((yaw - np.arctan2(d[1], d[0]) + np.pi) % (2 * np.pi) - np.pi)


def teacher(env, height: float = 1.0, speed: float = 1.0, avoid: bool = False):
    """Komenda wzorcowa (FlightCommand, konwencja BANC: yaw + = w prawo, pitch + = do przodu).
    ``avoid``: kurs na wolną lukę zamiast prosto na cel, wolniej przed blokiem."""
    from banc_control import FlightCommand

    roll, pitch, yaw = euler_zyx(env.data)
    b = bearing(env)
    v = env.data.qvel[0:3]
    v_fwd = np.cos(yaw) * v[0] + np.sin(yaw) * v[1]
    v_left = -np.sin(yaw) * v[0] + np.cos(yaw) * v[1]
    v_des = speed
    if avoid:
        b, front = avoid_heading(env, b)
        if np.isfinite(front):
            v_des *= float(np.clip((front - 1.0) / (AVOID_RANGE - 1.0), 0.2, 1.0))
    v_des *= max(np.cos(b), 0.0)
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
                 readout: str | None = None, yaw_init: Path | None = None, beacon_alpha: float | None = None,
                 vision_range: float = float("inf"), beacon_color: str = "scene", banc_axes=()) -> None:
        """``decoder_path``: wagi ``WorldDecoder`` (``train_world.py``, plik z ``w_yaw``) albo dekoder z
        ``train_decoder.py`` (tryb ze wspomaganiem). ``decoder_path=None`` + ``assist=False``: nowy
        ``WorldDecoder`` do treningu, wiersz yaw z ``yaw_init`` (dekoder zawisu, np. planB_distributed.npz);
        ``banc_axes``: osie thrust/roll/pitch tylko z BANC (bez czujników drona)."""
        from banc_control import BancController, Connectome
        from banc_control.readout import LinearDecoder
        from visual_pipeline import VisionBridge

        t0 = time.perf_counter()
        from sim.world_decoder import WorldDecoder

        self.decoder_path = Path(decoder_path) if decoder_path else None
        self.world = None  # WorldDecoder: BANC → yaw, BANC + czujniki drona → thrust/roll/pitch
        if self.decoder_path and WorldDecoder.is_world_file(self.decoder_path):
            self.world = WorldDecoder.load(self.decoder_path)
            if banc_axes:  # np. wagi uczone z czujnikami, puszczone bez nich (wagi czujników tych osi = 0)
                self.world = WorldDecoder.for_matrix(self.world.to_matrix(),
                                                     banc_only=tuple(set(self.world.banc_only) | set(banc_axes)))
            assist, readout = False, "dn"
        M = np.load(self.decoder_path)["M"] if self.decoder_path and self.world is None else None
        if readout is None:
            readout = "dn" if M is None or M.shape[1] > 7 else "mn"
        if assist is None:  # wagi bez wyrazu wolnego (train_decoder.py) → tryb ze wspomaganiem
            n = 6 + (375 if readout == "dn" else 0)
            assist = M is not None and M.shape[1] == n
        self.assist = assist
        self.ctrl = BancController(Connectome.from_banc(), decoder=LinearDecoder(lr=lr, bias=not assist),
                                   readout=readout)
        if not assist and self.world is None:
            self.world = WorldDecoder(self.ctrl.decoder.M.shape[1], banc_only=banc_axes)
            if yaw_init:  # yaw z dekodera zawisu (te same cechy; bez wyrazu wolnego → 0)
                w = np.load(yaw_init)["M"][3]
                self.world.w_yaw[:len(w)] = w
        self.bridge = VisionBridge(fps=30, fisheye=True)
        self.forward_speed, self.max_yaw_rate, self.cruise_height = forward_speed, max_yaw_rate, cruise_height
        self.calib, self.beacon_scale, self.beacon_alpha = calib, beacon_scale, beacon_alpha
        self.beacon_color = BEACON_DARK_RED if beacon_color == "dark-red" else None
        self.vision_range = vision_range  # warstwa zachowań: dalej od celu yaw z „GPS”, bliżej z BANC (inf = tylko BANC)
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
        set_beacon(env.model, self.beacon_scale, self.beacon_alpha, self.beacon_color)
        if self.calib == "world":
            render = self._world_render
        else:
            from sim.env import DroneEnv

            self._drone_env = DroneEnv()
            render = self._drone_env.calibration_render
        t = time.perf_counter()
        calib = LocalClient(ControlServer(self.bridge, self.ctrl)).calibrate(render)
        if self.decoder_path and self.world is None:
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

        t0 = time.perf_counter()
        left, right = obs["eyes"]
        gx, gy, gz = (float(v) for v in obs["imu"][:3])
        cmd = self.ctrl.step(self.bridge.step_batch(left, right), ImuState(gyro=(gx, gy, -gz)))  # yaw + = w prawo
        steer = None  # dla panelu BANC: skąd jest kurs
        if self.world is not None:  # BANC (normalizacja z kalibracji) + czujniki drona → komenda
            from sim.world_decoder import gps_yaw, sensors_from_obs, vision_weight

            x = self.ctrl.decoder.normalized(cmd.debug["motor_features"])
            sens = sensors_from_obs(obs)  # tylko odczyty czujników (sim/sensors.py), bez prawdziwego stanu
            cmd = self.world.decode(np.nan_to_num(x), sens)
            yaw_banc = cmd.yaw
            beacon = obs.get("beacon", np.zeros(2))
            w_vis = vision_weight(float(beacon[1]), self.vision_range)
            yaw_gps = gps_yaw(beacon)
            yaw_avoid = float(self.world.w_avoid @ np.nan_to_num(x))  # BANC: skręt omijania dokładany do GPS
            cmd.yaw = float(np.clip(w_vis * yaw_banc + (1 - w_vis) * (yaw_gps + yaw_avoid), -1, 1))
            cmd.debug = {"x": x, "sensors": sens, "yaw_banc": yaw_banc, "w_vis": w_vis, "yaw_gps": yaw_gps,
                         "yaw_avoid": yaw_avoid}
            if self.view is not None:
                from sim.brain_panel import steer_from_decoder

                steer = steer_from_decoder(self.world.w_yaw, x, yaw_banc, bias=True,
                                           yaw_gps=yaw_gps if np.isfinite(self.vision_range) else None, w_vis=w_vis,
                                           distance=float(beacon[1]) if np.isfinite(self.vision_range) else None)
        elif self.view is not None:
            from sim.brain_panel import steer_from_decoder

            dec = self.ctrl.decoder
            steer = steer_from_decoder(dec.M[3], dec.normalized(cmd.debug["motor_features"]), cmd.yaw, bias=dec.bias)
        if self.world is None and self.assist:
            cmd.roll = 0.0  # ze wspomaganiem roll z BANC nie idzie do drona
        self.cmd = cmd
        if self.view is not None:
            self.image = self.view.render(self.ctrl.dyn.rates_at(np.arange(self.ctrl.c.n)), cmd,
                                          (time.perf_counter() - t0) * 1e3, steer)
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

    def __init__(self, env, pilot: BancPilot, height: float = 1.0, speed: float = 1.0, obstacles: str = "clear",
                 path_blocks: int = 2) -> None:
        assert obstacles in ("clear", "path", "keep")
        self.env, self.pilot, self.height, self.speed = env, pilot, height, speed
        self.obstacles, self.path_blocks = obstacles, path_blocks
        self.avoid = obstacles != "clear"  # nauczyciel omija bloki

    def reset(self, world_seed: int, start_noise: bool = False):
        """Nowy epizod w świecie ``world_seed`` (korytarz do celu według ``obstacles``) → (obs, info)."""
        env = self.env
        env.start_noise = start_noise
        obs, info = env.reset(options={"world_seed": int(world_seed)})
        env._avoid_side = 0
        if self.obstacles != "keep":
            keep = self.path_blocks if self.obstacles == "path" else 0
            if clear_corridor(env, keep=keep, rng=np.random.default_rng(int(world_seed) + 7)) and env.eyes:
                obs["eyes"] = np.stack(env.eyes.render(env.data))  # oczy już bez schowanych bloków
        if self.pilot is not None:
            self.pilot.reset(obs)
        return obs, info

    def touching_block(self) -> bool:
        """Czy dron dotyka któregoś bloku."""
        env = self.env
        if not hasattr(env, "_block_geoms"):
            env._block_geoms = block_geoms(env.model)
        if not env._block_geoms or env.data.ncon == 0:
            return False
        c = env.data.contact[:env.data.ncon]
        drone, blk = env.crash.drone_geoms, np.fromiter(env._block_geoms, int)
        g1, g2 = c.geom1, c.geom2
        return bool(np.any((np.isin(g1, drone) & np.isin(g2, blk)) | (np.isin(g2, drone) & np.isin(g1, blk))))

    def episode(self, world_seed: int, learn: bool = False, beta: float = 0.0,
                rng: np.random.Generator | None = None, start_noise: bool = False) -> dict:
        env, pilot, dec = self.env, self.pilot, self.pilot.ctrl.decoder
        obs, info = self.reset(world_seed, start_noise)
        b0, d0 = np.rad2deg(bearing(env)), info["distance"]
        dmin, losses, heights, done = d0, [], [], False
        hits, touching = 0, False  # zderzenia z blokami (początki kontaktu)
        sq_avoid = []  # błąd poprawki omijania w klatkach z blokiem w zasięgu (daleko od celu)
        sq = []  # kwadraty błędów [thrust, roll, pitch, yaw] względem nauczyciela (WorldDecoder)
        while not done:
            cmd = pilot.decide(obs)
            act = cmd
            if learn:
                target = teacher(env, self.height, self.speed, avoid=self.avoid)
                if pilot.world is not None:  # DAgger: dane do statystyk, wagi dopasowuje wywołujący (fit)
                    from sim.world_decoder import sample_weight

                    sens = cmd.debug["sensors"]
                    # yaw z BANC uczymy tylko tam, gdzie cel jest w zasięgu wzroku (dalej kierunek daje GPS)
                    near = float(env._distance() < pilot.vision_range)
                    obstacle = self.avoid and bool(np.isfinite(scan_blocks(env)).any())
                    avoid_target = target.yaw - cmd.debug["yaw_gps"] if self.avoid else None
                    sq.append(pilot.world.add(cmd.debug["x"], sens, target,
                                              sample_weight(bearing(env), sens, obstacle), yaw_weight=near,
                                              avoid_target=avoid_target, avoid_weight=1.0 - near))
                    if obstacle and not near:
                        sq_avoid.append((cmd.debug["yaw_avoid"] - avoid_target) ** 2)
                else:
                    losses.append(dec.fit_step(cmd.debug["motor_features"], target, apply=False))
                if rng.random() < beta:
                    act = target
            obs, _, term, trunc, info = env.step(pilot.action(act))
            dmin = min(dmin, info["distance"])
            heights.append(env.data.xpos[env.drone_id][2] - ground_z(env))
            now = self.touching_block()
            hits += now and not touching
            touching = now
            done = term or trunc
        if learn and pilot.world is None:
            dec.apply_pending()  # wagi stałe w trakcie lotu, krok po epizodzie
        extra = {}
        if sq:
            from sim.world_decoder import AXES

            mse = np.mean(sq, axis=0)
            extra = {"loss_axes": dict(zip(AXES, map(float, mse)))}
            if sq_avoid:
                extra["loss_axes"]["avoid"] = float(np.mean(sq_avoid))
            losses = [float(mse.mean())]
        return {**extra, "world": int(world_seed), "outcome": info["outcome"], "reached": info["outcome"] == "cel",
                "time": float(info["time"]), "start_dist": float(d0), "min_dist": float(dmin),
                "bearing": float(b0), "final_deg": float(abs(np.rad2deg(bearing(env)))),
                "mean_height": float(np.mean(heights)), "min_height": float(np.min(heights)),
                "block_hits": int(hits), "loss": float(np.mean(losses)) if losses else float("nan")}

    def evaluate(self, seeds, log=None) -> dict:
        """``log``: np. ``print`` — linia po każdym świecie (ewaluacja trwa kilka minut)."""
        eps = []
        for i, s in enumerate(seeds):
            t = time.perf_counter()
            eps.append(self.episode(s))
            if log:
                e = eps[-1]
                log(f"  ewaluacja {i + 1}/{len(seeds)}: świat {s} → {e['outcome']}, najbliżej {e['min_dist']:.1f} m "
                    f"({time.perf_counter() - t:.0f} s)")
        return {"episodes": eps, "reached": int(sum(e["reached"] for e in eps)), "n": len(eps),
                "collided": int(sum(e["block_hits"] > 0 for e in eps)),
                "mean_height": float(np.mean([e["mean_height"] for e in eps])),
                "min_height": float(np.min([e["min_height"] for e in eps])),
                "mean_min_dist": float(np.mean([e["min_dist"] for e in eps])),
                "mean_final_deg": float(np.mean([e["final_deg"] for e in eps]))}


def show_world(tag: str, ev: dict) -> None:
    per = "  ".join(f"{e['world']}:{'CEL' if e['reached'] else e['outcome'].split(':')[0]}"
                    f"({e['min_dist']:.0f}m)" for e in ev["episodes"])
    extra = (f", zderzenia z blokami {ev['collided']}/{ev['n']}, wys. średnio {ev['mean_height']:.2f} m "
             f"(min {ev['min_height']:.2f})") if "collided" in ev else ""
    print(f"{tag}: cel {ev['reached']}/{ev['n']}, średnio najbliżej {ev['mean_min_dist']:.1f} m{extra}  ({per})",
          flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decoder", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105, 106])
    ap.add_argument("--calib", choices=("drone", "world"), default="drone")
    ap.add_argument("--beacon-scale", type=float, default=1.0)
    ap.add_argument("--beacon-alpha", type=float, default=None)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--obstacles", choices=("clear", "path", "keep"), default="clear",
                    help="czysty korytarz / bloki na trasie / wszystkie bloki")
    ap.add_argument("--path-blocks", type=int, default=2)
    args = ap.parse_args()

    from sim.world_env import WorldEnv

    pilot = BancPilot(args.decoder, calib=args.calib, beacon_scale=args.beacon_scale, brain=False, beacon_alpha=args.beacon_alpha)
    env = WorldEnv(control="acro" if pilot.assist else "angle", start_noise=False, max_time=args.max_time)
    env.reset(seed=0)
    pilot.bind(env)
    runner = WorldRunner(env, pilot, obstacles=args.obstacles, path_blocks=args.path_blocks)
    show_world("wynik", runner.evaluate(args.seeds))


if __name__ == "__main__":
    main()
