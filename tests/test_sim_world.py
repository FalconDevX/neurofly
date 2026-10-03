"""Testy symulatora Osoby 3 (sim/): regulator, świat, bloki, czujniki, metryki, WorldEnv.

Bez renderu (poza jednym testem oczu, pomijanym, gdy nie ma OpenGL). Wartości progowe z pomiarów
opisanych w docs/osoba3-plan.md — test pilnuje gwarancji, a nie dokładnych liczb.
"""

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

ASSETS = Path(__file__).resolve().parents[1] / "sim" / "assets"
HOVER = ASSETS / "scene_hover.xml"
BEACON = ASSETS / "scene_beacon.xml"


# --- pomocnicze --------------------------------------------------------------------------------------


@pytest.fixture
def hover():
    m = mujoco.MjModel.from_xml_path(str(HOVER))
    return m, mujoco.MjData(m)


@pytest.fixture(scope="module")
def beacon_model():
    from sim.terrain import load_scene

    return load_scene(BEACON)


def start(m, d, z=None, quat=None):
    mujoco.mj_resetDataKeyframe(m, d, m.key("hover").id)
    if z is not None:
        d.qpos[2] = z
    if quat is not None:
        d.qpos[3:7] = quat
    mujoco.mj_forward(m, d)


def axis_quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
    return q


def ground_height(m, d, x, y, exclude_body):
    geomid = np.zeros(1, np.int32)
    grp = np.zeros(6, np.uint8)
    grp[0] = 1  # teren i bloki (dron ma grupy 2-3)
    dist = mujoco.mj_ray(m, d, np.array([x, y, 20.0]), np.array([0, 0, -1.0]), grp, 1, exclude_body, geomid)
    return 20.0 - dist


# --- regulator ---------------------------------------------------------------------------------------


def test_hover_thrust_holds_altitude(hover):
    from sim.control import RateCommand, RateController

    m, d = hover
    start(m, d, z=2.0)
    rc = RateController(m)
    for _ in range(200):
        rc.apply(RateCommand(rc.hover_thrust), m, d)
        mujoco.mj_step(m, d)
    assert d.qpos[2] == pytest.approx(2.0, abs=0.02)
    assert np.linalg.norm(d.qvel) < 0.05


def test_rate_controller_tracks_roll_rate(hover):
    from sim.control import RateCommand, RateController

    m, d = hover
    start(m, d, z=3.0)
    rc = RateController(m)
    rates = []
    for _ in range(60):
        rc.apply(RateCommand(rc.hover_thrust, roll_rate=0.5), m, d)
        mujoco.mj_step(m, d)
        rates.append(d.qvel[3])
    assert np.mean(rates[-20:]) == pytest.approx(0.5, rel=0.05)


def test_acro_does_not_self_level(hover):
    """Plan A: przy zerowej prędkości kątowej dron zostaje w przechyle — nic go nie poziomuje."""
    from sim.control import RateCommand, RateController, euler_zyx

    m, d = hover
    start(m, d, z=3.0)
    rc = RateController(m)
    for k in range(120):
        rc.apply(RateCommand(rc.hover_thrust, pitch_rate=1.0 if k < 20 else 0.0), m, d)
        mujoco.mj_step(m, d)
    assert np.degrees(euler_zyx(d)[1]) > 5


@pytest.mark.parametrize("tau", [0.0, 0.04])
def test_motor_lag(hover, tau):
    from sim.control import RateCommand, RateController

    m, d = hover
    start(m, d, z=3.0)
    rc = RateController(m, motor_tau=tau)
    h = rc.hover_thrust
    rc.apply(RateCommand(1.5 * h), m, d)
    first = d.ctrl.sum()
    if tau == 0:
        assert first == pytest.approx(1.5 * h, rel=1e-6)
        return
    assert first < h + 0.5 * 0.5 * h  # nie od razu
    for _ in range(int(round(tau / m.opt.timestep)) - 1):
        mujoco.mj_step(m, d)
        rc.apply(RateCommand(1.5 * h), m, d)
    reached = (d.ctrl.sum() - h) / (0.5 * h)
    assert 0.5 < reached < 0.75  # po czasie tau ~63 % skoku


@pytest.mark.parametrize("tau", [0.0, 0.04])
def test_velocity_controller_flies_and_brakes(hover, tau):
    from sim.control import RateController, VelocityController

    m, d = hover
    start(m, d, z=3.0)
    vc = VelocityController(RateController(m, motor_tau=tau))
    vc.reset()
    for k in range(500):
        vc.apply(m, d, forward=2.0 if k < 300 else 0.0)
        mujoco.mj_step(m, d)
    assert d.qpos[0] > 3.0
    assert np.linalg.norm(d.qvel[:3]) < 0.3
    assert d.qpos[2] == pytest.approx(3.0, abs=0.15)


def test_nose_and_eyes_point_forward(hover):
    """Nos X2 (gimbal z kamerą) jest w +x, oczy siedzą na nosie (przód siatki, x ~0.15 m)."""
    m, d = hover
    start(m, d)
    base = d.xpos[m.body("x2").id]
    for eye in ("eye_left", "eye_right"):
        assert d.cam_xpos[m.camera(eye).id][0] - base[0] > 0.14
    gid = next(g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    mid = m.geom_dataid[gid]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    vb = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid] - base
    front = vb[vb[:, 0] > 0.13]
    assert len(front) and np.abs(front[:, 2] - 0.065).min() < 0.01  # soczewki gimbala z przodu


# --- świat, bloki, wywrotka ---------------------------------------------------------------------------


def test_randomize_is_reproducible(beacon_model):
    from sim.terrain import randomize

    m = beacon_model
    d = mujoco.MjData(m)
    randomize(m, d, 11)
    hf, target, blocks = m.hfield_data.copy(), m.body_pos[m.body("target").id].copy(), m.body_pos.copy()
    randomize(m, d, 12)
    assert not np.array_equal(hf, m.hfield_data)
    randomize(m, d, 11)
    assert np.array_equal(hf, m.hfield_data)
    assert np.array_equal(target, m.body_pos[m.body("target").id])
    assert np.array_equal(blocks, m.body_pos)


@pytest.mark.parametrize("seed", range(4))
def test_world_guarantees(beacon_model, seed):
    """Płaski start, cel 15-23 m od startu i >= 6 m od ścian, na wyrównanym placu; dron nie stoi w bloku."""
    from sim import terrain
    from sim.terrain import randomize

    m = beacon_model
    d = mujoco.MjData(m)
    randomize(m, d, seed)
    start(m, d)
    x2, tb = m.body("x2").id, m.body("target").id
    for x, y in [(0, 0), (1.0, 0), (0, -1.0)]:
        assert abs(ground_height(m, d, x, y, x2)) < 0.01
    tx, ty, tz = m.body_pos[tb]
    assert terrain.TARGET_DIST[0] - 0.1 <= np.hypot(tx, ty) <= terrain.TARGET_DIST[1] + 0.1
    half = m.hfield_size[m.hfield("terrain").id, 0]
    assert max(abs(tx), abs(ty)) <= half - terrain.WALL_MARGIN + 1e-6
    for dx, dy in [(0, 0), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5)]:
        assert ground_height(m, d, tx + dx, ty + dy, tb) == pytest.approx(tz, abs=0.02)
    assert not any(x2 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) for c in d.contact[:d.ncon])


_BOX_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def _block_parts(m, d, i):
    """[(geom id, narożniki w świecie (8, 3), czy stoi na gruncie)] używanych części bloku i."""
    from sim import blocks

    out = []
    for k in range(blocks.PARTS):
        g = m.geom(f"block{i}_part{k}").id
        size = m.geom_size[g]
        if size.max() < 0.01:  # nieużyta część
            continue
        world = d.geom_xpos[g] + (_BOX_SIGNS * size) @ d.geom_xmat[g].reshape(3, 3).T
        out.append((g, world, m.geom_pos[g][2] - size[2] <= -blocks.SINK + 1e-6))
    return out


def _terrain_z(m, x, y):
    hid = m.hfield("terrain").id
    n, half = m.hfield_nrow[hid], m.hfield_size[hid, 0]
    j, i = (int(round((v + half) / (2 * half) * (n - 1))) for v in (x, y))
    return m.hfield_data[m.hfield_adr[hid] + i * n + j] * m.hfield_size[hid, 2] - 1


@pytest.mark.parametrize("seed", range(3))
def test_blocks_keep_clear_and_stand_on_ground(beacon_model, seed):
    """Start i cel wolne; szczyt albo do przelecenia, albo wyraźnie ponad granicą; podstawa zawsze w ziemi."""
    from sim import blocks
    from sim.terrain import WALL_HEIGHT, randomize

    m = beacon_model
    d = mujoco.MjData(m)
    randomize(m, d, seed)
    start(m, d)
    tx, ty, _ = m.body_pos[m.body("target").id]
    placed = tall = tilted = 0
    for i in range(blocks.POOL_SIZE):
        b = m.body(f"block{i}").id
        if m.body_pos[b][2] <= blocks.HIDDEN_Z + 1:
            continue
        placed += 1
        tilted += d.xmat[b][8] < np.cos(np.deg2rad(5))
        parts = _block_parts(m, d, i)
        corners = np.vstack([c for _, c, _ in parts])
        assert np.hypot(corners[:, 0], corners[:, 1]).min() >= blocks.START_CLEARANCE - 0.05
        assert np.hypot(corners[:, 0] - tx, corners[:, 1] - ty).min() >= blocks.TARGET_CLEARANCE - 0.05
        top = corners[:, 2].max()
        tall += top > WALL_HEIGHT
        assert top <= WALL_HEIGHT - blocks.TOP_CLEARANCE + 0.05 or top >= WALL_HEIGHT + blocks.TALL_MARGIN - 0.05
        for _, c, grounded in parts:  # dolne narożniki części stojących na gruncie: pod ziemią
            if grounded:
                for p in c[np.argsort(c[:, 2])[:4]]:
                    assert p[2] <= _terrain_z(m, p[0], p[1]) + 0.02
    assert blocks.COUNT[0] * 0.8 <= placed <= blocks.COUNT[1]
    assert tilted > 0


def test_some_blocks_tall_and_tilted(beacon_model):
    from sim import blocks
    from sim.terrain import WALL_HEIGHT, randomize

    m = beacon_model
    d = mujoco.MjData(m)
    tall = tilted = 0
    for seed in range(5):
        randomize(m, d, seed)
        start(m, d)
        for i in range(blocks.POOL_SIZE):
            b = m.body(f"block{i}").id
            if m.body_pos[b][2] <= blocks.HIDDEN_Z + 1:
                continue
            tilt = np.degrees(np.arccos(np.clip(d.xmat[b][8], -1, 1)))
            tilted += tilt > 5
            assert tilt <= 90 - blocks.TILT_GROUND_ANGLE[0] + 0.5  # najwyżej 60° od pionu (30° do podłoża)
            tall += max(c[:, 2].max() for _, c, _ in _block_parts(m, d, i)) > WALL_HEIGHT
    assert tall >= 3 and tilted >= 10


def test_drone_collides_with_block(beacon_model):
    """Regresja: bvh_aabb z kompilacji gubił kolizje z przestawionymi częściami (obwiednia puli),
    także po pochyleniu bloku (obrót całego ciała)."""
    from sim import blocks
    from sim.terrain import randomize

    m = beacon_model
    d = mujoco.MjData(m)
    randomize(m, d, 3)
    start(m, d)
    checked = set()
    for i in range(blocks.POOL_SIZE):
        b = m.body(f"block{i}").id
        if m.body_pos[b][2] <= blocks.HIDDEN_Z + 1:
            continue
        kind = "pochylony" if d.xmat[b][8] < np.cos(np.deg2rad(5)) else "pionowy"
        if kind in checked:
            continue
        g = m.geom(f"block{i}_part0").id
        point = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][2] * 0.5  # nad ziemią
        start(m, d)
        d.qpos[:3] = point
        mujoco.mj_forward(m, d)
        assert any(g in (c.geom1, c.geom2) for c in d.contact[:d.ncon]), kind
        checked.add(kind)
        start(m, d)
    assert checked == {"pochylony", "pionowy"}


def test_outside_arena(beacon_model):
    from sim.terrain import WALL_HEIGHT, outside_arena, randomize

    m = beacon_model
    d = mujoco.MjData(m)
    randomize(m, d, 1)
    x2 = m.body("x2").id
    half = m.hfield_size[m.hfield("terrain").id, 0]
    for pos, expected in [((0, 0, 0.3), False), ((half - 0.5, 0, 3), False), ((half - 0.1, 0, 3), True),
                          ((0, 0, WALL_HEIGHT - 0.1), False), ((0, 0, WALL_HEIGHT + 0.1), True)]:
        start(m, d)
        d.qpos[:3] = pos
        mujoco.mj_forward(m, d)
        assert outside_arena(m, d, x2) is expected


def test_target_box_is_penetrable_and_touch_counts(beacon_model):
    """Cel = przenikalny prostopadłościan: brak kolizji, sukces przy zetknięciu (obrys + zasięg łopat)."""
    from sim.target import DRONE_REACH, Target
    from sim.terrain import randomize

    m = beacon_model
    d = mujoco.MjData(m)
    randomize(m, d, 4)
    start(m, d)
    target, x2 = Target(m), m.body("x2").id
    box = m.geom("target_box").id
    assert m.geom_contype[box] == 0 and m.geom_conaffinity[box] == 0
    half = m.geom_size[box].copy()
    center = d.geom_xpos[box].copy()
    for offset, expected in [((0, 0, 0), True), ((half[0] + DRONE_REACH - 0.02, 0, 0.5), True),
                             ((half[0] + DRONE_REACH + 0.1, 0, 0.5), False), ((0, 0, half[2] + DRONE_REACH + 0.2), False)]:
        d.qpos[:3] = center + offset
        mujoco.mj_forward(m, d)
        assert target.reached(d, x2) is expected
        assert not any(box in (c.geom1, c.geom2) for c in d.contact[:d.ncon])  # przelatuje przez cel
    size = m.geom_size[box].copy()
    m.geom_size[box, :2] *= 4  # beacon_scale w banc_pilot: szerszy cel -> łatwiej dotknąć
    d.qpos[:3] = center + (half[0] + DRONE_REACH + 0.1, 0, 0.5)
    mujoco.mj_forward(m, d)
    assert target.reached(d, x2)
    m.geom_size[box] = size


def test_crash_detector(hover):
    from sim.control import RateCommand, RateController
    from sim.episode import UPSIDE_DOWN_TIME, CrashDetector

    m, d = hover
    crash = CrashDetector(m)
    rc = RateController(m)
    start(m, d, z=2.0)
    for _ in range(300):
        rc.apply(RateCommand(rc.hover_thrust), m, d)
        mujoco.mj_step(m, d)
        assert crash.update(d, m.opt.timestep) is None
    start(m, d, z=5.0, quat=axis_quat([1, 0, 0], np.pi))
    crash.reset()
    reason, t = None, 0.0
    while reason is None and t < 3:
        mujoco.mj_step(m, d)
        t += m.opt.timestep
        reason = crash.update(d, m.opt.timestep)
    assert reason == "do góry nogami"
    assert t == pytest.approx(UPSIDE_DOWN_TIME, abs=0.05)


def test_wind_pushes_drone(hover):
    from sim.control import RateCommand, RateController
    from sim.wind import Wind

    m, d = hover
    for enabled in (False, True):
        wind = Wind(seed=3)
        wind.enabled = enabled
        start(m, d, z=3.0)
        rc = RateController(m)
        p0 = d.qpos[:2].copy()
        for _ in range(200):
            wind.step(m, m.opt.timestep)
            rc.apply(RateCommand(rc.hover_thrust), m, d)
            mujoco.mj_step(m, d)
        drift = d.qpos[:2] - p0
        if enabled:
            direction = wind.velocity[:2] / np.linalg.norm(wind.velocity[:2])
            assert drift @ direction > 0.5
            assert 3.0 < np.linalg.norm(wind.velocity) < 13.0
        else:
            assert np.linalg.norm(drift) < 0.05
            assert not m.opt.wind.any()
    m.opt.wind[:] = 0


# --- czujniki, GPS, metryki ---------------------------------------------------------------------------


def _hover_with_sensors(m, d, sensors, seconds, z=2.0, target=None):
    from sim.control import RateCommand, RateController

    start(m, d, z=z)
    sensors.reset(7, d, target=target)
    rc = RateController(m)
    out = []
    for _ in range(int(seconds / m.opt.timestep)):
        rc.apply(RateCommand(rc.hover_thrust), m, d)
        mujoco.mj_step(m, d)
        out.append((sensors.update(d), sensors.latest))
    return out


def test_sensors_ideal_equal_truth(hover):
    from sim.sensors import KEYS, DroneSensors

    m, d = hover
    for reading, true in _hover_with_sensors(m, d, DroneSensors(m, "ideal"), 0.3, target=[5, 0, 0]):
        for key in KEYS:
            np.testing.assert_allclose(reading[key], true[key])


def test_sensors_real_noise_range_and_latency(hover):
    from sim.sensors import GYRO_NOISE, LATENCY, RANGE_MAX, DroneSensors

    m, d = hover
    out = _hover_with_sensors(m, d, DroneSensors(m, "real"), 3.0)
    gyro_err = np.array([r["gyro"] - t["gyro"] for r, t in out])
    rng_err = np.array([r["range"] - t["range"] for r, t in out if np.isfinite(r["range"])])
    assert 0.5 * GYRO_NOISE < gyro_err.std(0).mean() < 2 * GYRO_NOISE
    assert rng_err.std() < 0.06
    assert 0.0 < np.mean([not np.isfinite(r["range"]) for r, _ in out]) < 0.05  # zgubione odczyty ~1 %

    high = _hover_with_sensors(m, d, DroneSensors(m, "real"), 0.1, z=RANGE_MAX + 0.6)
    assert not np.isfinite(high[-1][0]["range"]) and np.isnan(high[-1][0]["flow"]).all()

    s = DroneSensors(m, "real")
    start(m, d, z=2.0)
    s.reset(1, d)
    d.qvel[3] = 5.0  # skok prędkości kątowej
    seen = None
    for k in range(10):
        mujoco.mj_forward(m, d)
        if seen is None and s.update(d)["gyro"][0] > 2.5:
            seen = k
    assert seen * m.opt.timestep == pytest.approx(LATENCY, abs=m.opt.timestep)


def test_sensors_reproducible(hover):
    from sim.sensors import DroneSensors

    m, d = hover
    a = _hover_with_sensors(m, d, DroneSensors(m, "real"), 0.5, target=[8, 3, 0])
    b = _hover_with_sensors(m, d, DroneSensors(m, "real"), 0.5, target=[8, 3, 0])
    for (ra, _), (rb, _) in zip(a, b):
        np.testing.assert_array_equal(ra["gyro"], rb["gyro"])
        np.testing.assert_array_equal(ra["beacon"], rb["beacon"])


def test_gps_beacon(hover):
    from sim.sensors import GPS_RATE, DroneSensors

    m, d = hover
    target = np.array([20.0, -6.0, 0.0])
    ideal = _hover_with_sensors(m, d, DroneSensors(m, "ideal"), 0.1, target=target)[-1][0]["beacon"]
    assert ideal[0] == pytest.approx(np.arctan2(-6, 20), abs=1e-3)  # + = w lewo, cel w prawo -> ujemny
    assert ideal[1] == pytest.approx(np.hypot(20, 6), abs=0.05)

    out = _hover_with_sensors(m, d, DroneSensors(m, "real"), 2.0, target=target)
    errs = np.degrees([abs((r["beacon"][0] - t["beacon"][0] + np.pi) % (2 * np.pi) - np.pi) for r, t in out])
    assert np.median(errs) < 15  # daleko (~21 m): wie mniej więcej, dokąd
    assert np.median(errs) > 0.1  # ale nie dokładnie
    readings = [tuple(r["beacon"]) for r, _ in out]
    changes = sum(1 for a, b in zip(readings, readings[1:]) if a != b)
    assert changes == pytest.approx(2.0 * GPS_RATE, abs=2)


def test_metrics_summary_and_csv(hover, tmp_path):
    from sim.metrics import SUMMARY_FIELDS, EpisodeMetrics, sensors_panel
    from sim.sensors import DroneSensors

    m, d = hover
    path = tmp_path / "metryki.csv"
    metrics = EpisodeMetrics(m, csv_path=path)
    target = np.array([6.0, 0.0, 0.0])
    start(m, d, z=1.0)
    metrics.start(d, target, world=5)
    for _ in range(100):
        d.qpos[0] += 0.02  # 2 m w stronę celu
        mujoco.mj_forward(m, d)
        metrics.update(d, target, np.array([4.0, 0, 0]))
    summary = metrics.finish("cel")
    assert summary["wynik"] == "cel" and summary["swiat"] == 5
    assert summary["droga_m"] == pytest.approx(2.0, abs=0.05)
    assert summary["postep_proc"] == pytest.approx(100 * 2 / np.hypot(6, 0.7), abs=3)
    assert summary["wiatr_m_s"] == pytest.approx(4.0)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 1 and list(rows[0]) == list(SUMMARY_FIELDS)
    left, right = metrics.panel()
    assert "Cel" in left and len(left.splitlines()) == len(right.splitlines())
    s = DroneSensors(m, "real")
    s.reset(1, d, target=target)
    s.update(d)
    left, right = sensors_panel(s.read(), s.latest)
    assert "GPS" in left and len(left.splitlines()) == len(right.splitlines())


# --- WorldEnv ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("sensors", ["ideal", "real"])
def test_world_env_gymnasium_api(sensors):
    from gymnasium.utils.env_checker import check_env

    from sim.world_env import WorldEnv

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(WorldEnv(eyes=False, sensors=sensors, motor_tau=0.04), skip_render_check=True)


def test_world_env_outcomes():
    from sim.world_env import WorldEnv

    env = WorldEnv(eyes=False, start_noise=False)
    for place, expected in [("target", "cel"), ("outside", "poza planszą"), ("upside", "wywrotka: do góry nogami")]:
        obs, info = env.reset(seed=3)
        if place == "target":
            env.data.qpos[:3] = info["target"] + [0.4, 0.0, 0.5]
        elif place == "outside":
            env.data.qpos[:3] = (0, 0, 9.0)
        else:
            env.data.qpos[2] = 4.0
            env.data.qpos[3:7] = axis_quat([1, 0, 0], np.pi)
        mujoco.mj_forward(env.model, env.data)
        outcome = None
        for _ in range(60):
            obs, reward, terminated, truncated, info = env.step([0.5, 0, 0, 0])
            if terminated:
                outcome = info["outcome"]
                break
        assert outcome == expected
        assert (reward > 0) == (expected == "cel")


def test_world_env_reproducible_and_observation():
    from sim.world_env import WorldEnv

    env = WorldEnv(eyes=False, sensors="real")
    runs = []
    for _ in range(2):
        obs, info = env.reset(seed=4)
        for _ in range(10):
            obs, *_ = env.step([0.55, 0.1, 0, 0])
        runs.append(obs)
    for key in ("imu", "sensors", "beacon"):
        np.testing.assert_array_equal(runs[0][key], runs[1][key])
    assert obs["sensors"].shape == (5,) and obs["beacon"].shape == (2,)
    assert 15 <= obs["beacon"][1] <= 26  # GPS: cel 15-23 m (+ błąd)


def test_world_env_eyes_see_new_terrain():
    """Regresja: renderer oczu trzymał teren z GPU sprzed randomize() (upload_terrain)."""
    from sim.world_env import WorldEnv
    from visual_pipeline.drone_eyes import MujocoEyes

    try:
        env = WorldEnv(eyes=True)
    except Exception as exc:  # brak OpenGL w środowisku
        pytest.skip(f"brak renderu: {exc}")
    for seed in (5, 6):
        obs, _ = env.reset(seed=seed)
        fresh = np.stack(MujocoEyes(env.model).render(env.data))
        assert (np.abs(obs["eyes"].astype(int) - fresh.astype(int)).max(-1) > 10).mean() < 0.01


# --- wizualizacja (bez renderu) --------------------------------------------------------------------------


def test_trail(hover):
    from sim.trail import MIN_STEP, Trail

    m, _ = hover
    trail = Trail(max_points=10)
    for x in np.arange(0, 2, MIN_STEP / 2):
        trail.add([x, 0, 1])
    assert len(trail.points) == 10  # co MIN_STEP, najstarsze wypadają
    scn = mujoco.MjvScene(m, maxgeom=100)
    trail.draw(scn)
    assert scn.ngeom == 9
    trail.visible = False
    scn.ngeom = 0
    trail.draw(scn)
    assert scn.ngeom == 0
    trail.reset()
    assert not trail.points


def test_propellers(hover):
    from sim.propellers import BLADES_PER_PROP, PropellerVisuals

    m, d = hover
    start(m, d)
    props = PropellerVisuals(m)
    assert sorted(props.spin_dir) == [-1, -1, 1, 1]  # pary przeciwbieżne
    scn = mujoco.MjvScene(m, maxgeom=100)
    props.draw(scn, d)
    assert scn.ngeom == 4 * (BLADES_PER_PROP + 1)
