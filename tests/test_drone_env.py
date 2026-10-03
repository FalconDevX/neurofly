import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from banc_control import FlightCommand  # noqa: E402


def _env(**kw):
    from sim.env import DroneEnv

    return DroneEnv(eyes=False, **kw)


def _fly(env, cmd, seconds):
    for _ in range(int(seconds * env.fps)):
        obs, reward, terminated, truncated, info = env.step(cmd)
    return info


def test_reset_puts_target_at_requested_bearing():
    env = _env()
    for b in (-60, 0, 45):
        _, info = env.reset(bearing_deg=b, rng=np.random.default_rng(b + 100))
        assert np.rad2deg(info["bearing"]) == pytest.approx(b, abs=1e-6)
        assert info["z"] == pytest.approx(env.start_z)


def test_neutral_command_holds_hover():
    env = _env()
    env.reset("hover")
    info = _fly(env, FlightCommand(thrust=0.5), 2.0)
    assert abs(info["z"] - env.start_z) < 0.02
    assert abs(info["bearing"]) < np.deg2rad(1)
    assert np.hypot(*info["xy"]) < 0.02


def test_yaw_plus_turns_right_and_thrust_climbs():
    env = _env()
    env.reset("hover")
    info = _fly(env, FlightCommand(thrust=0.5, yaw=1.0), 1.0)
    assert info["heading"] > np.deg2rad(30)        # + = w prawo
    assert info["bearing"] < -np.deg2rad(30)       # cel na wprost zostaje po lewej
    env.reset("hover")
    info = _fly(env, FlightCommand(thrust=0.8), 1.0)
    assert info["z"] > env.start_z + 0.1


def test_steering_toward_target_closes_bearing_and_gust_disturbs():
    env = _env()
    _, info = env.reset("turn_right")
    while True:
        _, _, term, trunc, info = env.step(FlightCommand(thrust=0.5, yaw=float(np.clip(1.5 * info["bearing"], -1, 1))))
        if term or trunc:
            break
    assert not term and abs(info["bearing"]) < np.deg2rad(2)

    env.reset("gust")
    info = _fly(env, FlightCommand(thrust=0.5), 3.0)  # bez korekty podmuch obraca drona
    assert abs(info["bearing"]) > np.deg2rad(10)


def test_summarize_reports_settle_time():
    from sim.env import summarize

    log = [{"bearing": np.deg2rad(b), "z": 1.0, "xy": [0.0, 0.0], "t": k / 30}
           for k, b in enumerate([60, 30, 15, 5, 3, 1])]
    m = summarize(log)
    assert m["settle_time_s"] == pytest.approx(3 / 30) and m["final_deg"] == pytest.approx(1)


def _renderer_env():
    from sim.env import DroneEnv

    try:
        return DroneEnv()
    except Exception as e:  # brak kontekstu OpenGL (np. CI bez GPU)
        pytest.skip(f"renderowanie MuJoCo niedostępne: {e}")


def test_eyes_see_target_on_its_side():
    env = _renderer_env()

    def darkest(img):
        return img[200:300].mean(axis=(0, 2)).min()

    left, right = env.calibration_render(np.deg2rad(60))
    assert darkest(right) < darkest(left) - 10
    left, right = env.calibration_render(np.deg2rad(-60))
    assert darkest(left) < darkest(right) - 10


def test_closed_loop_through_local_client():
    """DroneEnv → LocalClient → ControlServer (fałszywy most, mały graf w schemacie BANC) → DroneEnv."""
    from banc_control import BancController, Connectome
    from tests.test_banc_control import mini_banc
    from tests.test_visual_pipeline import _FakeBridge
    from visual_pipeline.server import ControlServer, LocalClient

    env = _renderer_env()
    c = Connectome.from_banc_tables(*mini_banc())
    client = LocalClient(ControlServer(_FakeBridge(c), BancController(c)))
    calib = client.calibrate(env.calibration_render, turn_frames=5)
    assert calib["ok"]
    obs, _ = env.reset("turn_right")
    for k in range(10):
        reply = client.step(obs.left, obs.right, obs.imu, reset=(k == 0))
        obs, _, term, _, info = env.step(reply)
        assert not term
    assert set(info["cmd"]) == {"thrust", "roll", "pitch", "yaw"}


def test_saturating_yaw_does_not_change_altitude():
    """Yaw ±1 co klatkę nasyca mixer; ciąg zbiorczy ma zostać (wcześniej dron wznosił się ~2 m/s)."""
    env = _env()
    env.reset("hover")
    for k in range(60):
        _, _, _, _, info = env.step(FlightCommand(thrust=0.5, yaw=1.0 if k % 2 else -1.0))
    assert abs(info["z"] - env.start_z) < 0.05


def test_thrust_hold_ignores_banc_thrust():
    env = _env(thrust_mode="hold")
    env.reset("hover")
    info = _fly(env, FlightCommand(thrust=0.0), 2.0)
    assert abs(info["z"] - env.start_z) < 0.05
