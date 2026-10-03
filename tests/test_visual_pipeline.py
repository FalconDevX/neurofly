import numpy as np
import pandas as pd
import pytest

from banc_control import parse_visual_activity

MAP_FILE = "visual_pipeline/flyvis_banc_map.csv"


def test_map_file_schema():
    m = pd.read_csv(MAP_FILE)
    assert {"eye", "flyvis_index", "flyvis_type", "u", "v", "banc_888_id", "banc_cell_type"} <= set(m.columns)
    assert set(m.eye) == {"left", "right"}
    assert not m.duplicated(["eye", "flyvis_index"]).any()
    assert m.banc_888_id.notna().all()


def test_visual_batch_matches_record_list():
    from banc_control import BancActivation, BancController, Connectome, VisualBatch
    from tests.test_banc_control import mini_banc

    c = Connectome.from_banc_tables(*mini_banc())
    ctrl = BancController(c)
    ids = np.array([*c.root_ids[:4], 123456789], dtype=np.int64)  # ostatni nie istnieje
    act = np.array([0.5, -1.0, 2.0, 0.0, 7.0])

    from_list = ctrl.external_input([BancActivation(int(r), "", float(a)) for r, a in zip(ids, act)], None)
    unmatched_list = ctrl.unmatched_ids
    from_batch = ctrl.external_input(VisualBatch(ids, act), None)
    assert np.allclose(from_list, from_batch)
    assert ctrl.unmatched_ids == unmatched_list == 1


def test_prepare_frame_resizes_and_drops_alpha():
    pytest.importorskip("PIL")
    from visual_pipeline.frames import RETINA_SHAPE, prepare_frame

    assert prepare_frame(np.zeros((480, 640, 4), np.uint8)).shape == RETINA_SHAPE
    assert prepare_frame(np.ones((100, 100))).max() == 255


def test_fake_camera_puts_beacon_on_the_correct_side():
    from visual_pipeline.fake_camera import BAR, FakeStereoCamera

    cam = FakeStereoCamera(beacon_azimuth=0.0)
    left, right = cam.render(yaw=-np.deg2rad(45))  # cel 45° w prawo
    assert (right == BAR).any() and not (left == BAR).any()
    assert np.isclose(cam.bearing(-np.deg2rad(45)), np.deg2rad(45))


def test_zmq_request_roundtrip():
    from visual_pipeline.zmq_protocol import decode_request, encode_request

    left = np.random.default_rng(0).integers(0, 255, (512, 450, 3), dtype=np.uint8)
    right = left[::-1].copy()
    imu = {"gyro": [0.1, 0.0, -0.2], "accel": [0.0, 0.0, -9.81]}
    l2, r2, imu2, reset = decode_request(encode_request(left, right, imu, reset=True))
    assert np.array_equal(l2, left) and np.array_equal(r2, right)
    assert imu2 == imu and reset


def test_to_luminance_is_grey_and_close_to_bt601():
    from visual_pipeline.frames import to_luminance

    img = np.random.default_rng(1).integers(0, 255, (64, 48, 3), dtype=np.uint8)
    out = to_luminance(img)
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()
    ref = img.astype(float) @ [0.299, 0.587, 0.114]
    assert np.abs(out[..., 0] - ref).max() <= 1.5


def test_eye_cameras_look_sideways_with_front_toward_the_nose():
    import re

    from visual_pipeline.drone_eyes import eye_camera_xml

    for eye, side in (("right", -1), ("left", 1)):
        xyaxes = [float(x) for x in re.search(r'xyaxes="([^"]+)"', eye_camera_xml(eye)).group(1).split()]
        img_right, img_up = np.array(xyaxes[:3]), np.array(xyaxes[3:])
        view = -np.cross(img_right, img_up)  # kamera MuJoCo patrzy wzdłuż −z = −(x × y)
        assert np.sign(view[1]) == side  # +y = lewo w MuJoCo
        assert view[0] > 0  # lekko do przodu
        # prawe oko: prawo kadru do tyłu (przód po lewej); lewe oko: prawo kadru do przodu
        assert np.sign(img_right[0]) == (-1 if eye == "right" else 1)


def test_retina_mapper_is_bijection():
    pytest.importorskip("flygym")
    pytest.importorskip("flyvis")
    from visual_pipeline.retina_mapper import RetinaMapper

    mapper = RetinaMapper()
    for eye in ("left", "right"):
        assert sorted(mapper.flygym_to_flyvis_idx[eye]) == list(range(721))


def test_bridge_emits_contract_records():
    pytest.importorskip("flygym")
    flyvis = pytest.importorskip("flyvis")
    if not (flyvis.results_dir / "flow/0000/000").exists():
        pytest.skip("brak wag FlyVis: flyvis download-pretrained")
    from visual_pipeline import VisionBridge

    bridge = VisionBridge()
    frame = np.full((bridge.retina.nrows, bridge.retina.ncols, 3), 128, np.uint8)
    records = parse_visual_activity([r.__dict__ for r in bridge.step(frame, frame)])
    assert len(records) == len(bridge.root_ids) > 0
    assert len({r.banc_root_id for r in records}) == len(records)
    assert np.isfinite([r.activity for r in records]).all()


class _FakeBridge:
    """Interfejs VisionBridge bez FlyVis: aktywność = średnia jasność lewej/prawej połowy kadru."""

    def __init__(self, connectome):
        self.left = connectome.root_ids[connectome.group_indices("visual_L")]
        self.right = connectome.root_ids[connectome.group_indices("visual_R")]
        self.root_ids = np.concatenate([self.left, self.right]).astype(np.int64)
        self.last_timings = {}

    def reset(self):
        pass

    def step_batch(self, left, right):
        from banc_control import VisualBatch

        act = [1.0 - left.mean() / 255] * len(self.left) + [1.0 - right.mean() / 255] * len(self.right)
        return VisualBatch(self.root_ids, np.array(act))

    settle = step_batch


def test_server_calibrates_on_simulator_scenes():
    from banc_control import BancController, Connectome
    from tests.test_banc_control import mini_banc
    from visual_pipeline.server import ControlServer
    from visual_pipeline.zmq_protocol import decode, encode_request

    c = Connectome.from_banc_tables(*mini_banc())
    server = ControlServer(_FakeBridge(c), BancController(c))

    def render(bearing):  # ciemny cel po stronie bearing
        l, r = np.full((8, 8, 3), 200, np.uint8), np.full((8, 8, 3), 200, np.uint8)
        (r if bearing > 0 else l)[:] -= np.uint8(min(abs(bearing) * 150, 150))
        return l, r

    def send(*args, **kw):
        return server.handle(*decode(encode_request(*args, **kw)))[0]

    assert "error" in send(None, None, calib="finish")
    for scene, b in (("neutral", 0.0), ("left", -1.0), ("right", 1.0)):
        assert send(*render(b), calib=scene)["ok"]
    for k in range(10):
        send(*render(-0.1 * k), calib="turn_right")
        send(*render(0.1 * k), calib="turn_left")
    out = send(None, None, calib="finish")
    assert out["ok"] and out["optomotor_yaw_diff"] <= 0
    for _ in range(30):  # po resecie dynamika potrzebuje kilku klatek do stanu spoczynku
        reply = send(*render(0.0), imu={"gyro": [0.0, 0.0, 0.0], "accel": [0.0, 0.0, -9.81]})
    assert abs(reply["thrust"] - 0.5) < 0.05 and reply["unmatched_ids"] == 0
