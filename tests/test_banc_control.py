import json

import numpy as np
import pandas as pd
import pytest

from banc_control import BancController, Connectome, ImuState, parse_visual_activity
from banc_control.connectome import DEFAULT_DATA_DIR, EDGES_FILE, META_FILE, flight_groups
from banc_control.stubs import FakeVision, ToyDrone


def mini_banc():
    """Mały graf w oficjalnym schemacie BANC v888: wzrok → DN → MN skrzydeł, po obu stronach."""
    rows = []
    for side in ("left", "right"):
        s = side[0]
        rows += [
            dict(banc_888_id=f"1{s}1", super_class="visual_projection", side=side, cell_type="LC4"),
            dict(banc_888_id=f"1{s}2", super_class="descending", side=side, cell_type="DNx",
                 super_cluster="flight power"),
            dict(banc_888_id=f"1{s}3", super_class="motor", side=side, cell_type="DLM1-4", cell_function="wing_power"),
            dict(banc_888_id=f"1{s}4", super_class="motor", side=side, cell_type="b1", cell_function="wing_steering"),
            dict(banc_888_id=f"1{s}5", super_class="sensory", side=side, cell_type="haltere_cs", flow="afferent",
                 body_part_sensory="haltere"),
        ]
    rows.append(dict(banc_888_id="999", super_class="glia", side="left", cell_type="glia"))
    meta = pd.DataFrame(rows).fillna(np.nan)
    meta["banc_888_id"] = meta["banc_888_id"].str.replace("l", "1").str.replace("r", "2")
    meta["neurotransmitter_predicted"] = "acetylcholine"
    ids = dict(zip(meta["cell_type"] + meta["side"], meta["banc_888_id"]))
    edges = []
    for side in ("left", "right"):
        edges += [
            (ids["LC4" + side], ids["DNx" + side], 40),
            (ids["DNx" + side], ids["DLM1-4" + side], 30),
            (ids["DNx" + side], ids["b1" + side], 30),
            (ids["haltere_cs" + side], ids["DLM1-4" + side], 20),
        ]
    edges.append((ids["LC4left"], ids["DNxright"], 3))  # poniżej progu count ≥ 5
    return meta, pd.DataFrame(edges, columns=["pre", "post", "count"])


def make_controller():
    c = Connectome.from_banc_tables(*mini_banc())
    ctrl = BancController(c)
    ctrl.calibrate_rest()
    return c, ctrl


def test_groups_come_from_official_annotations():
    meta, _ = mini_banc()
    g = set(flight_groups(meta)) - {""}
    assert g == {f"{n}_{s}" for n in ("visual", "dn_flight_power", "wing_power", "wing_steering", "haltere_aff")
                 for s in "LR"}


def test_loader_drops_non_neurons_and_weak_edges():
    c, _ = make_controller()
    assert "glia" not in set(c.super_class)
    assert c.W.nnz == 8


def test_rest_gives_hover_command():
    _, ctrl = make_controller()
    cmd = ctrl.step([])
    assert abs(cmd.thrust - 0.5) < 1e-6 and abs(cmd.roll) < 1e-6


def test_beacon_side_flips_roll_and_yaw():
    c, _ = make_controller()
    vision = FakeVision(c)
    out = {}
    for bearing in (-0.8, 0.8):
        ctrl = BancController(c)
        ctrl.calibrate_rest()
        for _ in range(30):
            cmd = ctrl.step(vision(bearing))
        out[bearing] = (cmd.roll, cmd.yaw)
    assert np.sign(out[-0.8][0]) == -np.sign(out[0.8][0]) != 0
    assert np.sign(out[-0.8][1]) == -np.sign(out[0.8][1]) != 0


def test_dynamics_stay_bounded():
    c, ctrl = make_controller()
    vision = FakeVision(c)
    for _ in range(200):
        ctrl.step(vision(0.3, brightness=5.0), ImuState(gyro=(3.0, 0.0, 0.0)))
    assert np.all(np.isfinite(ctrl.dyn.r)) and ctrl.dyn.r.max() <= 1.0


def test_parse_contract_and_unmatched_ids():
    c, ctrl = make_controller()
    payload = json.dumps([
        {"banc_root_id": int(c.root_ids[0]), "cell_type": "LC4", "activity": 0.8},
        {"banc_root_id": 1, "cell_type": "??", "activity": 0.1},
    ])
    assert ctrl.step(parse_visual_activity(payload)).debug["unmatched_ids"] == 1


def test_closed_loop_with_toy_drone_runs():
    c, ctrl = make_controller()
    vision, drone = FakeVision(c), ToyDrone()
    imu = None
    for _ in range(100):
        imu = drone.step(ctrl.step(vision(-drone.yaw), imu))
    assert np.isfinite(drone.roll) and np.isfinite(drone.z)


@pytest.mark.skipif(not (DEFAULT_DATA_DIR / META_FILE).exists() or not (DEFAULT_DATA_DIR / EDGES_FILE).exists(),
                    reason="brak danych BANC — python scripts/download_banc.py")
def test_official_banc_v888_loads():
    c = Connectome.from_banc()
    assert c.n > 170_000
    for g in ("wing_power_L", "wing_power_R", "wing_steering_L", "wing_steering_R", "haltere_aff_L", "visual_R"):
        assert len(c.group_indices(g)) > 0, g


def test_calibrate_scale_brings_commands_to_unit_range():
    c, ctrl = make_controller()
    vision = FakeVision(c)
    ctrl.calibrate_scale([vision(-0.8), vision(0.8)])
    for _ in range(30):
        cmd = ctrl.step(vision(0.8))
    assert 0.3 < abs(cmd.roll) <= 1.0


def test_haltere_sign_calibration_makes_loop_corrective():
    c, ctrl = make_controller()
    ctrl.haltere_roll_weight = 1.0  # domyślnie roll nie idzie przez haltery
    vision = FakeVision(c)
    ctrl.calibrate_scale([vision(-0.8), vision(0.8)])
    ctrl.calibrate_haltere_sign()
    ctrl.dyn.reset()
    for _ in range(30):
        cmd = ctrl.step([], ImuState(gyro=(1.5, 0.0, 0.0)))
    assert cmd.roll < 0


@pytest.mark.skipif(not __import__("banc_control.dynamics").dynamics.default_device() == "cuda",
                    reason="brak CUDA")
def test_gpu_matches_cpu():
    c = Connectome.from_banc_tables(*mini_banc())
    stim = FakeVision(c)(0.5)  # FakeVision losuje szum przy każdym wywołaniu
    feats = {}
    for dev in ("cpu", "cuda"):
        ctrl = BancController(c, device=dev)
        for _ in range(20):
            ctrl.step(stim, ImuState(gyro=(1.0, 0.0, 0.0)))
        feats[dev] = ctrl.motor_features()
    assert np.allclose(feats["cpu"], feats["cuda"], atol=1e-5)


def test_yaw_sign_from_rotation_makes_optomotor_corrective():
    c, ctrl = make_controller()
    vision = FakeVision(c)
    ctrl.calibrate_scale([vision(-0.8), vision(0.8)])
    # W mini grafie pobudzenie lewej strony daje yaw > 0. Udajemy, że tak BANC odpowiada na
    # obrót drona w prawo → bez kalibracji odruch optomotoryczny wzmacniałby obrót.
    turn_right = [vision(-0.8)] * 20
    turn_left = [vision(0.8)] * 20
    assert ctrl._mean_yaw(turn_right) > ctrl._mean_yaw(turn_left)
    diff = ctrl.calibrate_yaw_sign(turn_right, turn_left)
    assert diff < 0 and ctrl.yaw_axis_sign == -1
    assert ctrl._mean_yaw(turn_right) < 0 < ctrl._mean_yaw(turn_left)
    assert ctrl.calibrate_yaw_sign(turn_right, turn_left) < 0 and ctrl.yaw_axis_sign == -1  # stabilne


def test_haltere_yaw_rate_has_own_sign():
    c, ctrl = make_controller()
    ctrl.calibrate_scale([FakeVision(c)(-0.8), FakeVision(c)(0.8)])
    ctrl.calibrate_haltere_sign()
    assert ctrl.haltere_yaw_sign in (-1.0, 1.0)
    ctrl.dyn.reset()
    for _ in range(30):
        cmd = ctrl.step([], ImuState(gyro=(0.0, 0.0, 1.5)))
    assert cmd.yaw <= 1e-9  # obrót w prawo nie może dawać komendy w prawo
