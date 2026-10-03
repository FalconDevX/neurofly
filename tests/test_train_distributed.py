"""Logika mastera treningu rozproszonego bez sieci i GPU: dwa fałszywe workery."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _args(tmp_path, episodes=6):
    return argparse.Namespace(plan="B", episodes=episodes, batch=2, max_bearing=90.0, init=None, seed=0,
                              out=tmp_path / "w.npz", port=0)


def _hello(name, sign=-1.0):
    return {"type": "hello", "name": name, "M": np.eye(4, 6).tolist(),
            "calib": {"yaw_axis_sign": sign, "hover_thrust": 0.5, "pitch_trim": 0.0, "haltere_gain": 1e-4}}


def _eval(name, tag):
    ev = {"+30": {"final_deg": 5.0}, "mean_final_deg": 5.0, "turns_toward": 1}
    return {"type": "result", "kind": "eval", "name": name, "tag": tag, "eval": ev}


def _train(name, task, dM):
    metrics = [{**e, "loss": 0.0, "final_deg": 1.0} for e in task["episodes"]]
    return {"type": "result", "kind": "train", "name": name, "dM": dM.tolist(), "metrics": metrics}


def test_master_sums_worker_updates_and_saves(tmp_path):
    from train_distributed import Master

    m = Master(_args(tmp_path))
    a = m.handle(_hello("A"))
    assert a["kind"] == "eval" and a["tag"] == "before"
    b = m.handle(_hello("B"))
    assert b["kind"] == "train" and len(b["episodes"]) == 2
    assert m.handle(_hello("C", sign=1.0))["kind"] == "stop"  # inny znak osi yaw → odrzucony

    dM = np.full((4, 6), 0.1)
    a = m.handle(_eval("A", "before"))
    assert a["kind"] == "train"
    b = m.handle(_train("B", b, dM))
    a = m.handle(_train("A", a, dM))
    b = m.handle(_train("B", b, dM))
    assert m.done_episodes == 6 and b["kind"] == "eval" and b["tag"] == "after"
    assert a["kind"] == "wait" or m.handle({"type": "poll", "name": "A"})["kind"] == "wait"
    np.testing.assert_allclose(m.M, np.eye(4, 6) + 3 * dM)
    assert m.handle(_eval("B", "after"))["kind"] == "stop"
    d = np.load(tmp_path / "w.npz")
    np.testing.assert_allclose(d["M"], m.M)
    assert [e["index"] for e in m.history] == [0, 1, 2, 3, 4, 5] or len(m.history) == 6
