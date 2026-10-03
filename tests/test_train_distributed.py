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


def test_master_waits_for_all_workers_and_aborts_without_slave(tmp_path):
    from train_distributed import Master

    a = _args(tmp_path)
    a.workers, a.wait_join, a.wait_ready = 2, 120.0, 900.0
    m = Master(a)
    assert m.handle({"type": "ping", "name": "A", "gpu": "RTX"})["kind"] == "pong"
    assert m.handle(_hello("A"))["kind"] == "wait"  # B jeszcze nie ma — nie startujemy
    m.check_deadlines(60.0)
    assert not m.finished
    m.check_deadlines(121.0)  # slave nie zgłosił się w czasie
    assert m.finished and "1/2" in m.error
    reply = m.handle({"type": "poll", "name": "A"})
    assert reply["kind"] == "stop" and reply["error"]
    assert not (tmp_path / "w.npz").exists()  # nic nie zapisano

    m = Master(a)  # obaj są → trening rusza
    for n in ("A", "B"):
        m.handle({"type": "ping", "name": n})
    assert m.handle(_hello("A"))["kind"] == "wait"
    assert m.handle(_hello("B"))["kind"] == "eval"
    m.check_deadlines(1000.0)
    assert not m.finished


def test_worker_labels_are_gpu_names(tmp_path):
    from train_distributed import Master, short_gpu

    assert short_gpu("NVIDIA GeForce RTX 3070 Ti Laptop GPU") == "RTX 3070 Ti"
    m = Master(_args(tmp_path))
    m.handle({"type": "ping", "name": "DESKTOP-A", "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU"})
    m.handle({"type": "ping", "name": "DESKTOP-B", "gpu": "NVIDIA GeForce RTX 3070 Ti Laptop GPU"})
    assert m.label("DESKTOP-A") == "RTX 4060" and m.label("DESKTOP-B") == "RTX 3070 Ti"
    m.handle({"type": "ping", "name": "DESKTOP-C", "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU"})
    assert m.label("DESKTOP-A") == "RTX 4060 (DESKTOP-A)"  # dwie takie same karty → + komputer


def test_master_takes_initial_weights_from_worker_with_init_file(tmp_path):
    from train_distributed import Master

    a = _args(tmp_path)
    a.workers = 2
    m = Master(a)
    m.handle({"type": "ping", "name": "slave"})
    m.handle({"type": "ping", "name": "laptop"})
    m.handle({**_hello("slave"), "M": np.zeros((4, 6)).tolist(), "init": False})  # pierwszy, ale bez pliku
    m.handle({**_hello("laptop"), "init": True})
    np.testing.assert_array_equal(m.M0, np.eye(4, 6))


def test_master_rejects_worker_in_other_mode(tmp_path):
    from train_distributed import Master

    a = _args(tmp_path)
    a.world = True
    m = Master(a)
    reply = m.handle({"type": "ping", "name": "slave", "gpu": "RTX 3070 Ti"})  # worker bez --world
    assert reply["kind"] == "stop" and "--world" in reply["error"]
    assert "slave" not in m.pinged
    assert m.handle({"type": "ping", "name": "laptop", "world": True})["kind"] == "pong"

    a = _args(tmp_path)
    m = Master(a)
    m.handle(_hello("A"))
    bad = {**_hello("B"), "M": np.zeros((4, 7)).tolist()}  # inny rozmiar wag
    assert m.handle(bad)["kind"] == "stop"
