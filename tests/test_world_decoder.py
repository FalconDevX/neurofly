"""WorldDecoder (DAgger + regresja grzbietowa) i jego obsługa w masterze treningu rozproszonego — bez GPU."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sim.world_decoder import SENSORS, WorldDecoder  # noqa: E402


def _data(n_banc, n=600, seed=0):
    rng = np.random.default_rng(seed)
    w_yaw = rng.normal(size=n_banc) * 0.3
    W_s = rng.normal(size=(3, len(SENSORS))) * 0.5
    for _ in range(n):
        x = np.r_[rng.normal(size=n_banc - 1), 1.0]  # ostatnia = wyraz wolny
        s = rng.normal(size=len(SENSORS))
        t = SimpleNamespace(thrust=0.5 + W_s[0] @ s, roll=W_s[1] @ s, pitch=W_s[2] @ s, yaw=float(w_yaw @ x))
        yield x, s, t


def test_ridge_recovers_teacher_and_yaw_ignores_sensors():
    dec = WorldDecoder(12, lam=1e-4)
    *train, (x, s, t) = list(_data(12, n=601))  # ostatnia próbka (ten sam nauczyciel) do sprawdzenia
    for xi, si, ti in train:
        dec.add(xi, si, ti)
    dec.fit()
    M = dec.to_matrix()
    np.testing.assert_array_equal(M[3, 12:], 0.0)  # yaw: czujniki drona mają wagę 0 z konstrukcji
    raw = dec.decode_raw(x, s)
    np.testing.assert_allclose(raw, [t.thrust, t.roll, t.pitch, t.yaw], atol=0.05)
    other = WorldDecoder.for_matrix(M)
    np.testing.assert_allclose(other.decode_raw(x, s), raw)


def test_stats_merge_equals_single_pass():
    a, b, both = WorldDecoder(8), WorldDecoder(8), WorldDecoder(8)
    for i, (x, s, t) in enumerate(_data(8, n=200)):
        (a if i % 2 else b).add(x, s, t)
        both.add(x, s, t)
    a.merge(b.stats(as_lists=True))  # jak master: statystyki od dwóch workerów przez JSON
    a.fit(), both.fit()
    np.testing.assert_allclose(a.to_matrix(), both.to_matrix(), rtol=1e-3, atol=1e-4)


def test_master_world_mode_fits_from_worker_stats(tmp_path):
    import argparse

    from train_distributed import Master

    a = argparse.Namespace(plan="B", episodes=2, batch=2, max_bearing=90.0, init=None, seed=0,
                           out=tmp_path / "w.npz", port=0, world=True, workers=1)
    m = Master(a)
    dec = WorldDecoder(8)
    m.handle({"type": "ping", "name": "A", "world": True})
    task = m.handle({"type": "hello", "name": "A", "M": dec.to_matrix().tolist(), "init": True,
                     "calib": {"yaw_axis_sign": -1.0, "hover_thrust": 0.5, "pitch_trim": 0.0}})
    assert task["kind"] == "eval"
    for x, s, t in _data(8, n=100):
        dec.add(x, s, t)
    ev = {"episodes": [], "reached": 0, "n": 0, "mean_min_dist": 1.0, "mean_final_deg": 1.0}
    task = m.handle({"type": "result", "kind": "eval", "name": "A", "tag": "before", "eval": ev})
    metrics = [{**e, "world": 1, "outcome": "cel", "min_dist": 0.5, "loss": 0.1} for e in task["episodes"]]
    task = m.handle({"type": "result", "kind": "train", "name": "A", "stats": dec.stats(as_lists=True),
                     "metrics": metrics})
    assert m.wdec is not None and m.wdec.n == 100
    assert np.abs(np.array(m.M)).sum() > 0 and task["kind"] == "eval"
    m.handle({"type": "result", "kind": "eval", "name": "A", "tag": "after", "eval": ev})
    assert WorldDecoder.is_world_file(tmp_path / "w.npz")


def test_master_mid_validation_keeps_best_weights(tmp_path):
    import argparse

    from train_distributed import Master

    a = argparse.Namespace(plan="B", episodes=8, batch=2, max_bearing=90.0, init=None, seed=0,
                           out=tmp_path / "w.npz", port=0, world=True, workers=1, eval_every=4)
    m = Master(a)
    dec = WorldDecoder(8)
    m.handle({"type": "ping", "name": "A", "world": True})
    m.handle({"type": "hello", "name": "A", "M": dec.to_matrix().tolist(), "init": True,
              "calib": {"yaw_axis_sign": -1.0, "hover_thrust": 0.5, "pitch_trim": 0.0}})
    ev = lambda r, d: {"episodes": [{"world": 1, "reached": False, "outcome": "x", "min_dist": d}],  # noqa: E731
                       "reached": r, "n": 6, "mean_min_dist": d, "mean_final_deg": 0.0}
    task = m.handle({"type": "result", "kind": "eval", "name": "A", "tag": "before", "eval": ev(0, 18.0)})
    tags = []
    for _ in range(20):
        if task["kind"] == "train":
            st = WorldDecoder(8)
            for x, s, t in _data(8, n=20):
                st.add(x, s, t)
            metrics = [{**e, "world": 1, "outcome": "cel", "min_dist": 0.5, "loss": 0.1} for e in task["episodes"]]
            task = m.handle({"type": "result", "kind": "train", "name": "A", "stats": st.stats(as_lists=True),
                             "metrics": metrics})
        elif task["kind"] == "eval":
            tags.append(task["tag"])
            good = task["tag"] == "mid@4"  # najlepsza walidacja w środku, końcowa gorsza
            task = m.handle({"type": "result", "kind": "eval", "name": "A", "tag": task["tag"],
                             "eval": ev(4 if good else 1, 3.0 if good else 9.0)})
        else:
            break
    assert "mid@4" in tags and tags[-1] == "after"
    assert m.best["tag"] == "mid@4"
    best = WorldDecoder.load(tmp_path / "w_best.npz")
    assert best.to_matrix().shape == m.M.shape
