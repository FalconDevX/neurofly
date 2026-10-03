"""Trening dekodera BANC do lotu do celu w świecie Osoby 3 (``WorldEnv``), Plan B z nauczycielem.

    python scripts/train_world.py [--episodes 120] [--init data/decoders/planB_dn.npz] [--out data/decoders/world.npz]
    python scripts/train_distributed.py master --world …   # to samo na kilku GPU (workerzy z --world)

BANC steruje thrust, roll, pitch i yaw (``sim/banc_pilot.py``, ``assist=False``), ``WorldEnv(control="angle")``
tylko utrzymuje zadany przechył. Uczy się wyłącznie liniowy dekoder (z wyrazem wolnym) na 6 grupach MN
+ pojedynczych DN lotu; BANC się nie zmienia. Nauczyciel (``teacher``) widzi prawdziwy stan: kąt do celu,
wysokość nad terenem, prędkość. DAgger: udział nauczyciela w sterowaniu maleje od 1 do 0 w pierwszej
połowie treningu. Wagi stałe w trakcie epizodu, krok po epizodzie. Światy losowe, korytarz do celu bez
drzew (``clear_corridor``), maszt celu pogrubiony ``--beacon-scale`` (domyślnie ×4, ~30 cm jak w treningu
na ``DroneEnv``). Start: wiersz yaw z ``--init`` (dekoder z ``train_decoder.py``), reszta od zera.

Ewaluacja (stałe światy ``EVAL_WORLDS``, bez nauczyciela): ile razy dron doleciał do celu.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_WORLDS = (101, 102, 103, 104, 105, 106)
TRAIN_SEED_OFFSET = 1000  # światy treningowe ≠ ewaluacyjne


def beta_schedule(episode: int, total: int) -> float:
    return max(0.0, 1.0 - episode / max(1, total // 2))


def setup(init: Path | None, lr: float = 0.5, beacon_scale: float = 4.0, max_time: float = 40.0):
    """→ (env, pilot, runner, calib): pilot BANC bez wspomagania, skalibrowany, z wagami startowymi."""
    from sim.banc_pilot import BancPilot, WorldRunner
    from sim.world_env import WorldEnv

    pilot = BancPilot(init, assist=False, brain=False, lr=lr, beacon_scale=beacon_scale, readout="dn")
    env = WorldEnv(control="angle", start_noise=False, max_time=max_time)
    env.reset(seed=0)
    calib = pilot.bind(env)
    M = pilot.ctrl.decoder.M
    if init is not None and np.load(init)["M"].shape[1] < M.shape[1]:
        M[[0, 1, 2]] = 0.0  # z dekodera zawisu bierzemy tylko yaw; thrust/roll/pitch uczone od zera
    return env, pilot, WorldRunner(env, pilot), calib


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from sim.banc_pilot import show_world

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--init", type=Path, default=ROOT / "data" / "decoders" / "planB_dn.npz")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--beacon-scale", type=float, default=4.0)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "decoders" / "world.npz")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    env, pilot, runner, calib = setup(args.init if args.init and args.init.exists() else None,
                                      args.lr, args.beacon_scale, args.max_time)
    dec = pilot.ctrl.decoder
    before = runner.evaluate(EVAL_WORLDS)
    show_world("przed treningiem", before)
    history = []
    for ep in range(args.episodes):
        t = time.perf_counter()
        beta = beta_schedule(ep, args.episodes)
        world = TRAIN_SEED_OFFSET + int(rng.integers(1_000_000))
        m = runner.episode(world, learn=True, beta=beta, rng=rng, start_noise=True)
        history.append({"episode": ep, "beta": beta, **m})
        print(f"ep {ep:3d}: świat {world} → {m['outcome']:<12s} najbliżej {m['min_dist']:4.1f} m  "
              f"wys. {m['mean_height']:.2f} m  beta {beta:.2f}  loss {m['loss']:.4f}  ({time.perf_counter() - t:.0f} s)",
              flush=True)
        if args.eval_every and (ep + 1) % args.eval_every == 0 and ep + 1 < args.episodes:
            ev = runner.evaluate(EVAL_WORLDS)
            history[-1]["eval"] = ev
            show_world(f"  po {ep + 1} epizodach", ev)
    after = runner.evaluate(EVAL_WORLDS)
    show_world("przed treningiem", before)
    show_world("po treningu     ", after)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dec.save(args.out, mode="world", episodes=args.episodes, reached_before=before["reached"], reached_after=after["reached"])
    args.out.with_suffix(".json").write_text(json.dumps(
        {"args": vars(args), "calibration": calib, "before": before, "after": after, "history": history},
        default=str, indent=1))
    print(f"zapisano {args.out} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
