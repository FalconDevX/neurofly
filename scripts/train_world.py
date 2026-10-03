"""Trening lotu do celu w świecie Osoby 3 (``WorldEnv``): DAgger z regresją grzbietową, osobno metryki osi.

    python scripts/train_world.py [--episodes 120] [--yaw-init data/decoders/planB_distributed.npz]
    python scripts/train_distributed.py master --world …   # to samo na kilku GPU (workerzy z --world)

Dekoder ``sim/world_decoder.py``: yaw tylko z BANC (6 grup MN + pojedyncze DN lotu), thrust/roll/pitch z BANC
+ czujników drona (wysokość nad terenem, v_z, prędkość przód/bok, żyroskop). ``WorldEnv(control="angle")``
utrzymuje zadany przechył. BANC się nie zmienia.

Nauczyciel (``sim.banc_pilot.teacher``) widzi prawdziwy stan. DAgger: epizody w partiach po ``--batch``; dron
leci mieszanką (z prawdopodobieństwem beta komendą nauczyciela, beta 1 → 0 w pierwszej połowie), wszystkie
pary (cechy, komenda nauczyciela) idą do wspólnych statystyk, po partii wagi liczone od nowa na całości.
Światy losowe, korytarz do celu bez bloków, maszt celu ×``--beacon-scale``. Start: yaw z dekodera zawisu
(``--yaw-init``), thrust = zawis, reszta 0.

Ewaluacja (stałe światy ``EVAL_WORLDS``, bez nauczyciela) przed, co ``--eval-every`` i po. Wynik:
``--out`` (.npz WorldDecoder + .json z przebiegiem) i wykresy ``*_wykresy.png`` (``plot_training.py``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EVAL_WORLDS = (101, 102, 103, 104, 105, 106)
TRAIN_SEED_OFFSET = 1000  # światy treningowe ≠ ewaluacyjne


def default_yaw_init() -> Path | None:
    for name in ("planB_distributed.npz", "planB_dn.npz"):  # najlepszy dostępny dekoder zawisu (cechy DN)
        p = ROOT / "data" / "decoders" / name
        if p.exists() and np.load(p)["M"].shape[1] > 7:
            return p
    return None


def setup(yaw_init: Path | None, beacon_scale: float = 4.0, max_time: float = 40.0):
    """→ (env, pilot, runner, calib): pilot BANC bez wspomagania z nowym WorldDecoder, skalibrowany."""
    from sim.banc_pilot import BancPilot, WorldRunner
    from sim.world_env import WorldEnv

    pilot = BancPilot(None, assist=False, brain=False, beacon_scale=beacon_scale, readout="dn", yaw_init=yaw_init)
    env = WorldEnv(control="angle", start_noise=False, max_time=max_time)
    env.reset(seed=0)
    calib = pilot.bind(env)
    return env, pilot, WorldRunner(env, pilot), calib


def axes_line(m: dict) -> str:
    la = m.get("loss_axes")
    return "  ".join(f"{k} {v:.3f}" for k, v in la.items()) if la else f"loss {m['loss']:.4f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from sim.banc_pilot import show_world
    from train_decoder import beta_schedule

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4, help="epizodów między dopasowaniami wag")
    ap.add_argument("--yaw-init", type=Path, default=default_yaw_init())
    ap.add_argument("--beacon-scale", type=float, default=4.0)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "decoders" / "world.npz")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    env, pilot, runner, calib = setup(args.yaw_init, args.beacon_scale, args.max_time)
    dec = pilot.world
    print(f"yaw na start: {args.yaw_init or 'brak (0)'}", flush=True)
    before = runner.evaluate(EVAL_WORLDS)
    show_world("przed treningiem", before)
    history = []
    for ep in range(args.episodes):
        t = time.perf_counter()
        beta = beta_schedule(ep, args.episodes, "B")
        world = TRAIN_SEED_OFFSET + int(rng.integers(1_000_000))
        m = runner.episode(world, learn=True, beta=beta, rng=rng, start_noise=True)
        history.append({"episode": ep, "beta": beta, "gpu": "lokalnie", **m})
        print(f"ep {ep:3d}: świat {world} → {m['outcome']:<12s} najbliżej {m['min_dist']:4.1f} m  "
              f"wys. {m['mean_height']:.2f} m  beta {beta:.2f}  {axes_line(m)}  ({time.perf_counter() - t:.0f} s)",
              flush=True)
        if (ep + 1) % args.batch == 0 or ep + 1 == args.episodes:
            dec.fit()
        if args.eval_every and (ep + 1) % args.eval_every == 0 and ep + 1 < args.episodes:
            ev = runner.evaluate(EVAL_WORLDS)
            history[-1]["eval"] = ev
            show_world(f"  po {ep + 1} epizodach", ev)
    after = runner.evaluate(EVAL_WORLDS)
    show_world("przed treningiem", before)
    show_world("po treningu     ", after)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dec.save(args.out, episodes=args.episodes, reached_before=before["reached"], reached_after=after["reached"],
             samples=dec.n)
    js = args.out.with_suffix(".json")
    js.write_text(json.dumps({"args": vars(args), "calibration": calib, "before": before, "after": after,
                              "history": history}, default=str, indent=1))
    print(f"zapisano {args.out} ({dec.n} próbek, {time.perf_counter() - t0:.0f} s)")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_training.py"), str(js)], check=False)


if __name__ == "__main__":
    main()
