"""Trening lotu do celu w świecie Osoby 3 (``WorldEnv``): DAgger z regresją grzbietową, osobno metryki osi.

    python scripts/train_world.py [--episodes 120] [--yaw-init data/decoders/planB_distributed.npz]
    python scripts/train_distributed.py master --world …   # to samo na kilku GPU (workerzy z --world)

Dekoder ``sim/world_decoder.py``: yaw tylko z BANC (6 grup MN + pojedyncze DN lotu), thrust/roll/pitch z BANC
+ czujników drona (wysokość nad terenem, v_z, prędkość przód/bok, żyroskop). ``WorldEnv(control="angle")``
utrzymuje zadany przechył. BANC się nie zmienia.

Nauczyciel (``sim.banc_pilot.teacher``) widzi prawdziwy stan. DAgger: epizody w partiach po ``--batch``; dron
leci mieszanką (z prawdopodobieństwem beta komendą nauczyciela, beta 1 → 0 w pierwszej połowie), wszystkie
pary (cechy, komenda nauczyciela) idą do wspólnych statystyk, po partii wagi liczone od nowa na całości.
Światy losowe, korytarz do celu bez bloków (``--obstacles clear``) albo z ``--path-blocks`` blokami na trasie
(``--obstacles path``; nauczyciel wtedy je omija, ewaluacja liczy zderzenia), znacznik celu (czerwony
prostopadłościan) ×``--beacon-scale``, przezroczystość ``--beacon-alpha``. Start: yaw z dekodera zawisu
(``--yaw-init``), thrust = zawis, reszta 0. ``--banc-axes thrust``: wysokość tylko z BANC (czujniki = 0).

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
# --obstacles path: światy, w których blok stoi na drodze (nauczyciel bez omijania wpada w niego; z omijaniem
# 43/44 światów 101–150 do celu bez zderzenia)
EVAL_WORLDS_PATH = (101, 103, 113, 115, 130, 135)


def eval_worlds(obstacles: str) -> tuple:
    return EVAL_WORLDS if obstacles == "clear" else EVAL_WORLDS_PATH
TRAIN_SEED_OFFSET = 1000  # światy treningowe ≠ ewaluacyjne


def default_yaw_init() -> Path | None:
    for name in ("planB_distributed.npz", "planB_dn.npz"):  # najlepszy dostępny dekoder zawisu (cechy DN)
        p = ROOT / "data" / "decoders" / name
        if p.exists() and np.load(p)["M"].shape[1] > 7:
            return p
    return None


def setup(yaw_init: Path | None, beacon_scale: float = 1.0, max_time: float = 40.0, beacon_alpha: float | None = None,
          sensors: str = "real", motor_tau: float = 0.04, vision_range: float = float("inf"), beacon_color: str = "dark-red",
          obstacles: str = "clear", path_blocks: int = 2, banc_axes=()):
    """→ (env, pilot, runner, calib): pilot BANC bez wspomagania z nowym WorldDecoder, skalibrowany."""
    from sim.banc_pilot import BancPilot, WorldRunner
    from sim.world_env import WorldEnv

    pilot = BancPilot(None, assist=False, brain=False, beacon_scale=beacon_scale, readout="dn", yaw_init=yaw_init,
                      beacon_alpha=beacon_alpha, vision_range=vision_range, beacon_color=beacon_color,
                      banc_axes=banc_axes)
    env = WorldEnv(control="angle", start_noise=False, max_time=max_time, sensors=sensors, motor_tau=motor_tau)
    env.reset(seed=0)
    calib = pilot.bind(env)
    return env, pilot, WorldRunner(env, pilot, obstacles=obstacles, path_blocks=path_blocks), calib


def add_world_args(ap) -> None:
    """Flagi wspólne z train_distributed.py --world."""
    ap.add_argument("--obstacles", choices=("clear", "path", "keep"), default="clear",
                    help="clear = czysty korytarz, path = bloki na trasie (nauczyciel omija), keep = wszystkie bloki")
    ap.add_argument("--path-blocks", type=int, default=2, help="--obstacles path: ile bloków zostaje na trasie")
    ap.add_argument("--banc-axes", nargs="*", choices=("thrust", "roll", "pitch"), default=[],
                    help="osie tylko z BANC (bez czujników drona), np. --banc-axes thrust")


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
    ap.add_argument("--beacon-scale", type=float, default=1.0, help="rozmiar znacznika celu (1 = 1.2 × 1.2 × 2 m)")
    ap.add_argument("--beacon-alpha", type=float, default=None, help="przezroczystość znacznika (1 = pełny; domyślnie ze sceny)")
    ap.add_argument("--sensors", choices=("real", "ideal"), default="real", help="czujniki drona: z szumem (sim/sensors.py) albo idealne")
    ap.add_argument("--beacon-color", choices=("scene", "dark-red"), default="dark-red",
                    help="kolor celu: ciemnoczerwony (kontrast jasności dla FlyVis) albo ze sceny (pomarańczowy)")
    ap.add_argument("--motor-tau", type=float, default=0.04, help="opóźnienie silników [s] (0 = natychmiast)")
    ap.add_argument("--vision-range", type=float, default=float("inf"),
                    help="m: bliżej celu yaw z BANC, dalej z „GPS” (inf = zawsze BANC); z check_beacon_visibility.py")
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "decoders" / "world.npz")
    ap.add_argument("--init", type=Path, default=None, help="wagi startowe WorldDecoder (.npz z train_world/train_distributed)")
    ap.add_argument("--eval-only", action="store_true", help="tylko ewaluacja wag z --init (np. punkt odniesienia)")
    add_world_args(ap)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    env, pilot, runner, calib = setup(args.yaw_init, args.beacon_scale, args.max_time, args.beacon_alpha,
                                      args.sensors, args.motor_tau, args.vision_range, args.beacon_color,
                                      args.obstacles, args.path_blocks, args.banc_axes)
    dec = pilot.world
    if args.init:
        from sim.world_decoder import WorldDecoder

        dec.from_matrix(WorldDecoder.load(args.init).to_matrix())  # maska banc_axes z bieżących ustawień
    print(f"wagi na start: {args.init or args.yaw_init or 'brak (0)'}", flush=True)
    before = runner.evaluate(eval_worlds(args.obstacles), log=print if args.eval_only else None)
    show_world("przed treningiem", before)
    if args.eval_only:
        js = args.out.with_name(args.out.stem + "_eval.json")
        js.write_text(json.dumps({"args": vars(args), "eval": before}, default=str, indent=1))
        print(f"zapisano {js}")
        return
    best = {"score": (before["reached"], -before["mean_min_dist"]), "M": dec.to_matrix(), "tag": "before"}
    evals = [{"tag": "before", "after_episodes": 0, "reached": before["reached"], "n": before["n"],
              "mean_min_dist": before["mean_min_dist"]}]

    def consider(tag: str, ev: dict, done: int) -> None:
        evals.append({"tag": tag, "after_episodes": done, "reached": ev["reached"], "n": ev["n"],
                      "mean_min_dist": ev["mean_min_dist"]})
        score = (ev["reached"], -ev["mean_min_dist"])
        if score > best["score"]:
            best.update(score=score, M=dec.to_matrix(), tag=tag)
            print(f"  nowe najlepsze wagi ({tag})", flush=True)

    history = []
    for ep in range(args.episodes):
        t = time.perf_counter()
        beta = beta_schedule(ep, args.episodes, "B")
        world = TRAIN_SEED_OFFSET + int(rng.integers(1_000_000))
        m = runner.episode(world, learn=True, beta=beta, rng=rng, start_noise=True)
        history.append({"episode": ep, "beta": beta, "gpu": "lokalnie", **m})
        print(f"ep {ep:3d}: świat {world} → {m['outcome']:<12s} najbliżej {m['min_dist']:4.1f} m  "
              f"wys. {m['mean_height']:.2f} m  zderzenia {m['block_hits']}  beta {beta:.2f}  {axes_line(m)}  "
              f"({time.perf_counter() - t:.0f} s)",
              flush=True)
        if (ep + 1) % args.batch == 0 or ep + 1 == args.episodes:
            dec.fit()
        if args.eval_every and (ep + 1) % args.eval_every == 0 and ep + 1 < args.episodes:
            ev = runner.evaluate(eval_worlds(args.obstacles))
            history[-1]["eval"] = ev
            show_world(f"  po {ep + 1} epizodach", ev)
            consider(f"mid@{ep + 1}", ev, ep + 1)
    after = runner.evaluate(eval_worlds(args.obstacles))
    consider("after", after, args.episodes)
    show_world("przed treningiem", before)
    show_world("po treningu     ", after)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dec.save(args.out, episodes=args.episodes, reached_before=before["reached"], reached_after=after["reached"],
             samples=dec.n)
    np.savez(args.out.with_name(args.out.stem + "_stats.npz"), **dec.stats())  # do przeliczenia wag bez lotów
    from sim.world_decoder import WorldDecoder

    best_path = args.out.with_name(args.out.stem + "_best.npz")
    WorldDecoder.for_matrix(best["M"], banc_only=args.banc_axes).save(best_path, best_from=best["tag"])
    print(f"najlepsze wagi ({best['tag']}) → {best_path}")
    js = args.out.with_suffix(".json")
    js.write_text(json.dumps({"args": vars(args), "calibration": calib, "before": before, "after": after,
                              "evals": evals, "best": best["tag"], "history": history}, default=str, indent=1))
    print(f"zapisano {args.out} ({dec.n} próbek, {time.perf_counter() - t0:.0f} s)")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_training.py"), str(js)], check=False)


if __name__ == "__main__":
    main()
