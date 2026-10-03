"""Czy z aktywności BANC da się odczytać, z której strony jest cel? (statyczne sceny z DroneEnv)

    python scripts/check_side_decoding.py [--gains 1 30 100]          # środowisko .venv312

Dla celu pod kątami −90…+90° (co 5°) i w kilku odległościach: FlyVis → BANC (``_settle``), potem
regresja grzbietowa kąt ~ aktywność dla różnych odczytów (6 średnich MN, MN pojedynczo, DN lotu
pojedynczo, neurony projekcyjne wzroku). Walidacja: odległości osobno (trenuj na dwóch, testuj na
trzeciej), więc model nie może zapamiętać konkretnej klatki. Wynik: korelacja przewidzianego kąta
z prawdziwym i trafność strony (znak) dla |kąt| ≥ 15°.

``visual_gain`` > 1 to nasze założenie (siła wejścia wzrokowego), nie wynik BANC.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BEARINGS = np.arange(-90, 91, 5)
DISTANCES = (3.0, 4.0, 6.0)


def readouts(ctrl) -> dict[str, np.ndarray]:
    c = ctrl.c
    idx = lambda *groups: np.concatenate([c.group_indices(g) for g in groups])  # noqa: E731
    return {
        "MN 6 średnich": None,  # specjalnie: motor_features()
        "MN pojedynczo": idx(*[f"{g}_{s}" for g in ("wing_power", "wing_steering", "wing_tension") for s in "LR"]),
        "DN lotu pojedynczo": idx("dn_flight_power_L", "dn_flight_power_R",
                                  "dn_flight_steering_L", "dn_flight_steering_R"),
        "wzrok projekcyjne": idx("visual_L", "visual_R"),
    }


def ridge_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Przewidywania dla każdej próbki z modelu uczonego bez jej grupy (odległości)."""
    pred = np.zeros_like(y)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        A = (X[tr] - mu) / sd
        w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
        pred[te] = (X[te] - mu) / sd @ w + y[tr].mean()
    return pred


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains", type=float, nargs="+", default=[1.0, 30.0, 100.0])
    ap.add_argument("--steps", type=int, default=40, help="klatki BANC na scenę")
    args = ap.parse_args()

    from banc_control import BancController, Connectome
    from sim.env import DroneEnv
    from visual_pipeline.bridge import VisionBridge

    t0 = time.perf_counter()
    env, bridge = DroneEnv(), VisionBridge(fps=30, fisheye=True)
    scenes, y, groups = [], [], []
    for gi, dist in enumerate(DISTANCES):
        env.beacon_distance = dist
        for b in BEARINGS:
            scenes.append(bridge.settle(*env.calibration_render(np.deg2rad(b))))
            y.append(b)
            groups.append(gi)
    y, groups = np.array(y, float), np.array(groups)
    ctrl = BancController(Connectome.from_banc())
    ro = readouts(ctrl)
    print(f"{len(scenes)} scen, FlyVis + BANC gotowe po {time.perf_counter() - t0:.0f} s; "
          f"neurony: " + ", ".join(f"{k} {1 if v is None else len(v)}" for k, v in ro.items() if v is not None),
          flush=True)

    side = np.abs(y) >= 15
    for gain in args.gains:
        ctrl.visual_gain = gain
        feats = {k: [] for k in ro}
        for s in scenes:
            ctrl._settle(s, None, args.steps)
            for k, i in ro.items():
                feats[k].append(ctrl.motor_features() if i is None else ctrl.dyn.rates_at(i))
        print(f"\nvisual_gain {gain:g}:")
        for k, f in feats.items():
            X = np.array(f, float)
            X = X[:, X.std(0) > 1e-12]
            pred = ridge_cv(X, y, groups)
            r = np.corrcoef(pred, y)[0, 1]
            acc = np.mean(np.sign(pred[side]) == np.sign(y[side]))
            print(f"  {k:22s} cech {X.shape[1]:5d}   korelacja kąta {r:+.2f}   trafność strony {acc:.0%}", flush=True)


if __name__ == "__main__":
    main()
