"""Czy BANC widzi cel (prostopadłościan ``target_box`` Osoby 3) w świecie — i z jakiej odległości?

    python scripts/check_beacon_visibility.py [--scales 1 2 4] [--alphas 1.0 0.5] [--distances 3 6 10 15 20]

Dron w zawisie 1 m nad terenem, w odległości d od celu, obrócony tak, żeby cel był pod kątem −60…+60°
(w polu widzenia oczu). Klatki oczu → FlyVis → BANC (ustalony stan) → aktywność 375 pojedynczych DN lotu.
Regresja grzbietowa kąt ~ DN, walidacja „bez jednego świata” (inny teren i bloki w tle). Wynik dla każdej
szerokości, przezroczystości i odległości: trafność strony celu (|kąt| ≥ 15°) i korelacja kąta. Bloki między
dronem a celem schowane (jak w treningu), żeby mierzyć widoczność celu, a nie zasłanianie.

Od wyniku zależy: szerokość i przezroczystość celu oraz odległość, poniżej której yaw bierzemy z BANC
(dalej — z „GPS” celu, sim/sensors.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def ridge_cv(X, y, groups, lam=1.0):
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
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 4.0], help="mnożniki szerokości celu (0.3 m)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5], help="przezroczystość (1 = pełny)")
    ap.add_argument("--distances", type=float, nargs="+", default=[3, 6, 10, 15, 20])
    ap.add_argument("--worlds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--bearings", type=float, nargs="+", default=list(range(-60, 61, 10)))
    ap.add_argument("--color", choices=("scene", "dark-red"), default="dark-red",
                    help="kolor celu: ze sceny (pomarańczowy) albo ciemnoczerwony (kontrast jasności dla FlyVis)")
    ap.add_argument("--save-frames", type=Path, help="zapisz przykładowe klatki oczu (PNG) do katalogu")
    args = ap.parse_args()

    import mujoco

    from banc_control import BancController, Connectome
    from sim.banc_pilot import BEACON_DARK_RED, clear_corridor, ground_z, set_beacon
    from sim.world_env import WorldEnv
    from visual_pipeline import VisionBridge

    t0 = time.perf_counter()
    env = WorldEnv(control="angle", start_noise=False)
    bridge = VisionBridge(fps=30, fisheye=True)
    ctrl = BancController(Connectome.from_banc(), readout="dn")
    dn = slice(6, None)  # cechy: 6 średnich MN + 375 DN — bierzemy DN
    gid = env.model.geom("target_box").id
    base_size, base_rgba = env.model.geom_size[gid].copy(), env.model.geom_rgba[gid].copy()
    combos = [(sc, al) for sc in args.scales for al in args.alphas]
    n = len(combos) * len(args.distances) * len(args.bearings) * len(args.worlds)
    print(f"gotowe po {time.perf_counter() - t0:.0f} s; scen: {n}", flush=True)

    results = {}
    for scale, alpha in combos:
        X, y, dist, world = [], [], [], []
        for w in args.worlds:
            env.reset(options={"world_seed": w})
            env.model.geom_size[gid], env.model.geom_rgba[gid] = base_size, base_rgba  # od bazy, nie kumulatywnie
            set_beacon(env.model, scale, alpha, BEACON_DARK_RED if args.color == "dark-red" else None)
            tgt = env.target.position(env.data)
            for d in args.distances:
                for b in args.bearings:
                    phi = np.deg2rad(37.0 * w)  # kierunek podejścia zależny od świata (inne tło)
                    env.data.qpos[0:2] = tgt[:2] - d * np.array([np.cos(phi), np.sin(phi)])
                    env.data.qpos[2] = 0.0
                    mujoco.mj_forward(env.model, env.data)
                    env.data.qpos[2] = ground_z(env) + 1.0  # 1 m nad terenem
                    yaw = phi + np.deg2rad(b)  # cel pod kątem b w prawo (MuJoCo yaw + = w lewo)
                    env.data.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
                    env.data.qvel[:] = 0
                    mujoco.mj_forward(env.model, env.data)
                    clear_corridor(env)  # bloki między TĄ pozycją drona a celem
                    L, R = env.eyes.render(env.data)
                    if args.save_frames and b in (-30, 30) and w == args.worlds[0]:
                        import cv2

                        args.save_frames.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(args.save_frames / f"s{scale:g}_a{alpha:g}_d{d:g}_b{b:+.0f}.png"),
                                    cv2.cvtColor(np.hstack([L, R]), cv2.COLOR_RGB2BGR))
                    ctrl._settle(bridge.settle(L, R), None, 40)
                    X.append(ctrl.motor_features()[dn])
                    y.append(b)
                    dist.append(d)
                    world.append(w)
        X, y, dist, world = np.array(X), np.array(y, float), np.array(dist), np.array(world)
        X = X[:, X.std(0) > 1e-12]
        pred = ridge_cv(X, y, world)
        side = np.abs(y) >= 15
        print(f"\nszerokość ×{scale:g} ({0.3 * scale:.1f} m), alpha {alpha:g} (1 = pełny):", flush=True)
        for d in args.distances:
            m = dist == d
            acc = np.mean(np.sign(pred[m & side]) == np.sign(y[m & side]))
            r = np.corrcoef(pred[m], y[m])[0, 1]
            results[(scale, alpha, d)] = acc
            print(f"  {d:5.0f} m: trafność strony {acc:4.0%}   korelacja kąta {r:+.2f}", flush=True)
    print("\nwniosek: do jakiej odległości trafność strony celu ≥ 90%:")
    for sc, al in combos:
        ok = [d for d in args.distances if results[(sc, al, d)] >= 0.9]
        print(f"  ×{sc:g}, alpha {al:g}: " + (f"do {max(ok):g} m" if ok else "nigdzie ≥ 90%"))


if __name__ == "__main__":
    main()
