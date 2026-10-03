"""Kontrola: czy informację o celu niesie struktura BANC, czy wystarczy dowolny graf + dekoder?

    python scripts/connectome_controls.py [--seeds 3] [--steps 40]        # środowisko .venv312, GPU

Ten sam test co ``check_side_decoding.py`` (cel pod kątem −90…+90°, 3 odległości, FlyVis → BANC,
regresja grzbietowa kąt ~ aktywność DN lotu, walidacja „bez jednej odległości”), ale powtórzony na:

* prawdziwym BANC v888,
* BANC z przetasowanymi połączeniami (te same neurony, stopnie, znaki NT i wagi — losowe cele krawędzi),
* BANC z przetasowanymi znakami NT,
* BANC z lezją neuronów projekcyjnych wzroku (VPN, łącznik płat wzrokowy → mózg centralny),
* samym wejściu FlyVis (bez sieci) — górna granica informacji, którą sieć może przenieść.

Sceny FlyVis liczone raz, wszystkie warianty dostają identyczne wejście. Odczyt i dekoder bez zmian,
więc różnica wynika wyłącznie z połączeń. Wynik: tabela + ``data/controls/controls.{json,png}``.
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

BEARINGS = np.arange(-90, 91, 5)  # jak w check_side_decoding.py
DISTANCES = (3.0, 4.0, 6.0)
DN = ("dn_flight_power_L", "dn_flight_power_R", "dn_flight_steering_L", "dn_flight_steering_R")


def ridge_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Jak w ``check_side_decoding``, w postaci dualnej (działa też dla ~20k cech wejścia FlyVis)."""
    pred = np.zeros_like(y)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        alpha = np.linalg.solve(A @ A.T + lam * len(A) * np.eye(len(A)), y[tr] - ym)
        pred[te] = B @ (A.T @ alpha) + ym
    return pred


def score(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    X = X[:, X.std(0) > 1e-12]
    side = np.abs(y) >= 15
    if X.shape[1] == 0:  # sygnał nie dochodzi do odczytu
        return {"features": 0, "corr": 0.0, "side_acc": 0.5, "abs_err_deg": float(np.abs(y).mean())}
    pred = ridge_cv(X, y, groups)
    return {"features": int(X.shape[1]), "corr": float(np.corrcoef(pred, y)[0, 1]),
            "side_acc": float(np.mean(np.sign(pred[side]) == np.sign(y[side]))),
            "abs_err_deg": float(np.abs(pred - y).mean())}


def lateralization(X: np.ndarray, y: np.ndarray, is_right: np.ndarray, n_perm: int = 2000) -> dict:
    """Bez uczenia: czy DN z prawej i lewej strony ciała odpowiadają na kąt celu przeciwnie?

    r_i = korelacja aktywności neuronu i z kątem; indeks = średnie r_i (DN prawe) − średnie r_i (DN lewe).
    Strona L/P to anatomia z BANC, nie nic dopasowanego. p z testu permutacyjnego etykiet L/P."""
    ok = X.std(0) > 1e-12
    if ok.sum() < 4:
        return {"lat_index": 0.0, "lat_p": 1.0, "lr_corr": 0.0}
    Xc = (X[:, ok] - X[:, ok].mean(0)) / X[:, ok].std(0)
    r = Xc.T @ ((y - y.mean()) / y.std()) / len(y)
    right = is_right[ok]
    idx = r[right].mean() - r[~right].mean()
    rng = np.random.default_rng(0)
    null = np.array([(lambda p: r[p].mean() - r[~p].mean())(rng.permutation(right)) for _ in range(n_perm)])
    lr = X[:, is_right].mean(1) - X[:, ~is_right].mean(1)  # średnia DN P − L, zero parametrów
    return {"lat_index": float(idx), "lat_p": float((np.abs(null) >= abs(idx)).mean() + 1 / n_perm),
            "lr_corr": float(np.corrcoef(lr, y)[0, 1]) if lr.std() > 0 else 0.0}


def dn_features(connectome, scenes: list, steps: int) -> np.ndarray:
    from banc_control import BancController

    ctrl = BancController(connectome)
    idx = np.concatenate([connectome.group_indices(g) for g in DN])
    X = []
    for s in scenes:
        ctrl._settle(s, None, steps)
        X.append(ctrl.dyn.rates_at(idx))
    del ctrl
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    return np.array(X, float)


def plot(results: list[dict], path: Path) -> None:
    """Dwa panele: trafność uczonego dekodera i lateralizacja bez uczenia (punkty = seedy)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["BANC v888", "przetasowane połączenia", "przetasowane znaki NT", "lezja VPN (wzrok → mózg)",
             "samo wejście FlyVis"]
    names = [n for n in order if any(r["variant"] == n for r in results)]
    y = np.arange(len(names))[::-1]
    blue, gray, ink = "#2a78d6", "#8a939d", "#222"
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.6), dpi=200, sharey=True,
                             gridspec_kw={"wspace": 0.08})
    panels = [("side_acc", 100, "trafność strony celu, uczony dekoder [%]", (50, 102), 50, "losowo"),
              ("lat_index", 1, "lateralizacja DN lotu, bez uczenia (P − L)", None, 0, "brak")]
    for ax, (key, k, label, xlim, ref, ref_txt) in zip(axes, panels):
        for yi, n in zip(y, names):
            v = [r[key] * k for r in results if r["variant"] == n and key in r]
            if not v:
                ax.text(ref, yi, " nie dotyczy", va="center", fontsize=8, color=gray)
                continue
            c = blue if n == "BANC v888" else gray
            ax.scatter(v, [yi] * len(v), s=46, color=c, zorder=3, edgecolor="white", linewidth=1)
            m = np.mean(v)
            ax.text(m, yi + 0.28, f"{m:.0f}%" if k == 100 else f"{m:+.3f}", ha="center", fontsize=8, color=ink)
        ax.axvline(ref, color="#666", ls="--", lw=0.8)
        ax.text(ref, y[-1] - 0.55, ref_txt, ha="center", fontsize=7, color="#666")
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel(label, fontsize=8.5)
        ax.grid(axis="x", color="#e8e8e8", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        for sp_ in ("top", "right", "left"):
            ax.spines[sp_].set_visible(False)
    axes[0].set_yticks(y, names, fontsize=9)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylim(y[-1] - 0.8, y[0] + 0.6)
    fig.suptitle("Czy kierunek do celu niesie struktura BANC? Te same wejście FlyVis i odczyt (375 DN lotu)",
                 x=0.01, ha="left", fontsize=10.5, weight="bold")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.86, bottom=0.16, wspace=0.08)
    fig.savefig(path)
    print(f"wykres: {path}")


def replot(out: Path) -> None:
    plot(json.loads((out / "controls.json").read_text(encoding="utf-8")), out / "controls.png")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3, help="powtórzenia wariantów losowych")
    ap.add_argument("--steps", type=int, default=40, help="klatki BANC na scenę")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "controls")
    ap.add_argument("--replot", action="store_true", help="tylko wykres z zapisanego controls.json")
    args = ap.parse_args()
    if args.replot:
        return replot(args.out)

    from banc_control import Connectome
    from banc_control.controls import lesion, shuffle_signs, shuffle_targets
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
    print(f"{len(scenes)} scen FlyVis po {time.perf_counter() - t0:.0f} s", flush=True)

    banc = Connectome.from_banc()
    variants = [("BANC v888", 0, lambda s: banc)]
    variants += [("przetasowane połączenia", s, lambda s: shuffle_targets(banc, s)) for s in range(args.seeds)]
    variants += [("przetasowane znaki NT", s, lambda s: shuffle_signs(banc, s)) for s in range(args.seeds)]
    variants += [("lezja VPN (wzrok → mózg)", 0, lambda s: lesion(banc, ["visual"]))]

    is_right = np.concatenate([np.full(len(banc.group_indices(g)), g.endswith("_R")) for g in DN])
    results, feats = [], {}
    # Wejście bez sieci: aktywność FlyVis zmapowana na neurony BANC (~22k cech).
    X_in = np.array([np.asarray(s.activity, float) for s in scenes])
    results.append({"variant": "samo wejście FlyVis", "seed": 0, **score(X_in, y, groups)})
    for name, seed, make in variants:
        t = time.perf_counter()
        X = dn_features(make(seed), scenes, args.steps)
        feats[f"{name}|{seed}"] = X
        r = {"variant": name, "seed": seed, **score(X, y, groups), **lateralization(X, y, is_right)}
        results.append(r)
        print(f"  {name:28s} seed {seed}  cech {r['features']:4d}  korelacja {r['corr']:+.2f}  "
              f"strona {r['side_acc']:.0%}  błąd kąta {r['abs_err_deg']:.1f}°  |  lateralizacja "
              f"{r['lat_index']:+.3f} (p {r['lat_p']:.3f})  DN P−L ~ kąt {r['lr_corr']:+.2f}  "
              f"({time.perf_counter() - t:.0f} s)", flush=True)

    print("\nPodsumowanie (średnia ± odch. std po seedach):")
    for name in dict.fromkeys(r["variant"] for r in results):
        rs = [r for r in results if r["variant"] == name]
        a, c, e = (np.array([r[k] for r in rs]) for k in ("side_acc", "corr", "abs_err_deg"))
        lat = np.array([r.get("lat_index", np.nan) for r in rs])
        print(f"  {name:28s} strona {a.mean():.0%} ± {a.std():.0%}   korelacja {c.mean():+.2f}   "
              f"błąd kąta {e.mean():.1f}°   lateralizacja {np.nanmean(lat):+.3f} ± {np.nanstd(lat):.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    plot(results, args.out / "controls.png")
    np.savez_compressed(args.out / "dn_features.npz", y=y, groups=groups, is_right=is_right, **feats)
    (args.out / "controls.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
