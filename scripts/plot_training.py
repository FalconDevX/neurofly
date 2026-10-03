"""Wykresy przebiegu treningu dekodera z pliku JSON (train_decoder / train_world / train_distributed).

    python scripts/plot_training.py data/decoders/planB_distributed.json      # → obok: *_wykresy.png
    python scripts/plot_training.py data/decoders/world_distributed.json --out wykresy.png

Panele (każdy z jedną osią Y):
  - zawis (DroneEnv): końcowy |kąt| do celu w epizodzie (kropki = epizody, kolor = GPU; linia = mediana krocząca),
  - świat (WorldEnv): najbliższa odległość do celu i udział epizodów zakończonych celem (kroczący),
  - strata dekodera względem nauczyciela (średnia krocząca; przy --world osobno dla każdej osi),
  - udział nauczyciela w sterowaniu (beta),
  - ewaluacja przed / po treningu na stałych zadaniach.
Styl: ciemne tło zinc jak eksplorator; kolory GPU z walidowanej palety (blue, orange, aqua — dark mode).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, PANEL, GRID, INK, MUTED = "#09090b", "#09090b", "#27272a", "#fafafa", "#a1a1aa"
SERIES = ["#3987e5", "#d95926", "#199e70"]  # kategoryczne, kolejność stała (walidowane na #09090b)
AXIS_COLORS = {"yaw": "#3987e5", "thrust": "#d95926", "pitch": "#199e70", "roll": "#c98500"}
BEFORE = "#52525b"  # „przed” = szarość, „po” = akcent


def rolling(y: np.ndarray, w: int, fn=np.nanmean) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        out[i] = fn(y[max(0, i - w + 1):i + 1])
    return out


def style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, loc="left", fontsize=11, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def legend(ax) -> None:
    # legenda obok wykresu (po prawej), żeby nie zasłaniała danych ani tytułu
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for h in leg.legend_handles:
        h.set_alpha(1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--window", type=int, default=0, help="okno średniej kroczącej (domyślnie ~5% epizodów)")
    args = ap.parse_args()

    d = json.loads(args.json.read_text(encoding="utf-8"))
    hist = d["history"]
    n = len(hist)
    w = args.window or max(5, n // 20)
    ep = np.array([h.get("index", h.get("episode", i)) for i, h in enumerate(hist)])
    order = np.argsort(ep)
    hist, ep = [hist[i] for i in order], ep[order]
    gpus = list(dict.fromkeys(h.get("gpu") or h.get("worker") or "GPU" for h in hist))
    gpu = np.array([h.get("gpu") or h.get("worker") or "GPU" for h in hist])
    world = "outcome" in hist[0]

    panels = ["main", "loss", "beta", "eval"] if not world else ["dist", "rate", "loss", "beta", "eval"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.6 * len(panels)), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.55})
    ax = dict(zip(panels, axes))
    counts = ", ".join(f"{g}: {int((gpu == g).sum())}" for g in gpus)
    fig.suptitle(f"{args.json.name} — {n} epizodów ({counts})", color=INK, x=0.06, ha="left", fontsize=13)

    def per_gpu_dots(a, y, label_suffix=""):
        for k, g in enumerate(gpus):
            m = gpu == g
            a.scatter(ep[m], y[m], s=9, color=SERIES[k % len(SERIES)], alpha=0.55, linewidths=0,
                      label=f"{g}{label_suffix}")

    if not world:
        y = np.array([h["final_deg"] for h in hist], float)
        per_gpu_dots(ax["main"], y)
        ax["main"].plot(ep, rolling(y, w, np.nanmedian), color=INK, linewidth=2, label=f"mediana ({w} ep.)")
        style(ax["main"], "Końcowy |kąt| do celu w epizodzie treningowym", "stopnie")
        legend(ax["main"])
    else:
        y = np.array([h["min_dist"] for h in hist], float)
        per_gpu_dots(ax["dist"], y)
        ax["dist"].plot(ep, rolling(y, w, np.nanmedian), color=INK, linewidth=2, label=f"mediana ({w} ep.)")
        style(ax["dist"], "Najbliżej celu w epizodzie", "m")
        legend(ax["dist"])
        hit = np.array([h["outcome"] == "cel" for h in hist], float)
        crash = np.array([str(h["outcome"]).startswith(("wywrotka", "poza")) for h in hist], float)
        ax["rate"].plot(ep, 100 * rolling(hit, w), color=SERIES[0], linewidth=2, label="cel")
        ax["rate"].plot(ep, 100 * rolling(crash, w), color=SERIES[1], linewidth=2, label="wywrotka / poza planszą")
        evs = d.get("evals") or []
        if evs:  # walidacje na stałych światach (przed / w trakcie / po) — bez nauczyciela
            xe = [e["after_episodes"] for e in evs]
            ye = [100 * e["reached"] / max(e["n"], 1) for e in evs]
            ax["rate"].plot(xe, ye, color=INK, linewidth=1, linestyle=":", marker="o", markersize=7,
                            markerfacecolor=SURFACE, markeredgewidth=2, label="walidacja: cel (6 światów)")
        ax["rate"].set_ylim(0, 100)
        style(ax["rate"], f"Wynik epizodu (udział w oknie {w} ep.)", "%")
        legend(ax["rate"])

    if world and "loss_axes" in hist[0]:
        for name, col in AXIS_COLORS.items():
            y = np.array([h["loss_axes"].get(name, np.nan) for h in hist], float)
            ax["loss"].plot(ep, rolling(y, w), color=col, linewidth=2, label=name)
        legend(ax["loss"])
        style(ax["loss"], f"Błąd dekodera względem nauczyciela, osobno dla osi (średnia {w} ep.)", "MSE")
    else:
        y = np.array([h.get("loss", np.nan) for h in hist], float)
        ax["loss"].scatter(ep, y, s=6, color=MUTED, alpha=0.35, linewidths=0)
        ax["loss"].plot(ep, rolling(y, w), color=SERIES[0], linewidth=2)
        style(ax["loss"], f"Błąd dekodera względem nauczyciela (średnia {w} ep.)", "MSE")
    ax["loss"].set_yscale("log")

    beta = np.array([h.get("beta", np.nan) for h in hist], float)
    ax["beta"].plot(ep, beta, color=MUTED, linewidth=2)
    ax["beta"].set_ylim(0, 1.05)
    style(ax["beta"], "Udział nauczyciela w sterowaniu (beta) — od 0 dron leci sam", "beta")
    ax["beta"].set_xlabel("epizod", color=MUTED, fontsize=9)

    b, a = d.get("before"), d.get("after")
    e = ax["eval"]
    if b and a and "episodes" in b:  # świat: najbliżej celu w każdym świecie ewaluacyjnym
        names = [str(x["world"]) for x in b["episodes"]]
        vb, va = [x["min_dist"] for x in b["episodes"]], [x["min_dist"] for x in a["episodes"]]
        unit, title = "m", f"Ewaluacja: najbliżej celu (cel przed {b['reached']}/{b['n']}, po {a['reached']}/{a['n']})"
    elif b and a:
        names = [k for k in b if k[:1] in "+-"]
        vb, va = [b[k]["final_deg"] for k in names], [a[k]["final_deg"] for k in names]
        names = [f"cel {k}°" for k in names]
        unit, title = "stopnie", (f"Ewaluacja: końcowy |kąt| (średnio {b['mean_final_deg']:.1f}° → "
                                  f"{a['mean_final_deg']:.1f}°)")
    else:
        names = []
    if names:
        x = np.arange(len(names))
        bars_b = e.bar(x - 0.2, vb, 0.38, color=BEFORE, label="przed treningiem")
        bars_a = e.bar(x + 0.2, va, 0.38, color=SERIES[0], label="po treningu")
        for bars in (bars_b, bars_a):
            e.bar_label(bars, fmt="%.1f", color=INK, fontsize=8, padding=2)
        e.set_xticks(x, names)
        style(e, title, unit)
        legend(e)
    else:
        e.axis("off")

    out = args.out or args.json.with_name(args.json.stem + "_wykresy.png")
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print(f"zapisano {out}")


if __name__ == "__main__":
    main()
