"""Log mastera treningu (tekst z terminala) → JSON dla ``plot_training.py``.

    python scripts/log_to_json.py master.log [--out data/decoders/world_distributed_log.json]
    python scripts/plot_training.py data/decoders/world_distributed_log.json

Przydaje się, gdy trening przerwano przed zapisem JSON (master zapisuje go dopiero po końcowej ewaluacji).
Czyta linie epizodów (tryb --world i zawis) oraz ewaluacji „przed / po / po N epizodach”.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

WORLD = re.compile(r"ep\s+(\d+)\s+\[([^\]]+)\]:\s+świat\s+(\d+)\s+→\s+(.+?)\s+najbliżej\s+([\d.]+)\s+m\s+beta\s+([\d.]+)\s+(.*?)\s+\((\d+)/(\d+)\)")
HOVER = re.compile(r"ep\s+(\d+)\s+\[([^\]]+)\]:\s+cel\s+([+-]?\d+)°\s+→\s+([\d.]+)°\s+beta\s+([\d.]+)\s+loss\s+([\d.e+-]+)")
AXES = re.compile(r"(thrust|roll|pitch|yaw)\s+([\d.e+-]+)")
EVAL = re.compile(r"(przed treningiem|po treningu|po\s+(\d+)\s+epizodach)\s*:\s+cel\s+(\d+)/(\d+),\s+średnio najbliżej\s+([\d.]+)\s+m\s+\((.*)\)")
EVAL_EP = re.compile(r"(\d+):(CEL|[^()]+?)\((\d+)m\)")


def parse_eval(m) -> dict:
    eps = [{"world": int(w), "reached": o == "CEL", "outcome": "cel" if o == "CEL" else o.strip(), "min_dist": float(d)}
           for w, o, d in EVAL_EP.findall(m.group(6))]
    return {"episodes": eps, "reached": int(m.group(3)), "n": int(m.group(4)), "mean_min_dist": float(m.group(5)),
            "mean_final_deg": float("nan")}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    history, evals, seen = [], [], set()
    before = after = None
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := WORLD.search(line):
            i = int(m.group(1))
            if i in seen:
                continue
            seen.add(i)
            axes = {k: float(v) for k, v in AXES.findall(m.group(7))}
            history.append({"index": i, "gpu": m.group(2), "world": int(m.group(3)), "outcome": m.group(4).strip(),
                            "min_dist": float(m.group(5)), "beta": float(m.group(6)), **({"loss_axes": axes} if axes else {}),
                            "loss": sum(axes.values()) / len(axes) if axes else float("nan")})
        elif m := HOVER.search(line):
            i = int(m.group(1))
            if i not in seen:
                seen.add(i)
                history.append({"index": i, "gpu": m.group(2), "bearing": float(m.group(3)), "final_deg": float(m.group(4)),
                                "beta": float(m.group(5)), "loss": float(m.group(6))})
        elif m := EVAL.search(line):
            ev = parse_eval(m)
            if m.group(1) == "przed treningiem":
                before = before or ev
                tag, done = "before", 0
            elif m.group(1).startswith(("po treningu", "wynik")):  # wynik = sim.banc_pilot (ocena zapisanego modelu)
                after, tag, done = ev, "after", len(seen)
            else:
                tag, done = f"mid@{m.group(2)}", int(m.group(2))
            if not any(e["tag"] == tag for e in evals):
                evals.append({"tag": tag, "after_episodes": done, "reached": ev["reached"], "n": ev["n"],
                              "mean_min_dist": ev["mean_min_dist"]})
    if not history:
        sys.exit("nie znalazłem linii epizodów (ep N [GPU]: …) w logu")
    out = args.out or args.log.with_suffix(".json")
    out.write_text(json.dumps({"source": str(args.log), "before": before, "after": after, "evals": evals,
                               "history": history}, indent=1), encoding="utf-8")
    print(f"{len(history)} epizodów, {len(evals)} ewaluacji → {out}")


if __name__ == "__main__":
    main()
