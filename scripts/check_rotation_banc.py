"""Kierunek ruchu w neuronach BANC przy obrocie drona, bez dekodera Osoby 2.

Skręt w prawo: świat przesuwa się w lewo względem drona, więc prawe oko widzi ruch tył→przód
(BANC T4b/T5b), a lewe oko przód→tył (BANC T4a/T5a). Skręt w lewo odwrotnie. Porównujemy
średnią aktywność typów a i b na wyjściu VisionBridge, osobno dla każdej strony.

    python scripts/check_rotation_banc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visual_pipeline import VisionBridge  # noqa: E402
from visual_pipeline.bridge import MAP_FILE  # noqa: E402
from visual_pipeline.fake_camera import FakeStereoCamera  # noqa: E402

FPS = 30


def rotation_check(bridge: VisionBridge, render) -> bool:
    """``render(yaw)`` → (lewa, prawa) klatka dla kursu drona ``yaw`` (rad, + = w prawo)."""
    side_of = pd.read_csv(MAP_FILE).drop_duplicates("banc_888_id").set_index("banc_888_id").eye
    side = side_of.reindex(bridge.root_ids).to_numpy()
    kind = np.array([t[-1] if t[:2] in ("T4", "T5") and len(t) == 3 else "" for t in bridge.cell_types])
    a_minus_b = {}
    for rate_deg, turn in ((90, "w prawo"), (-90, "w lewo")):
        bridge.settle(*render(0.0))
        yaw, acc = 0.0, []
        for k in range(45):
            yaw += np.deg2rad(rate_deg) / FPS
            act = bridge.step_arrays(*render(yaw))
            if k >= 15:
                acc.append(act)
        act = np.maximum(np.mean(acc, axis=0), 0)
        for eye in ("right", "left"):
            a = act[(side == eye) & (kind == "a")].mean()
            b = act[(side == eye) & (kind == "b")].mean()
            a_minus_b[turn, eye] = a - b
            print(f"skręt {turn}, oko {eye:5s}: T4/T5 a {a:.3f}  b {b:.3f}", flush=True)
    # Skręt w prawo: prawe oko widzi tył→przód (b), lewe przód→tył (a); w lewo odwrotnie.
    # Porównujemy a−b między skrętami, co jest odporne na stałą różnicę poziomów a i b.
    ok = True
    for eye, sign in (("right", -1), ("left", 1)):
        shift = a_minus_b["w prawo", eye] - a_minus_b["w lewo", eye]
        good = np.sign(shift) == sign
        ok &= good
        print(f"oko {eye:5s}: (a−b) skręt w prawo − skręt w lewo = {shift:+.3f} "
              f"(oczekiwany znak {'+' if sign > 0 else '−'}) {'OK' if good else 'ŹLE'}")
    print("WYNIK:", "orientacja ruchu zgodna z anatomią" if ok else "orientacja ruchu NIEZGODNA")
    return bool(ok)


def main() -> None:
    bars = [FakeStereoCamera(np.deg2rad(a)) for a in range(0, 360, 30)]

    def render(yaw):
        frames = [c.render(yaw) for c in bars]
        return np.minimum.reduce([f[0] for f in frames]), np.minimum.reduce([f[1] for f in frames])

    rotation_check(VisionBridge(fps=FPS), render)


if __name__ == "__main__":
    main()
