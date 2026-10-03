"""Kierunki preferowane T4/T5 w FlyVis względem obrazu z kamery (prawe oko).

W BANC podtypy są anatomiczne: a = przód→tył, b = tył→przód, c = w górę, d = w dół.
Dla naturalnej prawej kamery (przód po lewej stronie kadru, góra na górze) oczekujemy
w kadrze: a → 0° (w prawo), b → 180°, c → 90°, d → 270°. Skrypt mierzy faktyczne kierunki
kratką sinusoidalną w 12 kierunkach i dopasowuje przekształcenie kadru (obrót / odbicie).
Przy poprawnym ``RetinaMapper`` najlepsze jest „obrót 0°" bez odbicia.

Wynik (FlyVis flow/0000/000): a, b, c oraz T5d zgodne w granicach ~30°; T4d tego modelu
preferuje ruch w górę (słaba selektywność 0,51), więc średni błąd ~35° wynika głównie z T4d.

    python scripts/check_motion_directions.py [--flip]   # --flip: dodatkowo odbij kadr poziomo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from flygym.vision.retina import Retina

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visual_pipeline.flyvis_step import FlyVisStepper  # noqa: E402
from visual_pipeline.retina_mapper import RetinaMapper  # noqa: E402

SUBTYPES = [f"T{n}{s}" for n in (4, 5) for s in "abcd"]
EXPECTED = {"a": 0.0, "b": 180.0, "c": 90.0, "d": 270.0}  # stopnie w kadrze, x w prawo, y w górę
FPS, WAVELENGTH_PX, TEMP_FREQ = 30, 130.0, 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip", action="store_true")
    args = ap.parse_args()

    retina, mapper, stepper = Retina(), RetinaMapper(), FlyVisStepper(fps=FPS)
    types, _, _ = stepper.cell_index()
    H, W = retina.nrows, retina.ncols
    yy, xx = np.mgrid[0:H, 0:W]
    x, y = xx - W / 2, -(yy - H / 2)  # y w górę

    def frame(theta, t):
        phase = (x * np.cos(theta) + y * np.sin(theta)) / WAVELENGTH_PX - TEMP_FREQ * t
        img = (127 + 100 * np.sin(2 * np.pi * phase)).astype(np.uint8)
        img = np.repeat(img[..., None], 3, axis=2)
        return np.ascontiguousarray(img[:, ::-1]) if args.flip else img

    def lum(img):
        return mapper.to_flyvis(retina.raw_image_to_hex_pxls(img), "right")[None]

    thetas = np.deg2rad(np.arange(0, 360, 30))
    resp = np.zeros((len(thetas), len(SUBTYPES)))
    for i, th in enumerate(thetas):
        stepper.reset(1)
        for _ in range(15):  # adaptacja do nieruchomej kratki
            stepper.step(lum(frame(th, 0.0)))
        acc = np.zeros(len(SUBTYPES))
        for k in range(45):
            act = stepper.step(lum(frame(th, k / FPS)))[0]
            if k >= 15:
                acc += [np.maximum(act[types == s], 0).mean() for s in SUBTYPES]
        resp[i] = acc / 30

    print("kierunek preferowany w kadrze (0° = w prawo, 90° = w górę):")
    pds = {}
    for j, s in enumerate(SUBTYPES):
        r = resp[:, j] - resp[:, j].mean()
        vec = (r * np.exp(1j * thetas)).sum()
        pds[s] = np.rad2deg(np.angle(vec)) % 360
        dsi = abs(vec) / np.abs(r).sum()
        print(f"  {s}: {pds[s]:5.0f}°  (oczekiwane {EXPECTED[s[-1]]:3.0f}°, selektywność {dsi:.2f})")

    # Najlepsze przekształcenie kadru: obrót o k*90° z odbiciem lub bez.
    best = []
    for mirror in (False, True):
        for rot in range(0, 360, 15):
            err = []
            for s, pd in pds.items():
                p = (180 - pd) % 360 if mirror else pd
                err.append(abs(((p + rot) - EXPECTED[s[-1]] + 180) % 360 - 180))
            best.append((np.mean(err), mirror, rot))
    best.sort()
    for e, mirror, rot in best[:3]:
        print(f"  kadr {'odbity poziomo + ' if mirror else ''}obrót {rot:3d}°: średni błąd {e:5.1f}°")


if __name__ == "__main__":
    main()
