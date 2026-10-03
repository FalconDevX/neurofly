"""Generuje tekstury sceny beacon: sim/assets/textures/*.png.

    python sim/tools/build_blueprint_textures.py

- ground.png — brązowe podłoże z siatką: cienkie linie co 0.5 m, grube co 4 m
  (kafel = 4 x 4 m, scena powtarza go 15 x 15 razy na 60 m),
- block_side.png — ściany boczne bloków: jasny panel z siatką co 1/4 boku i niebieskim prostokątem
  („oknem”) w każdym polu, z odstępem od linii siatki,
- block_top.png — góra i spód bloków: ta sama siatka, bez okien.

Kolory bloków różnicuje geom_rgba (mnoży teksturę), więc tła tekstur bloków są prawie białe.
(Nazwa skryptu zostaje po stylu "blueprint" — wcześniej granatowe podłoże.)
"""

from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "assets" / "textures"

GROUND_BG = np.array([112, 80, 52])        # brąz ziemi
GROUND_MINOR = np.array([138, 104, 72])    # linie co 0.5 m
GROUND_MAJOR = np.array([196, 160, 118])   # linie co 2 i 4 m
BLOCK_BG = np.array([236, 240, 245])
BLOCK_LINE = np.array([130, 140, 155])
BLOCK_EDGE = np.array([70, 80, 95])
WINDOW = np.array([70, 125, 205])          # niebieskie prostokąty na ścianach bocznych
WINDOW_GAP = (0.22, 0.18)                  # odstęp okna od linii siatki: część szerokości / wysokości pola


def _lines(n, step, width):
    """Maska linii co `step` px o grubości `width` px (wyśrodkowana na granicy kafla)."""
    idx = np.arange(n)
    d = np.minimum(idx % step, step - idx % step)
    return d < width / 2


def ground(n=1024):
    img = np.tile(GROUND_BG, (n, n, 1)).astype(float)
    minor = _lines(n, n // 8, 2.0)    # 0.5 m
    major = _lines(n, n, 6.0)         # 4 m (krawędź kafla — połowa linii z każdej strony)
    mid = _lines(n, n // 2, 3.0)      # 2 m
    for mask, color, alpha in ((minor, GROUND_MINOR, 0.7), (mid, GROUND_MAJOR, 0.4), (major, GROUND_MAJOR, 0.8)):
        m = (mask[:, None] | mask[None, :])[..., None]
        img = np.where(m, (1 - alpha) * img + alpha * color, img)
    # delikatna winieta i ziarno, żeby podłoże nie było płaskie jak plansza
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    img *= (1 - 0.06 * (xx ** 2 + yy ** 2) * 4)[..., None]
    img += np.random.default_rng(0).normal(0, 4, (n, n, 1))
    return img.clip(0, 255).astype(np.uint8)


def _panel(n, windows):
    img = np.tile(BLOCK_BG, (n, n, 1)).astype(float)
    cell = n // 4
    if windows:
        gx, gy = (int(g * cell) for g in WINDOW_GAP)
        for r in range(4):
            for c in range(4):
                img[r * cell + gy:(r + 1) * cell - gy, c * cell + gx:(c + 1) * cell - gx] = WINDOW
    grid = _lines(n, cell, 3.0)
    m = (grid[:, None] | grid[None, :])[..., None]
    img = np.where(m, BLOCK_LINE, img)
    edge = _lines(n, n, 14.0)
    m = (edge[:, None] | edge[None, :])[..., None]
    img = np.where(m, BLOCK_EDGE, img)
    return img.clip(0, 255).astype(np.uint8)


def block_side(n=512):
    return _panel(n, windows=True)


def block_top(n=512):
    return _panel(n, windows=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ground()).save(OUT / "ground.png")
    Image.fromarray(block_side()).save(OUT / "block_side.png")
    Image.fromarray(block_top()).save(OUT / "block_top.png")
    print("zapisano", *sorted(p.name for p in OUT.glob("*.png")))
