"""Generuje tekstury "blueprint" sceny beacon: sim/assets/textures/*.png.

    python sim/tools/build_blueprint_textures.py

- blueprint_ground.png — granatowe podłoże z siatką: cienkie linie co 0.5 m, grube co 4 m
  (kafel = 4 x 4 m, scena powtarza go 15 x 15 razy na 60 m),
- blueprint_block.png — jasny panel testowy bloków: obramowanie, siatka co 1/4 boku, znacznik w rogu.

Kolory bloków różnicuje geom_rgba (mnoży teksturę), więc tekstura bloku jest prawie biała.
"""

from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "assets" / "textures"

GROUND_BG = np.array([22, 62, 128])
GROUND_MINOR = np.array([70, 120, 190])
GROUND_MAJOR = np.array([170, 210, 255])
BLOCK_BG = np.array([236, 242, 250])
BLOCK_LINE = np.array([120, 160, 215])
BLOCK_EDGE = np.array([40, 85, 160])


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
    for mask, color, alpha in ((minor, GROUND_MINOR, 0.8), (mid, GROUND_MAJOR, 0.45), (major, GROUND_MAJOR, 0.9)):
        m = (mask[:, None] | mask[None, :])[..., None]
        img = np.where(m, (1 - alpha) * img + alpha * color, img)
    # delikatna winieta w kaflu, żeby podłoże nie było płaskie jak plansza
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    img *= (1 - 0.06 * (xx ** 2 + yy ** 2) * 4)[..., None]
    return img.clip(0, 255).astype(np.uint8)


def block(n=512):
    img = np.tile(BLOCK_BG, (n, n, 1)).astype(float)
    grid = _lines(n, n // 4, 3.0)
    m = (grid[:, None] | grid[None, :])[..., None]
    img = np.where(m, BLOCK_LINE, img)
    edge = _lines(n, n, 14.0)
    m = (edge[:, None] | edge[None, :])[..., None]
    img = np.where(m, BLOCK_EDGE, img)
    # znacznik w rogu (jak na makiecie testowej): pełny kwadrat 1/8 boku
    k = n // 8
    img[10:10 + k // 2, 10:10 + k // 2] = BLOCK_EDGE
    return img.clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ground()).save(OUT / "blueprint_ground.png")
    Image.fromarray(block()).save(OUT / "blueprint_block.png")
    print("zapisano", *sorted(p.name for p in OUT.glob("*.png")))
