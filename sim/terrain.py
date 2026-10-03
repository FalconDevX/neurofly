"""Losowy świat sceny beacon: teren (pagórki i zagłębienia), drzewa (sim/trees.py) + pozycja celu.

Scenę beacon ładujemy przez load_scene() (dokłada pulę drzew) i zawsze losujemy przez randomize().

Teren to heightfield "terrain" z scene_beacon.xml (60 x 60 m, otoczony ścianami). Wysokości liczymy
w numpy i wpisujemy do model.hfield_data — każde ziarno daje inny świat, to samo ziarno ten sam.

Gwarancje (żeby dało się dolecieć do flagi):
- start (0, 0) jest płaski na poziomie gruntu z = 0,
- cel leży 15–23 m od startu i co najmniej 6 m od ścian, na wyrównanym placu (pole lądowania
  leży płasko także na zboczu, szczycie pagórka albo w dołku),
- drzewa nie stoją na starcie, na placu celu ani na sobie nawzajem,
- teren mieści się w [-1, +2] m, a nad czubkiem każdego drzewa zostaje >= 1.5 m do górnej granicy
  planszy (8 m) — nad wszystkim da się przelecieć.

Granica planszy (outside_arena): wnętrze ścian do ich wysokości. Dron, który dotknie ściany albo
wzleci ponad nią, jest poza planszą — podgląd resetuje go wtedy na start (ten sam świat).
"""

import mujoco
import numpy as np

from sim import trees

GROUND_DEPTH = 1.0         # m, najgłębsze zagłębienie poniżej gruntu (= -pos z geomu terrain)
START = (0.0, 0.0)
START_FLAT = (1.5, 3.0)    # m: płasko do pierwszej wartości, łagodne przejście do drugiej
TARGET_FLAT = (1.0, 2.2)
TARGET_DIST = (15.0, 23.0)  # m od startu
WALL_MARGIN = 6.0          # m od ściany
WALL_HEIGHT = 8.0          # m, góra ścian w scene_beacon.xml
DRONE_RADIUS = 0.35        # m, zasięg łopat od środka drona — dotyk ściany = wyjście z planszy


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _blend_to(height, xx, yy, center, level, flat):
    """Wyrównuje teren do `level` w promieniu flat[0], z łagodnym przejściem do flat[1]."""
    w = _smoothstep((np.hypot(xx - center[0], yy - center[1]) - flat[0]) / (flat[1] - flat[0]))
    return level + (height - level) * w


def generate(half_size, n, max_height, rng):
    """Zwraca (wysokości [m] względem gruntu, wiersz 0 = najmniejsze y; (x, y) celu)."""
    axis = np.linspace(-half_size, half_size, n)
    xx, yy = np.meshgrid(axis, axis)
    area = (2 * half_size / 60.0) ** 2  # liczba pagórków proporcjonalna do powierzchni

    height = np.zeros_like(xx)

    def bump(amplitude, sigma):
        # Gauss liczony tylko w oknie 3 sigma — ~10x szybciej niż na całej siatce.
        x, y = rng.uniform(-half_size, half_size, 2)
        r = 3 * sigma
        i0, i1 = np.searchsorted(axis, [y - r, y + r])
        j0, j1 = np.searchsorted(axis, [x - r, x + r])
        gx, gy = axis[j0:j1] - x, axis[i0:i1] - y
        height[i0:i1, j0:j1] += amplitude * np.exp(-(gy[:, None] ** 2 + gx[None, :] ** 2) / (2 * sigma * sigma))

    for _ in range(int(110 * area)):  # pagórki
        bump(rng.uniform(0.3, 2.0), rng.uniform(0.8, 3.2))
    for _ in range(int(35 * area)):  # zagłębienia
        bump(-rng.uniform(0.3, 1.0), rng.uniform(0.8, 2.5))
    # Łagodne nasycenie zamiast obcięcia: góra do max_height, dół do -GROUND_DEPTH.
    height = np.where(height > 0, max_height * np.tanh(height / max_height),
                      GROUND_DEPTH * np.tanh(height / GROUND_DEPTH))

    height = _blend_to(height, xx, yy, START, 0.0, START_FLAT)

    limit = half_size - WALL_MARGIN
    while True:
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(*TARGET_DIST)
        tx, ty = START[0] + dist * np.cos(angle), START[1] + dist * np.sin(angle)
        if abs(tx) <= limit and abs(ty) <= limit:
            break
    near = np.hypot(xx - tx, yy - ty) <= TARGET_FLAT[0]
    height = _blend_to(height, xx, yy, (tx, ty), float(height[near].mean()), TARGET_FLAT)
    return height, (tx, ty)


def randomize(model, data, seed=None):
    """Losuje teren i cel w modelu sceny beacon. Zwraca użyte ziarno (do odtworzenia świata)."""
    if seed is None:
        seed = int(np.random.default_rng().integers(1_000_000))
    rng = np.random.default_rng(seed)

    hid = model.hfield("terrain").id
    nrow, ncol = model.hfield_nrow[hid], model.hfield_ncol[hid]
    half_x, half_y, z_range, _ = model.hfield_size[hid]
    assert nrow == ncol and half_x == half_y, "teren musi być kwadratowy"
    max_height = z_range - GROUND_DEPTH

    height, (tx, ty) = generate(half_x, nrow, max_height, rng)
    adr = model.hfield_adr[hid]
    # hfield trzyma 0..1 skalowane przez size[2]; geom terrain stoi na z = -GROUND_DEPTH.
    model.hfield_data[adr:adr + nrow * ncol] = ((height + GROUND_DEPTH) / z_range).ravel()

    ix = int(round((tx + half_x) / (2 * half_x) * (ncol - 1)))
    iy = int(round((ty + half_y) / (2 * half_y) * (nrow - 1)))
    model.body_pos[model.body("target").id] = (tx, ty, height[iy, ix])

    if trees.has_pool(model):
        def ground(x, y):
            j = int(round((x + half_x) / (2 * half_x) * (ncol - 1)))
            i = int(round((y + half_y) / (2 * half_y) * (nrow - 1)))
            return height[i, j]
        trees.place(model, rng, ground, half_x, START, (tx, ty), WALL_HEIGHT)
    mujoco.mj_forward(model, data)
    return seed


def load_scene(path):
    """Ładuje scenę; jeśli ma teren (scena beacon), dokłada pulę drzew. Zwraca MjModel."""
    spec = mujoco.MjSpec.from_file(str(path))
    if any(h.name == "terrain" for h in spec.hfields):
        trees.add_pool(spec)
    return spec.compile()


def outside_arena(model, data, body_id):
    """True, gdy dron dotyka ściany albo jest ponad nią (poza planszą)."""
    half = model.hfield_size[model.hfield("terrain").id, 0]
    x, y, z = data.xpos[body_id]
    return max(abs(x), abs(y)) > half - DRONE_RADIUS or z > WALL_HEIGHT
