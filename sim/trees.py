"""Drzewa na losowym terenie sceny beacon — przeszkody o różnych gatunkach i rozmiarach.

MuJoCo nie dodaje obiektów do skompilowanego modelu, a podgląd jest związany z jednym modelem, więc:
- add_pool(spec) przy ładowaniu sceny dodaje POOL_SIZE drzew (pień + 3 elipsoidy korony),
- place(...) przy każdym losowaniu świata ustawia ich pozycje, rozmiary i kolory; nieużyte
  drzewa chowa głęboko pod terenem.

Gatunki (wysokość / grubość pnia / promień korony losowane w zakresach gatunku):
świerk (smukły, warstwowa ciemna korona), dąb (gruby pień, szeroka korona), brzoza (cienki biały
pień, wąska korona), krzak (niski, bez pnia). Drzewa nie nachodzą na siebie, na start ani na plac
celu, a nad czubkiem zostaje >= 1.5 m do górnej granicy planszy, więc nad każdym da się przelecieć.
Pień i korona kolidują z dronem.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

POOL_SIZE = 90
CROWN_PARTS = 3
HIDDEN_Z = -30.0          # m, schowek na nieużyte drzewa (głęboko pod terenem)
COUNT = (55, 80)          # liczba drzew na świat
START_CLEARANCE = 3.0     # m wolnego miejsca między koroną a startem drona
TARGET_CLEARANCE = 2.5    # m między koroną a środkiem pola lądowania
WALL_CLEARANCE = 0.5      # m między koroną a ścianą
TOP_CLEARANCE = 1.5       # m wolnego miejsca nad czubkiem drzewa do górnej granicy planszy
ENVELOPE_RADIUS = 3.0     # m, obwiednia największego drzewa (zasięg korony w poziomie)
ENVELOPE_HEIGHT = 9.0     # m, obwiednia największego drzewa (od podstawy pnia)
TRUNK_SINK = 0.15         # m, pień wchodzi w ziemię, żeby na zboczu nie wisiał w powietrzu
TRUNK_RGBA = np.array([0.36, 0.23, 0.12, 1.0])


@dataclass(frozen=True)
class Species:
    name: str
    weight: float           # udział w losowaniu
    height: tuple           # m
    trunk_radius: tuple     # m
    crown_radius: tuple     # m, największa elipsoida korony
    crown_rgba: tuple
    trunk_rgba: tuple = tuple(TRUNK_RGBA)


SPECIES = (
    Species("świerk", 0.35, (3.5, 7.0), (0.08, 0.15), (0.9, 1.6), (0.07, 0.27, 0.11, 1)),
    Species("dąb", 0.25, (4.0, 7.5), (0.18, 0.35), (1.4, 2.4), (0.17, 0.40, 0.13, 1)),
    Species("brzoza", 0.2, (4.5, 7.5), (0.06, 0.11), (0.6, 1.0), (0.42, 0.60, 0.18, 1), (0.88, 0.88, 0.82, 1)),
    Species("krzak", 0.2, (0.6, 1.4), (0.0, 0.0), (0.5, 1.0), (0.20, 0.36, 0.11, 1)),
)


def add_pool(spec):
    """Dodaje do MjSpec pulę drzew (wywołać raz, przed compile).

    Każda część jest kompilowana jako obwiednia największego możliwego drzewa (ENVELOPE_*), bo MuJoCo
    liczy przy kompilacji drzewo brył otaczających ciała (bvh_aabb) i nie odświeża go po zmianie
    geom_pos/geom_size — za mała obwiednia odrzucałaby kolizje z dronem. Do tego pos != 0, bo geom
    w środku ciała dostaje geom_sameframe i MuJoCo ignoruje potem zmiany geom_pos.
    """
    half_h = ENVELOPE_HEIGHT / 2
    for i in range(POOL_SIZE):
        body = spec.worldbody.add_body(name=f"tree{i}", pos=[0, 0, HIDDEN_Z])
        body.add_geom(name=f"tree{i}_trunk", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[ENVELOPE_RADIUS, half_h, 0], pos=[0, 0, half_h])
        for k in range(CROWN_PARTS):
            body.add_geom(name=f"tree{i}_crown{k}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          size=[ENVELOPE_RADIUS, ENVELOPE_RADIUS, half_h], pos=[0, 0, half_h])


def has_pool(model):
    try:
        model.body("tree0")
    except KeyError:
        return False
    return True


def _shape(species, rng):
    """Zwraca (pień (r, pół-długość, z środka), korona [(pos, pół-osie)], zasięg korony w poziomie, czubek)."""
    h = rng.uniform(*species.height)
    r_trunk = rng.uniform(*species.trunk_radius)
    big = rng.uniform(*species.crown_radius)
    jitter = lambda: rng.uniform(0.9, 1.1)  # noqa: E731
    if species.name == "świerk":  # piętra coraz węższe ku górze
        crown = [((0, 0, h * (0.35 + 0.22 * k)), (big * (1 - 0.3 * k), big * (1 - 0.3 * k), h * 0.16))
                 for k in range(3)]
        trunk_top = h * 0.9
    elif species.name == "dąb":  # szeroka główna korona + dwie boczne
        side = rng.uniform(0, 2 * np.pi)
        off = 0.45 * big
        crown = [((0, 0, h * 0.68), (big, big * jitter(), h * 0.32))]
        for a in (side, side + np.pi * rng.uniform(0.6, 1.4)):
            crown.append(((off * np.cos(a), off * np.sin(a), h * 0.6), (0.7 * big, 0.7 * big, h * 0.25)))
        trunk_top = h * 0.6
    elif species.name == "brzoza":  # wąska, wysoka korona
        crown = [((0, 0, h * z), (big * s, big * s * jitter(), h * 0.14))
                 for z, s in ((0.5, 1.0), (0.68, 0.9), (0.84, 0.7))]
        trunk_top = h * 0.9
    else:  # krzak: trzy niskie kępy, bez pnia
        crown = [((0, 0, h * 0.5), (big, big * jitter(), h * 0.5))]
        for a in rng.uniform(0, 2 * np.pi, 2):
            crown.append(((0.4 * big * np.cos(a), 0.4 * big * np.sin(a), h * 0.4), (0.7 * big, 0.7 * big, h * 0.4)))
        trunk_top = 0.0
    half = (trunk_top + TRUNK_SINK) / 2
    reach = max(np.hypot(pos[0], pos[1]) + max(axes[:2]) for pos, axes in crown)
    top = max(pos[2] + axes[2] for pos, axes in crown)
    assert reach <= ENVELOPE_RADIUS and top + TRUNK_SINK <= ENVELOPE_HEIGHT, "drzewo większe niż obwiednia"
    return (r_trunk, half, trunk_top - half), crown, reach, top


def _set_geom(model, gid, pos, size, rgba):
    model.geom_sameframe[gid] = 0  # patrz add_pool
    model.geom_pos[gid] = pos
    model.geom_size[gid] = size
    model.geom_rgba[gid] = rgba
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = size[0], size[1]
        model.geom_rbound[gid] = np.hypot(r, h)
        model.geom_aabb[gid] = (0, 0, 0, r, r, h)
    else:
        model.geom_rbound[gid] = max(size)
        model.geom_aabb[gid] = (0, 0, 0, *size)


def place(model, rng, ground, half_size, start, target, ceiling):
    """Rozstawia drzewa. ground(x, y) -> wysokość terenu [m], ceiling = górna granica planszy [m].

    Zwraca liczbę postawionych drzew.
    """
    weights = np.array([s.weight for s in SPECIES])
    wanted = int(rng.integers(*COUNT))
    placed = []  # (x, y, zasięg korony)
    for i in range(POOL_SIZE):
        body = model.body(f"tree{i}").id
        geoms = [model.geom(f"tree{i}_trunk").id] + [model.geom(f"tree{i}_crown{k}").id for k in range(CROWN_PARTS)]
        if len(placed) >= wanted:
            model.body_pos[body] = (0, 0, HIDDEN_Z)
            continue
        species = SPECIES[rng.choice(len(SPECIES), p=weights / weights.sum())]
        trunk, crown, reach, top = _shape(species, rng)
        for _ in range(200):  # losowanie miejsca z odrzucaniem
            x, y = rng.uniform(-half_size, half_size, 2)
            if max(abs(x), abs(y)) > half_size - reach - WALL_CLEARANCE:
                continue
            if np.hypot(x - start[0], y - start[1]) < reach + START_CLEARANCE:
                continue
            if np.hypot(x - target[0], y - target[1]) < reach + TARGET_CLEARANCE:
                continue
            if any(np.hypot(x - px, y - py) < reach + pr for px, py, pr in placed):
                continue
            if ground(x, y) + top > ceiling - TOP_CLEARANCE:  # drzewo na pagórku nie sięga granicy planszy
                continue
            break
        else:  # brak miejsca — chowamy to drzewo
            model.body_pos[body] = (0, 0, HIDDEN_Z)
            continue
        placed.append((x, y, reach))
        model.body_pos[body] = (x, y, ground(x, y) - TRUNK_SINK)
        r, half, zc = trunk
        crown_rgba = np.clip(np.array(species.crown_rgba) * rng.uniform(0.85, 1.15), 0, 1)  # odcień
        crown_rgba[3] = 1.0
        _set_geom(model, geoms[0], (0, 0, zc + TRUNK_SINK), (max(r, 1e-3), max(half, 1e-3), 0), species.trunk_rgba)
        for gid, (pos, axes) in zip(geoms[1:], crown):
            _set_geom(model, gid, (pos[0], pos[1], pos[2] + TRUNK_SINK), axes, crown_rgba)
    return len(placed)
