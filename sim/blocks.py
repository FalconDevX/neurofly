"""Bloki na losowym terenie sceny beacon — przeszkody w stylu poligonu testowego ("blueprint").

Zastępują drzewa. MuJoCo nie dodaje obiektów do skompilowanego modelu, a podgląd jest związany
z jednym modelem, więc:
- add_pool(spec) przy ładowaniu sceny dodaje POOL_SIZE bloków (do PARTS prostopadłościanów każdy),
- place(...) przy każdym losowaniu świata ustawia ich pozycje, obrót, rozmiary i kolory; nieużyte
  bloki chowa głęboko pod terenem.

Rodzaje (rozmiary losowane w zakresach): kostka, filar, ściana (płyta), wieża (2–3 piętra coraz
węższe), schodki (2–3 stopnie). Materiał "block" (sim/assets/textures/blueprint_block.png) to jasny
panel z siatką; odcień daje geom_rgba. Bloki nie nachodzą na siebie, na start ani na plac celu,
a nad szczytem zostaje >= 1.5 m do górnej granicy planszy, więc nad każdym da się przelecieć.
Wszystkie części kolidują z dronem.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

POOL_SIZE = 70
PARTS = 3
HIDDEN_Z = -30.0          # m, schowek na nieużyte bloki (głęboko pod terenem)
COUNT = (40, 60)          # liczba bloków na świat
START_CLEARANCE = 3.0     # m wolnego miejsca między blokiem a startem drona
TARGET_CLEARANCE = 2.5    # m między blokiem a środkiem pola lądowania
WALL_CLEARANCE = 0.5      # m między blokiem a ścianą
TOP_CLEARANCE = 1.5       # m wolnego miejsca nad blokiem do górnej granicy planszy
ENVELOPE_RADIUS = 3.0     # m, obwiednia największego bloku (zasięg w poziomie od środka)
ENVELOPE_HEIGHT = 6.5     # m, obwiednia największego bloku (od podstawy)
SINK = 0.3                # m, blok wchodzi w ziemię, żeby na zboczu nie wisiał w powietrzu
TINY = 1e-3               # m, rozmiar nieużytej części (schowana w środku podstawy)

# Odcienie (mnożą jasną teksturę): biel makiety, błękit, stal, akcent bursztynowy (rzadko).
PALETTE = np.array([
    (1.00, 1.00, 1.00, 1),
    (0.80, 0.90, 1.00, 1),
    (0.62, 0.78, 1.00, 1),
    (0.78, 0.82, 0.90, 1),
    (1.00, 0.82, 0.45, 1),
])
PALETTE_P = np.array([0.35, 0.25, 0.15, 0.17, 0.08])


@dataclass(frozen=True)
class Kind:
    name: str
    weight: float   # udział w losowaniu


KINDS = (Kind("kostka", 0.25), Kind("filar", 0.2), Kind("ściana", 0.2), Kind("wieża", 0.2), Kind("schodki", 0.15))


def add_pool(spec):
    """Dodaje do MjSpec pulę bloków (wywołać raz, przed compile).

    Każda część jest kompilowana jako obwiednia największego możliwego bloku (ENVELOPE_*), bo MuJoCo
    liczy przy kompilacji drzewo brył otaczających ciała (bvh_aabb) i nie odświeża go po zmianie
    geom_pos/geom_size — za mała obwiednia odrzucałaby kolizje z dronem. Do tego pos != 0, bo geom
    w środku ciała dostaje geom_sameframe i MuJoCo ignoruje potem zmiany geom_pos.
    """
    half_h = ENVELOPE_HEIGHT / 2
    r = ENVELOPE_RADIUS / np.sqrt(2)  # po obrocie wokół z narożnik nie wychodzi poza ENVELOPE_RADIUS
    for i in range(POOL_SIZE):
        body = spec.worldbody.add_body(name=f"block{i}", pos=[0, 0, HIDDEN_Z])
        for k in range(PARTS):
            body.add_geom(name=f"block{i}_part{k}", type=mujoco.mjtGeom.mjGEOM_BOX, material="block",
                          size=[r, r, half_h], pos=[0, 0, half_h])


def has_pool(model):
    try:
        model.body("block0")
    except KeyError:
        return False
    return True


def _shape(kind, rng):
    """Zwraca listę części [(środek (x, y, z), pół-boki)] w układzie bloku (z = 0 na gruncie)."""
    u = rng.uniform
    if kind.name == "kostka":
        a = u(0.4, 0.9)
        parts = [((0, 0, a), (a, a * u(0.8, 1.2), a))]
    elif kind.name == "filar":
        a, h = u(0.25, 0.45), u(1.8, 3.0)
        parts = [((0, 0, h), (a, a, h))]
    elif kind.name == "ściana":
        h = u(0.8, 1.6)
        parts = [((0, 0, h), (u(1.0, 2.0), u(0.15, 0.25), h))]
    elif kind.name == "wieża":  # piętra coraz węższe
        a, z, parts = u(0.8, 1.3), 0.0, []
        for _ in range(int(rng.integers(2, PARTS + 1))):
            h = u(0.5, 1.0)
            parts.append(((0, 0, z + h), (a, a, h)))
            z += 2 * h
            a *= u(0.55, 0.75)
    else:  # schodki: stopnie wzdłuż x, coraz wyższe
        w, d, parts = u(0.7, 1.2), u(0.35, 0.55), []
        for k in range(int(rng.integers(2, PARTS + 1))):
            h = 0.35 * (k + 1)
            parts.append((((2 * k - 1) * d, 0, h), (d, w, h)))
    return parts


def _set_geom(model, gid, pos, size, rgba):
    model.geom_sameframe[gid] = 0  # patrz add_pool
    model.geom_pos[gid] = pos
    model.geom_size[gid] = size
    model.geom_rgba[gid] = rgba
    model.geom_rbound[gid] = np.linalg.norm(size)
    model.geom_aabb[gid] = (0, 0, 0, *size)


def place(model, rng, ground, half_size, start, target, ceiling):
    """Rozstawia bloki. ground(x, y) -> wysokość terenu [m], ceiling = górna granica planszy [m].

    Zwraca liczbę postawionych bloków.
    """
    weights = np.array([k.weight for k in KINDS])
    wanted = int(rng.integers(*COUNT))
    placed = []  # (x, y, zasięg)
    for i in range(POOL_SIZE):
        body = model.body(f"block{i}").id
        geoms = [model.geom(f"block{i}_part{k}").id for k in range(PARTS)]
        if len(placed) >= wanted:
            model.body_pos[body] = (0, 0, HIDDEN_Z)
            continue
        kind = KINDS[rng.choice(len(KINDS), p=weights / weights.sum())]
        parts = _shape(kind, rng)
        reach = max(np.hypot(abs(p[0]) + s[0], abs(p[1]) + s[1]) for p, s in parts)
        top = max(p[2] + s[2] for p, s in parts)
        assert reach <= ENVELOPE_RADIUS and top + SINK <= ENVELOPE_HEIGHT, "blok większy niż obwiednia"
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
            if ground(x, y) + top > ceiling - TOP_CLEARANCE:  # blok na pagórku nie sięga granicy planszy
                continue
            break
        else:  # brak miejsca — chowamy ten blok
            model.body_pos[body] = (0, 0, HIDDEN_Z)
            continue
        placed.append((x, y, reach))
        # Podstawa na najniższym punkcie terenu pod blokiem, żeby na zboczu nie wisiała.
        base = min(ground(x + dx, y + dy) for dx in (-reach, 0, reach) for dy in (-reach, 0, reach))
        model.body_pos[body] = (x, y, base - SINK)
        yaw = rng.uniform(0, np.pi)
        model.body_quat[body] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        rgba = PALETTE[rng.choice(len(PALETTE), p=PALETTE_P)]
        for k, gid in enumerate(geoms):
            if k < len(parts):
                (px, py, pz), (sx, sy, sz) = parts[k]
                lo, hi = pz - sz + SINK, pz + sz + SINK  # w układzie ciała grunt jest na z = SINK
                if lo <= SINK:  # część stojąca na gruncie sięga SINK w głąb ziemi
                    lo = 0.0
                _set_geom(model, gid, (px, py, (lo + hi) / 2), (sx, sy, (hi - lo) / 2), rgba)
            else:  # nieużyta część: maleńka, w środku podstawy
                _set_geom(model, gid, (0, 0, SINK + TINY), (TINY, TINY, TINY), rgba)
    return len(placed)
