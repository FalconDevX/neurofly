"""Bloki na losowym terenie sceny beacon — przeszkody w stylu poligonu testowego ("blueprint").

Zastępują drzewa. MuJoCo nie dodaje obiektów do skompilowanego modelu, a podgląd jest związany
z jednym modelem, więc:
- add_pool(spec) przy ładowaniu sceny dodaje POOL_SIZE bloków (do PARTS prostopadłościanów każdy),
- place(...) przy każdym losowaniu świata ustawia ich pozycje, obrót, rozmiary i kolory; nieużyte
  bloki chowa głęboko pod terenem.

Rodzaje (rozmiary losowane w zakresach): kostka, filar, ściana (płyta), wieża (2–3 piętra coraz
węższe), schodki (2–3 stopnie). Materiał "block" (sim/assets/textures/block_side.png na bokach z niebieskimi
prostokątami, block_top.png na górze i spodzie) to jasny
panel z siatką; odcień daje geom_rgba. Bloki nie nachodzą na siebie, na start ani na plac celu.

Wysokość: zwykłe bloki zostawiają nad sobą >= TOP_CLEARANCE do górnej granicy planszy (da się przelecieć
górą); ~TALL_P filarów, ścian i wież to bloki WYSOKIE (TALL_HEIGHT) — ponad granicę planszy (8 m), trzeba je
ominąć. Nie ma bloków „prawie do przelecenia” (szczyt między granicą − TOP_CLEARANCE a granicą + TALL_MARGIN).
Pochylenie: ~TILT_P zwykłych bloków stoi pod kątem TILT_GROUND_ANGLE do podłoża (reszta pionowo). Obracamy
całe ciało (body_quat), więc obwiednia z kompilacji (w układzie ciała) dalej obejmuje części.
Podstawa zawsze jest w ziemi: blok jest obniżony tak, że najwyżej uniesiony róg podstawy leży >= SINK pod
gruntem — pochylony blok wygląda, jakby wyrastał z ziemi pod kątem, nigdy nie wisi w powietrzu. Pochyla się
w stronę wąskiego boku, a co najmniej MIN_ABOVE jego objętości (próbki 4x4x4 na część, względem prawdziwego
terenu pod każdą próbką) jest nad gruntem — żaden blok nie znika pod ziemią.
Wszystkie części kolidują z dronem.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

POOL_SIZE = 95
PARTS = 3
HIDDEN_Z = -40.0          # m, schowek na nieużyte bloki (głęboko pod terenem, poniżej najwyższego bloku)
COUNT = (60, 85)          # liczba bloków na świat (gęsto: trzeba lawirować)
START_CLEARANCE = 3.0     # m wolnego miejsca między blokiem a startem drona
TARGET_CLEARANCE = 2.5    # m między blokiem a środkiem celu
WALL_CLEARANCE = 0.5      # m między blokiem a ścianą
TOP_CLEARANCE = 1.5       # m wolnego miejsca nad zwykłym blokiem do górnej granicy planszy
TALL_MARGIN = 0.5         # m, wysoki blok wystaje co najmniej tyle ponad granicę planszy
TALL_P = 0.15             # udział wysokich wśród filarów, ścian i wież
TALL_HEIGHT = (9.5, 13.0)  # m, wysokość wysokiego bloku (od gruntu)
TILT_P = 0.3              # udział pochylonych wśród zwykłych bloków
TILT_GROUND_ANGLE = (30.0, 80.0)  # stopnie między osią bloku a podłożem (90 = pionowo)
ENVELOPE_RADIUS = 4.6     # m, obwiednia największego bloku w jego układzie (zasięg w poziomie od środka)
ENVELOPE_HEIGHT = 14.0    # m, obwiednia największego bloku w jego układzie (od podstawy)
SINK = 0.3                # m, podstawa wchodzi tyle w ziemię (na zboczu i przy pochyleniu nie wisi)
MIN_ABOVE = 0.5           # co najmniej taki udział objętości bloku musi być nad gruntem
ABOVE_MARGIN = 0.05       # zapas na zgrubne próbkowanie (4x4x4) — sprawdzamy MIN_ABOVE + ABOVE_MARGIN
TINY = 1e-3               # m, rozmiar nieużytej części (schowana pod ziemią)

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
    w środku ciała dostaje geom_sameframe i MuJoCo ignoruje potem zmiany geom_pos. Obwiednia jest w układzie
    ciała, więc obrót całego ciała (pochylenie) jej nie psuje.
    """
    half_h = ENVELOPE_HEIGHT / 2
    r = ENVELOPE_RADIUS / np.sqrt(2)  # narożnik kwadratu nie wychodzi poza ENVELOPE_RADIUS
    for i in range(POOL_SIZE):
        body = spec.worldbody.add_body(name=f"block{i}", pos=[0, 0, HIDDEN_Z])
        for k in range(PARTS):
            body.add_geom(name=f"block{i}_part{k}", type=mujoco.mjtGeom.mjGEOM_BOX, material="block",
                          size=[r, r, half_h], pos=[0, 0, half_h - SINK],
                          rgba=[1, 1, 1, 0])  # niewidoczny, dopóki place() go nie postawi


def hide(model, body):
    """Chowa blok (id ciała): pod ziemię, maleńki i przezroczysty — nic nie prześwituje pod mapą.

    Samo przeniesienie na HIDDEN_Z zostawiało pełny rozmiar (do ~13 m, a nieużyte z kompilacji to obwiednia
    4.6 x 14 m), więc pod środkiem mapy był widoczny blok.
    """
    name = model.body(body).name
    model.body_pos[body] = (0, 0, HIDDEN_Z)
    model.body_quat[body] = (1, 0, 0, 0)
    for k in range(PARTS):
        _set_geom(model, model.geom(f"{name}_part{k}").id, (0, 0, 0), (TINY, TINY, TINY), (1, 1, 1, 0))


def has_pool(model):
    try:
        model.body("block0")
    except KeyError:
        return False
    return True


def _shape(kind, rng, tall=False):
    """Zwraca listę części [(środek (x, y, z), pół-boki)] w układzie bloku (z = 0 na gruncie).

    tall: wysoki wariant (filar, ściana, wieża) — całkowita wysokość z TALL_HEIGHT.
    """
    u = rng.uniform
    if kind.name == "kostka":
        a = u(0.7, 1.5)
        parts = [((0, 0, a), (a, a * u(0.8, 1.2), a))]
    elif kind.name == "filar":
        a, h = u(0.35, 0.7), u(1.5, 3.0)
        parts = [((0, 0, h), (a, a, h))]
    elif kind.name == "ściana":
        h = u(1.0, 2.2)
        parts = [((0, 0, h), (u(1.5, 3.0), u(0.2, 0.35), h))]
    elif kind.name == "wieża":  # piętra coraz węższe
        a, z, parts = u(1.2, 2.0), 0.0, []
        for _ in range(int(rng.integers(2, PARTS + 1))):
            h = u(0.7, 1.4)
            parts.append(((0, 0, z + h), (a, a, h)))
            z += 2 * h
            a *= u(0.55, 0.75)
    else:  # schodki: stopnie wzdłuż x, coraz wyższe
        w, d, parts = u(1.0, 2.0), u(0.6, 1.0), []
        for k in range(int(rng.integers(2, PARTS + 1))):
            h = 0.6 * (k + 1)
            parts.append((((2 * k - 1) * d, 0, h), (d, w, h)))
    if tall:  # rozciągamy w pionie do zadanej wysokości (proporcje pięter zostają)
        scale = u(*TALL_HEIGHT) / max(p[2] + sz[2] for p, sz in parts)
        parts = [((p[0], p[1], p[2] * scale), (sz[0], sz[1], sz[2] * scale)) for p, sz in parts]
    return parts


def _sink(p, size):
    """Część stojąca na gruncie (dół na z = 0) wydłużona o SINK w głąb ziemi; reszta bez zmian."""
    (px, py, pz), (sx, sy, sz) = p, size
    if pz - sz > 1e-9:
        return (px, py, pz), (sx, sy, sz)
    lo, hi = -SINK, pz + sz
    return (px, py, (lo + hi) / 2), (sx, sy, (hi - lo) / 2)


_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def corners(parts):
    """(wszystkie narożniki (N, 3), narożniki podstawy (M, 3)) w układzie bloku.

    Podstawa = dolne narożniki części stojących na gruncie (dół na z = -SINK po _sink).
    """
    all_c, base_c = [], []
    for p, size in parts:
        c = np.asarray(p, float) + _SIGNS * size
        all_c.append(c)
        if p[2] - size[2] <= -SINK + 1e-9:
            base_c.append(c[c[:, 2] <= -SINK + 1e-9])
    return np.vstack(all_c), np.vstack(base_c)


_GRID = (np.arange(4) + 0.5) / 4 * 2 - 1  # 4 próbki na oś, środki komórek w [-1, 1]
_CELLS = np.array([[a, b, c] for a in _GRID for b in _GRID for c in _GRID])


def volume_samples(parts):
    """Punkty wewnątrz części (N, 3) w układzie bloku i ich wagi objętości (N,) — do udziału nad ziemią."""
    points, weights = [], []
    for p, size in parts:
        points.append(np.asarray(p, float) + _CELLS * size)
        weights.append(np.full(len(_CELLS), 8 * np.prod(size) / len(_CELLS)))
    return np.vstack(points), np.concatenate(weights)


def _rotation(yaw, tilt, axis="y"):
    """Pochylenie o tilt wokół osi bloku (x albo y), potem obrót o yaw wokół pionu. Zwraca (quat, macierz 3x3)."""
    s = np.sin(tilt / 2)
    q_tilt = np.array([np.cos(tilt / 2), s, 0, 0]) if axis == "x" else np.array([np.cos(tilt / 2), 0, s, 0])
    q_yaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, q_tilt)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    return q, mat.reshape(3, 3)


def _above_fraction(z, ground_z, weights):
    """Udział objętości (wag próbek) nad gruntem."""
    return float(weights[np.asarray(z) > ground_z].sum() / weights.sum())


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
            hide(model, body)
            continue
        kind = KINDS[rng.choice(len(KINDS), p=weights / weights.sum())]
        tall = kind.name in ("filar", "ściana", "wieża") and rng.random() < TALL_P
        parts = [_sink(p, sz) for p, sz in _shape(kind, rng, tall)]
        body_corners, base_corners = corners(parts)
        assert np.max(np.hypot(body_corners[:, 0], body_corners[:, 1])) <= ENVELOPE_RADIUS + 1e-9
        assert body_corners[:, 2].max() + SINK <= ENVELOPE_HEIGHT - SINK, "blok większy niż obwiednia"

        samples, sample_w = volume_samples(parts)
        # Pochylenie w stronę wąskiego boku (obrót wokół dłuższej osi podstawy): obniżenie zależy wtedy od
        # grubości, nie szerokości — ściana opiera się bokiem, zamiast zakopywać się na całą szerokość.
        axis = "x" if np.ptp(base_corners[:, 0]) >= np.ptp(base_corners[:, 1]) else "y"
        yaw = rng.uniform(0, 2 * np.pi)
        tilt = 0.0
        if not tall and rng.random() < TILT_P:
            for _ in range(5):  # na płaskim gruncie musi zostać nad ziemią wyraźnie więcej niż MIN_ABOVE
                tilt = np.deg2rad(90 - rng.uniform(*TILT_GROUND_ANGLE))
                _, rot = _rotation(yaw, tilt, axis)
                lift = (base_corners @ rot.T)[:, 2].max() + SINK
                if _above_fraction((samples @ rot.T)[:, 2] - lift, 0.0, sample_w) >= MIN_ABOVE + 0.1:
                    break
            else:
                tilt = 0.0
        quat, rot = _rotation(yaw, tilt, axis)
        world = body_corners @ rot.T
        base_world = base_corners @ rot.T
        samples_world = samples @ rot.T
        reach = float(np.max(np.hypot(world[:, 0], world[:, 1])))
        # Ciało stawiamy na z = grunt - SINK - (najwyżej uniesiony róg podstawy względem dołu podstawy):
        # wtedy cała podstawa jest >= SINK pod gruntem.
        lift = float(base_world[:, 2].max() + SINK)
        height = float(world[:, 2].max()) - lift  # szczyt ponad gruntem pod podstawą
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
            base = min(ground(x + bx, y + by) for bx, by in base_world[:, :2])
            top = base + height
            if tall and top < ceiling + TALL_MARGIN:  # wysoki: na pewno ponad granicę planszy
                continue
            if not tall and top > ceiling - TOP_CLEARANCE:  # zwykły: da się przelecieć górą
                continue
            terrain = ground(x + samples_world[:, 0], y + samples_world[:, 1])
            if _above_fraction(base - lift + samples_world[:, 2], terrain, sample_w) < MIN_ABOVE + ABOVE_MARGIN:
                continue  # np. przy pagórku: blok zakopany w ponad połowie
            break
        else:  # brak miejsca — chowamy ten blok
            hide(model, body)
            continue
        placed.append((x, y, reach))
        model.body_pos[body] = (x, y, base - lift)
        model.body_quat[body] = quat
        rgba = PALETTE[rng.choice(len(PALETTE), p=PALETTE_P)]
        for k, gid in enumerate(geoms):
            if k < len(parts):
                _set_geom(model, gid, *parts[k], rgba)
            else:  # nieużyta część: maleńka, pod ziemią w środku podstawy
                _set_geom(model, gid, (0, 0, -SINK), (TINY, TINY, TINY), rgba)
    return len(placed)
