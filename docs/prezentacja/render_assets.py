"""Grafiki do prezentacji z prawdziwych danych: render 3D connectomu BANC v888, obwód lotu, dron w MuJoCo, oczy.

    .venv312/Scripts/python docs/prezentacja/render_assets.py      # → docs/prezentacja/assets/*.jpg, logo.png, data.json
    node docs/prezentacja/build.js                                  # → NeuroFly.pptx (pptxgenjs)

Wymaga data/viz/banc_anatomy.json (scripts/export_anatomy.py), FlyVis/flygym i data/decoders/planB_distributed.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\NeuroFly")
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent / "assets" / "png"
OUT.mkdir(parents=True, exist_ok=True)
BG = np.array([9, 9, 11], np.float32)

d = json.loads((ROOT / "data" / "viz" / "banc_anatomy.json").read_text(encoding="utf-8"))
xyz = np.array(d["somas"]["xyz"], np.float32).reshape(-1, 3)
cls = np.array(d["somas"]["super_class"])
classes = d["classes"]
center = (xyz.min(0) + xyz.max(0)) / 2

# paleta jak w eksploratorze (HTML): żywe kolory klas, interneurony fioletowe zamiast szarych
TEAL, AMBER, PINK, BLUE, VIOLET, SLATE = (47, 211, 196), (255, 181, 71), (255, 79, 176), (79, 140, 255), (169, 139, 255), (111, 120, 150)
CLASS_COLOR = {}
for i, c in enumerate(classes):
    if c in ("optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"):
        CLASS_COLOR[i] = TEAL
    elif c in ("descending", "ascending"):
        CLASS_COLOR[i] = AMBER
    elif c in ("motor", "visceral_circulatory", "ascending_visceral_circulatory"):
        CLASS_COLOR[i] = PINK
    elif c.startswith("sensory"):
        CLASS_COLOR[i] = BLUE
    elif c == "unknown":
        CLASS_COLOR[i] = SLATE
    else:
        CLASS_COLOR[i] = VIOLET
GROUP_COLOR = {"dn_flight_power": AMBER, "dn_flight_steering": (255, 138, 61), "wing_power": PINK,
               "wing_steering": (255, 122, 217), "wing_tension": (214, 92, 255), "haltere_aff": BLUE}

# kotwice etykiet (współrzędne BANC, µm) — jak w explorer/lib/data.ts
region = np.array(d["somas"]["region"])
regions = d["regions"]
def _c(mask):
    return xyz[mask].mean(0)
_ol = region == regions.index("optic_lobe")
_cb = _c(region == regions.index("central_brain"))
_vnc = _c(region == regions.index("ventral_nerve_cord"))
NECK_Y = 375.0
_skel = {}
for n in d["neurons"]:
    g = n["group"].rsplit("_", 1)[0]
    pts = np.concatenate([np.array(l, np.float32) for l in n["lines"]]) if n["lines"] else np.zeros((0, 3), np.float32)
    _skel.setdefault(g, []).append(pts)
_skel = {g: np.concatenate(v) for g, v in _skel.items()}
_dn = np.concatenate([_skel["dn_flight_power"], _skel["dn_flight_steering"]])
_dn = _dn[_dn[:, 1] < NECK_Y]  # tylko część w mózgu, aksony biegną do VNC
_mn = np.concatenate([_skel["wing_power"], _skel["wing_steering"], _skel["wing_tension"]])
# w BANC prawa strona muchy ma mniejsze x
REGION_LABELS = [
    ("Optic lobe R", "right eye, FlyVis input", _c(_ol & (xyz[:, 0] < center[0])), TEAL, "L"),
    ("Optic lobe L", "left eye, only 36% typed in v888", _c(_ol & (xyz[:, 0] >= center[0])), TEAL, "R"),
    ("Central brain", "sensory integration, DN somas", _cb - np.array([0, 60, 0]), VIOLET, "L"),
    ("Neck", "DN axons from brain to VNC", np.array([_cb[0], NECK_Y, _cb[2]]), AMBER, "R"),
    ("VNC", "wing, leg and haltere motor neurons", _vnc + np.array([0, 80, 0]), PINK, "R"),
]
CIRCUIT_LABELS = [
    ("Flight DNs", "heading (yaw) read from single DNs", _dn.mean(0), AMBER, "L"),
    ("Wing motor neurons", "power, steering, tension: drone commands", _mn.mean(0), PINK, "L"),
    ("Haltere afferents", "rotation sensors, fed by the drone gyro", _skel["haltere_aff"].mean(0), BLUE, "R"),
    ("Neck", "the only path for commands to the wings", np.array([_cb[0], NECK_Y, _cb[2]]), (250, 250, 250), "R"),
]


def project(p, yaw, pitch, scale, W, H, dist=2600.0):
    """Perspektywa: obrót wokół osi pionowej BANC (y w dół → w górę na obrazie) i pochylenie."""
    q = (p - center) * np.array([1, -1, 1], np.float32)  # y BANC rośnie w stronę VNC → na dół obrazu
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    x = cy * q[:, 0] + sy * q[:, 2]
    z = -sy * q[:, 0] + cy * q[:, 2]
    y = cp * q[:, 1] - sp * z
    z = sp * q[:, 1] + cp * z
    f = dist / (dist + z)
    return W / 2 + x * f * scale, H / 2 - y * f * scale, z


def fit(u, v, W, H, margin=0.06, horizontal=False, margin_x=None):
    """Wpasowuje rzut w kadr po percentylach (punkty odstające nie przesuwają środka); horizontal: mózg z lewej."""
    if horizontal:
        u, v = v.copy(), -u.copy()
    lo_u, hi_u = np.percentile(u, [0.3, 99.7])
    lo_v, hi_v = np.percentile(v, [0.3, 99.7])
    mx = margin if margin_x is None else margin_x
    s = min(W * (1 - 2 * mx) / (hi_u - lo_u), H * (1 - 2 * margin) / (hi_v - lo_v))
    cu, cv_ = (lo_u + hi_u) / 2, (lo_v + hi_v) / 2

    def tf(pu, pv):
        if horizontal:
            pu, pv = pv, -pu
        return (pu - cu) * s + W / 2, (pv - cv_) * s + H / 2
    return (u - cu) * s + W / 2, (v - cv_) * s + H / 2, tf


def splat(img, u, v, color, alpha, r=1):
    """Dodaje punkty (kolor × alpha) do bufora float — gęste regiony świecą mocniej (render jak w mikroskopii)."""
    H, W = img.shape[:2]
    ui, vi = u.astype(int), v.astype(int)
    for dx in range(-r + 1, r):
        for dy in range(-r + 1, r):
            uu, vv = ui + dx, vi + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            np.add.at(img, (vv[ok], uu[ok]), color[ok] * alpha[ok, None])


def _font(name, fallback):
    """Roboto (ten sam font co w prezentacji) z czcionek użytkownika albo systemu; gdy go nie ma — Segoe UI."""
    import os

    for d in (os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts"), r"C:\Windows\Fonts"):
        f = os.path.join(d, name)
        if os.path.exists(f):
            return f
    return fallback


FONT_B = _font("Roboto-Medium.ttf", r"C:\Windows\Fonts\seguisb.ttf")
FONT_R = _font("Roboto-Regular.ttf", r"C:\Windows\Fonts\segoeui.ttf")


def draw_labels(out, labels, tf, yaw, pitch, size=44):
    """Etykiety: linia od kotwicy do boku (bez kropki), nazwa + krótki opis (Roboto).
    labels: (nazwa, opis, punkt BANC, kolor, strona "L"/"R"); etykiety po jednej stronie nie nachodzą na siebie."""
    from PIL import Image, ImageDraw, ImageFont

    H, W = out.shape[:2]
    im = Image.fromarray(out)
    dr = ImageDraw.Draw(im)
    fb, fr = ImageFont.truetype(FONT_B, size), ImageFont.truetype(FONT_R, int(size * 0.68))
    pad, gap = int(size * 0.35), int(size * 0.4)
    th = size + int(size * 0.68) + pad * 3
    boxes = []
    for name, desc, p, col, side in labels:
        pu, pv, _ = project(np.asarray(p, np.float32)[None], yaw, pitch, 1.0, 0, 0)
        ax, ay = (float(a[0]) for a in tf(pu, pv))
        bw = max(dr.textlength(name, font=fb), dr.textlength(desc, font=fr)) + 2 * pad
        bx = W * 0.025 if side == "L" else W * 0.975 - bw
        boxes.append(dict(name=name, desc=desc, col=col, side=side, ax=ax, ay=ay, bx=bx, bw=bw, by=ay - th / 2))
    for side in "LR":
        bs = sorted((b for b in boxes if b["side"] == side), key=lambda b: b["ay"])
        y = 4
        for b in bs:
            b["by"] = y = max(b["by"], y)
            y += th + gap
        over = (bs[-1]["by"] + th + 4 - H) if bs else 0
        if over > 0:
            for b in bs:
                b["by"] -= over
    for b in boxes:
        col, ax, ay, bx, by, bw = b["col"], b["ax"], b["ay"], b["bx"], b["by"], b["bw"]
        ex, ey = (bx + bw if b["side"] == "L" else bx), float(np.clip(ay, by + pad, by + th - pad))
        dr.line([(ax, ay), (ex, ey)], fill=col, width=3)
        dr.rounded_rectangle([bx, by, bx + bw, by + th], radius=pad, fill=(16, 16, 20), outline=col, width=3)
        dr.text((bx + pad, by + pad), b["name"], font=fb, fill=(250, 250, 250))
        dr.text((bx + pad, by + pad * 2 + size), b["desc"], font=fr, fill=(170, 170, 180))
    return np.asarray(im)


def connectome(W=2400, H=1500, yaw=0.55, pitch=0.28, name="connectome_3d.png", horizontal=False, labels=None, margin_x=None):
    img = np.zeros((H, W, 3), np.float32)
    u, v, z = project(xyz, yaw, pitch, 1.0, 0, 0)
    u, v, tf = fit(u, v, W, H, horizontal=horizontal, margin_x=margin_x)
    depth = (z - z.min()) / (z.max() - z.min())
    col = np.array([CLASS_COLOR[c] for c in cls], np.float32)
    alpha = 0.14 + 0.26 * (1 - depth)  # bliższe jaśniejsze
    teal = np.all(col == TEAL, axis=1)
    alpha[teal] *= 0.38  # płaty wzrokowe są bardzo gęste — bez tego przepalają się do jednolitej plamy
    violet = np.all(col == VIOLET, axis=1)
    alpha[violet] *= 0.6
    splat(img, u, v, col, alpha, r=2)
    img = BG + img * 0.9
    # miękkie przepalenie zamiast obcinania do bieli: kolor zostaje w gęstych miejscach
    img = BG + (255 - BG) * (1 - np.exp(-(img - BG) / 180.0))
    out = np.clip(img, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (0, 0), 0.6)
    if labels:
        out = draw_labels(out, labels, tf, yaw, pitch)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def circuit(W=2000, H=1500, yaw=0.55, pitch=0.28, horizontal=False, labels=None, margin_x=None, name="circuit_3d.png"):
    """Szkielety 863 neuronów lotu (SWC) na przygaszonym tle som w kolorach klas."""
    img = np.zeros((H, W, 3), np.float32)
    u0, v0, z = project(xyz, yaw, pitch, 1.0, 0, 0)
    u, v, tf = fit(u0, v0, W, H, horizontal=horizontal, margin_x=margin_x)
    col = np.array([CLASS_COLOR[c] for c in cls], np.float32)
    splat(img, u, v, col, np.full(len(u), 0.035), r=1)
    layer = np.zeros_like(img)
    order = {"haltere_aff": 0, "dn_flight_power": 1, "dn_flight_steering": 1}
    for n in sorted(d["neurons"], key=lambda n: order.get(n["group"].rsplit("_", 1)[0], 2)):
        c = GROUP_COLOR.get(n["group"].rsplit("_", 1)[0], (161, 161, 170))
        for line in n["lines"]:
            pu, pv, _ = project(np.array(line, np.float32), yaw, pitch, 1.0, 0, 0)
            pu, pv = tf(pu, pv)
            pts = np.stack([pu, pv], 1).astype(np.int32)
            cv2.polylines(layer, [pts], False, tuple(float(x) * 0.85 for x in c), 2, cv2.LINE_AA)
    img = BG + img + layer
    img = BG + (255 - BG) * (1 - np.exp(-(img - BG) / 200.0))
    out = np.clip(img, 0, 255).astype(np.uint8)
    if labels:
        out = draw_labels(out, labels, tf, yaw, pitch)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def sim_images():
    import mujoco

    from sim.env import DroneEnv
    from sim.world_env import WorldEnv

    env = DroneEnv()
    L, R = env.calibration_render(np.deg2rad(25))
    cv2.imwrite(str(OUT / "eye_left.png"), cv2.cvtColor(L, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / "eye_right.png"), cv2.cvtColor(R, cv2.COLOR_RGB2BGR))
    obs, info = env.reset("turn_right")
    for _ in range(20):
        obs, *_ = env.step({"thrust": 0.5, "roll": 0, "pitch": 0, "yaw": 0.0})
    cv2.imwrite(str(OUT / "drone_chase.png"), cv2.cvtColor(env.render_chase(), cv2.COLOR_RGB2BGR))

    w = WorldEnv(control="angle", eyes=False, start_noise=False)
    w.reset(options={"world_seed": 104})
    from sim.banc_pilot import teacher
    for _ in range(90):
        c = teacher(w)
        w.step([c.thrust, c.roll, c.pitch, -c.yaw])
    r = mujoco.Renderer(w.model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = w.drone_id
    cam.distance, cam.azimuth, cam.elevation = 6.0, 135, -22
    r.update_scene(w.data, cam)
    cv2.imwrite(str(OUT / "world.png"), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))


def retina():
    """Siatki ommatidiów (721 na oko) z obrazu kamer; dwa typy fotoreceptorów → maksimum kanałów."""
    from flygym.vision.retina import Retina

    from visual_pipeline.frames import prepare_frame, to_luminance

    r = Retina()
    for eye in ("left", "right"):
        img = cv2.cvtColor(cv2.imread(str(OUT / f"eye_{eye}.png")), cv2.COLOR_BGR2RGB)
        hx = np.asarray(r.raw_image_to_hex_pxls(r.correct_fisheye(to_luminance(prepare_frame(img))))).max(1, keepdims=True)
        hx = np.repeat(hx, 2, axis=1)
        hr = np.asarray(r.hex_pxls_to_human_readable(hx)).max(axis=2)
        inside = np.asarray(r.hex_pxls_to_human_readable(np.ones((721, 2)))).max(axis=2) > 0
        v = np.clip((hr - hx.min()) / (hx.max() - hx.min() + 1e-9), 0, 1)
        lo, hi = np.array([24, 24, 27], np.float32), np.array([45, 212, 191], np.float32)
        out = np.where(inside[..., None], lo + (hi - lo) * v[..., None], BG)
        cv2.imwrite(str(OUT / f"retina_{eye}.png"), cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _glass(img, pts2, tri, depth, tint, gain=120, wing=None):
    """Półprzezroczysta siatka jak szkło: jaśniejsze krawędzie (Fresnel), ściany w poziomach jasności (szybkie fillPoly)."""
    H, W = img.shape[:2]
    q3 = np.concatenate([pts2, depth[:, None]], 1)
    a, b, c = q3[tri[:, 0]], q3[tri[:, 1]], q3[tri[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    fres = (1 - np.abs(nrm[:, 2])) ** 2.2
    alpha = (0.025 + 0.42 * fres) * (np.where(wing, 0.55, 1.0) if wing is not None else 1.0)
    p2 = pts2[tri].astype(np.int32)
    levels = np.linspace(alpha.min(), alpha.max(), 14)
    idx = np.clip(np.digitize(alpha, levels) - 1, 0, len(levels) - 1)
    layer = np.zeros((H, W), np.float32)
    for k in range(len(levels)):
        sel = idx == k
        if not sel.any():
            continue
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, list(p2[sel]), 1, cv2.LINE_AA)
        layer += mask.astype(np.float32) * levels[k] * gain
    img += layer[..., None] * np.array(tint, np.float32)


def _fly(img, box, yaw=0.15, pitch=0.22):
    """Ciało muszki (NeuroMechFly, scripts/export_fly_body.py) z connectomem BANC w środku, wpisane w prostokąt box.
    Zwraca punkty obwodu lotu na obrazie z kolorami grup (kotwice nitek do drona). Dopasowanie ciała do BANC jest
    ilustracyjne (głowa → mózg, tułów → VNC)."""
    x0, y0, bw, bh = box
    meta = json.loads((ROOT / "data" / "viz" / "fly_body.json").read_text())
    raw = (ROOT / "data" / "viz" / "fly_body.bin").read_bytes()
    n, m = meta["vertices"], meta["triangles"]
    bpos = np.frombuffer(raw, np.float32, n * 3).reshape(n, 3)
    tri = np.frombuffer(raw, np.uint32, m * 3, offset=n * 12).reshape(m, 3)
    part = np.frombuffer(raw, np.uint8, n, offset=n * 12 + m * 12)

    bu0, bv0, bz = project(bpos, yaw, pitch, 1.0, 0, 0)
    bu, bv, tf0 = fit(bu0, bv0, bw, bh, margin=0.04)

    def tf(pu, pv):
        u, v = tf0(pu, pv)
        return u + x0, v + y0

    _glass(img, np.stack([bu + x0, bv + y0], 1), tri, bz, (0.82, 0.84, 0.9), wing=part[tri[:, 0]] == 3)

    # somy (przygaszone, w kolorach klas)
    su, sv, _ = project(xyz, yaw, pitch, 1.0, 0, 0)
    su, sv = tf(su, sv)
    col = np.array([CLASS_COLOR[c] for c in cls], np.float32)
    splat(img, su, sv, col, np.full(len(su), 0.06), r=1)

    # obwód lotu: poświata (szeroko, rozmyte) + cienka jasna linia
    glow, line = np.zeros_like(img), np.zeros_like(img)
    anchors = []
    for nr in d["neurons"]:
        cc = GROUP_COLOR.get(nr["group"].rsplit("_", 1)[0], (161, 161, 170))
        for ln in nr["lines"]:
            pu, pv, _ = project(np.array(ln, np.float32), yaw, pitch, 1.0, 0, 0)
            pu, pv = tf(pu, pv)
            pts = np.stack([pu, pv], 1).astype(np.int32)
            cv2.polylines(glow, [pts], False, tuple(float(x) * 0.25 for x in cc), 6, cv2.LINE_AA)
            cv2.polylines(line, [pts], False, tuple(float(x) * 0.8 for x in cc), 2, cv2.LINE_AA)
            anchors += [(u, v, cc) for u, v in zip(pu[::7], pv[::7])]
    img += cv2.GaussianBlur(glow, (0, 0), 6) + line
    return anchors


def fly_xray(W=2600, H=1500, yaw=0.15, pitch=0.22, name="fly_xray.png"):
    """Półprzezroczyste ciało muszki z connectomem BANC w środku (sama muszka)."""
    img = np.zeros((H, W, 3), np.float32)
    _fly(img, (0, 0, W, H), yaw, pitch)
    img = BG + (255 - BG) * (1 - np.exp(-img / 210.0))
    cv2.imwrite(str(OUT / name), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def _drone_mesh():
    """Siatka Skydio X2 (MuJoCo Menagerie) w układzie świata: wierzchołki (m) i trójkąty."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(ROOT / "third_party" / "mujoco_menagerie" / "skydio_x2" / "x2.xml"))
    dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    V, F, off = [], [], 0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        V.append(m.mesh_vert[a:a + n] @ dd.geom_xmat[g].reshape(3, 3).T + dd.geom_xpos[g])
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        F.append(m.mesh_face[fa:fa + fn] + off)
        off += n
    return np.concatenate(V).astype(np.float32), np.concatenate(F)


def _drone_props(V, F):
    """Oddziela tarcze śmigieł X2 (płaskie, ~26 cm, wachlarz trójkątów) od reszty siatki: zwraca F bez tarcz
    i środki śmigieł (cztery, każda tarcza jest w siatce kilka razy)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), (n, n)), directed=False)
    disc = np.zeros(lab.max() + 1, bool)
    centers, radius = [], 0.0
    for c in np.unique(lab):
        q = V[lab == c]
        ext = q.max(0) - q.min(0)
        if ext[:2].min() > 0.2 and ext[2] < 0.005:
            disc[c] = True
            radius = max(radius, float(ext[:2].max()) / 2)
            ctr = (q.max(0) + q.min(0)) / 2
            if not any(np.linalg.norm(ctr[:2] - o[:2]) < 0.05 for o in centers):
                centers.append(ctr)
    return F[~disc[lab[F[:, 0]]]], np.array(centers, np.float32), radius


def fly_drone(W=2700, H=1850, name="fly_drone.png", n_threads=26, seed=3,
              yaw=0.5, tilt=0.95, roll=-0.25, fly_box=(0.1, 0.02, 0.6, 0.9), drone_box=(0.63, 0.1, 0.36, 0.72)):
    """Slajd tytułowy: muszka z BANC w środku, obok półprzezroczysty dron X2, między nimi cienkie nitki od neuronów
    obwodu lotu do drona. Ilustracja pętli („mózg muchy steruje dronem”), nitki nie są danymi."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    fx, fy, fw, fh = fly_box
    fly_pts = _fly(img, (int(W * fx), int(H * fy), int(W * fw), int(H * fh)))

    # dron: nosem do góry jak głowa muchy, pochylony i obrócony, żeby było widać bok i śmigła
    V, F = _drone_mesh()
    F, props, R = _drone_props(V, F)
    mid = (V.min(0) + V.max(0)) / 2
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    ct, st = np.cos(tilt), np.sin(tilt)
    cr, sr = np.cos(roll), np.sin(roll)

    def view(P):
        P = P - mid
        x = cy_ * P[:, 0] - sy_ * P[:, 1]
        y = sy_ * P[:, 0] + cy_ * P[:, 1]
        u, v, z = -y, -x, P[:, 2]  # widok z góry: bok w poziomie, przód w górę
        v, z = ct * v - st * z, st * v + ct * z
        return cr * u - sr * v, sr * u + cr * v, z

    u, v, dz = view(V)
    bx, by, bw, bh = drone_box
    size = min(W * bw / (u.max() - u.min()), H * bh / (v.max() - v.min()))
    ou = W * (bx + bw / 2) - (u.min() + u.max()) / 2 * size
    ov = H * (by + bh / 2) - (v.min() + v.max()) / 2 * size

    def to_img(P):
        pu, pv, pz = view(P)
        return np.stack([ou + pu * size, ov + pv * size], 1).astype(np.float32), pz

    pts, dz = to_img(V)
    used = np.unique(F)

    # nitki: cienkie krzywe Béziera od punktów obwodu lotu do drona, kolor od grupy neuronów do akcentu
    red = np.array((255, 45, 111), np.float32)
    glow, line = np.zeros_like(img), np.zeros_like(img)
    fly_arr = np.array([(a, b) for a, b, _ in fly_pts], np.float32)
    fly_col = np.array([c for _, _, c in fly_pts], np.float32)
    src = rng.choice(len(fly_arr), n_threads, replace=False)
    near = used[dz[used] < np.percentile(dz[used], 55)]  # wierzchołki od strony patrzącego
    dst = rng.choice(near, n_threads, replace=False)
    t = np.linspace(0, 1, 120)[:, None]
    for i, j in zip(src, dst):
        p0, p3 = fly_arr[i], pts[j]
        dx = p3[0] - p0[0]
        sag = rng.uniform(-0.28, 0.22) * H
        p1 = p0 + np.array([dx * rng.uniform(0.25, 0.45), sag])
        p2 = p3 - np.array([dx * rng.uniform(0.25, 0.45), -sag * rng.uniform(0.2, 0.7)])
        curve = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        k = float(rng.uniform(0.6, 1.0))
        for s_ in range(len(curve) - 1):
            f = s_ / (len(curve) - 2)
            col = (1 - f) * fly_col[i] + f * red
            fade = 0.6 + 0.4 * abs(2 * f - 1) ** 1.5  # przy końcach jaśniej, w połowie drogi lekko przygaszone
            a, b = tuple(map(int, curve[s_])), tuple(map(int, curve[s_ + 1]))
            cv2.line(glow, a, b, tuple(float(x) * 0.22 * k * fade for x in col), 5, cv2.LINE_AA)
            cv2.line(line, a, b, tuple(float(x) * 0.85 * k * fade for x in col), 2, cv2.LINE_AA)
    img += cv2.GaussianBlur(glow, (0, 0), 4) + line

    # dron jak szkło (płaskie ściany X2 słabo łapią Fresnela, więc do tego cienka siatka krawędzi)
    tint = np.array((0.95, 0.84, 0.9), np.float32)
    _glass(img, pts, F, dz, tint, gain=320)
    wire = np.zeros(img.shape[:2], np.float32)
    cv2.polylines(wire, list(pts[F].astype(np.int32)), True, 1.0, 1, cv2.LINE_AA)
    img += np.minimum(wire, 3.0)[..., None] * tint * 30

    # śmigła: rozmyty krąg obrotu (cienki obrys, lekkie wypełnienie) i dwie zwężane łopaty
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    for c, phi in zip(props, rng.uniform(0, np.pi, len(props))):
        ring, _ = to_img(c + np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros_like(ang)], 1))
        disc = np.zeros(img.shape[:2], np.float32)
        cv2.fillPoly(disc, [ring.astype(np.int32)], 1.0, cv2.LINE_AA)
        img += cv2.GaussianBlur(disc, (0, 0), 6)[..., None] * tint * 9
        rim = np.zeros(img.shape[:2], np.float32)
        cv2.polylines(rim, [ring.astype(np.int32)], True, 1.0, 1, cv2.LINE_AA)
        img += rim[..., None] * tint * 45
        r = np.linspace(0.012, R * 0.96, 24)
        wdt = 0.004 + 0.016 * np.sin(np.pi * (r / R) ** 0.8) * (1 - 0.5 * r / R)
        for side in (0, np.pi):
            a0 = phi + side
            d, nrm = np.array([np.cos(a0), np.sin(a0), 0]), np.array([-np.sin(a0), np.cos(a0), 0])
            sweep = 0.25 * (r / R) ** 2  # łopata lekko wygięta do tyłu
            edge1 = c + r[:, None] * d + (wdt - sweep * wdt)[:, None] * nrm
            edge2 = c + r[:, None] * d - (wdt + sweep * wdt)[:, None] * nrm
            blade, _ = to_img(np.concatenate([edge1, edge2[::-1]]).astype(np.float32))
            m = np.zeros(img.shape[:2], np.float32)
            cv2.fillPoly(m, [blade.astype(np.int32)], 1.0, cv2.LINE_AA)
            img += m[..., None] * tint * 38
            cv2.polylines(m, [blade.astype(np.int32)], True, 1.0, 1, cv2.LINE_AA)
            img += (m > 0.99)[..., None] * tint * 25
    img = BG + (255 - BG) * (1 - np.exp(-img / 210.0))
    cv2.imwrite(str(OUT / name), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def style_assets(seed=7):
    """Elementy stylu (jak plakat „Aquire”): tło z szarymi, rozmytymi odłamkami i czerwono-pomarańczowa linia akcentu."""
    rng = np.random.default_rng(seed)
    for name, W, H, side in (("shards_title.png", 1920, 1080, "both"), ("shards_corner.png", 1920, 1080, "corner")):
        img = np.zeros((H, W, 3), np.float32) + BG
        sharp, soft = np.zeros_like(img), np.zeros_like(img)
        anchors = [(0.0, 1.0), (1.0, 0.0), (0.15, 0.85), (0.92, 0.2)] if side == "both" else [(1.0, 0.0), (0.9, 0.12)]
        for ax, ay in anchors:
            for _ in range(5):
                c = np.array([ax * W, ay * H]) + rng.normal(0, [W * 0.12, H * 0.15])
                pts = (c + rng.normal(0, [W * 0.16, H * 0.22], (3, 2))).astype(np.int32)
                tone = float(rng.uniform(22, 70))
                layer = soft if rng.random() < 0.45 else sharp
                cv2.fillConvexPoly(layer, pts, (tone, tone, tone * 1.04), cv2.LINE_AA)
                if rng.random() < 0.5:  # jasna krawędź odłamka
                    a, b = pts[rng.integers(3)], pts[rng.integers(3)]
                    cv2.line(sharp, tuple(map(int, a)), tuple(map(int, b)), (95, 95, 100), 2, cv2.LINE_AA)
        # rozmyty snop światła jak na plakacie
        beam = np.zeros_like(img)
        x0 = int(W * (0.62 if side == "both" else 0.8))
        cv2.fillConvexPoly(beam, np.array([[x0, int(H * 0.25)], [x0 + 60, int(H * 0.22)], [x0 + 420, int(H * 0.62)], [x0 + 330, int(H * 0.68)]]),
                           (150, 150, 155), cv2.LINE_AA)
        img = np.maximum(img, sharp) + cv2.GaussianBlur(soft, (0, 0), 18) * 0.9 + cv2.GaussianBlur(beam, (0, 0), 28) * 0.55
        vign = np.exp(-(((np.arange(W) - W / 2) / (W * 0.75)) ** 2))[None, :, None] * np.exp(
            -(((np.arange(H) - H / 2) / (H * 0.9)) ** 2))[:, None, None]
        img = BG + (img - BG) * (0.55 + 0.45 * (1 - vign))  # środek ciemniejszy — tu idzie treść
        cv2.imwrite(str(OUT / name), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # linia akcentu: róż → pomarańcz, z poświatą (pionowa)
    H, W = 900, 40
    t = np.linspace(0, 1, H)[:, None]
    col = (1 - t) * np.array([255, 45, 111]) + t * np.array([255, 106, 43])
    line = np.zeros((H, W, 4), np.float32)
    xs = np.arange(W) - W / 2
    core = (np.abs(xs) <= 1.5).astype(np.float32)
    glow = np.exp(-(xs / 6.0) ** 2) * 0.35
    a = np.clip(core + glow, 0, 1)[None, :] * np.ones((H, 1))
    line[..., :3] = col[:, None, :]
    line[..., 3] = a * 255
    cv2.imwrite(str(OUT.parent / "accent_line.png"), cv2.cvtColor(line.astype(np.uint8), cv2.COLOR_RGBA2BGRA))
    # podwójne konturowe trójkąty (wierzchołkiem w dół), białe, półprzezroczyste
    ch = np.zeros((240, 360, 4), np.uint8)
    for off, w in ((0, 300), (34, 230)):
        cx, top = 180, 20 + off
        pts = np.array([[cx - w // 2, top], [cx + w // 2, top], [cx, top + int(w * 0.62)]], np.int32)
        cv2.polylines(ch, [pts], True, (235, 235, 240, 190), 4, cv2.LINE_AA)
    cv2.imwrite(str(OUT.parent / "chevron.png"), cv2.cvtColor(ch, cv2.COLOR_RGBA2BGRA))


def wiring(W=1400, H=1700, yaw=0.55, pitch=0.25, n_edges=60000, name="connectome_wiring.png", labels=None, seed=0, horizontal=False):
    """Somy + prawdziwe połączenia BANC v888 (edgelist v2, count >= 5, bez autapsów) jako linie soma → soma.
    Krawędzie losowane z wagą liczby synaps; kolor = klasa neuronu presynaptycznego. Linia prosta między somami
    jest uproszczeniem rysunku (prawdziwe aksony biegną inaczej), samo połączenie jest z BANC."""
    import pandas as pd

    from banc_control.connectome import DEFAULT_DATA_DIR, EDGES_FILE, META_FILE, NON_NEURONS

    meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE, columns=["banc_888_id", "position", "super_class"])
    meta = meta[~meta["super_class"].isin(NON_NEURONS) & meta["position"].notna()]
    pos = np.array([[float(v) for v in p.split(",")] for p in meta["position"]]) * np.array([4.0, 4.0, 45.0]) / 1000.0
    lo, hi = np.percentile(pos, 0.05, axis=0), np.percentile(pos, 99.95, axis=0)
    ok = np.all((pos >= lo) & (pos <= hi), axis=1)
    idx = pd.Series(np.arange(ok.sum()), index=meta["banc_888_id"].to_numpy()[ok])
    pos, sc = pos[ok].astype(np.float32), meta["super_class"].fillna("unknown").to_numpy()[ok]
    e = pd.read_feather(DEFAULT_DATA_DIR / EDGES_FILE, columns=["pre", "post", "count"])
    e = e[(e["count"] >= 5) & (e["pre"] != e["post"]) & e["pre"].isin(idx.index) & e["post"].isin(idx.index)]
    rng = np.random.default_rng(seed)
    p = e["count"].to_numpy(float)
    pick = rng.choice(len(e), size=min(n_edges, len(e)), replace=False, p=p / p.sum())
    a, b = idx[e["pre"].to_numpy()[pick]].to_numpy(), idx[e["post"].to_numpy()[pick]].to_numpy()
    print(f"krawędzie: {len(e):,} w grafie, narysowane {len(pick):,}")

    img = np.zeros((H, W, 3), np.float32)
    u, v, _ = project(xyz, yaw, pitch, 1.0, 0, 0)
    u, v, tf = fit(u, v, W, H, margin_x=None if horizontal else 0.12, horizontal=horizontal)
    col = np.array([CLASS_COLOR[c] for c in cls], np.float32)
    splat(img, u, v, col, np.full(len(u), 0.16), r=1)
    pu, pv, _ = project(pos, yaw, pitch, 1.0, 0, 0)
    pu, pv = tf(pu, pv)
    name_to_idx = {c: i for i, c in enumerate(classes)}
    ecol = np.array([CLASS_COLOR.get(name_to_idx.get(s, -1), SLATE) for s in sc[a]], np.float32)
    layer = np.zeros_like(img)
    for k in range(len(a)):
        cv2.line(layer, (int(pu[a[k]]), int(pv[a[k]])), (int(pu[b[k]]), int(pv[b[k]])), tuple(float(x) * 0.16 for x in ecol[k]), 1, cv2.LINE_AA)
    img = BG + img + layer * 1.3 + cv2.GaussianBlur(layer, (0, 0), 4) * 0.8
    img = BG + (255 - BG) * (1 - np.exp(-(img - BG) / 120.0))
    out = np.clip(img, 0, 255).astype(np.uint8)
    if labels:
        out = draw_labels(out, labels, tf, yaw, pitch)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def thumbs():
    """Miniatury do pipeline'u: płat wzrokowy (to, co modeluje FlyVis), wejście wzroku w BANC, wagi dekodera yaw."""
    # płat wzrokowy z bliska: tylko somy optic_lobe, prawa strona muchy
    ol = (region == regions.index("optic_lobe")) & (xyz[:, 0] < center[0])
    for name, mask, dim in (("thumb_flyvis.png", ol, False), ("thumb_map.png", None, True)):
        W, H = 900, 600
        img = np.zeros((H, W, 3), np.float32)
        u, v, z = project(xyz, 0.55, 0.25, 1.0, 0, 0)
        sel = mask if mask is not None else np.ones(len(u), bool)
        uu, vv, _ = fit(u[sel], v[sel], W, H, margin=0.05)
        col = np.array([CLASS_COLOR[c] for c in cls[sel]], np.float32)
        al = np.full(sel.sum(), 0.12)
        if dim:  # wejście wzroku: płaty wzrokowe jasno, reszta przygaszona
            teal = np.all(col == TEAL, axis=1)
            al = np.where(teal, 0.1, 0.03)
        splat(img, uu, vv, col, al, r=2)
        img = BG + (255 - BG) * (1 - np.exp(-img / 160.0))
        cv2.imwrite(str(OUT / name), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # wagi yaw najlepszego dekodera świata (pojedyncze DN lotu), posortowane
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = np.load(ROOT / "data" / "decoders" / "world_distributed_best.npz")["w_yaw"][6:-1]  # bez 6 średnich MN i wyrazu wolnego
    w = np.sort(w)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    fig.patch.set_facecolor("#09090B")
    ax.set_facecolor("#09090B")
    ax.bar(np.arange(len(w)), w, width=1.0, color=np.where(w > 0, "#FF2D6F", "#FFB547"))
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "thumb_decoder.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def finish():
    """PNG → JPG dla prezentacji, logo z eksploratora, dane wykresu ewaluacji."""
    import shutil

    dst = OUT.parent
    for f in OUT.glob("*.png"):
        im = cv2.imread(str(f))
        if f.stem.startswith("thumb_") and f.stem != "thumb_decoder":
            # miniatury: kadr 16:9 wokół treści (render zostawia dużo pustego tła)
            ys, xs = np.where(im.astype(int).sum(2) > 220)
            y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int)
            x0, x1 = np.percentile(xs, [0.5, 99.5]).astype(int)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            w = max(x1 - x0, (y1 - y0) * 16 // 9) + 40
            h = w * 9 // 16
            pad = cv2.copyMakeBorder(im, h, h, w, w, cv2.BORDER_CONSTANT, value=(11, 9, 9))
            im = cv2.resize(pad[cy + h - h // 2:cy + h + h // 2, cx + w - w // 2:cx + w + w // 2], (800, 450), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / f"{f.stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    shutil.copy(ROOT / "explorer" / "public" / "logo-fly@2x.png", dst / "logo.png")
    d = json.loads((ROOT / "data" / "decoders" / "planB_distributed.json").read_text(encoding="utf-8"))
    keys = ("-60", "-30", "+30", "+60")
    # lot do celu w świecie (train_distributed.py --world): walidacja co 50 epizodów i bloki historii treningu
    w = json.loads((ROOT / "data" / "decoders" / "world_distributed.json").read_text(encoding="utf-8"))
    hist = w["history"]
    blocks = []
    for a in range(0, len(hist), 50):
        s = [e for e in hist if a <= e["index"] < a + 50]
        blocks.append({"from": a, "n": len(s), "reached": sum(e["reached"] for e in s),
                       "beta": float(np.mean([e["beta"] for e in s])), "min_dist": float(np.mean([e["min_dist"] for e in s]))})
    model_only = [e for e in hist if e["beta"] == 0.0]
    world = {"evals": [{k: e[k] for k in ("tag", "after_episodes", "reached", "n", "mean_min_dist")} for e in w["evals"]],
             "best": w["best"], "blocks": blocks,
             "workers": {k: {"episodes": v["episodes"], "gpu": v["gpu"]} for k, v in w["workers"].items()},
             "episodes": len(hist), "reached_all": sum(e["reached"] for e in hist),
             "flips": sum(e["outcome"].startswith("wywrotka") for e in hist),
             "model_only": {"n": len(model_only), "reached": sum(e["reached"] for e in model_only)},
             "args": {k: w["args"][k] for k in ("beacon_scale", "vision_range", "sensors", "motor_tau", "max_time", "readout")}}
    (dst / "data.json").write_text(json.dumps({"before": {k: d["before"][k]["final_deg"] for k in keys},
                                               "after": {k: d["after"][k]["final_deg"] for k in keys},
                                               "world": world}, indent=1))



if __name__ == "__main__":
    what = sys.argv[1:] or ["connectome", "wiring", "thumbs", "circuit", "flybody", "style", "sim", "retina", "finish"]
    if "connectome" in what:
        connectome(2600, 1300, yaw=0.5, pitch=0.25, name="connectome_wide.png", horizontal=True)
        connectome(1400, 1700, yaw=0.55, pitch=0.25, name="connectome_3d.png", labels=REGION_LABELS, margin_x=0.12)
    if "wiring" in what:
        wiring(labels=REGION_LABELS)
    if "thumbs" in what:
        thumbs()
        wiring(1200, 700, n_edges=40000, name="thumb_wiring.png", horizontal=True)
        circuit(1200, 700, horizontal=True, name="thumb_circuit.png")
    if "circuit" in what:
        circuit(1400, 1700, labels=CIRCUIT_LABELS, margin_x=0.12)
    if "style" in what:
        style_assets()
    if "flybody" in what:
        fly_xray()
        fly_drone()
    if "sim" in what:
        sim_images()
    if "retina" in what:
        retina()
    if "finish" in what:
        finish()
    print("ok", sorted(p.name for p in OUT.iterdir()))
