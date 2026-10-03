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
    ("Płat wzrokowy P", "prawe oko · wejście z FlyVis", _c(_ol & (xyz[:, 0] < center[0])), TEAL, "L"),
    ("Płat wzrokowy L", "lewe oko · tylko 36% typów w v888", _c(_ol & (xyz[:, 0] >= center[0])), TEAL, "R"),
    ("Mózg centralny", "integracja zmysłów · somy DN", _cb - np.array([0, 60, 0]), VIOLET, "L"),
    ("Szyja", "aksony DN: mózg → VNC", np.array([_cb[0], NECK_Y, _cb[2]]), AMBER, "R"),
    ("VNC", "motoneurony skrzydeł, nóg i halter", _vnc + np.array([0, 80, 0]), PINK, "R"),
]
CIRCUIT_LABELS = [
    ("DN lotu", "odczyt kursu (yaw) z pojedynczych DN", _dn.mean(0), AMBER, "L"),
    ("Motoneurony skrzydeł", "moc, sterowanie, napięcie → dron", _mn.mean(0), PINK, "L"),
    ("Aferenty halter", "czujniki obrotu ← żyroskop drona", _skel["haltere_aff"].mean(0), BLUE, "R"),
    ("Szyja", "jedyna droga komend do skrzydeł", np.array([_cb[0], NECK_Y, _cb[2]]), (250, 250, 250), "R"),
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
    """Etykiety jak w eksploratorze: kropka na kotwicy, linia do boku, nazwa + krótki opis (Roboto).
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
        r = size * 0.22
        dr.ellipse([ax - r, ay - r, ax + r, ay + r], fill=col, outline=(9, 9, 11), width=3)
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


def circuit(W=2000, H=1500, yaw=0.55, pitch=0.28, horizontal=False, labels=None, margin_x=None):
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
    cv2.imwrite(str(OUT / "circuit_3d.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


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


def fly_xray(W=2600, H=1500, yaw=0.15, pitch=0.22, name="fly_xray.png"):
    """Półprzezroczyste ciało muszki (NeuroMechFly, scripts/export_fly_body.py) z connectomem BANC w środku:
    ciało jak szkło (jaśniejsze krawędzie — Fresnel), somy w kolorach klas, obwód lotu z poświatą.
    Dopasowanie ciała do BANC jest ilustracyjne (głowa → mózg, tułów → VNC)."""
    meta = json.loads((ROOT / "data" / "viz" / "fly_body.json").read_text())
    raw = (ROOT / "data" / "viz" / "fly_body.bin").read_bytes()
    n, m = meta["vertices"], meta["triangles"]
    bpos = np.frombuffer(raw, np.float32, n * 3).reshape(n, 3)
    tri = np.frombuffer(raw, np.uint32, m * 3, offset=n * 12).reshape(m, 3)
    part = np.frombuffer(raw, np.uint8, n, offset=n * 12 + m * 12)

    bu0, bv0, bz = project(bpos, yaw, pitch, 1.0, 0, 0)
    bu, bv, tf = fit(bu0, bv0, W, H, margin=0.04)
    img = np.zeros((H, W, 3), np.float32)

    # ciało: normalne ścian w układzie widoku → Fresnel; ściany pogrupowane w poziomy jasności (szybkie fillPoly)
    q3 = np.stack([bu, bv, bz], 1)
    a, b, c = q3[tri[:, 0]], q3[tri[:, 1]], q3[tri[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    fres = (1 - np.abs(nrm[:, 2])) ** 2.2
    wing = part[tri[:, 0]] == 3
    alpha = (0.025 + 0.42 * fres) * np.where(wing, 0.55, 1.0)
    pts2 = np.stack([bu, bv], 1)[tri].astype(np.int32)
    levels = np.linspace(alpha.min(), alpha.max(), 14)
    idx = np.clip(np.digitize(alpha, levels) - 1, 0, len(levels) - 1)
    body = np.zeros((H, W), np.float32)
    for k in range(len(levels)):
        sel = idx == k
        if not sel.any():
            continue
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, list(pts2[sel]), 1, cv2.LINE_AA)
        body += mask.astype(np.float32) * levels[k] * 120
    img += body[..., None] * np.array([0.82, 0.84, 0.9], np.float32)

    # somy (przygaszone, w kolorach klas)
    su, sv, _ = project(xyz, yaw, pitch, 1.0, 0, 0)
    su, sv = tf(su, sv)
    col = np.array([CLASS_COLOR[c] for c in cls], np.float32)
    splat(img, su, sv, col, np.full(len(su), 0.06), r=1)

    # obwód lotu: poświata (szeroko, rozmyte) + cienka jasna linia
    glow, line = np.zeros_like(img), np.zeros_like(img)
    for nr in d["neurons"]:
        cc = GROUP_COLOR.get(nr["group"].rsplit("_", 1)[0], (161, 161, 170))
        for ln in nr["lines"]:
            pu, pv, _ = project(np.array(ln, np.float32), yaw, pitch, 1.0, 0, 0)
            pu, pv = tf(pu, pv)
            pts = np.stack([pu, pv], 1).astype(np.int32)
            cv2.polylines(glow, [pts], False, tuple(float(x) * 0.25 for x in cc), 6, cv2.LINE_AA)
            cv2.polylines(line, [pts], False, tuple(float(x) * 0.8 for x in cc), 2, cv2.LINE_AA)
    img += cv2.GaussianBlur(glow, (0, 0), 6) + line
    img = BG + (255 - BG) * (1 - np.exp(-img / 210.0))
    out = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def finish():
    """PNG → JPG dla prezentacji, logo z eksploratora, dane wykresu ewaluacji."""
    import shutil

    dst = OUT.parent
    for f in OUT.glob("*.png"):
        cv2.imwrite(str(dst / f"{f.stem}.jpg"), cv2.imread(str(f)), [cv2.IMWRITE_JPEG_QUALITY, 90])
    shutil.copy(ROOT / "explorer" / "public" / "logo-fly@2x.png", dst / "logo.png")
    d = json.loads((ROOT / "data" / "decoders" / "planB_distributed.json").read_text(encoding="utf-8"))
    keys = ("-60", "-30", "+30", "+60")
    (dst / "data.json").write_text(json.dumps({"before": {k: d["before"][k]["final_deg"] for k in keys},
                                               "after": {k: d["after"][k]["final_deg"] for k in keys}}, indent=1))



if __name__ == "__main__":
    what = sys.argv[1:] or ["connectome", "circuit", "flybody", "sim", "retina", "finish"]
    if "connectome" in what:
        connectome(2600, 1300, yaw=0.5, pitch=0.25, name="connectome_wide.png", horizontal=True)
        connectome(1400, 1700, yaw=0.55, pitch=0.25, name="connectome_3d.png", labels=REGION_LABELS, margin_x=0.12)
    if "circuit" in what:
        circuit(1400, 1700, labels=CIRCUIT_LABELS, margin_x=0.12)
    if "flybody" in what:
        fly_xray()
    if "sim" in what:
        sim_images()
    if "retina" in what:
        retina()
    if "finish" in what:
        finish()
    print("ok", sorted(p.name for p in OUT.iterdir()))
