"""Intro do demo (pod pierwsze zdania lektora): półprzezroczysta muszka (NeuroMechFly) z układem nerwowym BANC v888,
który zapala się falą od głowy do VNC, najazd kamery na głowę, potem odjazd i garść ziaren maku obok głowy w skali.

    .venv312/Scripts/python docs/prezentacja/render_intro.py                 # → data/videos/intro_poppy.mp4
    .venv312/Scripts/python docs/prezentacja/render_intro.py --stills 2,8,15  # podgląd pojedynczych klatek (PNG)

Z BANC: pozycje som i szkielety neuronów lotu (µm). Ilustracyjne: dopasowanie ciała do BANC (głowa → mózg),
ziarna maku (proceduralne, ~1,1 × 0,85 mm, w tej samej skali µm co somy), kamera i efekt „zapalania”.
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # procesy puli liczą równolegle, wątki BLAS tylko by się biły
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from render_assets import BG, CLASS_COLOR, FONT_B, FONT_R, GROUP_COLOR, NECK_Y, ROOT, cls, d, xyz  # noqa: E402

W, H, FPS, DUR = 1920, 1080, 30, 18.0
F = (W / 2) / np.tan(np.radians(36 / 2))  # ogniskowa w px, poziome pole widzenia 36°
UP = np.array([0.0, 0.0, -1.0])  # grzbiet muszki = −z (nogi mają duże z), oś ciała = y (głowa przy małym y)

# --- dane ---------------------------------------------------------------------------------------------------------
_meta = __import__("json").loads((ROOT / "data" / "viz" / "fly_body.json").read_text())
_raw = (ROOT / "data" / "viz" / "fly_body.bin").read_bytes()
_n, _m = _meta["vertices"], _meta["triangles"]
BODY = np.frombuffer(_raw, np.float32, _n * 3).reshape(_n, 3).astype(np.float64)
TRI = np.frombuffer(_raw, np.uint32, _m * 3, offset=_n * 12).reshape(_m, 3)
PART = np.frombuffer(_raw, np.uint8, _n, offset=_n * 12 + _m * 12)[TRI[:, 0]]
# tinty części: ciało szkło, oczy lekko czerwone (żeby było widać muchę), skrzydła słabiej
PART_TINT = {0: ((0.82, 0.84, 0.9), 1.0), 1: ((1.0, 0.5, 0.45), 1.0), 2: ((0.82, 0.84, 0.9), 0.8),
             3: ((0.82, 0.84, 0.9), 0.5), 4: ((0.82, 0.84, 0.9), 0.9)}

SOMA = xyz.astype(np.float64)
SOMA_COL = np.array([CLASS_COLOR[c] for c in cls], np.float32)
_y0, _y1 = np.percentile(SOMA[:, 1], [1, 99])
_rng = np.random.default_rng(1)
SOMA_TON = 1.0 + 3.0 * np.clip((SOMA[:, 1] - _y0) / (_y1 - _y0), 0, 1) + _rng.normal(0, 0.12, len(SOMA))

# szkielety neuronów lotu: wszystkie punkty razem, linie jako zakresy, 8 koszy czasu zapalenia (wg średniego y)
_pts, _rngs, _bins = [], [], []
for nr in d["neurons"]:
    col = GROUP_COLOR.get(nr["group"].rsplit("_", 1)[0], (161, 161, 170))
    for ln in nr["lines"]:
        a = np.asarray(ln, np.float64)
        if len(a) < 2:
            continue
        _rngs.append((sum(len(p) for p in _pts), len(a), col))
        _pts.append(a)
SKEL = np.concatenate(_pts)
SKEL_TON = [3.2 + 2.0 * np.clip((SKEL[s:s + k, 1].mean() - _y0) / (_y1 - _y0), 0, 1) for s, k, _ in _rngs]

HEAD = SOMA[SOMA[:, 1] < NECK_Y].mean(0)
FLY_C = np.array([(BODY[:, 0].min() + BODY[:, 0].max()) / 2, 1000.0, 250.0])


# --- kamera -------------------------------------------------------------------------------------------------------
def camera(target, dist, az, el):
    off = np.cos(el) * np.array([np.cos(az), np.sin(az), 0.0]) + np.sin(el) * UP
    pos = target + dist * off
    fwd = -off
    right = np.cross(fwd, UP)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, np.stack([right, up, fwd])


def to_cam(P, cam):
    pos, R = cam
    return (P - pos) @ R.T


def to_px(q):
    z = np.maximum(q[..., 2], 1.0)
    return W / 2 + F * q[..., 0] / z, H / 2 - F * q[..., 1] / z


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a + (b - a) * x


# ziarna maku: ułożone w płaszczyźnie widoku końcowej kamery, z prawej strony głowy
AZ_HEAD, EL = np.radians(-40), np.radians(16)
AZ_END = np.radians(-52)
_cam_end = camera(HEAD, 1.0, AZ_END, EL)
_r, _u = _cam_end[1][0], _cam_end[1][1]
_offs = [(0.0, 0.0), (0.95, 0.35), (0.25, 1.0), (-0.2, -0.95), (0.9, -0.6), (1.6, 0.6), (1.45, -0.25)]
SEED_C = HEAD + _r * 1550 + _u * (-60)
SEEDS = []
for i, (a, b) in enumerate(_offs):
    rs = np.random.default_rng(10 + i)
    ang = rs.uniform(0, 2 * np.pi, 3)
    cx, sx = np.cos(ang), np.sin(ang)
    Rx = np.array([[1, 0, 0], [0, cx[0], -sx[0]], [0, sx[0], cx[0]]])
    Ry = np.array([[cx[1], 0, sx[1]], [0, 1, 0], [-sx[1], 0, cx[1]]])
    Rz = np.array([[cx[2], -sx[2], 0], [sx[2], cx[2], 0], [0, 0, 1]])
    s = rs.uniform(0.92, 1.08)
    SEEDS.append(dict(
        c=SEED_C + _r * a * 780 + _u * b * 640 + _cam_end[1][2] * rs.uniform(-250, 250),
        R=Rz @ Ry @ Rx, rad=np.array([560, 420, 300]) * s, bend=0.3,
        t0=10.0 + 0.2 * i, spin=rs.uniform(-2.5, 2.5, 3),
        vor=rs.normal(size=(420, 3)), seed=i))
for s in SEEDS:
    s["vor"] /= np.linalg.norm(s["vor"], axis=1, keepdims=True)
_ext = np.array([to_cam(s["c"], _cam_end)[:2] for s in SEEDS] + [to_cam(HEAD, _cam_end)[:2]])
_mid_q = (_ext.min(0) + _ext.max(0)) / 2
END_TARGET = HEAD + _r * (_mid_q[0] + 120) + _u * _mid_q[1]
_span = (_ext[:, 0].max() - _ext[:, 0].min()) + 1500
END_DIST = _span * 1.18 * F / W


def cam_at(t):
    if t < 4.0:
        x = t / 4.0
        return camera(FLY_C, lerp(7800, 7300, x), np.radians(lerp(18, 4, x)), np.radians(24))
    if t < 9.5:
        x = ease((t - 4.0) / 5.5)
        return camera(lerp(FLY_C, HEAD, x), np.exp(lerp(np.log(7300), np.log(2250), x)),
                      lerp(np.radians(4), AZ_HEAD, x), lerp(np.radians(24), EL, x))
    if t < 12.8:
        x = ease((t - 9.5) / 3.3)
        return camera(lerp(HEAD, END_TARGET, x), np.exp(lerp(np.log(2250), np.log(END_DIST), x)),
                      lerp(AZ_HEAD, AZ_END, x), EL)
    x = (t - 12.8) / (DUR - 12.8)
    return camera(END_TARGET, END_DIST * (1 - 0.04 * x), AZ_END - np.radians(5) * x, EL)


# --- render -------------------------------------------------------------------------------------------------------
def glass(acc, q, gain):
    u, v = to_px(q)
    pix = np.stack([u, v], 1)
    a, b, c = q[TRI[:, 0]], q[TRI[:, 1]], q[TRI[:, 2]]
    ok = (a[:, 2] > 20) & (b[:, 2] > 20) & (c[:, 2] > 20)
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    view = (a + b + c) / 3
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    fres = (1 - np.abs((nrm * view).sum(1))) ** 2.2
    p2 = np.round(pix[TRI] * 16).astype(np.int32)
    for part, (tint, wm) in PART_TINT.items():
        sel = ok & (PART == part)
        if not sel.any():
            continue
        alpha = (0.025 + 0.42 * fres[sel]) * wm
        levels = np.linspace(0.025 * wm, 0.445 * wm, 14)
        idx = np.clip(np.digitize(alpha, levels) - 1, 0, 13)
        layer = np.zeros((H, W), np.float32)
        ps = p2[sel]
        for k in range(14):
            kk = idx == k
            if not kk.any():
                continue
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, list(ps[kk]), 255, cv2.LINE_AA, shift=4)
            layer += mask.astype(np.float32) * (levels[k] * gain / 255)
        acc += layer[..., None] * np.array(tint, np.float32)


def seed_sdf(p, s):
    a, b, c = s["rad"]
    q = p.copy()
    q[..., 1] += s["bend"] * q[..., 0] ** 2 / a  # kształt nerki: wygięcie w płaszczyźnie dwóch dłuższych osi
    r = np.array([a, b, c])
    k0 = np.linalg.norm(q / r, axis=-1)
    k1 = np.linalg.norm(q / r ** 2, axis=-1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


def draw_seed(img, s, cam, t):
    """Ziarno maku: raymarching zgiętej elipsoidy, siateczka komórek (Voronoi) na powierzchni, krawędzie z pokrycia SDF."""
    x = np.clip((t - s["t0"]) / 1.3, 0, 1)
    if x <= 0:
        return
    fall = 1 - (1 - x) ** 3
    pos, Rc = cam
    up_w = Rc[1]
    center = s["c"] + up_w * 2600 * (1 - fall)
    ang = s["spin"] * (1 - fall) * 2.0
    ca, sa = np.cos(ang), np.sin(ang)
    Rs = (np.array([[ca[2], -sa[2], 0], [sa[2], ca[2], 0], [0, 0, 1]]) @
          np.array([[1, 0, 0], [0, ca[0], -sa[0]], [0, sa[0], ca[0]]])) @ s["R"]
    rmax = s["rad"].max() * 1.35
    qc = to_cam(center[None], cam)[0]
    if qc[2] < rmax:
        return
    cu, cv_ = to_px(qc)
    rp = F * rmax / qc[2]
    u0, u1 = int(max(cu - rp, 0)), int(min(cu + rp, W - 1))
    v0, v1 = int(max(cv_ - rp, 0)), int(min(cv_ + rp, H - 1))
    if u1 <= u0 or v1 <= v0:
        return
    k = max(1, int(np.ceil(2 * rp / 480)))  # duże ziarno (blisko kamery): liczymy rzadziej i skalujemy w górę
    uu, vv = np.meshgrid(np.arange(u0, u1 + 1, k) + k / 2, np.arange(v0, v1 + 1, k) + k / 2)
    dcam = np.stack([(uu - W / 2) / F, -(vv - H / 2) / F, np.ones_like(uu)], -1)
    dcam /= np.linalg.norm(dcam, axis=-1, keepdims=True)
    dw = dcam @ Rc
    o = Rs.T @ (pos - center)
    dl = dw @ Rs  # = (Rs.T @ dw.T).T
    tt = np.full(uu.shape, np.linalg.norm(o) - rmax)
    tmax = np.linalg.norm(o) + rmax
    mind = np.full(uu.shape, 1e9)
    for _ in range(48):
        p = o + dl * tt[..., None]
        sd = seed_sdf(p, s)
        mind = np.minimum(mind, sd)
        tt = np.minimum(tt + np.clip(sd * 0.6, 0.5, None), tmax)
    p = o + dl * tt[..., None]
    sd = seed_sdf(p, s)
    pixw = tt * k / F
    hit = (sd < pixw) & (tt < tmax)
    cov = np.where(hit, 1.0, np.clip(1 - mind / pixw, 0, 1))
    if not (cov > 0).any():
        return
    e = 2.0
    nrm = np.stack([seed_sdf(p + np.array(dv) * e, s) - seed_sdf(p - np.array(dv) * e, s)
                    for dv in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
    nw = nrm @ Rs.T  # normalne w świecie
    nc = nw @ Rc.T  # i w kamerze
    # siateczka: odległość do dwóch najbliższych punktów na sferze (lokalnie, więc wzór obraca się z ziarnem)
    sph = p / s["rad"]
    sph /= np.linalg.norm(sph, axis=-1, keepdims=True) + 1e-9
    edge = np.ones(sph.shape[:2])
    on = cov > 0
    dots = np.partition(sph[on] @ s["vor"].T, -2, axis=1)
    edge[on] = dots[:, -1] - dots[:, -2]
    ridge = np.exp(-(edge / 0.012) ** 2)
    base = lerp(np.array([58, 66, 84], np.float32), np.array([128, 138, 158], np.float32), ridge[..., None] * 0.85)
    vdir = -dcam
    L = np.array([-0.45, 0.65, -0.6])
    L /= np.linalg.norm(L)
    lam = np.clip((nc * L).sum(-1), 0, 1)
    hlf = L + vdir
    hlf /= np.linalg.norm(hlf, axis=-1, keepdims=True)
    spec = np.clip((nc * hlf).sum(-1), 0, 1) ** 40 * (0.12 + 0.3 * ridge)
    rim = (1 - np.clip((nc * vdir).sum(-1), 0, 1)) ** 3
    col = base * (0.25 + 0.85 * lam[..., None]) + spec[..., None] * 160 + rim[..., None] * np.array([40, 70, 90])
    col = np.clip(col, 0, 255).astype(np.float32)
    if k > 1:
        size = (u1 - u0 + 1, v1 - v0 + 1)
        col = cv2.resize(col, size, interpolation=cv2.INTER_LINEAR)
        cov = cv2.GaussianBlur(cv2.resize(cov.astype(np.float32), size, interpolation=cv2.INTER_LINEAR), (0, 0), k * 0.6)
    a = cov[..., None]
    sub = img[v0:v1 + 1, u0:u1 + 1]
    img[v0:v1 + 1, u0:u1 + 1] = sub * (1 - a) + col * a


def frame(t):
    cam = cam_at(t)
    dist = np.linalg.norm(cam[0] - to_world_target(cam))
    acc = np.zeros((H, W, 3), np.float32)

    # ciało
    gb = ease(t / 1.4) * (1 - 0.45 * ease((7300 - dist) / 4500))  # przy głowie szkło ciszej, mózg widać lepiej
    if gb > 0:
        glass(acc, to_cam(BODY, cam), 120 * gb)

    # somy: fala zapalania od głowy do VNC, z błyskiem na froncie
    q = to_cam(SOMA, cam)
    su, sv = to_px(q)
    on = ease((t - SOMA_TON) / 0.5)
    flash = np.exp(-((t - SOMA_TON - 0.15) / 0.22) ** 2)
    a = (on * 0.75 + flash * 1.4) * 0.30 * (2250 / dist) ** 1.7
    a *= q[:, 2] > 20
    pts = np.zeros_like(acc)
    ui, vi = su.astype(int), sv.astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (a > 1e-4)
    np.add.at(pts, (vi[ok], ui[ok]), SOMA_COL[ok] * a[ok, None])
    acc += pts + cv2.GaussianBlur(pts, (0, 0), 4) * 1.6

    # obwód lotu (szkielety z BANC): poświata + linia
    qs = to_cam(SKEL, cam)
    ku, kv = to_px(qs)
    P = np.round(np.stack([ku, kv], 1) * 16).astype(np.int32)
    glow, line = np.zeros_like(acc), np.zeros_like(acc)
    thick = 2 if dist < 4000 else 1
    for (s0, k, col), ton in zip(_rngs, SKEL_TON):
        al = ease((t - ton) / 0.8)
        if al <= 0 or qs[s0:s0 + k, 2].min() < 20:
            continue
        c = np.array(col, np.float32) * al
        cv2.polylines(glow, [P[s0:s0 + k]], False, tuple(float(x) * 0.18 for x in c), 5 * thick, cv2.LINE_AA, shift=4)
        cv2.polylines(line, [P[s0:s0 + k]], False, tuple(float(x) * 0.55 for x in c), thick, cv2.LINE_AA, shift=4)
    acc += cv2.GaussianBlur(glow, (0, 0), 5) + line

    img = BG + (255 - BG) * (1 - np.exp(-acc / 210.0))

    for s in sorted(SEEDS, key=lambda s: -to_cam(s["c"][None], cam)[0, 2]):
        draw_seed(img, s, cam, t)

    img *= ease(t / 0.8)  # wejście z czerni
    img = np.clip(img, 0, 255).astype(np.uint8)
    return labels(img, cam, t)


def to_world_target(cam):
    """Punkt, na który patrzy kamera, na wysokości głowy (do skali jasności som)."""
    pos, R = cam
    return pos + R[2] * np.dot(HEAD - pos, R[2])


def labels(img, cam, t):
    al = ease((t - 13.2) / 0.8)
    if al <= 0:
        return img
    im = Image.fromarray(img)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    fb, fr = ImageFont.truetype(FONT_B, 34), ImageFont.truetype(FONT_R, 24)
    A = int(255 * al)

    def tag(anchor, dx, dy, title, sub):
        u, v = to_px(to_cam(anchor[None], cam))
        u, v = float(u[0]), float(v[0])
        x, y = min(max(u + dx, 40), W - 520), min(max(v + dy, 70), H - 120)  # napis zawsze w kadrze
        dr.line([(u, v), (x + 20, y + 44)], fill=(250, 250, 250, int(A * 0.6)), width=2)
        dr.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(250, 250, 250, A))
        dr.text((x, y), title, font=fb, fill=(250, 250, 250, A), anchor="lb")
        dr.text((x, y + 8), sub, font=fr, fill=(170, 170, 180, A), anchor="lt")

    pos, R = cam
    tag(HEAD, -200, -300, "Fruit fly brain", "BANC v888, 175,401 neurons")
    top = max(SEEDS, key=lambda s: np.dot(s["c"], R[1]))
    tag(top["c"] + R[1] * 250, 160, -120, "Poppy seeds", "about 1 mm each")

    # pasek skali 1 mm na głębokości głowy
    z = to_cam(HEAD[None], cam)[0, 2]
    L = F * 1000 / z
    x1, y1 = W - 110, H - 90
    x0 = x1 - L
    dr.line([(x0, y1), (x1, y1)], fill=(250, 250, 250, A), width=3)
    for xx in (x0, x1):
        dr.line([(xx, y1 - 9), (xx, y1 + 9)], fill=(250, 250, 250, A), width=3)
    dr.text(((x0 + x1) / 2, y1 - 16), "1 mm", font=fr, fill=(250, 250, 250, A), anchor="mb")
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", help="czasy w sekundach, np. 2,8,15 → PNG zamiast wideo")
    ap.add_argument("--out", default=str(ROOT / "data" / "videos" / "intro_poppy.mp4"))
    ap.add_argument("--still-dir", default=str(ROOT / "data" / "videos" / "intro_stills"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.stills:
        out = Path(args.still_dir)
        out.mkdir(parents=True, exist_ok=True)
        for ts in args.stills.split(","):
            f = frame(float(ts))
            cv2.imwrite(str(out / f"t{float(ts):05.2f}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            print("still", ts)
        return
    import imageio_ffmpeg

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
                          stdin=subprocess.PIPE)
    times = np.arange(int(DUR * FPS)) / FPS
    with Pool(args.workers) as pool:
        for i, f in enumerate(pool.imap(frame, times, chunksize=2)):
            ff.stdin.write(f.tobytes())
            if i % 30 == 0:
                print(f"klatka {i}/{len(times)}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("zapisano", args.out)


if __name__ == "__main__":
    main()
