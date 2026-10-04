"""Animacja logo NeuroFly (10 s): chmura som BANC v888 zlatuje się w ikonę muszki-drona, oczy zapalają się
komórka po komórce, impuls przez „mózg”, napis neuro wjeżdża literami, fly z efektem glitch, pod spodem hasło
z błyskiem światła. Czerwień logo: karmin lekko wpadający w róż (#E8163A → #EE2458 na „fly”).

    .venv312/Scripts/python docs/prezentacja/render_logo.py                  # → data/videos/logo_intro.mp4
    .venv312/Scripts/python docs/prezentacja/render_logo.py --stills 1,2.5,4.7,9

Z BANC v888: pozycje som (start animacji) i ich klasy (kolory). Reszta to grafika (logo z docs/assets/logo-dark.png).
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from render_assets import BG, CLASS_COLOR, ROOT, cls, xyz  # noqa: E402

W, H, FPS, DUR = 1920, 1080, 30, 10.0
ACC0, ACC1 = np.array((232, 22, 58), np.float32), np.array((238, 36, 88), np.float32)  # czerwień lekko w róż
RED = ACC0  # kolor poświat
WHITE = np.array((236, 239, 246), np.float32)


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3


# --- logo: warstwy -------------------------------------------------------------------------------------------------
LOGO = np.asarray(Image.open(ROOT / "docs" / "assets" / "logo-dark.png").convert("RGBA")).astype(np.float32)
LH, LW = LOGO.shape[:2]
OX, OY = (W - LW) // 2, (H - 683) // 2
L_RGB, L_A = LOGO[..., :3], LOGO[..., 3] / 255
ICON_Y1, WORD_Y0 = 418, 450  # pusty pas między ikoną a napisem
_red = (L_A > 0.08) & (L_RGB[..., 0] > 150) & (L_RGB[..., 1] < 110)
_rows = np.arange(LH)[:, None]
# przekolorowanie czerwieni: udział czerwieni z kanału G (biel ~242, czerwień ~24), jasność z R, gradient w poziomie
_near = cv2.dilate(_red.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
_r = np.clip((242 - L_RGB[..., 1]) / (242 - 24), 0, 1)
_gx = (np.arange(LW)[None, :] - 620) / 260  # gradient tylko na „fly”, oczy w kolorze ACC0
_acc = ACC0 + (ACC1 - ACC0) * np.clip(_gx, 0, 1)[..., None] ** 1.3
_acc = _acc * np.clip(L_RGB[..., :1] / 215, 0.55, 1.05)
L_RGB = np.where(_near[..., None], _r[..., None] * _acc + (1 - _r[..., None]) * L_RGB, L_RGB).astype(np.float32)
ICON_LIGHT = np.where((_rows < ICON_Y1) & ~_red, L_A, 0).astype(np.float32)
ICON_RED = np.where((_rows < ICON_Y1) & _red, L_A, 0).astype(np.float32)
BRAIN_C = np.array([450.0, 250.0])  # środek „mózgu” na ikonie (px logo)
_yy, _xx = np.mgrid[0:LH, 0:LW]
DIST_BRAIN = np.hypot(_xx - BRAIN_C[0], _yy - BRAIN_C[1]).astype(np.float32)

# komórki oczu (składowe czerwieni), kolejność: od środka oka na zewnątrz + losowo
_n, _lab, _st, _cen = cv2.connectedComponentsWithStats((ICON_RED > 0.3).astype(np.uint8), connectivity=4)
_rng = np.random.default_rng(4)
EYE_CELLS = []
for i in range(1, _n):
    if _st[i, 4] < 15:
        continue
    cx, cy = _cen[i]
    eye_c = (392, 125) if cx < 450 else (508, 125)
    EYE_CELLS.append((_lab == i, 2.25 + 0.012 * np.hypot(cx - eye_c[0], cy - eye_c[1]) + _rng.uniform(0, 0.25)))
_eye_mask = np.zeros_like(ICON_RED, bool)
for m, _ in EYE_CELLS:
    _eye_mask |= m
EYE_REST = np.where(~_eye_mask, ICON_RED, 0)  # antyaliasing krawędzi oczu, wchodzi razem z ostatnimi komórkami

# litery napisu (składowe, od lewej): neuro jasne, fly czerwone
_wa = (L_A > 0.08) & (_rows >= WORD_Y0)
_n2, _lab2, _st2, _ = cv2.connectedComponentsWithStats(_wa.astype(np.uint8), connectivity=8)
LETTERS = []
for i in sorted(range(1, _n2), key=lambda i: _st2[i, 0]):
    if _st2[i, 4] < 30:
        continue
    x, y, w, h = _st2[i, :4]
    m = cv2.dilate((_lab2 == i).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (L_A > 0)
    LETTERS.append(dict(box=(x, y, w, h), mask=m, red=bool(_red[_lab2 == i].mean() > 0.5)))
WORD_Y1 = max(l["box"][1] + l["box"][3] for l in LETTERS)

# --- cząstki: somy BANC → piksele ikony -------------------------------------------------------------------------
_p = xyz[:, :2].astype(np.float64)
_mid = (np.percentile(_p, 1, 0) + np.percentile(_p, 99, 0)) / 2
_span = np.percentile(_p, 99, 0) - np.percentile(_p, 1, 0)
S0 = 820 / _span[1]
P_START = np.array([W / 2, H / 2]) + (_p - _mid) * S0
P_COL = np.array([CLASS_COLOR[c] for c in cls], np.float32)
_ty, _tx = np.nonzero(ICON_LIGHT > 0.5)
_tgt = np.stack([_tx, _ty], 1).astype(np.float64)
_tn = (_tgt - _tgt.min(0)) / np.ptp(_tgt, 0)
_sn = np.clip((_p - np.percentile(_p, 1, 0)) / _span, 0, 1)
_, _idx = cKDTree(_tn).query(_sn)
_rp = np.random.default_rng(1)
P_END = _tgt[_idx] + _rp.uniform(-0.5, 0.5, (len(_p), 2)) + [OX, OY]
P_DELAY = 0.35 + 0.55 * _rp.random(len(_p))
P_ARC = _rp.uniform(-1, 1, len(_p)) * 140
P_DIR = P_END - P_START
P_PERP = np.stack([-P_DIR[:, 1], P_DIR[:, 0]], 1) / (np.linalg.norm(P_DIR, axis=1, keepdims=True) + 1e-9)

# tło: delikatna winieta
_gy, _gx = np.mgrid[0:H, 0:W]
BG_IMG = (BG + 10 * np.exp(-(((_gx - W / 2) / 900) ** 2 + ((_gy - H / 2) / 560) ** 2))[..., None]).astype(np.float32)



def over(canvas, rgb, a, x0=OX, y0=OY):
    """Nakłada warstwę (kolor, krycie) w miejscu (x0, y0)."""
    h, w = a.shape
    sub = canvas[y0:y0 + h, x0:x0 + w]
    sub *= 1 - a[..., None]
    sub += rgb * a[..., None]


def frame(t):
    img = BG_IMG.copy()
    glow = np.zeros_like(img)

    # 1. chmura som BANC zlatuje się w ikonę
    if t < 2.7:
        x = ease((t - P_DELAY) / 1.15)
        pos = P_START + P_DIR * x[:, None] + P_PERP * (np.sin(np.pi * x) * P_ARC)[:, None]
        zoom = 1 + 0.06 * (1 - ease(t / 0.8))
        pos = (pos - [W / 2, H / 2]) * zoom + [W / 2, H / 2]
        col = P_COL * (1 - x[:, None]) + WHITE * x[:, None]
        a = 0.22 * ease(t / 0.5) * (1 - ease((t - 1.85) / 0.7)) * (1 + 0.8 * np.sin(np.pi * x))
        acc = np.zeros_like(img)
        ui, vi = pos[:, 0].astype(int), pos[:, 1].astype(int)
        ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        np.add.at(acc, (vi[ok], ui[ok]), col[ok] * a[ok, None])
        acc += cv2.GaussianBlur(acc, (0, 0), 3) * 1.4
        img += (255 - img) * (1 - np.exp(-acc / 180.0))

    # 2. jasna część ikony: front od mózgu na zewnątrz, z poświatą na krawędzi
    r_front = 520 * ease_out((t - 1.45) / 1.0)
    if r_front > 0:
        rev = np.clip((r_front - DIST_BRAIN) / 30, 0, 1) * ICON_LIGHT
        over(img, L_RGB, rev)
        edge = np.exp(-((DIST_BRAIN - r_front) / 14) ** 2) * ICON_LIGHT * (1 - ease((t - 2.4) / 0.4))
        glow[OY:OY + LH, OX:OX + LW] += edge[..., None] * np.array((150, 220, 255), np.float32) * 1.4

    # 3. oczy: komórki zapalają się po kolei, z białym błyskiem
    eye_a = np.zeros_like(ICON_RED)
    eye_flash = np.zeros_like(ICON_RED)
    last = max(tc for _, tc in EYE_CELLS)
    for m, tc in EYE_CELLS:
        if t > tc:
            eye_a[m] = ICON_RED[m] * ease((t - tc) / 0.08)
            eye_flash[m] = np.exp(-((t - tc - 0.05) / 0.09) ** 2)
    eye_a += EYE_REST * ease((t - last + 0.1) / 0.2)
    if eye_a.any():
        col = L_RGB * (1 - eye_flash[..., None]) + 255 * eye_flash[..., None]
        over(img, col, eye_a)
        breathe = 0.55 + 0.45 * np.sin(2 * np.pi * (t - 3.0) / 2.2) if t > 3.0 else 1.0
        glow[OY:OY + LH, OX:OX + LW] += eye_a[..., None] * RED * (0.35 * breathe + 1.2 * eye_flash[..., None])

    # 4. impuls przez mózg i skrzydła (czerwona fala po jasnej części ikony)
    if 2.9 < t < 4.2:
        r = 600 * ease((t - 2.9) / 1.1)
        band = np.exp(-((DIST_BRAIN - r) / 22) ** 2) * ICON_LIGHT * (1 - ease((t - 3.7) / 0.5))
        glow[OY:OY + LH, OX:OX + LW] += band[..., None] * np.array((255, 70, 90), np.float32) * 1.3

    # 5. napis: neuro literami (z dołu, z rozmyciem), fly z glitchem
    rng = np.random.default_rng(int(t * FPS))
    neuro = [l for l in LETTERS if not l["red"]]
    fly = [l for l in LETTERS if l["red"]]
    for i, l in enumerate(neuro):
        x = ease_out((t - (3.5 + 0.11 * i)) / 0.55)
        if x <= 0:
            continue
        bx, by, bw, bh = l["box"]
        pad = 24
        y0, y1, x0, x1 = max(by - pad, 0), min(by + bh + pad, LH), max(bx - pad, 0), min(bx + bw + pad, LW)
        a = np.where(l["mask"], L_A, 0)[y0:y1, x0:x1].astype(np.float32)
        rgb = L_RGB[y0:y1, x0:x1]
        sig = 10 * (1 - x)
        if sig > 0.3:
            a = cv2.GaussianBlur(a, (0, 0), sig)
        dy = int(round(46 * (1 - x)))
        over(img, rgb, a * x, OX + x0, OY + y0 + dy)
    t_fly = 4.25
    if t > t_fly:
        k = 1 - ease((t - t_fly) / 0.75)  # siła glitcha
        on = 1.0 if (k < 0.15 or rng.random() > 0.35 * k) else 0.15
        fa = np.zeros_like(L_A)
        for l in fly:
            fa = np.maximum(fa, np.where(l["mask"], L_A, 0))
        xs = [l["box"][0] for l in fly]
        x0, x1 = min(xs) - 40, min(max(l["box"][0] + l["box"][2] for l in fly) + 40, LW)
        y0, y1 = WORD_Y0 - 10, WORD_Y1 + 10
        a = fa[y0:y1, x0:x1] * on * ease((t - t_fly) / 0.1)
        rgb = L_RGB[y0:y1, x0:x1]
        if k > 0.02:  # paski przesunięte w poziomie + rozszczepienie kanałów
            out = np.zeros_like(a)
            edges = np.sort(rng.integers(0, a.shape[0], 7))
            for s0, s1 in zip(np.r_[0, edges], np.r_[edges, a.shape[0]]):
                out[s0:s1] = np.roll(a[s0:s1], int(rng.integers(-40, 41) * k), axis=1)
            a = out
            sh = int(10 * k) + 1
            over(img, np.full_like(rgb, (40, 220, 255)), np.roll(a, -sh, axis=1) * 0.45 * k, OX + x0, OY + y0)
        over(img, rgb, a, OX + x0, OY + y0)
        glow[OY + y0:OY + y1, OX + x0:OX + x1] += a[..., None] * RED * 0.25

    # 6. błysk światła po całym napisie
    if 5.15 < t < 6.0:
        wa = np.where(_rows >= WORD_Y0, L_A, 0)
        pos = -200 + 1300 * ease((t - 5.15) / 0.8)
        band = np.exp(-(((_xx + 0.35 * (_yy - WORD_Y0)) - pos) / 45) ** 2) * wa
        glow[OY:OY + LH, OX:OX + LW] += band[..., None] * 255 * 0.9

    # poświata (bloom) z elementów świecących
    img += cv2.GaussianBlur(glow, (0, 0), 9) + glow * 0.35
    img = np.clip(img, 0, 255)

    im = img
    out = im.astype(np.float32)
    out *= ease(t / 0.3)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", help="czasy w s → PNG zamiast wideo")
    ap.add_argument("--out", default=str(ROOT / "data" / "videos" / "logo_intro.mp4"))
    ap.add_argument("--still-dir", default=str(ROOT / "data" / "videos" / "logo_stills"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.stills:
        out = Path(args.still_dir)
        out.mkdir(parents=True, exist_ok=True)
        for ts in args.stills.split(","):
            cv2.imwrite(str(out / f"t{float(ts):05.2f}.png"), cv2.cvtColor(frame(float(ts)), cv2.COLOR_RGB2BGR))
            print("still", ts, flush=True)
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
