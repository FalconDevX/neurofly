"""Animacja „wycieczka po connectomie”: najazd na półprzezroczystą muszkę, szybkie porównanie z ziarnami maku,
zbliżenie na głowę z obrotem, przejście do widoku z góry i podświetlanie części układu nerwowego od głowy w dół
z animowanymi etykietami (po angielsku). Gdy pojawiają się etykiety, ciało muszki znika, zostaje układ nerwowy.

    .venv312/Scripts/python docs/prezentacja/render_tour.py                  # → data/videos/connectome_tour.mp4
    .venv312/Scripts/python docs/prezentacja/render_tour.py --stills 4,9,14,20
    .venv312/Scripts/python docs/prezentacja/render_tour.py --gif docs/img/connectome_tour.gif   # wersja do README

Z BANC v888: pozycje som, regiony, szkielety neuronów lotu, liczby neuronów w podpisach (liczone z danych).
Ilustracyjne: dopasowanie ciała muszki do BANC, ziarna maku, kamera, poświaty i kolejność podświetlania.
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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import render_intro as ri  # noqa: E402
from render_assets import (AMBER, BG, BLUE, FONT_B, FONT_R, NECK_Y, PINK, ROOT, TEAL, VIOLET,  # noqa: E402
                           _drone_mesh, _drone_props, center, d, regions, xyz)

W, H, FPS, DUR = ri.W, ri.H, 30, 33.0
F = ri.F
ease, lerp, camera, to_cam, to_px = ri.ease, ri.lerp, ri.camera, ri.to_cam, ri.to_px

# --- grupy do podświetlania ----------------------------------------------------------------------------------------
SOMA_KEY = np.array(d["somas"]["region"])  # 1 central brain, 2 optic lobe, 3 VNC (jak w banc_anatomy.json)
R_CB, R_OL, R_VNC = (regions.index(n) for n in ("central_brain", "optic_lobe", "ventral_nerve_cord"))
SKEL_GROUP = {"dn_flight_power": "dn", "dn_flight_steering": "dn", "wing_power": "mn", "wing_steering": "mn",
              "wing_tension": "mn", "haltere_aff": "hal"}
_skel_keys, _skel_cnt = [], {"dn": 0, "mn": 0, "hal": 0}
for nr in d["neurons"]:
    g = SKEL_GROUP.get(nr["group"].rsplit("_", 1)[0])
    if g:
        _skel_cnt[g] += 1
    for ln in nr["lines"]:
        if len(ln) >= 2:
            _skel_keys.append(g)  # ta sama kolejność co ri._rngs
assert len(_skel_keys) == len(ri._rngs)


def _pts(g):
    return np.concatenate([np.asarray(ri.SKEL[s:s + k]) for (s, k, _), kk in zip(ri._rngs, _skel_keys) if kk == g])


_ol = SOMA_KEY == R_OL
_ol_r = _ol & (xyz[:, 0] < center[0])  # w BANC prawa strona muchy ma mniejsze x
_ol_l = _ol & (xyz[:, 0] >= center[0])
_dn = _pts("dn")
_dn_brain = _dn[_dn[:, 1] < NECK_Y]
_cb = xyz[SOMA_KEY == R_CB].mean(0)
CNS_C = np.array([center[0], (np.percentile(xyz[:, 1], 1) + np.percentile(xyz[:, 1], 99)) / 2, center[2]])
CNS_SPAN = np.percentile(xyz[:, 1], 99) - np.percentile(xyz[:, 1], 1)


def n(mask):
    return f"{int(mask.sum()):,}"


# (czas startu podświetlenia w s, klucz, nazwa, podpis, kotwica w µm, kolor); kolejność od głowy w dół
T0, DT = 12.0, 1.15
LABELS = [
    (0, ("soma", R_OL), "Optic lobe, right", f"{n(_ol_r)} neurons, input from the right eye", xyz[_ol_r].mean(0), TEAL),
    (0, ("soma", R_OL), "Optic lobe, left", f"{n(_ol_l)} neurons, input from the left eye", xyz[_ol_l].mean(0), TEAL),
    (1, ("soma", R_CB), "Central brain", f"{n(SOMA_KEY == R_CB)} neurons, integrates the senses", _cb - [0, 60, 0], VIOLET),
    (2, ("skel", "dn"), "Flight descending neurons", f"{_skel_cnt['dn']} cells, heading is read out here",
     _dn_brain.mean(0), AMBER),
    (3, ("neck", None), "Neck connective", "every command to the wings passes here",
     np.array([_cb[0], NECK_Y, _cb[2]]), (250, 250, 250)),
    (4, ("soma", R_VNC), "Ventral nerve cord", f"{n(SOMA_KEY == R_VNC)} neurons, the fly's spinal cord",
     xyz[SOMA_KEY == R_VNC].mean(0) + [0, 80, 0], (200, 170, 255)),
    (5, ("skel", "mn"), "Wing motor neurons", f"{_skel_cnt['mn']} cells: power, steering, tension",
     _pts("mn").mean(0), PINK),
    (6, ("skel", "hal"), "Haltere afferents", f"{_skel_cnt['hal']} cells, the fly's gyroscope", _pts("hal").mean(0), BLUE),
]
LABELS = [(T0 + DT * k, *rest) for k, *rest in LABELS]
T_LAST = LABELS[-1][0]

# --- kamera --------------------------------------------------------------------------------------------------------
AZ0, EL0, DIST0 = np.radians(30), np.radians(28), 9500.0
AZ_CLOSE, EL_CLOSE, DIST_CLOSE = np.radians(40), np.radians(30), 1650.0
AZ_TOP, EL_TOP = np.radians(90), np.radians(82)  # z góry, głowa na górze kadru
TOP_DIST = CNS_SPAN * 1.42 * F / H
# jedno ziarno maku obok głowy; kadr obejmuje głowę i ziarno
SEEDS = [dict(ri.SEEDS[0], t0=2.3)]
_cp = camera(ri.HEAD, 1.0, ri.AZ_END, ri.EL)[1]
_sq = (SEEDS[0]["c"] - ri.HEAD) @ _cp.T
POP_TARGET = ri.HEAD + _cp[0] * _sq[0] / 2 + _cp[1] * _sq[1] / 2
POP_DIST = (abs(_sq[0]) + 2300) * F / W

# --- finał: dron obok muszki i nitki od obwodu lotu (ILUSTRACYJNE, skala drona dobrana do muszki) ----------------
T_LAB_OUT, T_FIN, T_DRONE, T_TH = 21.5, 22.0, 23.5, 25.0
_DV, _DF = _drone_mesh()
_DF, _PROPS, _PR = _drone_props(_DV, _DF)
_DMID = (_DV.min(0) + _DV.max(0)) / 2
D_SCALE = 2600 / (_DV.max(0) - _DV.min(0))[:2].max()  # m → µm, dron ~ długość muszki
_bx0, _bx1 = ri.BODY[:, 0].min(), ri.BODY[:, 0].max()
DRONE_C = np.array([_bx1 + 300 + 1300, ri.FLY_C[1] - 150, ri.FLY_C[2]])
FIN_T = np.array([(_bx0 + DRONE_C[0] + 1300) / 2, ri.FLY_C[1] + 100, ri.FLY_C[2]])
FIN_AZ, FIN_EL = np.radians(90), np.radians(58)
FIN_DIST = (DRONE_C[0] + 1300 - _bx0) * 1.08 * F / W
RED = np.array((255, 45, 111), np.float32)


def drone_world(P, t):
    """Punkty X2 (m, z w górę) → świat muszki (µm, grzbiet = −z), powolny obrót i opadanie z góry przy wejściu."""
    yaw = np.radians(35) + 0.12 * (t - T_DRONE)
    c, s_ = np.cos(yaw), np.sin(yaw)
    Q = ((P - _DMID) * D_SCALE) @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]).T
    Q = Q * np.array([1, -1, -1])  # obrót o 180° wokół x: góra drona → grzbiet muszki
    drop = 1200 * (1 - ease((t - T_DRONE) / 1.6)) ** 2
    return DRONE_C + Q + ri.UP * drop


_rt = np.random.default_rng(7)
_dnp, _mnp = _pts("dn"), _pts("mn")
N_TH = 30
_pick = _rt.choice(len(_dnp) + len(_mnp), N_TH, replace=False)
TH_SRC = np.concatenate([_dnp, _mnp])[_pick]
TH_COL = np.array([AMBER if i < len(_dnp) else PINK for i in _pick], np.float32)
_used = np.unique(_DF)
TH_DST = np.concatenate([_PROPS[_rt.integers(0, 4, 18)] + _rt.normal(0, 0.01, (18, 3)),
                         _DV[_rt.choice(_used, N_TH - 18, replace=False)]])
TH_H = _rt.uniform(500, 1500, (N_TH, 2))
TH_T0 = T_TH + _rt.uniform(0, 1.3, N_TH)
TH_PH = _rt.uniform(0, 1, N_TH)


def cam_at(t):
    if t < 3.0:  # najazd na muszkę
        x = ease(t / 3.0)
        return camera(lerp(ri.FLY_C, POP_TARGET, x), np.exp(lerp(np.log(DIST0), np.log(POP_DIST), x)),
                      lerp(AZ0, ri.AZ_END, x), lerp(EL0, ri.EL, x))
    if t < 5.8:  # ziarna maku, kamera prawie stoi
        x = (t - 3.0) / 2.8
        return camera(POP_TARGET, POP_DIST * (1 - 0.03 * x), ri.AZ_END - np.radians(3) * x, ri.EL)
    az_m = ri.AZ_END - np.radians(3)
    if t < 9.0:  # zbliżenie na głowę z obrotem
        x = ease((t - 5.8) / 3.2)
        xt = ease((t - 5.8) / 2.0)  # cel szybciej niż obrót, głowa od razu w środku kadru
        return camera(lerp(POP_TARGET, ri.HEAD, xt), np.exp(lerp(np.log(POP_DIST * 0.97), np.log(DIST_CLOSE), x)),
                      lerp(az_m, AZ_CLOSE, x), lerp(ri.EL, EL_CLOSE, x))
    if t < 11.6:  # odjazd do widoku z góry
        x = ease((t - 9.0) / 2.6)
        return camera(lerp(ri.HEAD, CNS_C, x), np.exp(lerp(np.log(DIST_CLOSE), np.log(TOP_DIST), x)),
                      lerp(AZ_CLOSE, AZ_TOP, x), lerp(EL_CLOSE, EL_TOP, x))
    if t < T_FIN:
        x = (t - 11.6) / (T_FIN - 11.6)
        return camera(CNS_C, TOP_DIST * (1 - 0.04 * x), AZ_TOP, EL_TOP - np.radians(6) * x)
    el_top = EL_TOP - np.radians(6)
    if t < T_FIN + 3.5:  # odjazd: muszka w lewo, miejsce na drona
        x = ease((t - T_FIN) / 3.5)
        return camera(lerp(CNS_C, FIN_T, x), np.exp(lerp(np.log(TOP_DIST * 0.96), np.log(FIN_DIST), x)),
                      AZ_TOP, lerp(el_top, FIN_EL, x))
    x = (t - T_FIN - 3.5) / (DUR - T_FIN - 3.5)
    return camera(FIN_T, FIN_DIST * (1 - 0.03 * x), FIN_AZ + np.radians(6) * ease(x), FIN_EL)


# --- render --------------------------------------------------------------------------------------------------------
def highlight(t, key):
    """Mnożnik jasności grupy: przed etykietami 1, potem nieopisane przygasają, opisana błyska i zostaje jaśniejsza."""
    dim = 1 - 0.5 * ease((t - (T0 - 0.6)) / 0.8)
    w = dim
    for tr, k, *_ in LABELS:
        if k == key:
            w = dim + 1.6 * np.exp(-((t - tr - 0.25) / 0.35) ** 2) + 0.75 * ease((t - tr) / 0.5)
            break
    return w


def frame(t):
    cam = cam_at(t)
    dist = np.linalg.norm(cam[0] - ri.to_world_target(cam))
    acc = np.zeros((H, W, 3), np.float32)

    # ciało: lekko przezroczyste szkło, przy zbliżeniu ciszej, przy etykietach prawie znika
    gain = 175 * ease(t / 1.0) * (1 - 0.3 * ease((t - 6.5) / 2.0)) * (1 - 0.85 * ease((t - (T0 - 0.5)) / (T_LAST - T0 + 1.5)))
    gain += 75 * ease((t - (T_FIN + 0.5)) / 2.0)  # w finale ciało wraca, jak na slajdzie tytułowym
    if gain > 0.5:
        ri.glass(acc, to_cam(ri.BODY, cam), gain)

    scale = min((2250 / dist) ** 1.7, 1.6) * (1 + 1.2 * ease((t - 10.0) / 2.0))  # z góry jaśniej, bo dalej

    q = to_cam(ri.SOMA, cam)
    su, sv = to_px(q)
    on = ease((t - ri.SOMA_TON) / 0.5)
    flash = np.exp(-((t - ri.SOMA_TON - 0.15) / 0.22) ** 2)
    hw = np.ones(len(q), np.float32) * highlight(t, None)
    for r in (R_CB, R_OL, R_VNC):
        hw[SOMA_KEY == r] = highlight(t, ("soma", r))
    a = (on * 0.75 + flash * 1.4) * 0.30 * scale * hw * (q[:, 2] > 20)
    pts = np.zeros_like(acc)
    ui, vi = su.astype(int), sv.astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (a > 1e-4)
    np.add.at(pts, (vi[ok], ui[ok]), ri.SOMA_COL[ok] * a[ok, None])
    acc += pts + cv2.GaussianBlur(pts, (0, 0), 4) * 1.6

    qs = to_cam(ri.SKEL, cam)
    ku, kv = to_px(qs)
    P = np.round(np.stack([ku, kv], 1) * 16).astype(np.int32)
    glow, line = np.zeros_like(acc), np.zeros_like(acc)
    thick = 2 if dist < 4500 else 1
    hk = {g: highlight(t, ("skel", g)) for g in ("dn", "mn", "hal")}
    for (s0, k, col), ton, g in zip(ri._rngs, ri.SKEL_TON, _skel_keys):
        al = ease((t - ton) / 0.8) * hk.get(g, 1.0)
        if al <= 0 or qs[s0:s0 + k, 2].min() < 20:
            continue
        c = np.array(col, np.float32) * al
        cv2.polylines(glow, [P[s0:s0 + k]], False, tuple(float(x) * 0.18 for x in c), 5 * thick, cv2.LINE_AA, shift=4)
        cv2.polylines(line, [P[s0:s0 + k]], False, tuple(float(x) * 0.55 for x in c), thick, cv2.LINE_AA, shift=4)
    acc += cv2.GaussianBlur(glow, (0, 0), 5) + line

    # poświata w miejscu nowej etykiety
    yy, xx = np.mgrid[0:H:4, 0:W:4]
    halo = np.zeros((H // 4, W // 4, 3), np.float32)
    for tr, _, _, _, anchor, col in LABELS:
        p = np.exp(-((t - tr - 0.2) / 0.45) ** 2)
        if p < 0.01:
            continue
        u, v = to_px(to_cam(anchor[None], cam))
        halo += p * np.exp(-((xx - u[0]) ** 2 + (yy - v[0]) ** 2) / (2 * 70.0 ** 2))[..., None] * np.array(col, np.float32) * 0.35
    acc += cv2.resize(halo, (W, H), interpolation=cv2.INTER_LINEAR)

    if t > T_DRONE:
        drone(acc, cam, t)

    img = BG + (255 - BG) * (1 - np.exp(-acc / 210.0))

    sa = ease((t - 2.2) / 0.3) * (1 - ease((t - 5.5) / 0.7))  # ziarna: szybko wpadają i znikają
    if sa > 0:
        base = img.copy()
        for s in sorted(SEEDS, key=lambda s: -to_cam(s["c"][None], cam)[0, 2]):
            ri.draw_seed(img, s, cam, t)
        img = base + (img - base) * sa

    img *= ease(t / 0.8) * (1 - ease((t - (DUR - 1.0)) / 1.0))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return overlay(img, cam, t)


def glass_tris(acc, q, tris, tint, gain):
    """Szkło jak w render_intro.glass, dla dowolnej siatki: krycie z Fresnela, 10 poziomów, cienka siatka krawędzi."""
    u, v = to_px(q)
    a, b, c = q[tris[:, 0]], q[tris[:, 1]], q[tris[:, 2]]
    ok = (a[:, 2] > 20) & (b[:, 2] > 20) & (c[:, 2] > 20)
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    view = (a + b + c) / 3
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    alpha = 0.025 + 0.42 * (1 - np.abs((nrm * view).sum(1))) ** 2.2
    levels = np.linspace(0.025, 0.445, 10)
    idx = np.clip(np.digitize(alpha, levels) - 1, 0, 9)
    p2 = np.round(np.stack([u, v], 1)[tris] * 16).astype(np.int32)
    layer = np.zeros((H, W), np.float32)
    for k in range(10):
        kk = ok & (idx == k)
        if kk.any():
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, list(p2[kk]), 255, cv2.LINE_AA, shift=4)
            layer += mask.astype(np.float32) * (levels[k] * gain / 255)
    wire = np.zeros((H, W), np.float32)
    cv2.polylines(wire, list(p2[ok]), True, 1.0, 1, cv2.LINE_AA, shift=4)
    acc += (layer + wire * gain * 0.12)[..., None] * np.asarray(tint, np.float32)


def drone(acc, cam, t):
    """Szklany X2 ze śmigłami i nitki od neuronów lotu muszki do drona (impulsy biegną od muchy do drona)."""
    ga = ease((t - T_DRONE) / 1.2)
    tint = np.array((0.95, 0.84, 0.9), np.float32)
    glass_tris(acc, to_cam(drone_world(_DV, t), cam), _DF, tint, 230 * ga)

    # śmigła: krąg obrotu i dwie łopaty z rozmyciem ruchu; po podłączeniu nitek kręcą się szybciej
    spin = 2 * np.pi * (0.5 * (t - T_DRONE) + 1.6 * max(0.0, t - (T_TH + 1.2)))
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    m = np.zeros((H, W), np.float32)
    for i, hub in enumerate(_PROPS):
        ring = hub + np.stack([_PR * np.cos(ang), _PR * np.sin(ang), np.zeros_like(ang)], 1)
        ru, rv = to_px(to_cam(drone_world(ring, t), cam))
        poly = np.round(np.stack([ru, rv], 1) * 16).astype(np.int32)
        m[:] = 0
        cv2.fillPoly(m, [poly], 1.0, cv2.LINE_AA, shift=4)
        acc += cv2.GaussianBlur(m, (0, 0), 5)[..., None] * tint * 10 * ga
        m[:] = 0
        cv2.polylines(m, [poly], True, 1.0, 1, cv2.LINE_AA, shift=4)
        acc += m[..., None] * tint * 50 * ga
        dirn = 1 if i % 2 else -1
        for k, w8 in enumerate((1.0, 0.45, 0.2)):
            phi = dirn * (spin - 0.25 * k) + i
            for side in (0, np.pi):
                dv = np.array([np.cos(phi + side), np.sin(phi + side), 0])
                nv = np.array([-dv[1], dv[0], 0])
                r = np.linspace(0.01, _PR * 0.95, 10)[:, None]
                wd = 0.004 + 0.014 * np.sin(np.pi * r / _PR)
                blade = np.concatenate([hub + r * dv + wd * nv, (hub + r * dv - wd * nv)[::-1]])
                bu, bv = to_px(to_cam(drone_world(blade, t), cam))
                m[:] = 0
                cv2.fillPoly(m, [np.round(np.stack([bu, bv], 1) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
                acc += m[..., None] * tint * 40 * w8 * ga

    # nitki: krzywe Béziera w 3D od punktów obwodu lotu (DN, MN skrzydeł) do drona, rosną, potem płyną impulsy
    dst = drone_world(TH_DST, t)
    f = np.linspace(0, 1, 90)
    B = np.stack([(1 - f) ** 3, 3 * (1 - f) ** 2 * f, 3 * (1 - f) * f ** 2, f ** 3], 1)
    glow, line = np.zeros_like(acc), np.zeros_like(acc)
    for i in range(N_TH):
        g = ease((t - TH_T0[i]) / 1.0)
        if g <= 0:
            continue
        p0, p3 = TH_SRC[i], dst[i]
        dv = p3 - p0
        ctrl = np.stack([p0, p0 + ri.UP * TH_H[i, 0] + dv * 0.3, p3 + ri.UP * TH_H[i, 1] - dv * 0.3, p3])
        n_ = max(2, int(g * 89) + 1)
        q = to_cam(B[:n_] @ ctrl, cam)
        if q[:, 2].min() < 20:
            continue
        u, v = to_px(q)
        P = np.round(np.stack([u, v], 1) * 16).astype(np.int32)
        ff = f[:n_]
        pp = ((t - TH_T0[i] - 1.0) * 0.6 + TH_PH[i]) % 1 if t > TH_T0[i] + 1.0 else -9.0
        bright = 0.65 + 2.4 * np.exp(-((ff - pp) / 0.035) ** 2) + 1.5 * np.exp(-((ff - ff[-1]) / 0.03) ** 2) * (g < 1)
        for k in range(n_ - 1):
            col = ((1 - ff[k]) * TH_COL[i] + ff[k] * RED) * bright[k]
            a, b = (int(P[k, 0]), int(P[k, 1])), (int(P[k + 1, 0]), int(P[k + 1, 1]))
            cv2.line(glow, a, b, tuple(float(x) * 0.22 for x in col), 5, cv2.LINE_AA, shift=4)
            cv2.line(line, a, b, tuple(float(x) * 0.7 for x in col), 1, cv2.LINE_AA, shift=4)
    acc += cv2.GaussianBlur(glow, (0, 0), 4) + line


def overlay(img, cam, t):
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    fb, fr, fs = ImageFont.truetype(FONT_B, 34), ImageFont.truetype(FONT_R, 23), ImageFont.truetype(FONT_B, 22)
    fade_all = 1 - ease((t - (DUR - 1.0)) / 1.0)
    fade_lab = 1 - ease((t - T_LAB_OUT) / 0.8)

    # ziarna maku i skala
    pa = ease((t - 3.6) / 0.4) * (1 - ease((t - 5.4) / 0.4))
    if pa > 0:
        A = int(255 * pa)
        R = cam[1]
        top = max(SEEDS, key=lambda s: np.dot(s["c"], R[1]))
        u, v = (float(x[0]) for x in to_px(to_cam((top["c"] + R[1] * 250)[None], cam)))
        dr.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(250, 250, 250, A))
        sub = "about 1 mm, the whole fly is about 2.5 mm"
        xt = min(u + 200, W - 60 - dr.textlength(sub, font=fr))  # napis zawsze w kadrze
        dr.line([(u, v), (xt - 50, v - 110), (xt - 10, v - 110)], fill=(250, 250, 250, int(A * 0.6)), width=2)
        dr.text((xt, v - 110), "Poppy seed", font=fb, fill=(250, 250, 250, A), anchor="lm")
        dr.text((xt, v - 82), sub, font=fr, fill=(170, 170, 180, A), anchor="lt")
        z = to_cam(ri.HEAD[None], cam)[0, 2]
        L = F * 1000 / z
        x1, y1 = W - 110, H - 90
        dr.line([(x1 - L, y1), (x1, y1)], fill=(250, 250, 250, A), width=3)
        for xx in (x1 - L, x1):
            dr.line([(xx, y1 - 9), (xx, y1 + 9)], fill=(250, 250, 250, A), width=3)
        dr.text((x1 - L / 2, y1 - 16), "1 mm", font=fr, fill=(250, 250, 250, A), anchor="mb")

    # tytuł widoku z góry
    ta = ease((t - 11.2) / 0.6) * fade_lab
    if ta > 0:
        dr.text((60, H - 96), "BANC v888", font=ImageFont.truetype(FONT_B, 44), fill=(250, 250, 250, int(255 * ta)), anchor="ls")
        dr.text((60, H - 82), "brain and nerve cord of one fruit fly, 175,401 neurons", font=fr,
                fill=(170, 170, 180, int(255 * ta)), anchor="lt")

    # etykiety: kolumny po bokach, linia z kotwicy, układ bez nachodzenia
    cu = float(to_px(to_cam(CNS_C[None], cam))[0][0])
    side = {"L": [], "R": []}
    for tr, _, title, sub, anchor, col in LABELS:
        if t < tr:
            continue
        u, v = (float(x[0]) for x in to_px(to_cam(anchor[None], cam)))
        s = "L" if (u < cu - 30 or (abs(u - cu) <= 30 and title in ("Central brain", "Wing motor neurons", "Flight descending neurons"))) else "R"
        side[s].append([v, u, tr, title, sub, col])
    for s, items in side.items():
        items.sort()
        ys = []
        for it in items:
            y = max(it[0], (ys[-1] + 96) if ys else 90)
            ys.append(y)
            it[0] = y
        # gdy zjechało za dół, przesuń całą kolumnę w górę
        over = (ys[-1] + (200 if s == "L" else 120) - H) if ys else 0
        for it in items:
            it[0] -= max(over, 0)
        colx = W * 0.27 if s == "L" else W * 0.73
        for y, u, tr, title, sub, col in items:
            _, vv = (float(x[0]) for x in to_px(to_cam(next(l[4] for l in LABELS if l[2] == title)[None], cam)))
            x = ease((t - tr) / 0.3)
            ln = ease((t - tr - 0.1) / 0.45)
            tx = ease((t - tr - 0.3) / 0.45)
            sx = ease((t - tr - 0.45) / 0.45)
            A = int(255 * fade_lab)
            if A <= 0:
                continue
            r = 6 * x
            dr.ellipse([u - r, vv - r, u + r, vv + r], fill=(*col, A))
            dr.ellipse([u - 2.5 * r, vv - 2.5 * r, u + 2.5 * r, vv + 2.5 * r], outline=(*col, int(A * 0.5)), width=2)
            sign = -1 if s == "L" else 1
            ex, ey = colx - sign * 40, y
            p1 = (lerp(u, ex, min(ln * 2, 1)), lerp(vv, ey, min(ln * 2, 1)))
            pts = [(u, vv), p1]
            if ln > 0.5:
                pts.append((lerp(ex, colx, (ln - 0.5) * 2), ey))
            if ln > 0:
                dr.line(pts, fill=(*col, int(A * 0.8)), width=2)
            if tx > 0:
                off = sign * 24 * (1 - tx)
                anc = "rb" if s == "L" else "lb"
                xt = colx + sign * 18 + off
                dr.text((xt, y - 2), title, font=fb, fill=(250, 250, 250, int(A * tx)), anchor=anc)
                bar_x = xt + (8 if s == "L" else -14)
                dr.rectangle([bar_x, y - 36, bar_x + 5, y + 30], fill=(*col, int(A * tx)))
            if sx > 0:
                off = sign * 24 * (1 - sx)
                anc = "rt" if s == "L" else "lt"
                dr.text((colx + sign * 18 + off, y + 6), sub, font=fr, fill=(175, 175, 185, int(A * sx)), anchor=anc)

    ca = ease((t - 19.5) / 0.8) * fade_lab
    if ca > 0:
        dr.text((W - 60, H - 50), "Data: BANC v888 (Bates et al. 2026). Fly body: NeuroMechFly, illustrative fit.",
                font=ImageFont.truetype(FONT_R, 18), fill=(130, 130, 140, int(255 * ca)), anchor="rb")

    # finał: podpisy pod muszką i dronem, tytuł, uczciwa uwaga o nitkach
    fa = ease((t - (T_TH + 1.3)) / 0.6) * fade_all
    if fa > 0:
        A = int(255 * fa)
        bu, bv = to_px(to_cam(ri.BODY, cam))
        du, dv_ = to_px(to_cam(drone_world(_DV, t), cam))
        for (u, v), title, sub in (((np.median(bu), np.percentile(bv, 99) + 40), "Fruit fly flight circuit",
                                    "real neurons from BANC v888"),
                                   (((du.min() + du.max()) / 2, dv_.max() + 40), "Drone",
                                    "Skydio X2, simulated in MuJoCo")):
            v = min(v, H - 110)
            dr.text((u, v), title, font=fb, fill=(250, 250, 250, A), anchor="mt")
            dr.text((u, v + 46), sub, font=fr, fill=(175, 175, 185, A), anchor="mt")
    ha = ease((t - (T_TH + 2.3)) / 0.7) * fade_all
    if ha > 0:
        A = int(255 * ha)
        dr.text((W / 2, 70), "NeuroFly", font=ImageFont.truetype(FONT_B, 64), fill=(250, 250, 250, A), anchor="mt")
        dr.text((W / 2, 150), "a fruit fly's connectome steers a drone", font=ImageFont.truetype(FONT_R, 28),
                fill=(255, 120, 165, A), anchor="mt")
        dr.text((W - 60, H - 50), "Threads are illustrative. The real link: cameras, FlyVis, BANC, decoder, motors.",
                font=ImageFont.truetype(FONT_R, 18), fill=(130, 130, 140, A), anchor="rb")
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", help="czasy w s, np. 4,9,14 → PNG zamiast wideo")
    ap.add_argument("--out", default=str(ROOT / "data" / "videos" / "connectome_tour.mp4"))
    ap.add_argument("--gif", help="dodatkowo GIF (960 px, 15 FPS) z gotowego MP4, np. docs/img/connectome_tour.gif")
    ap.add_argument("--still-dir", default=str(ROOT / "data" / "videos" / "tour_stills"))
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

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    times = np.arange(int(DUR * FPS)) / FPS
    with Pool(args.workers) as pool:
        for i, f in enumerate(pool.imap(frame, times, chunksize=2)):
            ff.stdin.write(f.tobytes())
            if i % 30 == 0:
                print(f"klatka {i}/{len(times)}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("zapisano", args.out)
    if args.gif:
        vf = ("fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", args.out, "-vf", vf, "-loop", "0", args.gif], check=True)
        print("zapisano", args.gif)


if __name__ == "__main__":
    main()
