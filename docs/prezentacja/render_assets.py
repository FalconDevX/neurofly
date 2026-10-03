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

TEAL, AMBER, ROSE, BLUE, ZINC = (45, 212, 191), (251, 191, 36), (251, 113, 133), (96, 165, 250), (161, 161, 170)
CLASS_COLOR = {}
for i, c in enumerate(classes):
    if c in ("optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"):
        CLASS_COLOR[i] = TEAL
    elif c in ("descending", "ascending"):
        CLASS_COLOR[i] = AMBER
    elif c in ("motor",):
        CLASS_COLOR[i] = ROSE
    elif c.startswith("sensory"):
        CLASS_COLOR[i] = BLUE
    else:
        CLASS_COLOR[i] = ZINC


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


def fit(u, v, W, H, margin=0.06, horizontal=False):
    """Wpasowuje rzut w kadr po percentylach (punkty odstające nie przesuwają środka); horizontal: mózg z lewej."""
    if horizontal:
        u, v = v.copy(), -u.copy()
    lo_u, hi_u = np.percentile(u, [0.3, 99.7])
    lo_v, hi_v = np.percentile(v, [0.3, 99.7])
    s = min(W * (1 - 2 * margin) / (hi_u - lo_u), H * (1 - 2 * margin) / (hi_v - lo_v))
    return (u - (lo_u + hi_u) / 2) * s + W / 2, (v - (lo_v + hi_v) / 2) * s + H / 2, s


def splat(img, u, v, color, alpha, r=1):
    """Dodaje punkty (kolor × alpha) do bufora float — gęste regiony świecą mocniej (render jak w mikroskopii)."""
    H, W = img.shape[:2]
    ui, vi = u.astype(int), v.astype(int)
    for dx in range(-r + 1, r):
        for dy in range(-r + 1, r):
            uu, vv = ui + dx, vi + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            np.add.at(img, (vv[ok], uu[ok]), color[ok] * alpha[ok, None])


def connectome(W=2400, H=1500, yaw=0.55, pitch=0.28, name="connectome_3d.png", scale=1.55, horizontal=False):
    img = np.zeros((H, W, 3), np.float32)
    u, v, z = project(xyz, yaw, pitch, 1.0, 0, 0)
    u, v, _ = fit(u, v, W, H, horizontal=horizontal)
    depth = (z - z.min()) / (z.max() - z.min())
    col = np.array([CLASS_COLOR[c] for c in cls], np.float32)
    alpha = 0.10 + 0.22 * (1 - depth)  # bliższe jaśniejsze
    gray = np.all(col == ZINC, axis=1)
    alpha[gray] *= 0.55
    teal = np.all(col == TEAL, axis=1)
    alpha[teal] *= 0.28  # płaty wzrokowe są bardzo gęste — bez tego przepalają się do jednolitej plamy
    splat(img, u, v, col, alpha, r=2)
    img = BG + img * 0.9
    out = np.clip(img, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (0, 0), 0.6)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def circuit(W=2000, H=1500, yaw=0.55, pitch=0.28, scale=1.3, horizontal=False):
    """Szkielety 863 neuronów lotu (SWC) na przygaszonym tle som."""
    img = np.zeros((H, W, 3), np.float32)
    u0, v0, z = project(xyz, yaw, pitch, 1.0, 0, 0)
    u, v, s_fit = fit(u0, v0, W, H, horizontal=horizontal)
    cu, cv_ = (np.percentile(u0 if not horizontal else v0, [0.3, 99.7]).mean(),
               np.percentile(v0 if not horizontal else -u0, [0.3, 99.7]).mean())

    def tf(pu, pv):
        if horizontal:
            pu, pv = pv, -pu
        return (pu - cu) * s_fit + W / 2, (pv - cv_) * s_fit + H / 2
    splat(img, u, v, np.full((len(u), 3), 120, np.float32), np.full(len(u), 0.05), r=1)
    gcol = {"dn_flight_power": AMBER, "dn_flight_steering": (245, 158, 11), "wing_power": ROSE,
            "wing_steering": (244, 63, 94), "wing_tension": (225, 29, 72), "haltere_aff": BLUE}
    layer = np.zeros_like(img)
    for n in d["neurons"]:
        g = n["group"].rsplit("_", 1)[0]
        c = gcol.get(g, ZINC)
        for line in n["lines"]:
            p = np.array(line, np.float32)
            pu, pv, _ = project(p, yaw, pitch, 1.0, 0, 0)
            pu, pv = tf(pu, pv)
            pts = np.stack([pu, pv], 1).astype(np.int32)
            cv2.polylines(layer, [pts], False, tuple(float(x) * 0.30 for x in c), 2, cv2.LINE_AA)
    img = BG + img + layer
    out = np.clip(img, 0, 255).astype(np.uint8)
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
    what = sys.argv[1:] or ["connectome", "circuit", "sim", "retina", "finish"]
    if "connectome" in what:
        connectome(2600, 1300, yaw=0.5, pitch=0.25, name="connectome_wide.png", horizontal=True)
        connectome(1400, 1700, yaw=0.55, pitch=0.25, name="connectome_3d.png")
    if "circuit" in what:
        circuit(1400, 1700)
    if "sim" in what:
        sim_images()
    if "retina" in what:
        retina()
    if "finish" in what:
        finish()
    print("ok", sorted(p.name for p in OUT.iterdir()))
