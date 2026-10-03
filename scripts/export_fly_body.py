"""Ciało muszki (NeuroMechFly z FlyGym) jako siatka w układzie BANC — do półprzezroczystej nakładki w eksploratorze.

    python scripts/export_fly_body.py        # .venv312 → data/viz/fly_body.bin + fly_body.json

Model: oficjalne siatki NeuroMechFly (flygym, ≤ 2000 ścian na część), poza spoczynkowa (keyframe „neutral”, jeśli
jest). Dopasowanie do BANC v888 jest ILUSTRACYJNE: obrót tak, jak ułożony jest preparat BANC (głowa do góry,
strona brzuszna do widza, prawa strona muchy po lewej stronie ekranu), skala i przesunięcie tak, żeby środek głowy
trafił w środek mózgu BANC, a środek tułowia w środek VNC. W BANC łącznik szyjny jest rozciągnięty (preparat
wyjęty z ciała), więc anatomiczna zgodność jest przybliżona.

Plik binarny: float32 pozycje [n × 3] w µm BANC, potem uint32 indeksy [m × 3], potem uint8 część ciała [n]
(0 tułów/głowa/odwłok, 1 oczy, 2 nogi, 3 skrzydła/haltery, 4 czułki/aparat gębowy). Metadane w .json.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "viz"
WING_SPREAD_DEG = 65  # skrzydła rozłożone na boki (w pozie „neutral” leżą złożone na odwłoku) — tylko wizualnie
PARTS = {"eye": 1, "coxa": 2, "trochanter": 2, "femur": 2, "tibia": 2, "tarsus": 2, "wing": 3, "haltere": 3,
         "arista": 4, "funiculus": 4, "pedicel": 4, "rostrum": 4, "haustellum": 4}


def part_of(name: str) -> int:
    return next((v for k, v in PARTS.items() if k in name), 0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import mujoco
    import pandas as pd

    from flygym.compose import Fly

    warnings.filterwarnings("ignore")
    m = Fly().compile()[0]
    d = mujoco.MjData(m)
    if m.nkey and "neutral" in [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) for k in range(m.nkey)]:
        mujoco.mj_resetDataKeyframe(m, d, m.key("neutral").id)
    mujoco.mj_forward(m, d)

    pos, tri, part = [], [], []
    n0 = 0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        v = m.mesh_vert[va:va + vn] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.endswith("_wing"):  # obrót wokół zawiasu (początek ciała skrzydła) i osi grzbietowej z
            hinge = d.xpos[m.geom_bodyid[g]]
            a = np.deg2rad(WING_SPREAD_DEG) * (-1 if name.startswith("l") else 1)  # lewe skrzydło na +y (lewo)
            c, sn = np.cos(a), np.sin(a)
            Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
            v = (v - hinge) @ Rz.T + hinge
        pos.append(v)
        tri.append(m.mesh_face[fa:fa + fn] + n0)
        part.append(np.full(vn, part_of(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")))
        n0 += vn
    pos, tri, part = np.vstack(pos), np.vstack(tri), np.concatenate(part).astype(np.uint8)

    # NeuroMechFly: x przód, y lewo, z góra (mm). Układ BANC (jak w eksploratorze przed toScene): x poprzecznie
    # (lewa strona muchy → większe x), y w stronę VNC (w dół ekranu), z w głąb. Głowa do góry, brzuszna strona do widza.
    R = np.array([[0, 1, 0],    # x_banc ← lewo
                  [-1, 0, 0],   # y_banc ← −przód (głowa = małe y, jak mózg w BANC)
                  [0, 0, -1]])  # z_banc ← −góra
    head = d.geom_xpos[m.geom("c_head").id] @ R.T
    thorax = d.geom_xpos[m.geom("c_thorax").id] @ R.T

    meta = pd.read_feather(ROOT / "data" / "banc_888" / "banc_888_meta.feather", columns=["position", "super_class"])
    xyz = np.array([[float(v) for v in p.split(",")] for p in meta.position]) * np.array([4, 4, 45]) / 1000.0
    brain = xyz[meta.super_class.isin(["central_brain_intrinsic", "optic_lobe_intrinsic", "visual_projection"])].mean(0)
    vnc = xyz[meta.super_class.isin(["ventral_nerve_cord_intrinsic"])].mean(0)
    s = np.linalg.norm(vnc[:2] - brain[:2]) / np.linalg.norm(thorax[:2] - head[:2])  # µm na mm modelu
    t = brain - s * head
    banc = (pos @ R.T) * s + t

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "fly_body.bin", "wb") as f:
        f.write(banc.astype(np.float32).tobytes())
        f.write(tri.astype(np.uint32).tobytes())
        f.write(part.tobytes())
    info = {"vertices": int(len(banc)), "triangles": int(len(tri)), "scale_um_per_mm": float(s),
            "parts": ["body", "eyes", "legs", "wings", "antennae_mouth"],
            "source": "NeuroMechFly (flygym) meshes; alignment to BANC v888 is illustrative (head→brain, thorax→VNC)"}
    (OUT / "fly_body.json").write_text(json.dumps(info, indent=1))
    print(f"{len(banc)} wierzchołków, {len(tri)} trójkątów, skala {s:.0f} µm/mm → {OUT / 'fly_body.bin'}")


if __name__ == "__main__":
    main()
