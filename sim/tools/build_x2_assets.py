"""Generuje zasoby drona X2 do sim/assets/x2/ z MuJoCo Menagerie.

Siatka Skydio X2 ma śmigła wypalone jako płaskie tarcze z teksturą rozmycia
(12 płaskich komponentów, po 3 na wirnik). Ten skrypt je wycina, żeby w ich
miejscu rysować obracające się łopaty (sim/propellers.py), i wypisuje środki
tarcz w układzie drona — trafiają one do x2.xml jako site'y prop1..prop4.

Użycie (raz, po sparse-checkout skydio_x2 do third_party/, patrz docs/osoba3-plan.md):
    python sim/tools/build_x2_assets.py
"""

import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "third_party" / "mujoco_menagerie" / "skydio_x2"
DST = ROOT / "sim" / "assets" / "x2"

# Tarcza śmigła: płaski komponent ~26.5 x 26.5 cm (jednostki OBJ to cm).
DISC_MIN_EXTENT = 20.0
DISC_MAX_THICKNESS = 0.5
MESH_SCALE = 0.01
# Obrót siatki z x2.xml (quat="0 0 1 1"): (x, y, z)_obj -> (-x, z, y)_model.
OBJ_TO_MODEL = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)


def read_obj(path):
    v, vt, vn, faces, header = [], [], [], [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v":
            v.append([float(x) for x in parts[1:4]])
        elif tag == "vt":
            vt.append([float(x) for x in parts[1:3]])
        elif tag == "vn":
            vn.append([float(x) for x in parts[1:4]])
        elif tag == "f":
            faces.append([tuple(int(i) - 1 for i in t.split("/")) for t in parts[1:]])
        elif tag in ("mtllib", "usemtl"):
            header.append(line)
    return np.array(v), np.array(vt), np.array(vn), faces, header


def face_components(n_verts, faces):
    parent = np.arange(n_verts)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for f in faces:
        root = find(f[0][0])
        for corner in f[1:]:
            other = find(corner[0])
            if other != root:
                parent[other] = root
    comps = {}
    for fi, f in enumerate(faces):
        comps.setdefault(find(f[0][0]), []).append(fi)
    return list(comps.values())


def write_obj(path, v, vt, vn, faces, header):
    used = [sorted({c[k] for f in faces for c in f}) for k in range(3)]
    remap = [{old: new for new, old in enumerate(u)} for u in used]
    lines = [header[0]] if header else []
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v[used[0]]]
    lines += [f"vt {a:.6f} {b:.6f}" for a, b in vt[used[1]]]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vn[used[2]]]
    lines += header[1:]
    for f in faces:
        lines.append("f " + " ".join("/".join(str(remap[k][c[k]] + 1) for k in range(3)) for c in f))
    path.write_text("\n".join(lines) + "\n")


def main():
    v, vt, vn, faces, header = read_obj(SRC / "assets" / "X2_lowpoly.obj")
    discs, keep = [], []
    for comp in face_components(len(v), faces):
        pts = v[sorted({c[0] for fi in comp for c in faces[fi]})]
        extent = pts.max(0) - pts.min(0)
        flat = np.sort(extent)
        if flat[0] < DISC_MAX_THICKNESS and flat[1] > DISC_MIN_EXTENT:
            discs.append((pts.min(0) + pts.max(0)) / 2)
        else:
            keep.extend(comp)
    assert len(discs) == 12, f"oczekiwano 12 tarcz śmigieł, znaleziono {len(discs)}"

    DST.mkdir(parents=True, exist_ok=True)
    write_obj(DST / "X2_noprops.obj", v, vt, vn, [faces[i] for i in sorted(keep)], header)
    shutil.copy(SRC / "assets" / "X2_lowpoly_texture_SpinningProps_1024.png", DST / "X2_texture.png")
    shutil.copy(SRC / "LICENSE", DST / "LICENSE")

    # 3 warstwy na wirnik -> uśrednij do 4 środków w układzie modelu.
    centers = np.array(discs) @ OBJ_TO_MODEL.T * MESH_SCALE
    rotors = []
    for c in centers:
        for group in rotors:
            if np.linalg.norm(group[0][:2] - c[:2]) < 0.05:
                group.append(c)
                break
        else:
            rotors.append([c])
    print(f"usunięto {len(faces) - len(keep)} ścian tarcz, zostało {len(keep)}")
    for group in rotors:
        x, y, z = np.mean(group, axis=0)
        print(f'środek wirnika: pos="{x:.4f} {y:.4f} {z:.4f}"')


if __name__ == "__main__":
    main()
