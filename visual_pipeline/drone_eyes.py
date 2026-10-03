"""Kamery-oczy drona w MuJoCo w konwencji FlyGym (specyfikacja dla Osoby 3).

FlyGym renderuje każde oko kamerą MuJoCo o ``fovy`` 157° i rozmiarze 512×450 (H×W),
potem stosuje ``Retina.correct_fisheye`` i dopiero wtedy Retina. Tu to samo dla drona:

- 2 kamery na ciele drona, osie 70° w lewo i w prawo od kierunku lotu (+x), poziomo,
  ~14° widzenia obuocznego z przodu; góra obrazu = góra drona (+z);
- kadr surowy (prostoliniowy): korekcję „rybiego oka" robi ``VisionBridge(fisheye=True)``.

Konwencje osi MuJoCo dla Skydio X2 z mujoco_menagerie: +x przód, +y lewo, +z góra.
Kamera MuJoCo patrzy wzdłuż −z swojego układu, ``xyaxes`` = (prawo obrazu, góra obrazu).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EYE_FOVY = 157.0          # stopnie, pionowo (jak FlyGym vision.yaml)
EYE_H, EYE_W = 512, 450   # flygym Retina raw image
EYE_AZIMUTH = 70.0        # stopnie od kierunku lotu, + = w prawo
EYE_POS = (0.12, 0.03, 0.03)  # m, przód drona, ±y dla oczu


def eye_camera_xml(eye: str) -> str:
    """Element <camera> oka ``left`` / ``right`` do wstawienia w <body> drona."""
    az = np.deg2rad(EYE_AZIMUTH if eye == "right" else -EYE_AZIMUTH)
    # Kierunek patrzenia d = (cos az, -sin az, 0) (az + = w prawo, a +y to lewo).
    # Prawo obrazu = d × góra; dla prawego oka wskazuje do tyłu, więc przód jest po lewej.
    right = (-np.sin(az), -np.cos(az), 0.0)
    x, y, z = EYE_POS[0], (-1 if eye == "right" else 1) * EYE_POS[1], EYE_POS[2]
    return (f'<camera name="eye_{eye}" pos="{x} {y} {z}" fovy="{EYE_FOVY}" '
            f'xyaxes="{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} 0 0 1"/>')


def x2_with_eyes(menagerie_dir: str | Path, extra_worldbody: str = "") -> Path:
    """Zapisuje obok modelu X2 wersję z kamerami-oczami i zwraca ścieżkę do sceny.

    ``extra_worldbody``: dodatkowe obiekty w świecie (np. cel do testów).
    Pliki trafiają do katalogu modelu, bo MJCF odwołuje się do assets względnie.
    """
    d = Path(menagerie_dir)
    body = '<body name="x2" pos="0 0 0.1" childclass="x2">'
    x2 = (d / "x2.xml").read_text()
    if body not in x2:
        raise ValueError(f"nie znaleziono {body} w x2.xml")
    eyes = "\n      ".join(eye_camera_xml(e) for e in ("left", "right"))
    (d / "x2_eyes.xml").write_text(x2.replace(body, f"{body}\n      {eyes}"))
    scene = (d / "scene.xml").read_text().replace('file="x2.xml"', 'file="x2_eyes.xml"')
    # Domyślny bufor offscreen ma 640×480, a oko potrzebuje 512 px wysokości.
    scene = scene.replace("<global ", f'<global offwidth="{max(EYE_W, 640)}" offheight="{max(EYE_H, 480)}" ', 1)
    if extra_worldbody:
        scene = scene.replace("</worldbody>", f"{extra_worldbody}\n  </worldbody>")
    out = d / "scene_eyes.xml"
    out.write_text(scene)
    return out


DRONE_GEOM_GROUPS = (2, 3)  # X2: 2 = siatki wizualne, 3 = kolizje; oczy ich nie widzą


class MujocoEyes:
    """Renderuje surowe klatki obu oczu z danego ``MjData`` (lewe, prawe), uint8 (512, 450, 3).

    Jak w FlyGym (oczy muchy nie widzą własnej głowy) oczy nie renderują geometrii drona:
    inaczej śmigło i kadłub zasłaniają środek kadru i nie ruszają się przy obrocie.
    """

    def __init__(self, model) -> None:
        import mujoco

        self.renderer = mujoco.Renderer(model, height=EYE_H, width=EYE_W)
        self.cam = {e: model.camera(f"eye_{e}").id for e in ("left", "right")}
        self.option = mujoco.MjvOption()
        for g in DRONE_GEOM_GROUPS:
            self.option.geomgroup[g] = 0

    def render(self, data) -> tuple[np.ndarray, np.ndarray]:
        frames = []
        for eye in ("left", "right"):
            self.renderer.update_scene(data, camera=self.cam[eye], scene_option=self.option)
            frames.append(self.renderer.render().copy())
        return frames[0], frames[1]
