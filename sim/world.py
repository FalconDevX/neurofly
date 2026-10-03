"""Scena dla pętli z BANC: X2 Osoby 3 (sim/assets/x2) + kamery-oczy Osoby 1 + przesuwalny cel.

Oczy: ``visual_pipeline.drone_eyes.eye_camera_xml`` (konwencja FlyGym, 512×450, fovy 157°).
Cel: ciemny słup jak w ``scripts/example_sim_client.py`` (na nim robiona jest kalibracja), ale na
ciele mocap, więc ``DroneEnv`` przestawia go w ``reset`` bez ponownej kompilacji modelu.

Pliki ``scene_eyes.xml`` i ``x2/x2_eyes.xml`` są generowane obok assets (MJCF odwołuje się
do siatek względnie) i są w ``.gitignore``.
"""

from __future__ import annotations

from pathlib import Path

from visual_pipeline.drone_eyes import EYE_H, EYE_W, eye_camera_xml

ASSETS = Path(__file__).parent / "assets"
X2_BODY = '<body name="x2" pos="0 0 0.1" childclass="x2">'

# Słup 4 m wysokości, promień 0.15 m; contype/conaffinity 0, żeby dron nie mógł w niego uderzyć.
BEACON = ('<body name="beacon" mocap="true" pos="4 0 0">\n'
          '      <geom type="cylinder" size="0.15 2" pos="0 0 2" rgba="0.05 0.05 0.05 1" '
          'contype="0" conaffinity="0"/>\n'
          '    </body>')


def build_scene(extra_worldbody: str = "") -> Path:
    """Zapisuje scenę z oczami i celem, zwraca ścieżkę do ``scene_eyes.xml``."""
    x2 = (ASSETS / "x2" / "x2.xml").read_text(encoding="utf-8")
    if X2_BODY not in x2:
        raise ValueError(f"nie znaleziono {X2_BODY} w sim/assets/x2/x2.xml")
    eyes = "\n      ".join(eye_camera_xml(e) for e in ("left", "right"))
    (ASSETS / "x2" / "x2_eyes.xml").write_text(x2.replace(X2_BODY, f"{X2_BODY}\n      {eyes}"), encoding="utf-8")

    scene = (ASSETS / "scene_hover.xml").read_text(encoding="utf-8")
    scene = scene.replace('file="x2/x2.xml"', 'file="x2/x2_eyes.xml"')
    # Domyślny bufor offscreen ma 640×480, a oko potrzebuje 512 px wysokości.
    scene = scene.replace("<global ", f'<global offwidth="{max(EYE_W, 1280)}" offheight="{max(EYE_H, 720)}" ', 1)
    scene = scene.replace("</worldbody>", f"  {BEACON}\n{extra_worldbody}\n  </worldbody>")
    out = ASSETS / "scene_eyes.xml"
    out.write_text(scene, encoding="utf-8")
    return out
