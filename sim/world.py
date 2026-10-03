"""Scena dla pętli z BANC: X2 Osoby 3 (sim/assets/x2) + kamery-oczy Osoby 1 + przesuwalny cel.

Oczy: ``visual_pipeline.drone_eyes.eye_camera_xml`` (konwencja FlyGym, 512×450, fovy 157°).
Cel: ciemny słup jak w ``scripts/example_sim_client.py`` (na nim robiona jest kalibracja), ale na
ciele mocap, więc ``DroneEnv`` przestawia go w ``reset`` bez ponownej kompilacji modelu.
Reszta świata (niebo, podłoże, światło) ze wspólnego ``common.xml`` Osoby 3, podłoga z ``scene_hover.xml``.

Pliki ``*_eyes.xml`` są generowane obok assets (MJCF odwołuje się do siatek względnie)
i są w ``.gitignore``.
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


def _replace(text: str, old: str, new: str, where: str) -> str:
    if old not in text:
        raise ValueError(f"nie znaleziono {old!r} w {where}")
    return text.replace(old, new, 1)


def build_scene(extra_worldbody: str = "") -> Path:
    """Zapisuje scenę z oczami i celem, zwraca ścieżkę do ``scene_eyes.xml``."""
    x2 = (ASSETS / "x2" / "x2.xml").read_text(encoding="utf-8")
    eyes = "\n      ".join(eye_camera_xml(e) for e in ("left", "right"))
    x2 = _replace(x2, X2_BODY, f"{X2_BODY}\n      {eyes}", "x2/x2.xml")
    (ASSETS / "x2" / "x2_eyes.xml").write_text(x2, encoding="utf-8")

    common = (ASSETS / "common.xml").read_text(encoding="utf-8")
    common = _replace(common, 'file="x2/x2.xml"', 'file="x2/x2_eyes.xml"', "common.xml")
    # Domyślny bufor offscreen ma 640×480, a oko potrzebuje 512 px wysokości.
    common = _replace(common, "<global ", f'<global offwidth="{max(EYE_W, 1280)}" offheight="{max(EYE_H, 720)}" ',
                      "common.xml")
    (ASSETS / "common_eyes.xml").write_text(common, encoding="utf-8")

    scene = (ASSETS / "scene_hover.xml").read_text(encoding="utf-8")
    scene = _replace(scene, 'file="common.xml"', 'file="common_eyes.xml"', "scene_hover.xml")
    scene = _replace(scene, "</worldbody>", f"  {BEACON}\n{extra_worldbody}\n  </worldbody>", "scene_hover.xml")
    out = ASSETS / "scene_eyes.xml"
    out.write_text(scene, encoding="utf-8")
    return out
