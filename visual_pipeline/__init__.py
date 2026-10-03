"""Osoba 1 — wzrok → BANC. Kamera RGB → FlyGym Retina → RetinaMapper → FlyVis → neurony BANC.

Wymaga zależności z ``pip install -e .[vision]`` (Python 3.12) i wag FlyVis
(``flyvis download-pretrained``). Importy są leniwe, żeby ``banc_control`` działał bez nich.
"""

__all__ = ["FlyVisStepper", "RetinaMapper", "VisionBridge"]


def __getattr__(name):
    if name == "VisionBridge":
        from .bridge import VisionBridge
        return VisionBridge
    if name == "FlyVisStepper":
        from .flyvis_step import FlyVisStepper
        return FlyVisStepper
    if name == "RetinaMapper":
        from .retina_mapper import RetinaMapper
        return RetinaMapper
    raise AttributeError(name)
