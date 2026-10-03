"""Przygotowanie klatek z dowolnej kamery (np. Gazebo Osoby 3) do flygym Retina."""

from __future__ import annotations

import numpy as np
from PIL import Image

RETINA_SHAPE = (512, 450, 3)  # H, W, C


def prepare_frame(img: np.ndarray) -> np.ndarray:
    """uint8/float, (H, W), (H, W, 3) lub (H, W, 4) → uint8 RGB (512, 450, 3).

    Inny rozmiar jest skalowany (bez zachowania proporcji: Retina i tak rzutuje kadr na
    oko muchy). Float zakłada zakres [0, 1].
    """
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3]
    if img.shape != RETINA_SHAPE:
        img = np.asarray(Image.fromarray(img).resize((RETINA_SHAPE[1], RETINA_SHAPE[0]), Image.BILINEAR))
    return img


def to_luminance(img: np.ndarray) -> np.ndarray:
    """RGB → szara luminancja powielona na 3 kanały.

    Retina czyta w każdym omatidium tylko kanał G albo B (typ yellow/pale). W kolorowej
    scenie (np. niebieskawe niebo MuJoCo) sąsiednie omatidia różnią się wtedy jasnością
    i FlyVis dostaje fałszywy kontrast w szachownicę. FlyVis był trenowany na luminancji.
    """
    # Wagi BT.601 ×256 w liczbach całkowitych: ~5× szybciej niż float32 @ wagi.
    r, g, b = (img[..., i].astype(np.uint16) for i in range(3))
    lum = ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
    out = np.empty(img.shape[:2] + (3,), np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = lum
    return out
