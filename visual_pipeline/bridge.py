"""Most FlyVis → BANC: klatki z dwóch kamer → aktywność neuronów BANC w kontrakcie Osoby 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flygym.vision.retina import Retina

from banc_control.contracts import BancActivation

from .flyvis_step import FlyVisStepper
from .retina_mapper import RetinaMapper

MAP_FILE = Path(__file__).with_name("flyvis_banc_map.csv")
EYES = ("left", "right")


class VisionBridge:
    """``step(frame_left, frame_right)`` → ``list[BancActivation]`` (``banc_root_id`` = ``banc_888_id``).

    Klatki: ``uint8`` RGB o kształcie (512, 450, 3), tyle oczekuje flygym Retina.
    Aktywność to surowa wartość z FlyVis; neuron BANC przypisany kilku komórkom FlyVis
    dostaje ich średnią.
    """

    def __init__(self, fps: float = 30.0, map_file: str | Path = MAP_FILE) -> None:
        self.retina = Retina()
        self.mapper = RetinaMapper()
        self.stepper = FlyVisStepper(fps=fps)
        self.stepper.reset(batch_size=len(EYES))

        m = pd.read_csv(map_file)
        self._eyes = []
        for eye in EYES:
            e = m[m.eye == eye]
            codes, root_ids = pd.factorize(e.banc_888_id)
            cell_types = e.groupby(codes).banc_cell_type.first().to_numpy()
            self._eyes.append((e.flyvis_index.to_numpy(), codes, np.bincount(codes),
                               root_ids.to_numpy(np.int64), cell_types))

        self.root_ids = np.concatenate([r for *_, r, _ in self._eyes])
        self.cell_types = np.concatenate([c for *_, c in self._eyes])

    def step_arrays(self, frame_left: np.ndarray, frame_right: np.ndarray) -> np.ndarray:
        """Szybka ścieżka: aktywność w kolejności ``self.root_ids`` / ``self.cell_types``."""
        shape = (self.retina.nrows, self.retina.ncols, 3)
        for f in (frame_left, frame_right):
            if f.shape != shape:
                raise ValueError(f"frame has shape {f.shape}, Retina expects {shape}")
        lum = np.stack([self.mapper.to_flyvis(self.retina.raw_image_to_hex_pxls(f), eye)
                        for f, eye in zip((frame_left, frame_right), EYES)])
        act = self.stepper.step(lum)
        return np.concatenate([
            np.bincount(codes, weights=act[b, fv_idx]) / counts
            for b, (fv_idx, codes, counts, _, _) in enumerate(self._eyes)
        ])

    def step(self, frame_left: np.ndarray, frame_right: np.ndarray) -> list[BancActivation]:
        act = self.step_arrays(frame_left, frame_right)
        return [BancActivation(int(r), str(c), float(a))
                for r, c, a in zip(self.root_ids, self.cell_types, act)]
