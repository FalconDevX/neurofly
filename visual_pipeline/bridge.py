"""Most FlyVis → BANC: klatki z dwóch kamer → aktywność neuronów BANC w kontrakcie Osoby 2."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from flygym.vision.retina import Retina

from banc_control.contracts import BancActivation, VisualBatch

from .flyvis_step import FlyVisStepper
from .frames import prepare_frame, to_luminance
from .retina_mapper import RetinaMapper

MAP_FILE = Path(__file__).with_name("flyvis_banc_map.csv")
EYES = ("left", "right")


class VisionBridge:
    """``step(frame_left, frame_right)`` → ``list[BancActivation]`` (``banc_root_id`` = ``banc_888_id``).

    Klatki: RGB z dowolnej kamery; ``prepare_frame`` skaluje je do (512, 450, 3) dla flygym Retina.
    Aktywność to surowa wartość z FlyVis; neuron BANC przypisany kilku komórkom FlyVis
    dostaje ich średnią.
    """

    def __init__(self, fps: float = 30.0, map_file: str | Path = MAP_FILE, fisheye: bool = False) -> None:
        """``fisheye=True`` dla surowych kamer MuJoCo (``drone_eyes``): ``Retina.correct_fisheye``
        jak w FlyGym. ``FakeStereoCamera`` renderuje już równokątnie, więc tam False."""
        self.fisheye = fisheye
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
        # Listy Pythona raz na starcie: rekordy co klatkę bez konwersji skalarów numpy.
        self._root_list = self.root_ids.tolist()
        self._type_list = [str(t) for t in self.cell_types]
        self.last_timings: dict[str, float] = {}  # ms ostatniego kroku, do profilowania

    def reset(self) -> None:
        """Stan FlyVis jak po 1 s szarego obrazu (np. przed nową sceną kalibracyjną)."""
        self.stepper.reset(batch_size=len(EYES))

    def step_arrays(self, frame_left: np.ndarray, frame_right: np.ndarray) -> np.ndarray:
        """Szybka ścieżka: aktywność w kolejności ``self.root_ids`` / ``self.cell_types``."""
        t0 = time.perf_counter()
        frame_left, frame_right = to_luminance(prepare_frame(frame_left)), to_luminance(prepare_frame(frame_right))
        if self.fisheye:
            frame_left, frame_right = self.retina.correct_fisheye(frame_left), self.retina.correct_fisheye(frame_right)
        lum = np.stack([self.mapper.to_flyvis(self.retina.raw_image_to_hex_pxls(f), eye)
                        for f, eye in zip((frame_left, frame_right), EYES)])
        t1 = time.perf_counter()
        act = self.stepper.step(lum)
        t2 = time.perf_counter()
        out = np.concatenate([
            np.bincount(codes, weights=act[b, fv_idx]) / counts
            for b, (fv_idx, codes, counts, _, _) in enumerate(self._eyes)
        ])
        t3 = time.perf_counter()
        self.last_timings = {"retina": (t1 - t0) * 1e3, "flyvis": (t2 - t1) * 1e3, "banc_map": (t3 - t2) * 1e3}
        return out

    def step(self, frame_left: np.ndarray, frame_right: np.ndarray) -> list[BancActivation]:
        act = self.step_arrays(frame_left, frame_right)
        t0 = time.perf_counter()
        records = list(map(BancActivation, self._root_list, self._type_list, act.tolist()))
        self.last_timings["records"] = (time.perf_counter() - t0) * 1e3
        return records

    def step_batch(self, frame_left: np.ndarray, frame_right: np.ndarray) -> VisualBatch:
        """Jak ``step``, ale bez tworzenia obiektów: dla ``BancController`` w tym samym procesie."""
        return VisualBatch(self.root_ids, self.step_arrays(frame_left, frame_right))

    def settle(self, frame_left: np.ndarray, frame_right: np.ndarray, n_frames: int = 30) -> VisualBatch:
        """Odpowiedź na scenę statyczną: reset FlyVis, potem ``n_frames`` tej samej klatki."""
        self.reset()
        for _ in range(n_frames - 1):
            self.step_arrays(frame_left, frame_right)
        return self.step_batch(frame_left, frame_right)
