"""Logika serwera wzroku bez gniazda ZMQ (protokół: ``zmq_protocol.py``), żeby dało się ją testować.

``bridge``: ``VisionBridge`` albo obiekt o tym samym interfejsie (``reset``, ``settle``,
``step_batch``, ``last_timings``). ``ctrl``: ``BancController`` (tryb "command") lub None ("activity").
"""

from __future__ import annotations

import time

import numpy as np

from banc_control import ImuState
from banc_control.contracts import VisualBatch

from .zmq_protocol import CALIB_SCENES, VisionClient, decode


def calibrate_controller(ctrl, scenes: dict[str, list[VisualBatch]]) -> dict:
    """Kalibracja w kolejności: spoczynek → skala → znak yaw (obrót) → znaki halter → wzmocnienie halter."""
    neutral = scenes["neutral"][-1]
    ctrl.calibrate_rest(30, visual=neutral)
    ctrl.calibrate_scale([scenes["left"][-1], scenes["right"][-1]])
    out = {}
    if scenes.get("turn_left") and scenes.get("turn_right"):
        out["optomotor_yaw_diff"] = ctrl.calibrate_yaw_sign(scenes["turn_right"], scenes["turn_left"])
    out["haltere_sign"] = ctrl.calibrate_haltere_sign(neutral)
    out["haltere_gain"] = ctrl.calibrate_haltere_gain(neutral)
    out["haltere_yaw_sign"] = ctrl.haltere_yaw_sign
    out["yaw_axis_sign"] = ctrl.yaw_axis_sign
    for name in ("left", "right"):  # kontrola: odpowiedź na cel z boku po kalibracji
        ctrl.dyn.reset()
        for _ in range(30):
            cmd = ctrl.step(scenes[name][-1])
        out[f"cmd_{name}"] = {"thrust": cmd.thrust, "roll": cmd.roll, "yaw": cmd.yaw}
    ctrl.dyn.reset()
    return out


class ControlServer:
    """``decoder_weights``: plik z ``train_decoder.py``, wczytywany po każdej kalibracji (Plan A/B)."""

    def __init__(self, bridge, ctrl=None, decoder_weights=None) -> None:
        self.bridge, self.ctrl = bridge, ctrl
        self.decoder_weights = decoder_weights
        self.scenes: dict[str, list[VisualBatch]] = {}

    def _calib(self, scene: str, left, right) -> dict:
        if self.ctrl is None:
            return {"error": "kalibracja tylko w trybie command"}
        if scene == "finish":
            missing = [s for s in ("neutral", "left", "right") if s not in self.scenes]
            if missing:
                return {"error": f"brak scen kalibracyjnych: {missing}"}
            out = calibrate_controller(self.ctrl, self.scenes)
            if self.decoder_weights:
                self.ctrl.decoder.load_weights(self.decoder_weights)
                out["decoder"] = str(self.decoder_weights)
            self.scenes = {}
            self.bridge.reset()
            return {"ok": True, "scene": "finish", **out}
        if scene not in CALIB_SCENES:
            return {"error": f"nieznana scena {scene!r}, dozwolone: {CALIB_SCENES} i 'finish'"}
        frames = self.scenes.setdefault(scene, [])
        if scene.startswith("turn"):
            if not frames:  # bez skoku szare tło → scena na początku obrotu
                self.bridge.settle(left, right)
            frames.append(self.bridge.step_batch(left, right))
        else:
            self.scenes[scene] = [self.bridge.settle(left, right)]
        return {"ok": True, "scene": scene, "frames": len(self.scenes[scene])}

    def handle(self, header: dict, left, right) -> tuple[dict, list[bytes]]:
        """Odpowiedź: (nagłówek JSON, dodatkowe części binarne)."""
        if header.get("calib"):
            return self._calib(header["calib"], left, right), []
        t = time.perf_counter()
        if header.get("reset"):
            self.bridge.reset()
            # Bez skoku szare tło → scena: wzrok i dynamika BANC dochodzą do stanu dla tej klatki.
            batch = self.bridge.settle(left, right)
            if self.ctrl is not None:
                self.ctrl.dyn.reset()
                self.ctrl.warm_start(batch)
        else:
            batch = self.bridge.step_batch(left, right)
        timing = {k: round(v, 2) for k, v in self.bridge.last_timings.items()}
        if self.ctrl is None:
            timing["total"] = round((time.perf_counter() - t) * 1e3, 2)
            return ({"n": len(batch.root_ids), "timing_ms": timing},
                    [np.asarray(batch.root_ids, np.int64).tobytes(),
                     np.asarray(batch.activity, np.float32).tobytes()])
        imu = header.get("imu")
        state = ImuState(**{k: tuple(v) for k, v in imu.items()}) if imu else None
        cmd = self.ctrl.step(batch, state).clipped()
        timing["total"] = round((time.perf_counter() - t) * 1e3, 2)
        return ({"thrust": cmd.thrust, "roll": cmd.roll, "pitch": cmd.pitch, "yaw": cmd.yaw,
                 "unmatched_ids": self.ctrl.unmatched_ids, "timing_ms": timing}, [])


class LocalClient(VisionClient):
    """``VisionClient`` bez gniazda: ``ControlServer`` w tym samym procesie (trening dekodera, testy).

    Wiadomości przechodzą przez te same ``encode_request`` / ``decode`` co po ZMQ.
    """

    def __init__(self, server: ControlServer) -> None:
        self.server = server

    def _send(self, parts: list[bytes]) -> dict:
        reply, _ = self.server.handle(*decode(parts))
        if "error" in reply:
            raise RuntimeError(f"serwer wzroku: {reply['error']}")
        return reply
