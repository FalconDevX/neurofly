"""Protokół ZMQ symulator (Osoba 3) ↔ serwer wzroku (Osoba 1, opcjonalnie z kontrolerem Osoby 2).

Lekki moduł: tylko numpy, pyzmq i json, więc działa w środowisku symulatora bez torch/FlyVis.

Żądanie (REQ → REP), wiadomość wieloczęściowa:
    [nagłówek JSON, klatka lewa, klatka prawa]   (``calib: "finish"``: sam nagłówek)
    nagłówek: {"shape": [H, W, C], "imu": {"gyro": [r, p, y], "accel": [x, y, z]} | null,
               "reset": bool, "calib": null | scena kalibracyjna}
    klatki: surowe bajty uint8 (C-order), np. z ``drone_eyes.MujocoEyes.render``.
    imu w konwencji ``banc_control.ImuState``: gyro = (roll, pitch, yaw) rad/s, yaw + = w prawo
    (w MuJoCo yaw = −ω_z), roll + = prawe skrzydło w dół.

Kalibracja na scenach z symulatora (tryb "command", ``VisionClient.calibrate``):
    "neutral" / "left" / "right": jedna klatka statyczna, cel na wprost / 60° w lewo / w prawo;
    "turn_left" / "turn_right": kolejne klatki wymuszonego obrotu drona (odruch optomotoryczny);
    "finish": serwer kalibruje kontroler na zebranych scenach i odpowiada podsumowaniem.

Odpowiedź:
    tryb "command":  [JSON {"thrust", "roll", "pitch", "yaw", "unmatched_ids", "timing_ms"}]
    tryb "activity": [JSON {"n", "timing_ms"}, root_ids int64, activity float32]
    kalibracja:      [JSON {"ok", "scene", "frames"}], po "finish" wynik kalibracji
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

DEFAULT_ADDRESS = "tcp://127.0.0.1:5555"
CALIB_SCENES = ("neutral", "left", "right", "turn_left", "turn_right")


def encode_request(left: np.ndarray | None, right: np.ndarray | None, imu: dict | None = None,
                   reset: bool = False, calib: str | None = None) -> list[bytes]:
    if left is None:
        return [json.dumps({"shape": None, "imu": imu, "reset": reset, "calib": calib}).encode()]
    left, right = np.ascontiguousarray(left, np.uint8), np.ascontiguousarray(right, np.uint8)
    if left.shape != right.shape:
        raise ValueError(f"klatki mają różne kształty: {left.shape} vs {right.shape}")
    header = {"shape": list(left.shape), "imu": imu, "reset": reset, "calib": calib}
    return [json.dumps(header).encode(), left.tobytes(), right.tobytes()]


def decode(parts: list[bytes]) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """Nagłówek i klatki (None, gdy wiadomość ma sam nagłówek)."""
    header = json.loads(parts[0])
    if len(parts) < 3:
        return header, None, None
    shape = tuple(header["shape"])
    left = np.frombuffer(parts[1], np.uint8).reshape(shape)
    right = np.frombuffer(parts[2], np.uint8).reshape(shape)
    return header, left, right


def decode_request(parts: list[bytes]) -> tuple[np.ndarray, np.ndarray, dict | None, bool]:
    header, left, right = decode(parts)
    return left, right, header.get("imu"), bool(header.get("reset", False))


class VisionClient:
    """Strona symulatora: ``step(lewa, prawa, imu)`` → słownik komendy (tryb "command")."""

    def __init__(self, address: str = DEFAULT_ADDRESS, timeout_ms: int = 10_000) -> None:
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(address)

    def _send(self, parts: list[bytes]) -> dict:
        self.sock.send_multipart(parts)
        parts = self.sock.recv_multipart()
        reply = json.loads(parts[0])
        if len(parts) == 3:  # tryb "activity"
            reply["root_ids"] = np.frombuffer(parts[1], np.int64)
            reply["activity"] = np.frombuffer(parts[2], np.float32)
        if "error" in reply:
            raise RuntimeError(f"serwer wzroku: {reply['error']}")
        return reply

    def step(self, left: np.ndarray, right: np.ndarray, imu: dict | None = None,
             reset: bool = False) -> dict:
        return self._send(encode_request(left, right, imu, reset))

    def calibrate(self, render: Callable[[float], tuple[np.ndarray, np.ndarray]], fps: float = 30.0,
                  side_deg: float = 60.0, turn_deg_s: float = 90.0, turn_frames: int = 45) -> dict:
        """Kalibracja kontrolera na scenach z symulatora.

        ``render(bearing)`` → (lewa, prawa) klatka, gdy cel jest pod kątem ``bearing`` [rad]
        (+ = cel w prawo) od kierunku lotu; dron poziomo, w zawisie. Obrót drona w prawo
        przesuwa cel w lewo, więc ``turn_right`` to malejący ``bearing``.
        """
        side = np.deg2rad(side_deg)
        for scene, bearing in (("neutral", 0.0), ("left", -side), ("right", side)):
            self._send(encode_request(*render(bearing), calib=scene))
        step = np.deg2rad(turn_deg_s) / fps
        for scene, direction in (("turn_right", 1.0), ("turn_left", -1.0)):
            for k in range(turn_frames):
                self._send(encode_request(*render(-direction * step * k), calib=scene))
        return self._send(encode_request(None, None, calib="finish"))
