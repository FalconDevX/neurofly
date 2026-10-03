"""Protokół ZMQ symulator (Osoba 3) ↔ serwer wzroku (Osoba 1, opcjonalnie z kontrolerem Osoby 2).

Lekki moduł: tylko numpy, pyzmq i json, więc działa w środowisku symulatora bez torch/FlyVis.

Żądanie (REQ → REP), wiadomość wieloczęściowa:
    [nagłówek JSON, klatka lewa, klatka prawa]
    nagłówek: {"shape": [H, W, C], "imu": {"gyro": [r, p, y], "accel": [x, y, z]} | null,
               "reset": bool}
    klatki: surowe bajty uint8 (C-order), np. z ``drone_eyes.MujocoEyes.render``.
    imu w konwencji ``banc_control.ImuState``: gyro = (roll, pitch, yaw) rad/s, yaw + = w prawo
    (w MuJoCo yaw = −ω_z), roll + = prawe skrzydło w dół.

Odpowiedź:
    tryb "command":  [JSON {"thrust", "roll", "pitch", "yaw", "unmatched_ids", "timing_ms"}]
    tryb "activity": [JSON {"n", "timing_ms"}, root_ids int64, activity float32]
"""

from __future__ import annotations

import json

import numpy as np

DEFAULT_ADDRESS = "tcp://127.0.0.1:5555"


def encode_request(left: np.ndarray, right: np.ndarray, imu: dict | None = None,
                   reset: bool = False) -> list[bytes]:
    left, right = np.ascontiguousarray(left, np.uint8), np.ascontiguousarray(right, np.uint8)
    if left.shape != right.shape:
        raise ValueError(f"klatki mają różne kształty: {left.shape} vs {right.shape}")
    header = {"shape": list(left.shape), "imu": imu, "reset": reset}
    return [json.dumps(header).encode(), left.tobytes(), right.tobytes()]


def decode_request(parts: list[bytes]) -> tuple[np.ndarray, np.ndarray, dict | None, bool]:
    header = json.loads(parts[0])
    shape = tuple(header["shape"])
    left = np.frombuffer(parts[1], np.uint8).reshape(shape)
    right = np.frombuffer(parts[2], np.uint8).reshape(shape)
    return left, right, header.get("imu"), bool(header.get("reset", False))


class VisionClient:
    """Strona symulatora: ``step(lewa, prawa, imu)`` → słownik komendy (tryb "command")."""

    def __init__(self, address: str = DEFAULT_ADDRESS, timeout_ms: int = 10_000) -> None:
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(address)

    def step(self, left: np.ndarray, right: np.ndarray, imu: dict | None = None,
             reset: bool = False) -> dict:
        self.sock.send_multipart(encode_request(left, right, imu, reset))
        parts = self.sock.recv_multipart()
        reply = json.loads(parts[0])
        if len(parts) == 3:  # tryb "activity"
            reply["root_ids"] = np.frombuffer(parts[1], np.int64)
            reply["activity"] = np.frombuffer(parts[2], np.float32)
        return reply
