"""Protokół ZMQ symulator (Osoba 3) ↔ serwer wzroku (Osoba 1, opcjonalnie z kontrolerem Osoby 2).

Lekki moduł: tylko numpy, pyzmq i json, więc działa w środowisku symulatora bez torch/FlyVis.

Każde żądanie (REQ → REP) to wiadomość wieloczęściowa [nagłówek JSON, klatka, klatka, ...];
klatki to surowe bajty uint8 (C-order) o kształcie ``header["shape"]``, parami (lewa, prawa),
np. z ``drone_eyes.MujocoEyes.render``.

Krok pętli:
    nagłówek {"shape", "imu": {"gyro": [r, p, y], "accel": [x, y, z]} | null, "reset": bool},
    klatki [lewa, prawa].
    imu w konwencji ``banc_control.ImuState``: gyro = (roll, pitch, yaw) rad/s, yaw + = w prawo
    (w MuJoCo yaw = −ω_z), roll + = prawe skrzydło w dół.
    Odpowiedź, tryb "command":  [JSON {"thrust", "roll", "pitch", "yaw", "unmatched_ids",
                                       "calibration", "timing_ms"}]
              tryb "activity": [JSON {"n", "timing_ms"}, root_ids int64, activity float32]

Kalibracja kontrolera na scenach z symulatora (tylko tryb "command"):
    nagłówek {"shape", "calibrate": ["rest", "left", "right"]},
    klatki [lewa, prawa] dla każdej sceny w tej kolejności: dron w zawisie, cel na wprost /
    60° w lewo / 60° w prawo, bez ruchu. Odpowiedź: [JSON {"calibration": "sim", "haltere_sign"}].
"""

from __future__ import annotations

import json

import numpy as np

DEFAULT_ADDRESS = "tcp://127.0.0.1:5555"
CALIBRATION_SCENES = ("rest", "left", "right")
CALIBRATION_BEARING_DEG = {"rest": 0.0, "left": -60.0, "right": 60.0}


def encode_message(header: dict, frames: list[np.ndarray]) -> list[bytes]:
    frames = [np.ascontiguousarray(f, np.uint8) for f in frames]
    if len({f.shape for f in frames}) > 1:
        raise ValueError(f"klatki mają różne kształty: {[f.shape for f in frames]}")
    header = {**header, "shape": list(frames[0].shape)}
    return [json.dumps(header).encode(), *(f.tobytes() for f in frames)]


def decode_message(parts: list[bytes]) -> tuple[dict, list[np.ndarray]]:
    header = json.loads(parts[0])
    shape = tuple(header["shape"])
    return header, [np.frombuffer(p, np.uint8).reshape(shape) for p in parts[1:]]


def encode_request(left: np.ndarray, right: np.ndarray, imu: dict | None = None,
                   reset: bool = False) -> list[bytes]:
    return encode_message({"imu": imu, "reset": reset}, [left, right])


def decode_request(parts: list[bytes]) -> tuple[np.ndarray, np.ndarray, dict | None, bool]:
    header, (left, right) = decode_message(parts)
    return left, right, header.get("imu"), bool(header.get("reset", False))


def encode_calibration(scenes: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[bytes]:
    """``scenes``: {"rest": (lewa, prawa), "left": ..., "right": ...}."""
    missing = set(CALIBRATION_SCENES) - set(scenes)
    if missing:
        raise ValueError(f"brak scen kalibracyjnych: {sorted(missing)}")
    frames = [f for name in CALIBRATION_SCENES for f in scenes[name]]
    return encode_message({"calibrate": list(CALIBRATION_SCENES)}, frames)


class VisionClient:
    """Strona symulatora: ``step(lewa, prawa, imu)`` → słownik komendy (tryb "command")."""

    def __init__(self, address: str = DEFAULT_ADDRESS, timeout_ms: int = 10_000) -> None:
        import zmq

        self._zmq = zmq
        self.timeout_ms = timeout_ms
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

    def calibrate(self, scenes: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
        """Kalibruje kontroler na scenach z symulatora (patrz ``CALIBRATION_BEARING_DEG``).

        Trwa kilka sekund (FlyVis ustala odpowiedź na każdą scenę, BANC się stabilizuje),
        więc na czas tego żądania timeout gniazda jest wydłużony do 2 minut.
        """
        self.sock.setsockopt(self._zmq.RCVTIMEO, 120_000)
        try:
            self.sock.send_multipart(encode_calibration(scenes))
            return json.loads(self.sock.recv_multipart()[0])
        finally:
            self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
