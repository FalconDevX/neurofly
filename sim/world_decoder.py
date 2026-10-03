"""Dekoder lotu w świecie Osoby 3: BANC (percepcja, kierunek) + czujniki drona (wysokość, prędkości, żyroskop).

Tylko numpy — używa go też master treningu rozproszonego (bez MuJoCo i GPU).

Podział (NASZE ZAŁOŻENIE, opisywać wprost):
  - yaw   ← wyłącznie cechy BANC (6 grup MN + pojedyncze DN lotu + wyraz wolny); czujniki drona mają tu
            wagę 0 z konstrukcji, więc kierunek do celu bierze się tylko z sieci,
  - thrust, roll, pitch ← cechy BANC + czujniki, które ma prawdziwy dron: wysokość nad terenem (dalmierz),
            prędkość pionowa (baro/IMU), prędkość przód/bok (czujnik przepływu optycznego), żyroskop 3 osie.
Bez czujników te osie były nieuczalne: nauczyciel steruje nimi z wysokości i prędkości, których BANC nie
dostaje (do BANC idzie tylko obraz i yaw z żyroskopu przez haltery).

Uczenie: DAgger — wszystkie pary (cechy, komenda nauczyciela) z kolejnych epizodów sumowane w statystykach
XᵀX, Xᵀy; po każdej partii wagi od nowa z regresji grzbietowej na całości. Statystyki da się sumować
między komputerami (``merge``), więc workerzy wysyłają tylko je, bez klatek.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SENSORS = ("wysokość−1m", "v_z", "v_przód", "v_bok", "gyro_x", "gyro_y", "gyro_z")
CTL_AXES = ("thrust", "roll", "pitch")
AXES = ("thrust", "roll", "pitch", "yaw")


def drone_sensors(height: float, vel_world, yaw: float, gyro) -> np.ndarray:
    """Czujniki drona → wektor ``SENSORS`` (skale ~1). ``yaw`` [rad] do obrotu prędkości do układu drona."""
    c, s = np.cos(yaw), np.sin(yaw)
    v_fwd = c * vel_world[0] + s * vel_world[1]
    v_left = -s * vel_world[0] + c * vel_world[1]
    return np.array([height - 1.0, vel_world[2], v_fwd, v_left, *gyro], dtype=float)


def sample_weight(bearing_rad: float, s: np.ndarray) -> float:
    """Waga próbki DAgger: większość klatek to „cel prawie na wprost, nic nie rób”, więc trudne stany
    dostają więcej: duży kąt do celu (> 20°), błąd wysokości > 0.3 m, szybki obrót (|gyro| > 1 rad/s,
    np. po skręcie albo podmuchu). Kolejne warunki się sumują: 1, 3, 5 albo 7."""
    w = 1.0
    w += 2.0 * (abs(bearing_rad) > np.deg2rad(20))
    w += 2.0 * (abs(s[0]) > 0.3)  # s[0] = wysokość − 1 m
    w += 2.0 * (np.linalg.norm(s[4:7]) > 1.0)
    return w


class WorldDecoder:
    def __init__(self, n_banc: int, lam: float = 1e-2) -> None:
        self.n_banc, self.n_s, self.lam = n_banc, len(SENSORS), lam
        m = n_banc + self.n_s
        self.w_yaw = np.zeros(n_banc)
        self.W_ctl = np.zeros((len(CTL_AXES), m))
        self.W_ctl[0, n_banc - 1] = 0.5  # wyraz wolny thrust = zawis, zanim cokolwiek się nauczy
        self.reset_stats()

    # --- komendy ---
    def decode_raw(self, x: np.ndarray, s: np.ndarray) -> np.ndarray:
        """[thrust, roll, pitch, yaw] przed przycięciem. ``x`` = znormalizowane cechy BANC z wyrazem wolnym."""
        z = np.r_[x, s]
        return np.r_[self.W_ctl @ z, self.w_yaw @ x]

    def decode(self, x: np.ndarray, s: np.ndarray):
        from banc_control import FlightCommand

        t, r, p, y = self.decode_raw(x, s)
        return FlightCommand(float(t), float(r), float(p), float(y)).clipped()

    # --- dane i dopasowanie ---
    def reset_stats(self) -> None:
        m = self.n_banc + self.n_s
        self.A_yaw = np.zeros((self.n_banc, self.n_banc))
        self.b_yaw = np.zeros(self.n_banc)
        self.A_ctl = np.zeros((m, m))
        self.b_ctl = np.zeros((m, len(CTL_AXES)))
        self.n = 0

    def add(self, x: np.ndarray, s: np.ndarray, target, weight: float = 1.0) -> np.ndarray:
        """Dopisuje parę (cechy, komenda nauczyciela) z wagą ``weight`` (trudne próbki > 1, patrz
        ``sample_weight``). Zwraca kwadraty błędów [thrust, roll, pitch, yaw] obecnych wag (przed dopasowaniem)."""
        x = np.nan_to_num(x)
        z = np.r_[x, s]
        y_ctl = np.array([target.thrust, target.roll, target.pitch])
        self.A_yaw += weight * np.outer(x, x)
        self.b_yaw += weight * x * target.yaw
        self.A_ctl += weight * np.outer(z, z)
        self.b_ctl += weight * np.outer(z, y_ctl)
        self.n += 1
        pred = np.clip(self.decode_raw(x, s), [0, -1, -1, -1], 1)  # błąd po przycięciu, jak w locie
        return (pred - np.r_[y_ctl, target.yaw]) ** 2

    def stats(self, as_lists: bool = False) -> dict:
        st = {"A_yaw": self.A_yaw, "b_yaw": self.b_yaw, "A_ctl": self.A_ctl, "b_ctl": self.b_ctl, "n": self.n}
        if as_lists:  # do JSON (trening rozproszony): float32 wystarcza, plik ~2× mniejszy
            st = {k: (np.asarray(v, np.float32).tolist() if k != "n" else v) for k, v in st.items()}
        return st

    def merge(self, st: dict) -> None:
        self.A_yaw += np.asarray(st["A_yaw"])
        self.b_yaw += np.asarray(st["b_yaw"])
        self.A_ctl += np.asarray(st["A_ctl"])
        self.b_ctl += np.asarray(st["b_ctl"])
        self.n += int(st["n"])

    @staticmethod
    def _ridge(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
        d = np.diag(A).copy()
        scale = np.sqrt(np.where(d > 1e-12, d, 1.0))  # standaryzacja kolumn (cechy BANC i czujniki mają różne skale)
        As = A / np.outer(scale, scale)
        reg = lam * np.trace(As) / len(As)
        sol = np.linalg.solve(As + reg * np.eye(len(As)), b / (scale if b.ndim == 1 else scale[:, None]))
        return sol / (scale if sol.ndim == 1 else scale[:, None])

    def fit(self) -> None:
        """Wagi z regresji grzbietowej na wszystkich zebranych danych (DAgger)."""
        if self.n < 10:
            return
        self.w_yaw = self._ridge(self.A_yaw, self.b_yaw, self.lam)
        self.W_ctl = self._ridge(self.A_ctl, self.b_ctl, self.lam).T

    # --- jedna macierz (trening rozproszony przesyła wagi jako 4 × m, jak dekoder zawisu) ---
    def to_matrix(self) -> np.ndarray:
        """Wiersze thrust, roll, pitch (BANC + czujniki) i yaw (BANC, czujniki = 0)."""
        return np.vstack([self.W_ctl, np.r_[self.w_yaw, np.zeros(self.n_s)]])

    def from_matrix(self, M) -> None:
        M = np.asarray(M, float)
        self.W_ctl, self.w_yaw = M[:3].copy(), M[3, :self.n_banc].copy()

    @classmethod
    def for_matrix(cls, M) -> "WorldDecoder":
        dec = cls(np.shape(M)[1] - len(SENSORS))
        dec.from_matrix(M)
        return dec

    # --- zapis ---
    def weights(self) -> dict:
        return {"w_yaw": self.w_yaw.tolist(), "W_ctl": self.W_ctl.tolist()}

    def set_weights(self, w: dict) -> None:
        self.w_yaw, self.W_ctl = np.asarray(w["w_yaw"], float), np.asarray(w["W_ctl"], float)

    def save(self, path: Path, **meta) -> None:
        np.savez(path, w_yaw=self.w_yaw, W_ctl=self.W_ctl, n_banc=self.n_banc, sensors=np.array(SENSORS),
                 meta=np.array(repr(meta)))

    @classmethod
    def load(cls, path: Path) -> "WorldDecoder":
        d = np.load(path)
        dec = cls(int(d["n_banc"]))
        dec.set_weights({"w_yaw": d["w_yaw"], "W_ctl": d["W_ctl"]})
        return dec

    @staticmethod
    def is_world_file(path: Path) -> bool:
        return "w_yaw" in np.load(path).files
