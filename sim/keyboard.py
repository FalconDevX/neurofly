"""Odczyt trzymanych klawiszy do ręcznego latania.

key_callback podglądu MuJoCo dostaje tylko naciśnięcia (bez puszczenia), więc stan klawiszy
czytamy z Windows (GetAsyncKeyState) — tylko gdy na wierzchu jest okno MuJoCo.
Na innych systemach ręczne latanie jest wyłączone (dron trzyma zawis).

W/A/S/D/Q/E to też skróty flag wbudowanego podglądu (wireframe, cienie, ...);
sim/viewer.py przywraca te flagi w każdej klatce.
"""

import ctypes
import sys

WINDOW_TITLE_PREFIX = "MuJoCo"

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_A, VK_D, VK_E, VK_Q, VK_S, VK_W = (ord(c) for c in "ADEQSW")


class FlightKeyboard:
    def __init__(self, speed=3.0, climb=1.5, yaw_rate=0.8):
        self.speed, self.climb, self.yaw_rate = speed, climb, yaw_rate
        self.available = sys.platform == "win32"
        if self.available:
            self._user32 = ctypes.windll.user32

    def _down(self, vk):
        return bool(self._user32.GetAsyncKeyState(vk) & 0x8000)

    def _window_focused(self):
        hwnd = self._user32.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(256)
        self._user32.GetWindowTextW(hwnd, buf, 256)
        return buf.value.startswith(WINDOW_TITLE_PREFIX)

    def setpoints(self):
        """Zwraca forward/left/up [m/s] i yaw_rate [rad/s] dla VelocityController.

        W/S przód/tył, A/D lewo/prawo, Shift wznoszenie, Ctrl opadanie, Q/E obrót w lewo/prawo.
        """
        sp = dict(forward=0.0, left=0.0, up=0.0, yaw_rate=0.0)
        if not self.available or not self._window_focused():
            return sp
        d = self._down
        sp["forward"] = self.speed * (d(VK_W) - d(VK_S))
        sp["left"] = self.speed * (d(VK_A) - d(VK_D))
        sp["up"] = self.climb * (d(VK_SHIFT) - d(VK_CONTROL))
        sp["yaw_rate"] = self.yaw_rate * (d(VK_Q) - d(VK_E))
        return sp
