"""Klawiatura i kółko myszy do ręcznego latania w podglądzie MuJoCo (Windows).

Wbudowany podgląd MuJoCo ma skrót pod każdą literą (W wireframe, S cienie, L additive, ...).
Trzymany klawisz lotu przełączał więc flagę renderu przy każdym autorepeat — klatki z wireframe
na czarnym tle migały, zanim pętla zdążyła przywrócić flagi. Dlatego klawisze lotu i kółko
przechwytujemy niskopoziomowym hookiem Windows (WH_KEYBOARD_LL / WH_MOUSE_LL), gdy aktywne jest
okno MuJoCo: zdarzenie nie dociera do podglądu, a stan trzymamy sami.

Na innych systemach ręczne latanie jest wyłączone (stały ciąg zawisu, zero prędkości kątowych).
"""

import ctypes
import sys
import threading
from ctypes import wintypes

from sim.control import RateCommand

WINDOW_TITLE_PREFIX = "MuJoCo"

VK_LSHIFT, VK_RSHIFT, VK_LCONTROL, VK_RCONTROL = 0xA0, 0xA1, 0xA2, 0xA3
VK_LMENU, VK_RMENU = 0xA4, 0xA5
VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12  # VK_MENU = Alt
VK_CAPITAL = 0x14  # CapsLock — tylko obserwowany (nie przechwytujemy, lampka działa normalnie)
VK_A, VK_C, VK_D, VK_E, VK_L, VK_M, VK_N, VK_Q, VK_S, VK_T, VK_W = (ord(c) for c in "ACDELMNQSTW")
# klawisze przechwytywane, gdy aktywne jest okno MuJoCo
CAPTURED = {VK_W, VK_A, VK_S, VK_D, VK_Q, VK_E, VK_L, VK_N, VK_C, VK_M, VK_T, VK_SHIFT, VK_CONTROL, VK_MENU}
ALIASES = {VK_LSHIFT: VK_SHIFT, VK_RSHIFT: VK_SHIFT, VK_LCONTROL: VK_CONTROL, VK_RCONTROL: VK_CONTROL,
           VK_LMENU: VK_MENU, VK_RMENU: VK_MENU}

WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x100, 0x101, 0x104, 0x105
WM_MOUSEWHEEL, WM_QUIT = 0x20A, 0x12
WHEEL_DELTA = 120


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)


class FlightKeyboard:
    """Stan trzymanych klawiszy lotu, liczba naciśnięć L / Alt / N i przewinięcia kółka (odwrócone)."""

    def __init__(self, tilt_rate=1.0, yaw_rate=0.8, throttle=0.3, speed=3.0, climb=1.5):
        self.tilt_rate, self.yaw_rate, self.throttle = tilt_rate, yaw_rate, throttle
        self.speed, self.climb = speed, climb  # tryb ze stabilizacją
        self.available = sys.platform == "win32"
        self._lock = threading.Lock()
        self._held: set[int] = set()
        self._lock_presses = 0
        self._stab_presses = 0
        self._world_presses = 0
        self._panel_presses = {VK_C: 0, VK_M: 0, VK_T: 0}  # C = czujniki, M = metryki, T = ślad lotu
        self._caps_lock = False
        self._caps_down = False
        self._wheel = 0.0
        self._thread_id = None
        if self.available:
            self._user32 = ctypes.windll.user32
            self._caps_lock = bool(self._user32.GetKeyState(VK_CAPITAL) & 1)  # stan na starcie
            self._user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
            self._user32.CallNextHookEx.restype = ctypes.c_ssize_t
            self._user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
            self._user32.SetWindowsHookExW.restype = wintypes.HHOOK
            self._ready = threading.Event()
            threading.Thread(target=self._hook_loop, daemon=True).start()
            self._ready.wait(2.0)

    # --- wątek hooka -----------------------------------------------------------------------------

    def _mujoco_focused(self):
        hwnd = self._user32.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(256)
        self._user32.GetWindowTextW(hwnd, buf, 256)
        return buf.value.startswith(WINDOW_TITLE_PREFIX)

    def _on_key(self, n_code, w_param, l_param):
        if n_code == 0:
            info = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk = ALIASES.get(info.vkCode, info.vkCode)
            if vk == VK_CAPITAL:  # każde naciśnięcie (bez autorepeat) przełącza CapsLock
                down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)
                with self._lock:
                    if down and not self._caps_down:
                        self._caps_lock = not self._caps_lock
                    self._caps_down = down
            if vk in CAPTURED:
                down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)
                with self._lock:
                    if down:
                        if vk == VK_L and vk not in self._held:  # bez autorepeat
                            self._lock_presses += 1
                        if vk == VK_MENU and vk not in self._held:
                            self._stab_presses += 1
                        if vk == VK_N and vk not in self._held:
                            self._world_presses += 1
                        if vk in self._panel_presses and vk not in self._held:
                            self._panel_presses[vk] += 1
                        self._held.add(vk)
                    else:
                        self._held.discard(vk)
                if self._mujoco_focused():
                    return 1  # nie przekazujemy do MuJoCo — żadnej zmiany flag renderu
        return self._user32.CallNextHookEx(None, n_code, w_param, l_param)

    def _on_mouse(self, n_code, w_param, l_param):
        if n_code == 0 and w_param == WM_MOUSEWHEEL and self._mujoco_focused():
            info = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
            hwnd = self._user32.GetAncestor(self._user32.WindowFromPoint(info.pt), 2)  # GA_ROOT
            if hwnd == self._user32.GetForegroundWindow():
                delta = ctypes.c_short(info.mouseData >> 16).value
                with self._lock:
                    self._wheel += delta / WHEEL_DELTA
                return 1
        return self._user32.CallNextHookEx(None, n_code, w_param, l_param)

    def _hook_loop(self):
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        self._kb_proc, self._ms_proc = HOOKPROC(self._on_key), HOOKPROC(self._on_mouse)  # referencje muszą żyć
        self._kb_hook = self._user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._kb_proc, None, 0)
        self._ms_hook = self._user32.SetWindowsHookExW(WH_MOUSE_LL, self._ms_proc, None, 0)
        if not self._kb_hook:
            self.available = False
        self._ready.set()
        msg = wintypes.MSG()
        while self._user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass
        self._user32.UnhookWindowsHookEx(self._kb_hook)
        self._user32.UnhookWindowsHookEx(self._ms_hook)

    def close(self):
        if self._thread_id:
            self._user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    # --- odczyt z pętli symulacji ----------------------------------------------------------------

    def command(self, hover_thrust):
        """RateCommand prosto z klawiszy (tryb acro, bez autostabilizacji).

        W/S pochylenie nosa w dół/górę, A/D przechył w lewo/prawo, Q/E obrót w lewo/prawo —
        trzymany klawisz = stała prędkość kątowa, puszczony = 0 (dron zostaje w bieżącym przechyle).
        Shift/Ctrl: ciąg powyżej/poniżej ciągu zawisu; bez nich stały ciąg zawisu (bez kompensacji
        przechyłu, więc przechylony dron opada).
        """
        cmd = RateCommand(thrust=hover_thrust)
        if not self.available or not self._mujoco_focused():
            return cmd
        with self._lock:
            d = lambda vk: vk in self._held  # noqa: E731
            cmd.pitch_rate = self.tilt_rate * (d(VK_W) - d(VK_S))
            cmd.roll_rate = self.tilt_rate * (d(VK_D) - d(VK_A))
            cmd.yaw_rate = self.yaw_rate * (d(VK_Q) - d(VK_E))
            cmd.thrust = hover_thrust * (1 + self.throttle * (d(VK_SHIFT) - d(VK_CONTROL)))
        return cmd

    def setpoints(self):
        """Tryb ze stabilizacją: forward/left/up [m/s] i yaw_rate [rad/s] dla VelocityController.

        Te same klawisze co w acro: W/S przód/tył, A/D lewo/prawo, Shift/Ctrl góra/dół, Q/E obrót.
        """
        sp = dict(forward=0.0, left=0.0, up=0.0, yaw_rate=0.0)
        if not self.available or not self._mujoco_focused():
            return sp
        with self._lock:
            d = lambda vk: vk in self._held  # noqa: E731
            sp["forward"] = self.speed * (d(VK_W) - d(VK_S))
            sp["left"] = self.speed * (d(VK_A) - d(VK_D))
            sp["up"] = self.climb * (d(VK_SHIFT) - d(VK_CONTROL))
            sp["yaw_rate"] = self.yaw_rate * (d(VK_Q) - d(VK_E))
        return sp

    def take_stabilization_toggles(self):
        """Ile razy naciśnięto Alt od ostatniego odczytu."""
        with self._lock:
            n, self._stab_presses = self._stab_presses, 0
        return n

    @property
    def caps_lock(self):
        """Stan CapsLocka (włączony = wiatr w podglądzie)."""
        with self._lock:
            return self._caps_lock

    def take_panel_toggles(self):
        """(ile razy C, ile razy M) od ostatniego odczytu — przełączniki paneli czujników i metryk."""
        with self._lock:
            c, m = self._panel_presses[VK_C], self._panel_presses[VK_M]
            self._panel_presses[VK_C] = self._panel_presses[VK_M] = 0
        return c, m

    def take_trail_toggles(self):
        """Ile razy naciśnięto T (ślad lotu wł./wył.) od ostatniego odczytu."""
        with self._lock:
            n, self._panel_presses[VK_T] = self._panel_presses[VK_T], 0
        return n

    def take_new_world_requests(self):
        """Ile razy naciśnięto N od ostatniego odczytu."""
        with self._lock:
            n, self._world_presses = self._world_presses, 0
        return n

    def take_camera_toggles(self):
        """Ile razy naciśnięto L od ostatniego odczytu."""
        with self._lock:
            n, self._lock_presses = self._lock_presses, 0
        return n

    def take_wheel(self):
        """Przewinięcie kółka w krokach (+ = od siebie) od ostatniego odczytu."""
        with self._lock:
            w, self._wheel = self._wheel, 0.0
        return w
