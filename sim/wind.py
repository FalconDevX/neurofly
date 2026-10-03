"""Wiatr ze zmiennymi podmuchami — przez wbudowany model płynu MuJoCo (model.opt.wind).

Scena ma ustawioną gęstość i lepkość powietrza (x2.xml: density, viscosity), więc MuJoCo liczy opór
drona względem ruchu powietrza; wystarczy ustawiać model.opt.wind. Zmierzone (płasko, 3 m, ze stabilizacją):
5 m/s spycha z zawisu o 1.8 m / 10 s, 8 m/s o 4.4 m, 11 m/s o 7.7 m; lot pod wiatr 3 m/s zwalnia o 30 / 40 / 60 %.
Domyślne 8 m/s z podmuchami ±3 m/s = wyraźna przeszkoda (5 m/s to lekka bryza).

Wiatr = średnia prędkość w losowym kierunku poziomym + podmuchy: prędkość i kierunek błądzą wokół
średniej (proces Ornsteina-Uhlenbecka, gładkie zmiany co ~2 s). To samo ziarno = ten sam przebieg.
"""

import numpy as np


class Wind:
    def __init__(self, mean_speed=8.0, gust_speed=3.0, gust_angle=np.deg2rad(20), gust_time=2.0, seed=None):
        self.mean_speed = mean_speed      # m/s
        self.gust_speed = gust_speed      # m/s, odchylenie standardowe podmuchów
        self.gust_angle = gust_angle      # rad, odchylenie standardowe kierunku
        self.gust_time = gust_time        # s, czas korelacji podmuchów
        self.enabled = False
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.direction = self.rng.uniform(0, 2 * np.pi)  # rad, dokąd wieje (0 = +x)
        self.speed_noise = 0.0
        self.angle_noise = 0.0

    @property
    def velocity(self):
        """Wektor wiatru [m/s] w układzie świata (0, gdy wyłączony)."""
        if not self.enabled:
            return np.zeros(3)
        speed = max(0.0, self.mean_speed + self.speed_noise)
        angle = self.direction + self.angle_noise
        return np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])

    def step(self, model, dt):
        """Przesuwa podmuchy o dt i ustawia model.opt.wind."""
        if self.enabled:
            decay = np.exp(-dt / self.gust_time)
            kick = np.sqrt(1 - decay ** 2)
            self.speed_noise = decay * self.speed_noise + kick * self.gust_speed * self.rng.standard_normal()
            self.angle_noise = decay * self.angle_noise + kick * self.gust_angle * self.rng.standard_normal()
        model.opt.wind[:] = self.velocity
