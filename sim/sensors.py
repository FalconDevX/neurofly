"""Realistyczne czujniki drona: szum, dryf, opóźnienie i ograniczenia zasięgu.

Symulacja zna prawdziwy stan, ale dron go nie zna — ma tylko czujniki, które się mylą. Model uczony na idealnych
wartościach opiera się na nich zamiast na BANC i nie radzi sobie z prawdziwymi warunkami (Plan A).

Czujniki (układ drona: x przód, y lewo, z góra):
    gyro      żyroskop [rad/s]       szum biały + dryf (błądzący bias)
    accel     akcelerometr [m/s^2]   szum biały + stały bias (jak MuJoCo: bez grawitacji = swobodny spadek)
    range     dalmierz w dół [m]     odległość po osi -z drona (przy przechyle rośnie), szum ~1 cm + 1 %,
                                     zasięg 0.05–4 m, poza zasięgiem i przy zgubionym odczycie: NaN
    flow      przepływ optyczny [m/s] prędkość przód/bok z kamery w dół; szum rośnie z wysokością,
                                     działa tylko, gdy dalmierz ma odczyt (inaczej NaN)
    baro      barometr [m]           wysokość względem startu, szum ~10 cm + powolny dryf
    vz        prędkość pionowa [m/s] — estymata z barometru: szum i opóźnienie filtra
    beacon    „GPS” celu: kierunek do celu względem nosa drona [rad, + = w lewo] i odległość w poziomie [m].
              Liczony jak z prawdziwego GPS + kompasu: błąd pozycji drona ~2.5 m, który powoli pływa
              (Ornstein-Uhlenbeck, ~15 s), kompas z błędem kilku stopni, odczyt 5 razy/s z opóźnieniem 0.2 s
              (między odczytami ostatni znany). Daleko od celu kierunek jest dość dokładny, blisko coraz mniej
              pewny — ostatni odcinek dron musi wypatrzyć oczami. O przeszkodach nic nie mówi: trasę wyznacza sam.
Wszystkie odczyty docierają z opóźnieniem LATENCY (bufor kroków fizyki). Ziarno = powtarzalne zakłócenia.
mode="ideal": te same klucze, prawdziwe wartości bez szumu i opóźnienia (do porównania i dla zgodności).

Metryki i nagroda NIE używają czujników — liczą się z prawdziwego stanu (sim/metrics.py, WorldEnv.info).
"""

from collections import deque

import mujoco
import numpy as np

GYRO_NOISE = 0.01          # rad/s, szum biały (na odczyt)
GYRO_BIAS_START = 0.01     # rad/s, początkowy bias (losowy, na epizod)
GYRO_BIAS_WALK = 0.002     # rad/s na sqrt(s), dryf biasu
ACCEL_NOISE = 0.1          # m/s^2
ACCEL_BIAS = 0.05          # m/s^2, stały bias (losowy, na epizod)
RANGE_MIN, RANGE_MAX = 0.05, 4.0   # m
RANGE_NOISE = 0.01         # m + RANGE_NOISE_REL * odległość
RANGE_NOISE_REL = 0.01
RANGE_DROPOUT = 0.01       # prawdopodobieństwo zgubionego odczytu
FLOW_NOISE = 0.05          # m/s + FLOW_NOISE_PER_M * wysokość
FLOW_NOISE_PER_M = 0.05
BARO_NOISE = 0.1           # m
BARO_DRIFT = 0.02          # m na sqrt(s)
VZ_NOISE = 0.3             # m/s, szum przed filtrem (po filtrze ~0.05 m/s)
VZ_SMOOTH = 0.15           # s, stała czasowa filtra prędkości pionowej (= jej opóźnienie)
LATENCY = 0.02             # s, opóźnienie wszystkich odczytów
GPS_RATE = 5.0             # Hz, nowy odczyt kierunku/odległości do celu
GPS_LATENCY = 0.2          # s
GPS_POS_SIGMA = 2.5        # m, błąd pozycji GPS (odchylenie standardowe, w każdej osi poziomej)
GPS_POS_TAU = 15.0         # s, jak szybko błąd pozycji się zmienia
COMPASS_BIAS = np.deg2rad(3)    # stały błąd kompasu (losowy, na epizod)
COMPASS_NOISE = np.deg2rad(2)   # szum kompasu na odczyt

KEYS = ("gyro", "accel", "range", "flow", "baro", "vz", "beacon")


class DroneSensors:
    def __init__(self, model, mode="real", body="x2", seed=None):
        assert mode in ("real", "ideal")
        self.model = model
        self.mode = mode
        self.body_id = model.body(body).id
        self.dt = model.opt.timestep
        self.gyro_adr = model.sensor_adr[model.sensor("body_gyro").id]
        self.accel_adr = model.sensor_adr[model.sensor("body_linacc").id]
        self.delay = max(0, int(round(LATENCY / self.dt)))
        self.reset(seed)

    def reset(self, seed=None, data=None, target=None):
        """Nowy epizod: nowe biasy, pusty bufor. target = pozycja celu (dla „GPS” celu; None = brak celu).

        Z data od razu wypełnia odczyt (bez czekania na opóźnienie), także pierwszy odczyt GPS.
        """
        self.rng = np.random.default_rng(seed)
        self.gyro_bias = self.rng.normal(0, GYRO_BIAS_START, 3)
        self.accel_bias = self.rng.normal(0, ACCEL_BIAS, 3)
        self.baro_drift = 0.0
        self.baro_ref = None
        self.vz = None
        self.target = None if target is None else np.asarray(target, float).copy()
        self.gps_error = self.rng.normal(0, GPS_POS_SIGMA, 2)
        self.compass_bias = self.rng.normal(0, COMPASS_BIAS)
        self.t = 0.0
        self.next_fix = 0.0
        self.pending_fixes = deque()
        self.fix = None
        self.buffer = deque(maxlen=self.delay + 1)
        self.latest = None
        if data is not None:
            if self.mode == "real":
                self.fix = self._gps_fix(data)  # na starcie dron ma już pierwszy odczyt
            self.update(data)
            while len(self.buffer) < self.buffer.maxlen:
                self.buffer.append(self.buffer[-1])

    # --- prawdziwe wartości ----------------------------------------------------------------------

    def true_values(self, data):
        """Prawdziwe wartości tego, co mierzą czujniki (do panelu C i trybu ideal)."""
        s = data.sensordata
        rot = data.xmat[self.body_id].reshape(3, 3)
        vel_body = rot.T @ data.qvel[0:3]
        dist = self._range(data, rot)
        if self.baro_ref is None:
            self.baro_ref = data.xpos[self.body_id][2]
        return {
            "gyro": s[self.gyro_adr:self.gyro_adr + 3].copy(),
            "accel": s[self.accel_adr:self.accel_adr + 3].copy(),
            "range": dist,
            "flow": vel_body[:2].copy(),
            "baro": data.xpos[self.body_id][2] - self.baro_ref,
            "vz": data.qvel[2],
            "beacon": self._beacon(data, data.xpos[self.body_id][:2], self._heading(rot)),
        }

    @staticmethod
    def _heading(rot):
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _beacon(self, data, xy, heading):
        """(kierunek do celu względem nosa [rad, + = w lewo], odległość w poziomie [m]); NaN bez celu."""
        if self.target is None:
            return np.array([np.nan, np.nan])
        d = self.target[:2] - xy
        bearing = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
        return np.array([bearing, float(np.hypot(d[0], d[1]))])

    def _gps_fix(self, data):
        """Odczyt „GPS” celu: pozycja drona z błędem GPS, kurs z kompasu z błędem."""
        rot = data.xmat[self.body_id].reshape(3, 3)
        heading = self._heading(rot) + self.compass_bias + self.rng.normal(0, COMPASS_NOISE)
        return self._beacon(data, data.xpos[self.body_id][:2] + self.gps_error, heading)

    def _range(self, data, rot):
        """Odległość wzdłuż osi -z drona do najbliższej przeszkody (teren, blok, pole celu); inf = brak."""
        geomid = np.zeros(1, np.int32)
        d = mujoco.mj_ray(self.model, data, data.xpos[self.body_id], -rot[:, 2], None, 1, self.body_id, geomid)
        return float(d) if d >= 0 else np.inf

    # --- krok -----------------------------------------------------------------------------------

    def update(self, data):
        """Wywołać po każdym kroku fizyki. Dokłada odczyt do bufora opóźnienia."""
        true = self.true_values(data)
        self.t += self.dt
        reading = true if self.mode == "ideal" else self._corrupt(true, data)
        self.buffer.append(reading)
        self.latest = true
        return self.read()

    def read(self):
        """Odczyt z opóźnieniem LATENCY (tryb ideal: bez opóźnienia)."""
        if not self.buffer:
            return None
        return self.buffer[-1] if self.mode == "ideal" else self.buffer[0]

    def _corrupt(self, true, data):
        rng, dt = self.rng, self.dt
        self.gyro_bias += rng.normal(0, GYRO_BIAS_WALK * np.sqrt(dt), 3)
        self.baro_drift += rng.normal(0, BARO_DRIFT * np.sqrt(dt))

        dist = true["range"]
        if not RANGE_MIN <= dist <= RANGE_MAX or rng.random() < RANGE_DROPOUT:
            rng_reading = np.nan
        else:
            rng_reading = dist + rng.normal(0, RANGE_NOISE + RANGE_NOISE_REL * dist)
        if np.isnan(rng_reading):
            flow = np.full(2, np.nan)
        else:
            flow = true["flow"] + rng.normal(0, FLOW_NOISE + FLOW_NOISE_PER_M * dist, 2)

        baro = true["baro"] + self.baro_drift + rng.normal(0, BARO_NOISE)
        raw_vz = true["vz"] + rng.normal(0, VZ_NOISE)  # różniczkowanie surowego barometru dałoby szum ~m/s
        self.vz = raw_vz if self.vz is None else self.vz + (1 - np.exp(-dt / VZ_SMOOTH)) * (raw_vz - self.vz)

        # GPS celu: błąd pozycji pływa (OU), odczyt co 1/GPS_RATE s, dociera po GPS_LATENCY, potem trzymany
        decay = np.exp(-dt / GPS_POS_TAU)
        self.gps_error = decay * self.gps_error + np.sqrt(1 - decay ** 2) * GPS_POS_SIGMA * rng.normal(size=2)
        if self.t >= self.next_fix:
            self.pending_fixes.append((self.t + GPS_LATENCY, self._gps_fix(data)))
            self.next_fix += 1 / GPS_RATE
        while self.pending_fixes and self.pending_fixes[0][0] <= self.t:
            self.fix = self.pending_fixes.popleft()[1]
        return {
            "gyro": true["gyro"] + self.gyro_bias + rng.normal(0, GYRO_NOISE, 3),
            "accel": true["accel"] + self.accel_bias + rng.normal(0, ACCEL_NOISE, 3),
            "range": rng_reading,
            "flow": flow,
            "baro": baro,
            "vz": self.vz,
            "beacon": self.fix if self.fix is not None else np.array([np.nan, np.nan]),
        }

    @staticmethod
    def as_vector(reading):
        """[range, flow_fwd, flow_left, baro, vz] jako float32 (NaN = brak odczytu)."""
        return np.array([reading["range"], *reading["flow"], reading["baro"], reading["vz"]], np.float32)
