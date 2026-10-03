"""Metryki lotu z PRAWDZIWEGO stanu symulacji (nie z czujników) — ocena, jak dron naprawdę leci.

EpisodeMetrics śledzi bieżący epizod (odległość i postęp do celu, błąd kursu, wysokość nad terenem, przechył,
prędkość, droga) i zbiera wyniki epizodów (cel / wywrotka / poza planszą / limit czasu / przerwany).
Opcjonalnie zapisuje podsumowanie każdego epizodu do CSV (csv_path).
"""

import csv
from pathlib import Path

import mujoco
import numpy as np

SUMMARY_FIELDS = ("epizod", "swiat", "wynik", "czas_s", "odl_start_m", "odl_min_m", "odl_koniec_m",
                  "postep_proc", "droga_m", "przechyl_max_deg", "blad_kursu_sredni_deg", "wiatr_m_s")


class EpisodeMetrics:
    def __init__(self, model, body="x2", csv_path=None):
        self.model = model
        self.body_id = model.body(body).id
        self.csv_path = Path(csv_path) if csv_path else None
        self.results = []  # podsumowania zakończonych epizodów
        self.episode = 0
        self.world = None
        self.current = None

    # --- bieżący epizod ---------------------------------------------------------------------------

    def start(self, data, target_pos=None, world=None):
        self.episode += 1
        self.world = world
        self.t0 = data.time
        self.pos_prev = data.xpos[self.body_id].copy()
        self.path = 0.0
        self.tilt_max = 0.0
        self.heading_errors = []
        self.wind_max = 0.0
        self.start_distance = self._distance(data, target_pos)
        self.min_distance = self.start_distance
        self.current = self._snapshot(data, target_pos, wind=np.zeros(3))

    def update(self, data, target_pos=None, wind=None):
        pos = data.xpos[self.body_id]
        self.path += float(np.linalg.norm(pos - self.pos_prev))
        self.pos_prev = pos.copy()
        snap = self._snapshot(data, target_pos, wind if wind is not None else np.zeros(3))
        self.tilt_max = max(self.tilt_max, snap["tilt"])
        if snap["distance"] is not None:
            self.min_distance = min(self.min_distance, snap["distance"])
            self.heading_errors.append(abs(snap["heading_error"]))
        self.wind_max = max(self.wind_max, snap["wind"])
        self.current = snap
        return snap

    def finish(self, outcome):
        """Zamyka epizod z wynikiem; zwraca podsumowanie (i dopisuje do CSV)."""
        if self.current is None:
            return None
        c = self.current
        progress = None
        if self.start_distance:
            progress = 100 * (self.start_distance - c["distance"]) / self.start_distance
        summary = {
            "epizod": self.episode, "swiat": self.world, "wynik": outcome, "czas_s": round(c["time"], 2),
            "odl_start_m": _r(self.start_distance), "odl_min_m": _r(self.min_distance),
            "odl_koniec_m": _r(c["distance"]), "postep_proc": _r(progress, 1), "droga_m": round(self.path, 2),
            "przechyl_max_deg": round(self.tilt_max, 1),
            "blad_kursu_sredni_deg": _r(np.mean(self.heading_errors) if self.heading_errors else None, 1),
            "wiatr_m_s": round(self.wind_max, 1),
        }
        self.results.append(summary)
        if self.csv_path is not None:
            new = not self.csv_path.exists()
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(summary)
        self.current = None
        return summary

    # --- pomiary ----------------------------------------------------------------------------------

    def _distance(self, data, target_pos):
        if target_pos is None:
            return None
        return float(np.linalg.norm(data.xpos[self.body_id] - target_pos))

    def _snapshot(self, data, target_pos, wind):
        pos = data.xpos[self.body_id]
        rot = data.xmat[self.body_id].reshape(3, 3)
        heading = np.arctan2(rot[1, 0], rot[0, 0])
        snap = {
            "time": data.time - self.t0,
            "position": pos.copy(),
            "height": self._height_above_ground(data),
            "speed": float(np.linalg.norm(data.qvel[0:2])),
            "climb": float(data.qvel[2]),
            "tilt": float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))),
            "heading": float(np.degrees(heading)),
            "wind": float(np.linalg.norm(wind[:2])),
            "distance": self._distance(data, target_pos),
            "heading_error": None,
        }
        if target_pos is not None:
            to_target = np.arctan2(target_pos[1] - pos[1], target_pos[0] - pos[0])
            snap["heading_error"] = float(np.degrees((to_target - heading + np.pi) % (2 * np.pi) - np.pi))
        return snap

    def _height_above_ground(self, data):
        """Pionowo w dół do terenu/bloku (prawdziwa wysokość, nie dalmierz)."""
        geomid = np.zeros(1, np.int32)
        d = mujoco.mj_ray(self.model, data, data.xpos[self.body_id], np.array([0, 0, -1.0]), None, 1,
                          self.body_id, geomid)
        return float(d) if d >= 0 else None

    # --- tekst do panelu M --------------------------------------------------------------------------

    def panel(self):
        """(lewa kolumna, prawa kolumna) do viewer.set_texts."""
        c = self.current
        rows = []
        if c is not None:
            rows += [("Epizod", f"{self.episode}  (świat {self.world})" if self.world is not None else str(self.episode)),
                     ("Czas", f"{c['time']:.1f} s")]
            if c["distance"] is not None:
                progress = 100 * (self.start_distance - c["distance"]) / self.start_distance if self.start_distance else 0
                rows += [("Odl. do celu", f"{c['distance']:.1f} m (min {self.min_distance:.1f})"),
                         ("Postęp", f"{progress:.0f} %"),
                         ("Błąd kursu", f"{c['heading_error']:+.0f}°")]
            rows += [("Wys. nad terenem", f"{c['height']:.2f} m" if c["height"] is not None else "—"),
                     ("Prędkość", f"{c['speed']:.1f} m/s  (pion {c['climb']:+.1f})"),
                     ("Przechył", f"{c['tilt']:.0f}°  (max {self.tilt_max:.0f}°)"),
                     ("Droga", f"{self.path:.1f} m"),
                     ("Wiatr", f"{c['wind']:.1f} m/s")]
        if self.results:
            outcomes = [r["wynik"] for r in self.results]
            goals = [r for r in self.results if r["wynik"] == "cel"]
            rows += [("", ""), ("Epizody", str(len(outcomes))),
                     ("Cel", f"{len(goals)}  ({100 * len(goals) / len(outcomes):.0f} %)"),
                     ("Wywrotki", str(sum(o.startswith("wywrotka") for o in outcomes))),
                     ("Poza planszą", str(outcomes.count("poza planszą")))]
            if goals:
                rows.append(("Najlepszy czas", f"{min(r['czas_s'] for r in goals):.1f} s"))
        return "METRYKI (prawdziwy stan)\n" + "\n".join(k for k, _ in rows), "[M]\n" + "\n".join(v for _, v in rows)


def sensors_panel(reading, true, mode="real"):
    """(lewa, prawa) kolumna do viewer.set_texts: odczyt czujnika i prawdziwa wartość obok."""
    def fmt(x, unit, digits=2):
        return "brak" if x is None or not np.isfinite(x) else f"{x:+.{digits}f}{unit}"

    rows = [
        ("Żyroskop x/y/z", " ".join(fmt(v, "", 2) for v in reading["gyro"]) + " rad/s",
         " ".join(fmt(v, "", 2) for v in true["gyro"])),
        ("Akcelerometr x/y/z", " ".join(fmt(v, "", 1) for v in reading["accel"]) + " m/s²",
         " ".join(fmt(v, "", 1) for v in true["accel"])),
        ("Dalmierz", fmt(reading["range"], " m"), fmt(true["range"] if true["range"] <= 50 else np.nan, " m")),
        ("Przepływ przód/bok", f"{fmt(reading['flow'][0], '')} {fmt(reading['flow'][1], '')} m/s",
         f"{fmt(true['flow'][0], '')} {fmt(true['flow'][1], '')}"),
        ("Barometr (wys.)", fmt(reading["baro"], " m"), fmt(true["baro"], " m")),
        ("Prędkość pionowa", fmt(reading["vz"], " m/s"), fmt(true["vz"], " m/s")),
        ("GPS celu: kierunek", fmt(np.degrees(reading["beacon"][0]), "°", 0) + " (+ = w lewo)",
         fmt(np.degrees(true["beacon"][0]), "°", 0)),
        ("GPS celu: odległość", fmt(reading["beacon"][1], " m", 1), fmt(true["beacon"][1], " m", 1)),
    ]
    title = "CZUJNIKI (odczyt | prawda)" + ("" if mode == "real" else " — tryb ideal, bez szumu")
    left = title + "\n" + "\n".join(r[0] for r in rows)
    right = "[C]\n" + "\n".join(f"{r[1]}  |  {r[2]}" for r in rows)
    return left, right


def _r(x, digits=2):
    return None if x is None else round(float(x), digits)
