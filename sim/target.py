"""Cel lotu ze sceny sim/assets/scene_beacon.xml: cienki pomarańczowy prostopadłościan (geom "target_box").

Prostopadłościan jest przenikalny (bez kolizji). Sukces = zetknięcie: jakakolwiek część drona (zasięg łopat
DRONE_REACH od środka) wchodzi w jego obrys. Wtedy robi się zielony (show_reached).
Pozycja celu służy tylko do nagrody i metryk — nie jest wejściem sterowania (dron ma go zobaczyć,
w przybliżeniu podpowiada mu „GPS” celu z sim/sensors.py).
"""

import numpy as np

REACHED_RGBA = np.array([0.1, 0.85, 0.25, 1.0], dtype=np.float32)
DRONE_REACH = 0.3  # m, od środka drona do końca łopat (dotknięcie liczymy od krawędzi drona, nie środka)


class Target:
    def __init__(self, model):
        self.model = model
        self.site_id = model.site("target").id
        self.box_id = model.geom("target_box").id
        self.base_rgba = model.geom_rgba[self.box_id].copy()

    @classmethod
    def from_model(cls, model):
        """Target albo None, jeśli scena nie ma celu (np. scene_hover.xml)."""
        try:
            return cls(model)
        except KeyError:
            return None

    def position(self, data):
        """Środek podstawy prostopadłościanu (na ziemi)."""
        return data.site_xpos[self.site_id].copy()

    def distance(self, data, body_id):
        return float(np.linalg.norm(data.xpos[body_id] - self.position(data)))

    def reached(self, data, body_id):
        """Czy dron dotyka prostopadłościanu (obrys poszerzony o DRONE_REACH; rozmiar czytany z modelu,
        więc działa też po poszerzeniu celu w pamięci, np. beacon_scale w sim/banc_pilot.py)."""
        half = self.model.geom_size[self.box_id]  # pół-wymiary: x, y, z
        offset = data.xpos[body_id] - data.geom_xpos[self.box_id]
        return bool(np.all(np.abs(offset) <= half + DRONE_REACH))

    def show_reached(self, model, reached):
        """Prostopadłościan robi się zielony, gdy dron go dotknął. Kolor sprzed zazielenienia jest zapamiętywany
        w tej chwili, nie w __init__ — inaczej reset przywracał pomarańczowy ze sceny zamiast koloru z set_beacon."""
        current = model.geom_rgba[self.box_id]
        if reached:
            if not np.allclose(current, REACHED_RGBA):
                self.base_rgba = current.copy()
            model.geom_rgba[self.box_id] = REACHED_RGBA
        elif np.allclose(current, REACHED_RGBA):
            model.geom_rgba[self.box_id] = self.base_rgba
