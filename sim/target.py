"""Cel lotu (pole lądowania z beaconem) ze sceny sim/assets/scene_beacon.xml.

Pozycja celu służy tylko do nagrody i metryk — nie jest wejściem sterowania (dron ma go zobaczyć).
"""

import numpy as np

REACHED_RGBA = np.array([0.1, 0.8, 0.2, 1.0], dtype=np.float32)


class Target:
    def __init__(self, model, max_height=1.5):
        self.site_id = model.site("target").id
        self.zone_id = model.geom("target_zone").id
        self.half_size = model.geom_size[self.zone_id, :2].copy()
        self.max_height = max_height  # nad polem liczy się tylko niski przelot / lądowanie
        self.base_rgba = model.geom_rgba[self.zone_id].copy()

    @classmethod
    def from_model(cls, model):
        """Target albo None, jeśli scena nie ma celu (np. scene_hover.xml)."""
        try:
            return cls(model)
        except KeyError:
            return None

    def position(self, data):
        return data.site_xpos[self.site_id].copy()

    def distance(self, data, body_id):
        return float(np.linalg.norm(data.xpos[body_id] - self.position(data)))

    def reached(self, data, body_id):
        offset = data.xpos[body_id] - self.position(data)
        return bool(np.all(np.abs(offset[:2]) <= self.half_size) and offset[2] <= self.max_height)

    def show_reached(self, model, reached):
        """Pole zmienia kolor na zielony, gdy dron jest nad nim."""
        model.geom_rgba[self.zone_id] = REACHED_RGBA if reached else self.base_rgba
