"""Interaktywny podgląd drona X2 w MuJoCo: ręczne latanie i obracające się śmigła.

    python -m sim.viewer [--scene sim/assets/scene_hover.xml]

Klawisze (w oknie podglądu, trzymane):
    W / S        lot do przodu / do tyłu
    A / D        lot w lewo / w prawo
    Shift / Ctrl wznoszenie / opadanie
    Q / E        powolny obrót w lewo / w prawo
    Backspace    reset symulacji do stanu startowego (keyframe "hover")
    Spacja       pauza / wznowienie

Po puszczeniu klawiszy dron hamuje i trzyma wysokość (sim/control.py).
Litery są też skrótami flag wbudowanego podglądu MuJoCo (W wireframe, S cienie, ...),
więc flagi wizualizacji są przywracane w każdej klatce — przełączanie ich z menu nie działa.
"""

import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from sim.control import RateController, VelocityController
from sim.keyboard import FlightKeyboard
from sim.propellers import PropellerVisuals

DEFAULT_SCENE = Path(__file__).parent / "assets" / "scene_hover.xml"

# Kody klawiszy GLFW.
KEY_SPACE = 32
KEY_BACKSPACE = 259


class Controls:
    """Zdarzenia z wątku okna podglądu, odbierane w pętli symulacji."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset = False
        self.paused = False

    def on_key(self, key):
        with self.lock:
            if key == KEY_BACKSPACE:
                self.reset = True
            elif key == KEY_SPACE:
                self.paused = not self.paused

    def take(self):
        with self.lock:
            events = (self.reset, self.paused)
            self.reset = False
        return events


def reset_to_start(model, data, props, flight, key_id):
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    props.reset()
    flight.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    props = PropellerVisuals(model)
    flight = VelocityController(RateController(model))
    keyboard = FlightKeyboard()
    key_id = model.key("hover").id
    controls = Controls()
    reset_to_start(model, data, props, flight, key_id)

    with mujoco.viewer.launch_passive(model, data, key_callback=controls.on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("x2").id
        viewer.cam.distance = 1.5
        vis_flags = viewer.opt.flags.copy()
        render_flags = viewer.user_scn.flags.copy()
        print(__doc__)
        if not keyboard.available:
            print("Ręczne latanie działa tylko na Windows — dron będzie trzymał zawis.")
        last = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            frame_dt = min(now - last, 0.05)  # bez nadrabiania po przycięciu okna
            last = now
            reset, paused = controls.take()
            setpoints = keyboard.setpoints()

            with viewer.lock():
                if reset:
                    reset_to_start(model, data, props, flight, key_id)
                if not paused:
                    target = data.time + frame_dt
                    while data.time < target:
                        flight.apply(model, data, **setpoints)
                        mujoco.mj_step(model, data)
                    props.advance(data, frame_dt)
                viewer.opt.flags[:] = vis_flags
                viewer.user_scn.flags[:] = render_flags
                viewer.user_scn.ngeom = 0
                props.draw(viewer.user_scn, data)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))


if __name__ == "__main__":
    main()
