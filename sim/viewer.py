"""Interaktywny podgląd drona X2 w MuJoCo: ręczne latanie i obracające się śmigła.

    python -m sim.viewer [--scene sim/assets/scene_beacon.xml] [--seed 1234]

Domyślna scena: losowy teren 60 x 60 m (pagórki i zagłębienia) z drzewami, otoczony ścianami, i cel —
pomarańczowe pole lądowania z masztem i flagą, 15–23 m od startu w losowym kierunku (sim/terrain.py).
Ziarno świata jest wypisywane w konsoli; --seed odtwarza ten sam świat.
Gdy dron jest nad polem (niżej niż 1.5 m), pole robi się zielone. Gdy dotknie ściany albo
wzleci ponad nią (8 m), wraca automatycznie na start w tym samym świecie.

Klawisze (w oknie podglądu, trzymane):
    Alt          autostabilizacja wł. / wył. (na starcie wyłączona — tak jak lata model)
    S / W        acro: pochylanie nosa w dół / w górę | stabilizacja: lot do przodu / do tyłu
    D / A        acro: przechylanie w lewo / w prawo  | stabilizacja: lot w lewo / w prawo
    Shift / Ctrl acro: ciąg +30% / -30% od zawisu     | stabilizacja: wznoszenie / opadanie
    Q / E        powolny obrót w lewo / w prawo
    L            kamera przypięta do drona / swobodna
    kółko        zoom (od siebie = przybliż)
    Backspace    reset drona do startu (ten sam świat)
    N            nowy losowy świat (teren, drzewa, cel) i reset drona
    Spacja       pauza / wznowienie

Acro (bez stabilizacji): klawisz zadaje prędkość kątową, nie kąt — po puszczeniu dron przestaje się
obracać, ale zostaje przechylony i leci dalej; nic go nie poziomuje ani nie trzyma wysokości.
Stabilizacja (tylko do ręcznych testów): po puszczeniu klawiszy dron hamuje i trzyma wysokość.
Klawisze lotu, Alt, L, N i kółko przechwytuje hook Windows (sim/keyboard.py), więc nie przełączają
skrótów podglądu MuJoCo (W wireframe, S cienie, L additive) — stąd brak migania na czarno.
Flagi wizualizacji i tak są przywracane w każdej klatce (pozostałe litery dalej są skrótami).
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
from sim.target import Target
from sim.terrain import load_scene, outside_arena, randomize

DEFAULT_SCENE = Path(__file__).parent / "assets" / "scene_beacon.xml"

# Kody klawiszy GLFW.
KEY_SPACE = 32
KEY_BACKSPACE = 259
ZOOM_STEP = 1.12  # mnożnik odległości kamery na jeden krok kółka


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


def set_camera_lock(viewer, model, locked):
    """Przypięta: kamera śledzi drona. Swobodna: zostaje w miejscu, sterowanie myszą jak zwykle."""
    if locked:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("x2").id
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


def new_world(model, data, seed=None):
    seed = randomize(model, data, seed)
    print(f"świat: ziarno {seed} (ten sam świat: --seed {seed})")
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--seed", type=int, default=None, help="ziarno losowego świata (domyślnie losowe)")
    args = parser.parse_args()

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    props = PropellerVisuals(model)
    flight = RateController(model)
    stabilizer = VelocityController(flight)
    stabilized = False
    keyboard = FlightKeyboard()
    goal = Target.from_model(model)
    drone_id = model.body("x2").id
    goal_reached = False
    key_id = model.key("hover").id
    controls = Controls()
    terrain_id = model.hfield("terrain").id if model.nhfield else None
    seed = None
    if terrain_id is not None:
        seed = new_world(model, data, args.seed)
    reset_to_start(model, data, props, stabilizer, key_id)

    with mujoco.viewer.launch_passive(model, data, key_callback=controls.on_key) as viewer:
        camera_locked = True
        set_camera_lock(viewer, model, camera_locked)
        viewer.cam.distance = 1.5
        vis_flags = viewer.opt.flags.copy()
        render_flags = viewer.user_scn.flags.copy()
        print(__doc__)
        if not keyboard.available:
            print("Ręczne latanie działa tylko na Windows — stały ciąg zawisu, bez sterowania.")
        last = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            frame_dt = min(now - last, 0.05)  # bez nadrabiania po przycięciu okna
            last = now
            reset, paused = controls.take()
            if keyboard.take_stabilization_toggles() % 2:
                stabilized = not stabilized
                stabilizer.reset()  # trzyma wysokość z chwili włączenia
                print("autostabilizacja:", "WŁĄCZONA" if stabilized else "wyłączona (acro)")
            setpoints = keyboard.setpoints() if stabilized else None
            command = None if stabilized else keyboard.command(flight.hover_thrust)
            regenerate = terrain_id is not None and keyboard.take_new_world_requests() > 0
            camera_toggles = keyboard.take_camera_toggles()
            wheel = keyboard.take_wheel()

            with viewer.lock():
                if regenerate:
                    seed = new_world(model, data)
                    goal_reached = False
                    goal.show_reached(model, False)
                if reset or regenerate:
                    reset_to_start(model, data, props, stabilizer, key_id)
                if not paused:
                    target = data.time + frame_dt
                    while data.time < target:
                        if stabilized:
                            stabilizer.apply(model, data, **setpoints)
                        else:
                            flight.apply(command, model, data)
                        mujoco.mj_step(model, data)
                    props.advance(data, frame_dt)
                    if terrain_id is not None and outside_arena(model, data, drone_id):
                        print(f"dron poza planszą — reset na start (ten sam świat, ziarno {seed})")
                        reset_to_start(model, data, props, stabilizer, key_id)
                if goal is not None and goal.reached(data, drone_id) != goal_reached:
                    goal_reached = not goal_reached
                    goal.show_reached(model, goal_reached)
                    if goal_reached:
                        print(f"cel osiągnięty po {data.time:.1f} s")
                if camera_toggles % 2:
                    camera_locked = not camera_locked
                    set_camera_lock(viewer, model, camera_locked)
                    print("kamera:", "przypięta do drona" if camera_locked else "swobodna")
                if wheel:
                    viewer.cam.distance = min(50.0, max(0.2, viewer.cam.distance * ZOOM_STEP ** -wheel))
                viewer.opt.flags[:] = vis_flags
                viewer.user_scn.flags[:] = render_flags
                viewer.user_scn.ngeom = 0
                props.draw(viewer.user_scn, data)
            if regenerate:
                viewer.update_hfield(terrain_id)  # blokuje sam — poza viewer.lock()
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))
    keyboard.close()


if __name__ == "__main__":
    main()
