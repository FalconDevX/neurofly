"""Uruchamia WorldEnv (sim/world_env.py) w oknie — sterujesz Ty z klawiatury albo (później) model.

    python -m sim.run_env                      # Ty sterujesz przez WorldEnv (tak jak będzie sterował model)
    python -m sim.run_env --seed 7             # konkretny świat
    python -m sim.run_env --model pakiet.modul:funkcja   # gdy będzie model: funkcja(obs) -> akcja
    python -m sim.run_env --banc data/decoders/planB_dn.npz   # BANC + dekoder pilotuje (.venv312), panel sieci

Bez --model dron NIE leci sam: akcję daje klawiatura (bez klawiszy = ciąg zawisu, zero obrotu),
a start jest w idealnym zawisie (--start-noise włącza losowe zaburzenie startu, jak przy uczeniu).
Różnica względem python -m sim.viewer: tu wszystko idzie przez WorldEnv — akcja w formacie
FlightCommand, obserwacja = oczy + IMU, koniec epizodu i nagroda jak przy uczeniu (wypisywane w konsoli).

Klawisze (jak w sim.viewer):
    Alt          autostabilizacja wł. / wył. (start: wyłączona = acro, jak model w Planie A)
    W / S        acro: nos w dół / w górę           | stabilizacja: lot do przodu / do tyłu
    A / D        acro: przechył w lewo / w prawo    | stabilizacja: lot w lewo / w prawo
    Shift / Ctrl acro: ciąg +30% / -30%             | stabilizacja: wznoszenie / opadanie
    Q / E        obrót w lewo / w prawo
    CapsLock     wiatr wł. / wył. (domyślnie 8 m/s ± podmuchy, --wind-speed; strzałka w prawym dolnym rogu)
    Backspace    od nowa w tym samym świecie        N   nowy losowy świat
    L            kamera za dronem / swobodna        kółko myszy   oddalanie / przybliżanie
Koniec epizodu (cel, wywrotka, poza planszą, limit czasu) — wynik w konsoli i start od nowa w tym samym świecie.
Autostabilizacja (Alt) to VelocityController (sim/control.py, prawdziwy stan z symulatora) — tylko do ręcznych
testów; jej wynik też idzie do WorldEnv jako zwykła akcja. Model jej nie dostaje.
"""

import argparse
from pathlib import Path
import importlib
import time

import mujoco
import mujoco.viewer
import numpy as np

from sim.control import VelocityController
from sim.world_env import WorldEnv, to_action
from sim.keyboard import FlightKeyboard
from sim.propellers import PropellerVisuals
from sim.viewer import CHASE_DISTANCE, KEY_BACKSPACE, ZOOM_STEP, Overlays, follow_drone, set_camera_lock


def keyboard_action(keyboard, env, stabilizer, stabilized):
    """Klawisze -> akcja WorldEnv (te same klawisze i tryby co sim.viewer)."""
    hover = env.rate_ctrl.hover_thrust
    if stabilized:
        return to_action(stabilizer.compute(env.data, **keyboard.setpoints()), hover)
    return to_action(keyboard.command(hover), hover)


def load_model(spec):
    """'pakiet.modul:funkcja' -> funkcja(obs) zwracająca akcję WorldEnv."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name or "act")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="ziarno świata (domyślnie losowe)")
    parser.add_argument("--model", default=None, help="pakiet.modul:funkcja — funkcja(obs) -> akcja; bez tego klawiatura")
    parser.add_argument("--banc", type=Path, default=None,
                        help="wagi dekodera (train_decoder.py): pilotem jest BANC (sim/banc_pilot.py)")
    parser.add_argument("--banc-forward", type=float, default=1.0, help="m/s do przodu z --banc")
    parser.add_argument("--beacon-scale", type=float, default=1.0, help="z --banc: grubszy maszt celu (w pamięci)")
    parser.add_argument("--wind-speed", type=float, default=8.0, help="średnia prędkość wiatru [m/s] (CapsLock)")
    parser.add_argument("--start-noise", action="store_true",
                        help="losowy przechył/prędkość na starcie (jak przy uczeniu); domyślnie start w idealnym zawisie")
    parser.add_argument("--control", choices=("acro", "angle"), default="acro",
                        help="acro = Plan A (domyślnie), angle = Plan C (symulator trzyma poziom)")
    args = parser.parse_args()

    policy = load_model(args.model) if args.model else None
    banc = None
    if args.banc:
        from sim.banc_pilot import BancPilot

        banc = policy = BancPilot(args.banc, args.banc_forward, beacon_scale=args.beacon_scale)
        if not banc.assist:
            args.control = "angle"  # wagi z train_world.py: BANC daje przechył, symulator go utrzymuje
    env = WorldEnv(control=args.control, start_noise=args.start_noise, wind_speed=args.wind_speed)
    if banc:
        env.reset(seed=args.seed)
        banc.bind(env)  # kalibracja zmienia stan env — dlatego reset poniżej
    keyboard = FlightKeyboard()
    stabilizer = VelocityController(env.rate_ctrl)
    stabilized = False
    requests = {"reset": False}
    on_key = lambda key: requests.update(reset=True) if key == KEY_BACKSPACE else None  # noqa: E731

    obs, info = env.reset(seed=args.seed)
    world = info["world_seed"]
    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key,
                                          show_right_ui=banc is None)  # panel BANC przy prawej krawędzi
    camera_locked = True
    set_camera_lock(viewer, env.model, camera_locked)
    viewer.cam.distance = CHASE_DISTANCE
    follow_drone(viewer, env.data, env.drone_id, snap=True)
    props = PropellerVisuals(env.model)
    overlays = Overlays(env.model, policy if hasattr(policy, "image") else None)  # panel BANC przy --banc
    print(__doc__)
    print(f"sterowanie: {'BANC ' + str(args.banc) if args.banc else 'model ' + args.model if policy else 'klawiatura (bez modelu dron sam nie leci)'}")
    reset_policy = None
    if banc:  # po każdym resecie: korytarz do celu bez drzew (jak w treningu), stan sieci od nowa
        from sim.banc_pilot import clear_corridor

        def banc_reset(o):
            if clear_corridor(env):
                o["eyes"] = np.stack(env.eyes.render(env.data))
            banc.reset(o)

        reset_policy = banc_reset
    if reset_policy:
        reset_policy(obs)
    print(f"świat {world}, cel {info['distance']:.1f} m od startu")

    total, steps, t0 = 0.0, 0, time.perf_counter()
    while viewer.is_running():
        env.wind.enabled = keyboard.caps_lock
        if keyboard.take_stabilization_toggles() % 2:
            stabilized = not stabilized
            stabilizer.z_ref = None  # trzyma wysokość z chwili włączenia
            print("autostabilizacja:", "WŁĄCZONA" if stabilized else "wyłączona (acro)")
        new_world = keyboard.take_new_world_requests() > 0
        if keyboard.take_camera_toggles() % 2:
            camera_locked = not camera_locked
            set_camera_lock(viewer, env.model, camera_locked)
        wheel = keyboard.take_wheel()
        if wheel:
            viewer.cam.distance = min(50.0, max(0.2, viewer.cam.distance * ZOOM_STEP ** -wheel))
        if new_world or requests["reset"]:
            requests["reset"] = False
            with viewer.lock():
                obs, info = env.reset(options={} if new_world else {"world_seed": world})
            if reset_policy:
                reset_policy(obs)
            world = info["world_seed"]
            stabilizer.z_ref = None
            viewer.update_hfield(env.model.hfield("terrain").id)
            print(f"{'nowy świat' if new_world else 'od nowa'}: świat {world}, cel {info['distance']:.1f} m od startu")
            total, steps, t0 = 0.0, 0, time.perf_counter()

        action = policy(obs) if policy else keyboard_action(keyboard, env, stabilizer, stabilized)
        with viewer.lock():
            obs, reward, terminated, truncated, info = env.step(action)
            props.advance(env.data, env.dt)
            viewer.user_scn.ngeom = 0
            props.draw(viewer.user_scn, env.data)
            follow_drone(viewer, env.data, env.drone_id, env.dt)
        total += reward
        steps += 1
        # oczy z obserwacji = dokładnie to, co dostaje model (co krok, 30 Hz)
        overlays.update(viewer, env.data, env.wind, refresh_eyes=True, frames=obs["eyes"])
        viewer.sync()
        time.sleep(max(0.0, t0 + steps * env.dt - time.perf_counter()))  # czas rzeczywisty

        if terminated or truncated:
            print(f"   koniec: {info['outcome']} po {info['time']:.1f} s, nagroda {total:.1f}, "
                  f"odległość do celu {info['distance']:.1f} m — od nowa w tym samym świecie")
            with viewer.lock():
                obs, info = env.reset(options={"world_seed": world})
            if reset_policy:
                reset_policy(obs)
            stabilizer.z_ref = None
            total, steps, t0 = 0.0, 0, time.perf_counter()
    keyboard.close()


if __name__ == "__main__":
    main()
