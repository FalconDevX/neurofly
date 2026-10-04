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
    C            panel czujników (to, co dostaje model; --sensors real/ideal), w tym „GPS” celu + strzałki
                 nad dronem: pomarańczowa = odczyt GPS, zielona = prawdziwy kierunek do celu
    T            ślad lotu wł. / wył. (linia za dronem)
    M            panel metryk z prawdziwego stanu (--metrics-csv zapisuje podsumowania epizodów)
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

from sim.control import MOTOR_TAU, VelocityController
from sim.world_env import WorldEnv, to_action
from sim.keyboard import FlightKeyboard
from sim.propellers import PropellerVisuals
from sim.trail import Trail
from sim.metrics import EpisodeMetrics
from sim.viewer import (CHASE_DISTANCE, KEY_BACKSPACE, ZOOM_STEP, Overlays, draw_beacon_arrows, follow_drone,
                        set_camera_lock, update_panels)


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
    parser.add_argument("--beacon-scale", type=float, default=1.0, help="z --banc: szerszy cel (prostopadłościan, w pamięci)")
    parser.add_argument("--beacon-alpha", type=float, default=None, help="z --banc: przezroczystość znacznika (1 = pełny)")
    parser.add_argument("--vision-range", type=float, default=float("inf"), help="z --banc: bliżej celu yaw z BANC, dalej z GPS [m]")
    parser.add_argument("--obstacles", choices=("clear", "path", "keep"), default="clear",
                        help="z --banc: czysty korytarz / bloki na trasie / wszystkie bloki")
    parser.add_argument("--path-blocks", type=int, default=2, help="z --banc --obstacles path: ile bloków na trasie")
    parser.add_argument("--beacon-color", choices=("scene", "dark-red"), default="dark-red",
                    help="kolor celu: ciemnoczerwony (kontrast jasności dla FlyVis) albo ze sceny (pomarańczowy)")
    parser.add_argument("--wind-speed", type=float, default=8.0, help="średnia prędkość wiatru [m/s] (CapsLock)")
    parser.add_argument("--start-noise", action="store_true",
                        help="losowy przechył/prędkość na starcie (jak przy uczeniu); domyślnie start w idealnym zawisie")
    parser.add_argument("--sensors", choices=("real", "ideal"), default=None,
                        help="czujniki w obserwacji: real = szum, dryf, opóźnienie (domyślnie), ideal = prawdziwe "
                             "wartości (domyślnie z --banc, bo na takich trenowano pilota)")
    parser.add_argument("--metrics-csv", type=Path, default=None, help="zapis podsumowań epizodów (M) do CSV")
    parser.add_argument("--motor-tau", type=float, default=None,
                        help="opóźnienie silników [s] (domyślnie 0.04 = realistycznie; z --banc 0, jak w treningu)")
    parser.add_argument("--banc-axes", nargs="*", choices=("thrust", "roll", "pitch"), default=[],
                        help="z --banc: te osie tylko z BANC (wagi czujników drona = 0), np. thrust roll pitch")
    parser.add_argument("--control", choices=("acro", "angle"), default="acro",
                        help="acro = Plan A (domyślnie), angle = Plan C (symulator trzyma poziom)")
    args = parser.parse_args()

    policy = load_model(args.model) if args.model else None
    banc = None
    if args.banc:
        from sim.banc_pilot import BancPilot

        banc = policy = BancPilot(args.banc, args.banc_forward, beacon_scale=args.beacon_scale, beacon_alpha=args.beacon_alpha,
                                  vision_range=args.vision_range, beacon_color=args.beacon_color,
                                  banc_axes=args.banc_axes)
        if not banc.assist:
            args.control = "angle"  # wagi z train_world.py: BANC daje przechył, symulator go utrzymuje
    sensors_mode = args.sensors or ("ideal" if args.banc else "real")
    motor_tau = args.motor_tau if args.motor_tau is not None else (0.0 if args.banc else MOTOR_TAU)
    env = WorldEnv(control=args.control, start_noise=args.start_noise, wind_speed=args.wind_speed,
                   sensors=sensors_mode, motor_tau=motor_tau)
    metrics = EpisodeMetrics(env.model, csv_path=args.metrics_csv)
    show_sensors = show_metrics = False
    metrics_pos = mujoco.mjtGridPos.mjGRID_TOP if args.banc else mujoco.mjtGridPos.mjGRID_TOPRIGHT
    trail = Trail()  # T: ślad lotu
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
    if banc:  # po każdym resecie: korytarz do celu według --obstacles (jak w treningu), stan sieci od nowa
        from sim.banc_pilot import clear_corridor

        def banc_reset(o):
            if args.obstacles != "keep":
                keep = args.path_blocks if args.obstacles == "path" else 0
                if clear_corridor(env, keep=keep, rng=np.random.default_rng(int(env.world_seed) + 7)):
                    o["eyes"] = np.stack(env.eyes.render(env.data))
            banc.reset(o)

        reset_policy = banc_reset
    if reset_policy:
        reset_policy(obs)
    print(f"świat {world}, cel {info['distance']:.1f} m od startu, czujniki: {sensors_mode}, "
          f"opóźnienie silników: {motor_tau * 1000:.0f} ms")
    metrics.start(env.data, info["target"], world)

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
        toggle_c, toggle_m = keyboard.take_panel_toggles()
        show_sensors ^= bool(toggle_c % 2)
        show_metrics ^= bool(toggle_m % 2)
        if keyboard.take_trail_toggles() % 2:
            trail.visible = not trail.visible
        if new_world or requests["reset"]:
            metrics.finish("nowy świat (N)" if new_world else "przerwany (Backspace)")
            requests["reset"] = False
            with viewer.lock():
                obs, info = env.reset(options={} if new_world else {"world_seed": world})
            if reset_policy:
                reset_policy(obs)
            world = info["world_seed"]
            stabilizer.z_ref = None
            viewer.update_hfield(env.model.hfield("terrain").id)
            print(f"{'nowy świat' if new_world else 'od nowa'}: świat {world}, cel {info['distance']:.1f} m od startu")
            metrics.start(env.data, info["target"], world)
            trail.reset()
            total, steps, t0 = 0.0, 0, time.perf_counter()

        action = policy(obs) if policy else keyboard_action(keyboard, env, stabilizer, stabilized)
        with viewer.lock():
            obs, reward, terminated, truncated, info = env.step(action)
            props.advance(env.data, env.dt)
            trail.add(env.data.xpos[env.drone_id])
            viewer.user_scn.ngeom = 0
            trail.draw(viewer.user_scn)
            props.draw(viewer.user_scn, env.data)
            if show_sensors:
                draw_beacon_arrows(viewer.user_scn, env.data, env.drone_id, env.sensors)
            follow_drone(viewer, env.data, env.drone_id, env.dt)
        metrics.update(env.data, info["target"], info["wind"])
        total += reward
        steps += 1
        if steps % 3 == 0 or toggle_c or toggle_m:  # panele C / M ~10 razy na sekundę
            update_panels(viewer, show_sensors, show_metrics, env.sensors, metrics, metrics_pos)
        # oczy z obserwacji = dokładnie to, co dostaje model (co krok, 30 Hz)
        overlays.update(viewer, env.data, env.wind, refresh_eyes=True, frames=obs["eyes"])
        viewer.sync()
        time.sleep(max(0.0, t0 + steps * env.dt - time.perf_counter()))  # czas rzeczywisty

        if terminated or truncated:
            if info["outcome"] == "cel":
                print(f"*** SUKCES! Dron dotknął celu po {info['time']:.1f} s ***")
                with viewer.lock():
                    env.target.show_reached(env.model, True)  # zielony cel przez chwilę przed resetem
                viewer.sync()
                time.sleep(1.5)
                with viewer.lock():
                    env.target.show_reached(env.model, False)
            print(f"   koniec: {info['outcome']} po {info['time']:.1f} s, nagroda {total:.1f}, "
                  f"odległość do celu {info['distance']:.1f} m — od nowa w tym samym świecie")
            metrics.finish(info["outcome"])
            with viewer.lock():
                obs, info = env.reset(options={"world_seed": world})
            if reset_policy:
                reset_policy(obs)
            stabilizer.z_ref = None
            metrics.start(env.data, info["target"], world)
            trail.reset()
            total, steps, t0 = 0.0, 0, time.perf_counter()
    keyboard.close()


if __name__ == "__main__":
    main()
