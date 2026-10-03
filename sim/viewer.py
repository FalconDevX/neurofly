"""Interaktywny podgląd drona X2 w MuJoCo: ręczne latanie i obracające się śmigła.

    python -m sim.viewer [--scene sim/assets/scene_beacon.xml] [--seed 1234]

Domyślna scena: losowy teren 60 x 60 m (pagórki i zagłębienia) z blokami (styl blueprint), otoczony ścianami, i cel —
cienki pomarańczowy prostopadłościan (przenikalny), 15–23 m od startu w losowym kierunku (sim/terrain.py).
Ziarno świata jest wypisywane w konsoli; --seed odtwarza ten sam świat.
Gdy dron go dotknie — SUKCES: cel robi się zielony, wynik w konsoli. Gdy dotknie ściany albo
wzleci ponad nią (8 m) albo się wywróci (sim/episode.py: > 1 s do góry nogami albo > 1.5 s na boku
na ziemi), wraca automatycznie na start w tym samym świecie — dla uczenia to nieudana próba.

Z --brain (środowisko .venv312 z FlyVis i danymi BANC): po prawej panel z siecią BANC v888 na żywo —
oczy drona → FlyVis → BANC, aktywność neuronów i grup lotu oraz komenda dekodera (sim/brain_panel.py).
Sieć tylko obserwuje, sterujesz dalej klawiszami. --brain-decoder: wagi z train_decoder.py.

Lewy dolny róg: obraz z kamer-oczu drona (lewe | prawe), dokładnie to, co dostaje wzrok Osoby 1
(visual_pipeline/drone_eyes.py: 157°, bez własnego drona w kadrze), odświeżany 30 razy na sekundę w osobnym wątku.

Klawisze (w oknie podglądu, trzymane):
    Alt          autostabilizacja wł. / wył. (na starcie wyłączona — tak jak lata model)
    W / S        acro: pochylanie nosa w dół / w górę | stabilizacja: lot do przodu / do tyłu
    A / D        acro: przechylanie w lewo / w prawo  | stabilizacja: lot w lewo / w prawo
    Shift / Ctrl acro: ciąg +30% / -30% od zawisu     | stabilizacja: wznoszenie / opadanie
    Q / E        powolny obrót w lewo / w prawo
    L            kamera przypięta za tyłem drona (obraca się z nim) / swobodna (mysz)
    kółko        zoom (od siebie = przybliż)
    Backspace    reset drona do startu (ten sam świat)
    N            nowy losowy świat (teren, bloki, cel) i reset drona
    Spacja       pauza / wznowienie
    C            panel czujników: odczyty realistycznych czujników (szum, dryf, opóźnienie) obok prawdy, w tym
                 „GPS” celu (kierunek i odległość do celu, przybliżone) + strzałki nad dronem:
                 pomarańczowa = gdzie według GPS jest cel, zielona = gdzie jest naprawdę
    T            ślad lotu wł. / wył. (linia za dronem, ~30 m trasy, starsze odcinki bledną)
    M            panel metryk: odległość i postęp do celu, błąd kursu, wysokość, przechył, wyniki epizodów
                 (z prawdziwego stanu; --metrics-csv zapisuje podsumowania epizodów)
    CapsLock     wiatr wł. / wył. (domyślnie 8 m/s w losowym kierunku, podmuchy ±3, --wind-speed; strzałka w prawym dolnym rogu
                 pokazuje, dokąd wieje względem widoku kamery — w górę = w głąb ekranu)

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

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from sim.control import MOTOR_TAU, RateController, VelocityController
from sim.episode import CrashDetector
from sim.metrics import EpisodeMetrics, sensors_panel
from sim.keyboard import FlightKeyboard
from sim.propellers import PropellerVisuals
from sim.sensors import DroneSensors
from sim.target import Target
from sim.terrain import load_scene, outside_arena, randomize, upload_terrain
from sim.trail import Trail
from sim.wind import Wind
from visual_pipeline.drone_eyes import EYE_H, EYE_W, MujocoEyes

DEFAULT_SCENE = Path(__file__).parent / "assets" / "scene_beacon.xml"

# Kody klawiszy GLFW.
KEY_SPACE = 32
KEY_BACKSPACE = 259
ZOOM_STEP = 1.12  # mnożnik odległości kamery na jeden krok kółka
CHASE_ELEVATION = -25.0  # stopnie, kamera za dronem lekko z góry
CHASE_DISTANCE = 1.5     # m na starcie (kółko myszy zmienia)
CHASE_FOLLOW_TIME = 0.25  # s, stała czasowa obrotu kamery za kursem drona (wygładzanie niezależne od FPS)
EYES_HEIGHT = (120, 240)  # px, wysokość podglądu oczu: ~1/3 wysokości sceny w tych granicach
EYES_EVERY = 2     # co ile klatek odświeżać podgląd oczu (60 / 2 = 30 Hz; całość w osobnym wątku)
WIND_SIZE = 160    # px, kwadrat ze strzałką wiatru w prawym dolnym rogu sceny


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


def reset_to_start(model, data, key_id, *parts):
    """Dron na start (keyframe), parts = obiekty ze stanem do wyzerowania (.reset())."""
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    for part in parts:
        part.reset()


class EyesWorker:
    """Render oczu drona w osobnym wątku (własny kontekst GPU).

    Render offscreen w głównym wątku czekał na GPU zajęte przez okno podglądu (~20-50 ms) i szarpał
    fizyką i kamerą. Tu główna pętla tylko kopiuje stan (request); gotowe klatki idą do on_frames
    (też w tym wątku — skalowanie i viewer.set_images kosztują ~4 ms).
    Zmiany modelu (nowy świat) robić pod `with worker.render_lock:`, potem upload_terrain().
    """

    def __init__(self, model, on_frames=None):
        self.model = model
        self.on_frames = on_frames  # on_frames(klatki (lewa, prawa)) — wywoływane w wątku oczu
        self.snapshot = mujoco.MjData(model)
        self.render_lock = threading.Lock()  # trzymany przez wątek w trakcie renderu
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._busy = self._upload = self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, data):
        """Zleca render obecnego stanu; ignoruje, jeśli poprzedni jeszcze trwa."""
        with self._lock:
            if self._busy:
                return
            mujoco.mj_copyData(self.snapshot, self.model, data)
            self._busy = True
        self._wake.set()

    def upload_terrain(self):
        with self._lock:
            self._upload = True

    def close(self):
        self._stop = True
        self._wake.set()

    def _run(self):
        eyes = MujocoEyes(self.model)  # kontekst GPU musi powstać w tym wątku
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                break
            with self._lock:
                upload, self._upload = self._upload, False
            with self.render_lock:
                if upload:
                    upload_terrain(eyes.renderer, self.model)
                frames = eyes.render(self.snapshot)
            if self.on_frames is not None:
                self.on_frames(frames)
            with self._lock:
                self._busy = False


class Overlays:
    """Obrazy na scenie: oczy drona (lewy dolny róg) i strzałka wiatru (prawy dolny róg).

    viewer.set_images zastępuje wszystkie obrazy naraz, więc oba idą w jednym wywołaniu.
    """

    def __init__(self, model, brain=None):
        self.model = model
        self.brain = brain  # BrainPanel (--brain) albo None
        self.eyes = None  # MujocoEyes tworzony dopiero, gdy nie dostajemy gotowych klatek
        self.eyes_image = None

    def update(self, viewer, data, wind, refresh_eyes, frames=None):
        """frames: gotowe klatki oczu (lewa, prawa), np. obs["eyes"] z WorldEnv; None = renderuj tutaj."""
        view = viewer.viewport  # obszar sceny 3D w oknie (bez paneli menu); współrzędne od dołu okna
        if view is None:
            return
        images = []
        # oczy: ~1/3 wysokości sceny, ale zmniejszone tak, żeby zmieściły się obok strzałki wiatru
        free_width = view.width - (WIND_SIZE + 8 if wind.enabled else 0)
        height = int(min(np.clip(view.height // 3, *EYES_HEIGHT), (free_width - 4) / 2 * EYE_H / EYE_W))
        width = round(EYE_W * height / EYE_H)
        if height >= 60 and view.height >= height:
            if refresh_eyes or self.eyes_image is None or self.eyes_image.shape[0] != height:
                if frames is None:
                    if self.eyes is None:
                        self.eyes = MujocoEyes(self.model)
                    frames = self.eyes.render(data)
                left, right = (cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) for img in frames)
                gap = np.full((height, 4, 3), 255, np.uint8)  # biały pasek między okiem lewym i prawym
                self.eyes_image = np.hstack([left, gap, right])
            images.append((mujoco.MjrRect(view.left, view.bottom, 2 * width + 4, height), self.eyes_image))
        if self.brain is not None and self.brain.image is not None:
            # panel sieci przy prawej krawędzi okna (panel ustawień MuJoCo schowany, Shift+Tab go przywraca),
            # na całą wysokość nad strzałką wiatru, najwyżej 40% szerokości sceny
            img = self.brain.image
            free_height = view.height - (WIND_SIZE + 8 if wind.enabled else 0)
            w = int(min(view.width * 0.4, free_height * img.shape[1] / img.shape[0]))
            h = round(w * img.shape[0] / img.shape[1])
            if w >= 120:
                images.append((mujoco.MjrRect(view.left + view.width - w, view.bottom + view.height - h, w, h),
                               cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)))
        if wind.enabled and view.width >= WIND_SIZE and view.height >= WIND_SIZE:
            rect = mujoco.MjrRect(view.left + view.width - WIND_SIZE, view.bottom, WIND_SIZE, WIND_SIZE)
            images.append((rect, wind_arrow(wind.velocity, viewer.cam.azimuth)))
        if images:
            viewer.set_images(images)
        else:
            viewer.clear_images()


def wind_arrow(velocity, camera_azimuth):
    """Kwadrat ze strzałką kierunku wiatru względem widoku kamery (góra = w głąb ekranu) i prędkością."""
    img = np.full((WIND_SIZE, WIND_SIZE, 3), 40, np.uint8)
    c = WIND_SIZE // 2
    cv2.circle(img, (c, c - 8), c - 22, (90, 90, 90), 2, cv2.LINE_AA)
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed > 0.05:
        # kąt wiatru względem kierunku patrzenia kamery (azymut 0 = kamera patrzy w +x)
        rel = np.arctan2(velocity[1], velocity[0]) - np.deg2rad(camera_azimuth)
        forward, left = np.cos(rel), np.sin(rel)
        r = c - 30
        tip = (int(c - left * r), int(c - 8 - forward * r))
        tail = (int(c + left * r), int(c - 8 + forward * r))
        cv2.arrowedLine(img, tail, tip, (90, 200, 255), 5, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(img, f"wiatr {speed:.1f} m/s", (10, WIND_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    return img


BEACON_ARROW_RGBA = np.array([1.0, 0.55, 0.1, 0.9], np.float32)   # gdzie według GPS jest cel
TRUE_ARROW_RGBA = np.array([0.2, 0.9, 0.3, 0.6], np.float32)      # gdzie jest naprawdę


def draw_beacon_arrows(scn, data, drone_id, sensors):
    """Nad dronem: strzałka z odczytu „GPS” celu (pomarańczowa) i prawdziwy kierunek (cienka zielona)."""
    reading, true = sensors.read(), sensors.latest
    if reading is None or true is None or not np.all(np.isfinite(reading["beacon"])):
        return
    rot = data.xmat[drone_id].reshape(3, 3)
    heading = np.arctan2(rot[1, 0], rot[0, 0])
    start = data.xpos[drone_id] + np.array([0, 0, 0.35])
    for beacon, rgba, width in ((reading["beacon"], BEACON_ARROW_RGBA, 0.025), (true["beacon"], TRUE_ARROW_RGBA, 0.012)):
        if scn.ngeom >= scn.maxgeom:
            return
        angle = heading + beacon[0]
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        z = direction
        x = np.array([0.0, 0.0, 1.0])
        y = np.cross(z, x)
        mat = np.column_stack([x, y, z])  # oś z strzałki = kierunek do celu
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW,
                            np.array([width, width, 0.8]), start, mat.ravel(), rgba)
        scn.ngeom += 1


def update_panels(viewer, show_sensors, show_metrics, sensors, metrics, metrics_pos):
    """Panele tekstowe: C = czujniki (lewy górny róg sceny), M = metryki (prawy górny / środek)."""
    texts = []
    font = mujoco.mjtFontScale.mjFONTSCALE_100
    if show_sensors and sensors.read() is not None:
        texts.append((font, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                      *sensors_panel(sensors.read(), sensors.latest, sensors.mode)))
    if show_metrics:
        texts.append((font, metrics_pos, *metrics.panel()))
    if texts:
        viewer.set_texts(texts)
    else:
        viewer.clear_texts()


def set_camera_lock(viewer, model, locked):
    """Przypięta: kamera za tyłem drona, obraca się z nim (follow_drone). Swobodna: sterowanie myszą jak zwykle."""
    if locked:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("x2").id
        viewer.cam.elevation = CHASE_ELEVATION
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


def follow_drone(viewer, data, drone_id, dt=0.0, snap=False):
    """Kamera przypięta: azymut za kursem drona (kamera zawsze za jego tyłem), wygładzony w czasie."""
    if viewer.cam.type != mujoco.mjtCamera.mjCAMERA_TRACKING:
        return
    nose = data.xmat[drone_id].reshape(3, 3)[:, 0]  # oś x drona = nos
    if np.hypot(nose[0], nose[1]) < 0.2:  # dron prawie pionowo — kurs nieokreślony, nie ruszamy kamery
        return
    heading = np.degrees(np.arctan2(nose[1], nose[0]))
    diff = (heading - viewer.cam.azimuth + 180) % 360 - 180
    viewer.cam.azimuth += diff if snap else (1 - np.exp(-dt / CHASE_FOLLOW_TIME)) * diff


def new_world(model, data, seed=None):
    seed = randomize(model, data, seed)
    print(f"świat: ziarno {seed} (ten sam świat: --seed {seed})")
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--seed", type=int, default=None, help="ziarno losowego świata (domyślnie losowe)")
    parser.add_argument("--wind-speed", type=float, default=8.0, help="średnia prędkość wiatru [m/s] (CapsLock)")
    parser.add_argument("--metrics-csv", type=Path, default=None, help="zapis podsumowań epizodów (M) do CSV")
    parser.add_argument("--motor-tau", type=float, default=MOTOR_TAU,
                        help="opóźnienie silników [s] (domyślnie 0.04 = realistycznie, 0 = natychmiast)")
    parser.add_argument("--brain", action="store_true", help="panel z siecią BANC na żywo (.venv312)")
    parser.add_argument("--brain-decoder", type=Path, help="wagi dekodera do panelu (train_decoder.py)")
    args = parser.parse_args()
    brain = None
    if args.brain or args.brain_decoder:
        from sim.brain_panel import BrainPanel

        brain = BrainPanel(decoder_path=args.brain_decoder)

    model = load_scene(args.scene)
    data = mujoco.MjData(model)
    props = PropellerVisuals(model)
    flight = RateController(model, motor_tau=args.motor_tau)
    stabilizer = VelocityController(flight)
    stabilized = False
    keyboard = FlightKeyboard()
    goal = Target.from_model(model)
    drone_id = model.body("x2").id
    goal_reached = False
    key_id = model.key("hover").id
    crash = CrashDetector(model)
    sensors = DroneSensors(model, mode="real")      # C: realistyczne czujniki (szum, dryf, opóźnienie)
    metrics = EpisodeMetrics(model, csv_path=args.metrics_csv)  # M: metryki z prawdziwego stanu
    show_sensors = show_metrics = False
    metrics_pos = mujoco.mjtGridPos.mjGRID_TOP if brain is not None else mujoco.mjtGridPos.mjGRID_TOPRIGHT
    trail = Trail()  # T: ślad lotu
    overlays = Overlays(model, brain)
    gyro_adr = model.sensor("body_gyro").adr[0]
    eyes_worker = EyesWorker(model)
    wind = Wind(mean_speed=args.wind_speed, gust_speed=0.375 * args.wind_speed)  # 8 m/s -> podmuchy ±3
    controls = Controls()
    terrain_id = model.hfield("terrain").id if model.nhfield else None
    seed = None
    if terrain_id is not None:
        seed = new_world(model, data, args.seed)
        wind.reset(seed)
    resettables = (props, stabilizer, crash)

    def restart(outcome=None):
        """Dron na start + nowy epizod metryk (outcome = wynik kończonego epizodu)."""
        if outcome is not None:
            summary = metrics.finish(outcome)
            if summary is not None and args.metrics_csv:
                print(f"   metryki zapisane: {args.metrics_csv}")
        reset_to_start(model, data, key_id, *resettables)
        trail.reset()
        sensors.reset(None if seed is None else seed + metrics.episode + 1, data,
                      target=goal.position(data) if goal is not None else None)
        metrics.start(data, goal.position(data) if goal is not None else None, seed)

    restart()

    # z panelem sieci chowamy prawy panel ustawień MuJoCo, żeby panel sieci stał przy prawej krawędzi okna
    with mujoco.viewer.launch_passive(model, data, key_callback=controls.on_key,
                                      show_right_ui=brain is None) as viewer:
        camera_locked = True
        set_camera_lock(viewer, model, camera_locked)
        viewer.cam.distance = CHASE_DISTANCE
        follow_drone(viewer, data, drone_id, snap=True)
        def on_frames(frames):
            if brain is not None:
                brain.submit(frames)
            overlays.update(viewer, None, wind, refresh_eyes=True, frames=frames)

        eyes_worker.on_frames = on_frames
        vis_flags = viewer.opt.flags.copy()
        render_flags = viewer.user_scn.flags.copy()
        print(__doc__)
        if not keyboard.available:
            print("Ręczne latanie działa tylko na Windows — stały ciąg zawisu, bez sterowania.")
        last = time.perf_counter()
        frame = 0
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
            if keyboard.caps_lock != wind.enabled:
                wind.enabled = keyboard.caps_lock
                v = wind.velocity
                print("wiatr:", f"WŁĄCZONY, {np.hypot(v[0], v[1]):.1f} m/s, wieje w kierunku "
                      f"{np.degrees(np.arctan2(v[1], v[0])) % 360:.0f}° (0° = +x)" if wind.enabled else "wyłączony")
            camera_toggles = keyboard.take_camera_toggles()
            wheel = keyboard.take_wheel()
            toggle_c, toggle_m = keyboard.take_panel_toggles()
            show_sensors ^= bool(toggle_c % 2)
            show_metrics ^= bool(toggle_m % 2)
            if keyboard.take_trail_toggles() % 2:
                trail.visible = not trail.visible

            with viewer.lock():
                if regenerate:
                    metrics.finish("nowy świat (N)")
                    with eyes_worker.render_lock:  # wątek oczu nie czyta modelu w trakcie zmiany świata
                        seed = new_world(model, data)
                    eyes_worker.upload_terrain()
                    wind.reset(seed)
                    goal_reached = False
                    goal.show_reached(model, False)
                    restart()
                elif reset:
                    restart("przerwany (Backspace)")
                if not paused:
                    wind.step(model, frame_dt)
                    target = data.time + frame_dt
                    while data.time < target:
                        if stabilized:
                            stabilizer.apply(model, data, **setpoints)
                        else:
                            flight.apply(command, model, data)
                        mujoco.mj_step(model, data)
                        sensors.update(data)
                    props.advance(data, frame_dt)
                    trail.add(data.xpos[drone_id])
                    metrics.update(data, goal.position(data) if goal is not None else None, wind.velocity)
                    if terrain_id is not None and outside_arena(model, data, drone_id):
                        print(f"dron poza planszą — reset na start (ten sam świat, ziarno {seed})")
                        restart("poza planszą")
                    crashed = crash.update(data, frame_dt)
                    if crashed:
                        print(f"dron się wywrócił ({crashed}) — nieudana próba, reset na start (ziarno {seed})")
                        restart(f"wywrotka: {crashed}")
                if goal is not None and goal.reached(data, drone_id) != goal_reached:
                    goal_reached = not goal_reached
                    goal.show_reached(model, goal_reached)
                    if goal_reached:
                        print(f"*** SUKCES! Dron dotknął celu po {metrics.current['time']:.1f} s ***")
                        metrics.finish("cel")  # nowy epizod liczy się dalej od tego miejsca
                        metrics.start(data, goal.position(data), seed)
                if camera_toggles % 2:
                    camera_locked = not camera_locked
                    set_camera_lock(viewer, model, camera_locked)
                    print("kamera:", "przypięta do drona" if camera_locked else "swobodna")
                if wheel:
                    viewer.cam.distance = min(50.0, max(0.2, viewer.cam.distance * ZOOM_STEP ** -wheel))
                follow_drone(viewer, data, drone_id, frame_dt)
                viewer.opt.flags[:] = vis_flags
                viewer.user_scn.flags[:] = render_flags
                viewer.user_scn.ngeom = 0
                trail.draw(viewer.user_scn)
                props.draw(viewer.user_scn, data)
                if show_sensors:
                    draw_beacon_arrows(viewer.user_scn, data, drone_id, sensors)
            if brain is not None:  # konwencja DroneEnv.imu: yaw + = w prawo
                gx, gy, gz = data.sensordata[gyro_adr:gyro_adr + 3]
                brain.set_gyro((gx, gy, -gz))
            if frame % 6 == 0 or toggle_c or toggle_m:  # panele C / M ~10 razy na sekundę
                update_panels(viewer, show_sensors, show_metrics, sensors, metrics, metrics_pos)
            if frame % EYES_EVERY == 0:
                eyes_worker.request(data)  # render, skalowanie i set_images w tle; tutaj tylko kopia stanu
            frame += 1
            if regenerate:
                viewer.update_hfield(terrain_id)  # blokuje sam — poza viewer.lock()
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))
    eyes_worker.close()
    if brain is not None:
        brain.close()
    keyboard.close()


if __name__ == "__main__":
    main()
