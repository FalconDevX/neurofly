# Osoba 3 — symulowany dron (MuJoCo)

## Zmiana silnika: Gazebo Sim → MuJoCo

Gazebo Sim wymaga WSL2, który na maszynie Osoby 3 nie startuje (`HCS_E_HYPERV_NOT_INSTALLED`).
MuJoCo działa natywnie na Windows, a FlyGym (Osoba 1) też jest zbudowany na MuJoCo — kamery-oczy
drona mogą renderować w tej samej konwencji, której oczekuje Retina.

## Ustalenia interfejsu

- **Wejście:** ciąg zbiorczy + zadane prędkości kątowe `[thrust, roll_rate, pitch_rate, yaw_rate]`
  (tryb acro). Regulator PID prędkości kątowych i mixer na 4 silniki są po stronie symulatora.
  `FlightCommand` z `banc_control/contracts.py` (thrust 0..1, roll/pitch/yaw −1..1) traktujemy jako
  znormalizowane prędkości kątowe.
- **Wyjście:** 2 kamery RGB (lewe/prawe oko — rozdzielczość i FOV do uzgodnienia z Osobą 1 pod FlyGym Retina),
  IMU (`ImuState`), stan drona. Pozycja świata tylko do nagrody i metryk.
- Komunikacja z procesami Osób 1 i 2 przez ZMQ (osobne środowiska, bez konfliktów wersji MuJoCo/PyTorch).

## Setup (Windows + Anaconda)

```powershell
conda env create -f environment.yml
conda activate neurofly-sim

git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
cd third_party/mujoco_menagerie
git sparse-checkout set skydio_x2
cd ../..

python -m mujoco.viewer --mjcf=third_party/mujoco_menagerie/skydio_x2/scene.xml
```

**Laptop z dwiema kartami (np. Intel UHD + NVIDIA):** Windows domyślnie odpala Pythona na zintegrowanej
karcie — scena z terenem i blokami renderuje się wtedy ~230 ms/klatkę (lag), na GTX 1650 ~10 ms.
Ustawienie „Wysoka wydajność” dla Pythona ze środowiska (jak Ustawienia → System → Ekran → Grafika):

```powershell
$py = (conda run -n neurofly-sim python -c "import sys; print(sys.executable)").Trim()
New-ItemProperty -Path "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences" -Name $py -Value "GpuPreference=2;" -PropertyType String -Force
```

Sprawdzenie: `python -c "import mujoco; from OpenGL import GL; c = mujoco.GLContext(64, 64); c.make_current(); print(GL.glGetString(GL.GL_RENDERER))"`
powinno wypisać kartę NVIDIA.

Paczki numeryczne w `environment.yml` są instalowane przez pip celowo: starsza conda (4.x) dobiera
binarki pod numpy 1.x, a `mujoco` wymaga numpy 2.x.

Sprawdzone: numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, mujoco 3.14.0, model Skydio X2
(4 silniki, 1 kamera), renderowanie offscreen działa.

## Model drona i podgląd

- `sim/assets/x2/` — Skydio X2 z Menagerie (Apache-2.0) bez wypalonych tarcz śmigieł; fizyka bez zmian.
  Generowane przez `python sim/tools/build_x2_assets.py` (potrzebne `third_party/` tylko do regeneracji).
- `sim/propellers.py` — łopaty dorysowywane do sceny, prędkość obrotu ∝ sqrt(ciąg silnika), kierunek
  zgodny ze znakiem momentu reakcji. Działa też z `mujoco.Renderer` (`props.draw(renderer.scene, data)`).
- `sim/control.py` — `RateController`: interfejs zespołu `RateCommand(thrust [N], roll/pitch/yaw_rate [rad/s])`
  → PID prędkości kątowych → mixer (macierz alokacji z geometrii silników) → `data.ctrl`.
  **Bez autostabilizacji** — nic nie poziomuje drona, nie hamuje i nie trzyma wysokości; tego ma się
  nauczyć model. Przy zerowych prędkościach kątowych dron przestaje się obracać, ale zostaje w przechyle.
  `VelocityController` (poziomowanie, hamowanie, trzymanie wysokości) jest tylko do ręcznych testów
  w podglądzie — włączany Altem, nie wchodzi do pętli modelu.
- Sceny: `sim/assets/common.xml` (dron, światło, niebo, zielona kratka) + `scene_hover.xml` (płaska podłoga)
  i `scene_beacon.xml` (domyślna w podglądzie): losowy teren 60 × 60 m (pagórki do +2 m, zagłębienia
  do −1 m) otoczony ścianami 8 m + cel: cienki pomarańczowy prostopadłościan 0.3 × 0.3 × 3 m, przenikalny (bez kolizji)
  — sukces w chwili zetknięcia. `sim/terrain.py`: `load_scene(path)` ładuje scenę z pulą bloków, a
  `randomize(model, data, seed)` losuje teren, bloki i cel (~0.1 s) — start zawsze płaski, cel 15–23 m
  od startu, ≥ 6 m od ścian, na wyrównanym placu (też na wzniesieniu albo w dołku). Scenę beacon zawsze
  ładujemy przez `load_scene` + `randomize` (bez nich nie ma bloków, a teren jest płaski na −1 m).
  Wygląd: brązowe podłoże z siatką 0.5 / 2 / 4 m; bloki z jasnych paneli z siatką, na ścianach bocznych niebieskie
  prostokąty („okna”) z odstępami, góra i spód sama siatka (tekstura cube: up/down = ±y, front/back = ±z);
  tekstury z `sim/tools/build_blueprint_textures.py` (`sim/assets/textures/`: ground, block_side, block_top).
  `sim/blocks.py` (zastąpił drzewa): 60–85 bloków — kostka, filar, ściana, wieża (2–3 piętra), schodki;
  losowy obrót i odcień (biel, błękit, stal, rzadko bursztyn); nie nachodzą na siebie, na start (≥ 3 m)
  ani na plac celu (≥ 2.5 m), nad szczytem zostaje ≥ 1.5 m do granicy planszy. Wszystkie kolidują z dronem.
  Pula bloków jest kompilowana jako obwiednia największego bloku — inaczej MuJoCo (bvh_aabb liczone przy
  kompilacji) gubi kolizje.
  `outside_arena()`: dotyk ściany albo lot ponad 8 m = poza planszą — podgląd resetuje wtedy drona na start
  (ten sam świat).
- `sim/episode.py`: `CrashDetector` — wywrotka = nieudana próba: > 1 s do góry nogami (przechył > 90°,
  w powietrzu albo na ziemi) albo > 1.5 s na ziemi z przechyłem > 60° prawie bez ruchu. Krótki przewrót
  w powietrzu nie kończy próby. Podgląd resetuje wtedy drona (ten sam świat); docelowo koniec epizodu w `WorldEnv`.
- **Dron obrócony o 180° wokół z** względem Menagerie: w oryginalnym X2 nos z kamerą jest po stronie −x, a cały
  projekt (Osoba 1, 2, regulator) przyjmuje +x = przód. Teraz nos jest w +x, W leci nosem do przodu.
  `x2_with_eyes()` Osoby 1 (używane w `example_sim_client.py`, `check_mujoco_eyes.py`, `demo_figure.py`) dokleja
  oczy do nieobróconego X2 z Menagerie: na siatce wypadają na ogonie, ale oczy nie widzą drona, a kierunek +x
  jest ten sam, więc obraz do FlyVis różni się tylko o kilka cm położenia kamery — kalibracja się nie psuje.
  Docelowo te skrypty i `WorldEnv` mają używać jednego modelu: `sim/assets` (oczy na soczewkach nosa).
- Kamery-oczy `eye_left` / `eye_right` w `sim/assets/x2/x2.xml`: orientacja wg specyfikacji Osoby 1
  (`visual_pipeline/drone_eyes.py`: 157°, ±70°, 512 × 450), pozycja na prawdziwych soczewkach kamery w gimbalu
  na nosie (x = 0.158, y = −0.002 / −0.018, z = 0.065). Podgląd pokazuje je w lewym dolnym rogu sceny
  (render `MujocoEyes` Osoby 1 — bez własnego drona w kadrze, jak FlyGym), 20 Hz (~23 ms na odświeżenie).
  `sim/target.py`: `Target.reached()` = dron dotyka prostopadłościanu (obrys + 0.3 m zasięgu łopat; robi się zielony),
  `distance()` do nagrody/metryk. Pozycja celu nie jest wejściem sterowania — dron ma go zobaczyć.
- `python -m sim.viewer` — `Alt` włącza/wyłącza autostabilizację (start: wyłączona). Bez niej tryb acro
  (Windows, klawisze trzymane = prędkość kątowa):
  `W`/`S` nos w dół/górę, `A`/`D` przechył w lewo/prawo, `Q`/`E` obrót, `Shift`/`Ctrl` ciąg ±30% od
  zawisu, `L` kamera przypięta/swobodna, kółko zoom, `Backspace` reset drona, `N` nowy losowy świat,
  `CapsLock` wiatr, `Spacja` pauza; `--seed` odtwarza świat (ziarno jest wypisywane w konsoli).
  Klawisze lotu przechwytuje hook Windows (`sim/keyboard.py`), żeby nie przełączały skrótów podglądu MuJoCo.

## Pętla z BANC: `DroneEnv` (`sim/env.py`)

- `sim/world.py` buduje scenę: X2 z `sim/assets/x2` + oczy Osoby 1 (`drone_eyes.eye_camera_xml`) + cel
  (ciemny słup na ciele mocap, przestawiany w `reset`). Generowane `scene_eyes.xml` / `x2/x2_eyes.xml` są w `.gitignore`.
- `DroneEnv.reset(scenario | bearing_deg)` → `(Observation(left, right, imu), info)`;
  `step(FlightCommand | dict)` → `(obs, reward, terminated, truncated, info)`. Krok = klatka 30 FPS = 5 kroków fizyki
  (timestep 1/150 s, żeby czas klatki zgadzał się z `fps` FlyVis w serwerze). `info` (kąt do celu, wysokość, pozycja)
  jest tylko do metryk/nagrody.
- Oczy renderują bez cieni: mapa cieni reflektora śledzącego drona dawała przy horyzoncie ciemną plamę jeżdżącą
  razem z dronem (fałszywy ruch dla FlyVis).
- `Pilot` (nasze założenie): `mode="angle"` (domyślnie, wariant demo Planu C) — symulator trzyma poziom i hamuje dryf
  (`VelocityController`), BANC daje yaw (±1 → ±1 rad/s) i thrust jako wznoszenie ((thrust − 0,5) · 1 m/s, martwa
  strefa 0,05), roll/pitch z BANC ignorowane (`use_roll` → ruch w bok). `mode="acro"` — znormalizowane prędkości kątowe.
- `DroneEnv.calibration_render(bearing)` = `render` dla `VisionClient.calibrate`.
- Scenariusze `SCENARIOS`: `hover`, `turn_right` / `turn_left` (cel ±60°), `gust` (moment odchylenia + siła boczna
  w t = 2 s). `summarize(log)` → błąd kursu (końcowy, średni z ostatniej s, czas ustalenia ≤ 10°), wysokość, dryf.
- Sprawdzone regulatorem P na prawdziwym kącie (zamiast BANC): cel 60° → 0,3° po 3 s, wysokość ±6 cm;
  podmuch → maks. 16° i powrót. Render obu oczu + fizyka ~3 ms/klatkę.

```bash
python scripts/vision_server.py                       # .venv312 (wzrok + BANC), osobny terminal
python scripts/fly_banc.py --video demo.mp4           # środowisko symulatora; wyniki w data/runs/
python scripts/fly_banc.py --local                    # wszystko w jednym procesie (.venv312)
python scripts/train_decoder.py --plan B              # trening dekodera (.venv312), wagi → data/decoders/
python scripts/vision_server.py --decoder data/decoders/planB.npz
```

## WorldEnv — środowisko treningowe (losowy świat) (Plan A)

- `sim/world_env.py`: `WorldEnv` (Gymnasium). Akcja = `FlightCommand` Osoby 2: `[thrust 0..1 (0.5 = zawis), roll, pitch, yaw −1..1]`;
  `control="acro"` (domyślnie, Plan A: prędkości kątowe, bez autostabilizacji) albo `"angle"` (Plan C, zapasowy:
  symulator trzyma przechył). Obserwacja tylko jak u muszki: `eyes` (2 × 512 × 450 × 3) + `imu` (żyroskop,
  akcelerometr). Prawdziwy stan i cel tylko w `info`. Koniec: `cel` / `wywrotka` / `poza planszą` (terminated),
  `limit czasu` (truncated). Nagroda: postęp w stronę celu − koszt czasu ± 10 na końcu. Opcje: `start_noise`
  (losowy przechył/prędkość na starcie), `wind`, `substeps` (3 × 10 ms = 30 Hz oczu).
  Determinizm: IMU i stan w pełni; obraz oczu z dokładnością do szumu GPU (≤ 2/255 na 0.25 % pikseli).
- `python -m sim.run_env` — `WorldEnv` w oknie. Bez modelu dron sam nie leci: sterujesz z klawiatury
  (W/S, A/D, Shift/Ctrl, Q/E, CapsLock = wiatr, Backspace = od nowa, N = nowy świat) dokładnie przez interfejs
  modelu (akcja `FlightCommand`, obserwacja oczy + IMU, koniec epizodu i nagroda w konsoli). Gdy będzie model:
  `--model pakiet.modul:funkcja` (funkcja(obs) → akcja). Start w idealnym zawisie, `--start-noise` jak przy uczeniu.
  Alt = autostabilizacja jak w `sim.viewer` (do testów; jej wynik też idzie do `WorldEnv` jako zwykła akcja).
- `sim/wind.py`: wiatr przez model płynu MuJoCo (`opt.wind`), domyślnie 8 m/s w losowym kierunku + podmuchy ±3 m/s
  (Ornstein-Uhlenbeck); `--wind-speed` w obu programach. Ze stabilizacją 8 m/s spycha z zawisu o 4.4 m / 10 s,
  lot pod wiatr 3 m/s zwalnia o ~40 % (5 m/s: 1.8 m i ~30 %, 11 m/s: 7.7 m i ~60 %). W podglądzie: CapsLock = wiatr wł./wył., strzałka w prawym dolnym rogu.
- Wydajność (GTX 1650): mapa cieni 2048 zamiast 4096 (`common.xml`) — render obu oczu 20 → 2.6 ms, widok główny
  9 → 5 ms; odbicie terenu wyłączone (wymuszało drugi przebieg z cieniami). W `sim.viewer` podgląd oczu renderuje
  się w osobnym wątku (`EyesWorker`) — główna pętla trzyma ~58 FPS (było ~29 z przeskokami do 70 ms).
  Kamera przypięta za tyłem drona obraca się z jego kursem (wygładzenie w czasie, 0.25 s), kółko = zoom, L = swobodna.
- `sim/terrain.py: upload_terrain()` — po `randomize()` trzeba wysłać teren do GPU każdego `mujoco.Renderer`
  (np. oczu), inaczej kamery widzą poprzedni teren. `WorldEnv` i podgląd robią to same.

## Czujniki drona i metryki

- `sim/sensors.py`: `DroneSensors(model, mode="real" | "ideal")` — realistyczne czujniki: żyroskop (szum 0.01 rad/s +
  dryf biasu), akcelerometr (szum + bias), dalmierz w dół wzdłuż osi drona (szum 1 cm + 1 %, zasięg 0.05–4 m,
  1 % zgubionych odczytów, poza zasięgiem brak), przepływ optyczny przód/bok (szum rośnie z wysokością, tylko z odczytem
  dalmierza), barometr (szum 0.1 m + dryf), prędkość pionowa (estymata z filtra), opóźnienie 20 ms. Ziarno = powtarzalnie.
  `mode="ideal"` = prawdziwe wartości (zgodność z dotychczasowym zachowaniem).
- `WorldEnv(sensors="ideal" | "real")` (domyślnie `ideal`, żeby nie zmieniać po cichu treningu `banc_pilot`):
  `obs["imu"]` z czujników, nowe `obs["sensors"]` = [dalmierz (−1 = brak), przepływ przód, bok, barometr, v_z].
  Pod Plan A trening powinien przejść na `sensors="real"` (zakres Osoby 2: `world_decoder.drone_sensors` liczy dziś
  wysokość i prędkość z prawdziwego stanu).
- `sim/metrics.py`: `EpisodeMetrics` — z PRAWDZIWEGO stanu (nie z czujników): odległość / postęp do celu, błąd kursu,
  wysokość nad terenem, prędkość, przechył (max), droga, wiatr; wyniki epizodów (cel, wywrotki, poza planszą, najlepszy
  czas); `--metrics-csv plik.csv` zapisuje podsumowanie każdego epizodu.
- „GPS” celu (`beacon` w `DroneSensors`, `obs["beacon"]` w `WorldEnv`): kierunek do celu względem nosa
  [rad, + = w lewo] i odległość w poziomie [m]. W trybie `real` jak prawdziwy GPS + kompas: błąd pozycji ~2.5 m
  (pływa, ~15 s), kompas ±3°, 5 Hz, opóźnienie 0.2 s. Zmierzone: błąd kierunku (mediana) 7° powyżej 15 m, 8° przy
  8–15 m, 18° przy 3–8 m, bezużyteczny poniżej 3 m; odległość ±1.9 m. Dron wie mniej więcej, dokąd lecieć — o
  przeszkodach nic nie wie, a ostatnie metry musi wypatrzyć oczami. Do wykorzystania w dekoderze (Osoba 2) zamiast
  prawdziwego kierunku do celu.
- Klawisze w `sim.viewer` i `sim.run_env`: **C** = panel czujników (odczyt | prawda), **M** = panel metryk.
  Przy włączonym C nad dronem strzałki: pomarańczowa = odczyt GPS, zielona = prawdziwy kierunek do celu.
- Bloki (`sim/blocks.py`): większe (kostki do 3 m, filary 3–6 m, ściany do 6 m szerokości, wieże do ~8 m),
  60–85 na świat (średnio ~70). ~15 % filarów/ścian/wież jest **wysokich** (9.5–13 m, ponad granicę planszy 8 m — trzeba
  ominąć); zwykłe zostawiają >= 1.5 m nad szczytem, bloków „prawie do przelecenia” nie ma. ~30 % zwykłych stoi
  **pochylonych** (30–80° do podłoża, obrót całego ciała — kolizje działają). Podstawa zawsze >= 0.3 m w ziemi
  (blok obniżony o najwyżej uniesiony róg podstawy), więc żaden blok nie wisi w powietrzu. Pochyla się w stronę
  wąskiego boku, a >= 50 % objętości każdego bloku jest nad gruntem (próbki 4x4x4 na część względem terenu
  pod każdą próbką, z zapasem 5 %; zmierzone na 2105 blokach: min 52 %, mediana 80 %).
- Opóźnienie silników: `RateController(motor_tau=...)` — siła silnika dochodzi do zadanej z tą stałą czasową
  (filtr 1. rzędu, jak rozpędzające się śmigło). `MOTOR_TAU = 0.04` s; `WorldEnv(motor_tau=0.0)` domyślnie (zgodność),
  `sim.viewer` / `sim.run_env` domyślnie 0.04 (`--motor-tau`; z `--banc` 0). Zmierzone: ciąg 63 % po τ, stabilizacja
  stabilna do 80 ms (co 10 i co 30 ms).
- Testy CI `tests/test_sim_world.py` (32): regulator (zawis, prędkości kątowe, acro bez poziomowania, opóźnienie silników,
  stabilizacja), nos i oczy w +x, świat (powtarzalność, płaski start, cel 15–23 m na placu, granice), bloki (wolny start
  i cel, prześwit, kolizja z dronem), wywrotka, wiatr, czujniki (szum, zasięg, opóźnienie, GPS celu), metryki + CSV,
  `WorldEnv` (Gymnasium, wyniki epizodu, powtarzalność), oczy widzą nowy teren, ślad, łopaty.
- `sim/trail.py`: ślad lotu — turkusowa linia za dronem (punkt co 10 cm, ostatnie ~30 m, starsze odcinki bledną),
  czyszczony przy resecie. **T** = wł./wył. w `sim.viewer` i `sim.run_env`. Tylko wizualizacja (~+3 ms renderu).
  `sim.run_env --sensors real|ideal` (domyślnie `real`, z `--banc` `ideal`).

## Dalej

1. Korytarz, kilka celów, wiatr ciągły.
2. Nagranie demo: oczy + aktywność BANC (explorer) + dron.
