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
karcie — scena z terenem i drzewami renderuje się wtedy ~230 ms/klatkę (lag), na GTX 1650 ~10 ms.
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
  do −1 m) otoczony ścianami 8 m + cel: pomarańczowe pole lądowania 1.2 × 1.2 m z czarnym masztem 2.5 m
  i czerwoną flagą. `sim/terrain.py`: `load_scene(path)` ładuje scenę z pulą drzew, a
  `randomize(model, data, seed)` losuje teren, drzewa i cel (~0.1 s) — start zawsze płaski, cel 15–23 m
  od startu, ≥ 6 m od ścian, na wyrównanym placu (też na wzniesieniu albo w dołku). Scenę beacon zawsze
  ładujemy przez `load_scene` + `randomize` (bez nich nie ma drzew, a teren jest płaski na −1 m).
  `sim/trees.py`: 55–80 drzew — świerk, dąb, brzoza, krzak (wysokość, pień, korona losowane w zakresach
  gatunku); nie nachodzą na siebie, na start (≥ 3 m) ani na plac celu (≥ 2.5 m), nad czubkiem zostaje
  ≥ 1.5 m do granicy planszy. Pień i korona kolidują z dronem. Pula drzew jest kompilowana jako
  obwiednia największego drzewa — inaczej MuJoCo (bvh_aabb liczone przy kompilacji) gubi kolizje.
  `outside_arena()`: dotyk ściany albo lot ponad 8 m = poza planszą — podgląd resetuje wtedy drona na start
  (ten sam świat).
  `sim/target.py`: `Target.reached()` = dron nad polem niżej niż 1.5 m (pole robi się zielone),
  `distance()` do nagrody/metryk. Pozycja celu nie jest wejściem sterowania — dron ma go zobaczyć.
- `python -m sim.viewer` — `Alt` włącza/wyłącza autostabilizację (start: wyłączona). Bez niej tryb acro
  (Windows, klawisze trzymane = prędkość kątowa):
  `S`/`W` nos w dół/górę, `D`/`A` przechył w lewo/prawo, `Q`/`E` obrót, `Shift`/`Ctrl` ciąg ±30% od
  zawisu, `L` kamera przypięta/swobodna, kółko zoom, `Backspace` reset drona, `N` nowy losowy świat,
  `Spacja` pauza; `--seed` odtwarza świat (ziarno jest wypisywane w konsoli).
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

## Dalej

1. Korytarz, kilka celów, wiatr ciągły.
2. Nagranie demo: oczy + aktywność BANC (explorer) + dron.
