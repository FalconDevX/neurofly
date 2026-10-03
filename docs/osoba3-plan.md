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
- `sim/episode.py`: `CrashDetector` — wywrotka = nieudana próba: > 1 s do góry nogami (przechył > 90°,
  w powietrzu albo na ziemi) albo > 1.5 s na ziemi z przechyłem > 60° prawie bez ruchu. Krótki przewrót
  w powietrzu nie kończy próby. Podgląd resetuje wtedy drona (ten sam świat); docelowo koniec epizodu w `DroneEnv`.
- **Dron obrócony o 180° wokół z** względem Menagerie: w oryginalnym X2 nos z kamerą jest po stronie −x, a cały
  projekt (Osoba 1, 2, regulator) przyjmuje +x = przód. Teraz nos jest w +x, W leci nosem do przodu.
  `x2_with_eyes()` Osoby 1 (używane w `example_sim_client.py`, `check_mujoco_eyes.py`, `demo_figure.py`) dokleja
  oczy do nieobróconego X2 z Menagerie: na siatce wypadają na ogonie, ale oczy nie widzą drona, a kierunek +x
  jest ten sam, więc obraz do FlyVis różni się tylko o kilka cm położenia kamery — kalibracja się nie psuje.
  Docelowo te skrypty i `DroneEnv` mają używać jednego modelu: `sim/assets` (oczy na soczewkach nosa).
- Kamery-oczy `eye_left` / `eye_right` w `sim/assets/x2/x2.xml`: orientacja wg specyfikacji Osoby 1
  (`visual_pipeline/drone_eyes.py`: 157°, ±70°, 512 × 450), pozycja na prawdziwych soczewkach kamery w gimbalu
  na nosie (x = 0.158, y = −0.002 / −0.018, z = 0.065). Podgląd pokazuje je w lewym dolnym rogu sceny
  (render `MujocoEyes` Osoby 1 — bez własnego drona w kadrze, jak FlyGym), 20 Hz (~23 ms na odświeżenie).
  `sim/target.py`: `Target.reached()` = dron nad polem niżej niż 1.5 m (pole robi się zielone),
  `distance()` do nagrody/metryk. Pozycja celu nie jest wejściem sterowania — dron ma go zobaczyć.
- `python -m sim.viewer` — `Alt` włącza/wyłącza autostabilizację (start: wyłączona). Bez niej tryb acro
  (Windows, klawisze trzymane = prędkość kątowa):
  `W`/`S` nos w dół/górę, `A`/`D` przechył w lewo/prawo, `Q`/`E` obrót, `Shift`/`Ctrl` ciąg ±30% od
  zawisu, `L` kamera przypięta/swobodna, kółko zoom, `Backspace` reset drona, `N` nowy losowy świat,
  `Spacja` pauza; `--seed` odtwarza świat (ziarno jest wypisywane w konsoli).
  Klawisze lotu przechwytuje hook Windows (`sim/keyboard.py`), żeby nie przełączały skrótów podglądu MuJoCo.

## Dalej

1. Dwie kamery-oczy na modelu X2.
2. Regulator prędkości kątowych + mixer.
3. `DroneEnv` (`reset`/`step`), światy: zawis → beacon → korytarz → wiatr → podmuchy.
4. Most ZMQ do `banc_control` (w miejsce `ToyDrone`).
