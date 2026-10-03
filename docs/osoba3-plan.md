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
  Nad nim `VelocityController` do ręcznego latania (prędkość zadana → przechylenie → prędkości kątowe,
  trzymanie wysokości). Używa prawdziwego stanu z symulatora — tylko do podglądu, nie jako wejście sieci.
- `python -m sim.viewer` — ręczne latanie (Windows, klawisze trzymane): `W`/`S` przód/tył,
  `A`/`D` lewo/prawo, `Shift`/`Ctrl` wznoszenie/opadanie, `Q`/`E` powolny obrót, `Backspace` reset,
  `Spacja` pauza. Bez klawiszy dron trzyma pozycję i wysokość. Litery są też skrótami flag
  wbudowanego podglądu MuJoCo (W wireframe, S cienie, ...), więc viewer przywraca flagi w każdej klatce.

## Dalej

1. Dwie kamery-oczy na modelu X2.
2. Regulator prędkości kątowych + mixer.
3. `DroneEnv` (`reset`/`step`), światy: zawis → beacon → korytarz → wiatr → podmuchy.
4. Most ZMQ do `banc_control` (w miejsce `ToyDrone`).
