# Osoba 2 — BANC → Sterowanie: plan na start

Źródła: tablica Miro „Drosophila Connectome → Drone — Hackathon Plan” i dokument
„Drosophila Vision → BANC — plan integracji” (Google Docs).

## Dane: oficjalny BANC v888

Pracujemy wyłącznie na oficjalnym wydaniu (materializacja v888, wersja z publikacji):
[Harvard Dataverse doi:10.7910/DVN/7WTH1N](https://doi.org/10.7910/DVN/7WTH1N),
publiczny bucket `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/`.

```bash
python scripts/download_banc.py   # banc_888_meta.feather + banc_888_edgelist_simple_v2.feather → data/banc_888/
```

Grupy funkcjonalne biorę z oficjalnych kolumn `banc_888_meta.feather` (`banc_control/connectome.py: flight_groups`):

| grupa | kryterium | L / R |
|---|---|---|
| visual | `super_class == visual_projection` | 3474 / 3762 |
| dn_flight_power | descending, `super_cluster == "flight power"` | 116 / 119 |
| dn_flight_steering | descending, `super_cluster` = flight steering 1/2 | 70 / 70 |
| haltere_aff | `flow == afferent`, `body_part_sensory` zawiera haltere | 216 / 212 |
| wing_power | motor, `cell_function == wing_power` (DLM, DVM) | 12 / 12 |
| wing_steering | motor, `cell_function == wing_steering` (b1, b2, i1, iii1, …) | 12 / 12 |
| wing_tension | motor, `cell_function == wing_tension` (tp, ps) | 6 / 6 |

Graf po odrzuceniu glejów/tchawek: 175 401 neuronów, 1 534 828 krawędzi (count ≥ 5, bez autapsów).

**Nasze założenia (nie wynik BANC):** model szybkości odpalania, znaki NT (ACh +, GABA/Glu/histamina −,
reszta +), wejście wzroku jako lewa/prawa strona `visual_projection`, kodowanie gyro → aferenty halter
L/R, przełożenie mięśni skrzydeł na thrust/roll/yaw.

## Pierwsze wyniki na pełnym BANC (`scripts/export_viz_data.py`)

- Sygnał dochodzi od wzroku do MN skrzydeł, ale jest słaby (aktywność MN ~1e-3) → `calibrate_scale()` w Planie C.
- Beacon z boku → roll i yaw w stronę beacona, stopniowane z kątem; thrust spada umiarkowanie (0.49 → ~0.36).
- Haltery: przy znaku dobranym przez `calibrate_haltere_sign()` roll jest korygowany, ale jest silne
  sprzężenie na yaw (±1) i spadek thrust. Do strojenia / uczenia.
- Czas: GPU (torch CSR) 0.3 ms/podkrok, ~2.6 ms/klatkę; CPU ~24 ms/klatkę. Pętla 50 Hz mieści się z zapasem.

## Zakres

**START:** aktywność przypisana do BANC root IDs od Osoby 1 + odczyt IMU od Osoby 3.
**STOP:** komenda lotu `thrust / roll / pitch / yaw` przekazana do mixera Osoby 3.

Poza zakresem: obraz → FlyVis → BANC (Osoba 1), fizyka i Gazebo (Osoba 3).

## Przepływ

```
Osoba 1 JSON ──► wejście na neurony wzrokowe BANC
IMU (Osoba 3) ──► aferenty halter w VNC
                     │
          RateDynamics (W rzadka, N ≈ 10⁵)
   wzrok → mózg centralny → DN → szyja → VNC → MN skrzydeł
                     │
          motor_features (6 grup MN)
                     │
   Dekoder: Plan A (Adaptive) / B (Linear, LMS) / C (Manual)
                     │
              FlightCommand ──► Osoba 3
```

Analogia skrzydło → quadcopter (`banc_control/readout.py`):

| komenda | źródło w VNC |
|---|---|
| thrust | średnia aktywność MN `wing_power` L+R |
| roll | asymetria MN `wing_power` L − R |
| pitch | brak ręcznego mapowania: trim w Planie C, uczony w Planach A/B |
| yaw | asymetria MN `wing_steering` L − R |

## Harmonogram (12 h, zgodny z checkpointami z Miro)

**0–4 h — moduły równolegle**
- [x] Pobrać oficjalny BANC v888 (`scripts/download_banc.py`), `Connectome.from_banc()`.
- [x] Grupy z oficjalnych adnotacji (tabela wyżej).
- [x] Zmierzyć czas kroku.
- [x] Plan C: `calibrate_rest(visual=scena neutralna)`, `calibrate_scale()`, `calibrate_haltere_sign()`.
- [ ] Ustalić z Osobą 1: mapowanie FlyVis → **root ID v888** (stare ID przez kolumny `root_626` / `root_850`).
- [x] Przyspieszyć krok: GPU, 0.3 ms / podkrok.
- [ ] Dostroić sprzężenie halter → yaw.

**4–6 h — pełna pętla ze stubami**
- [ ] Zamienić `FakeVision` na prawdziwy JSON od Osoby 1.
- [ ] Zamienić `ToyDrone` na Gazebo Osoby 3 (`FlightCommand.to_json()`).
- [ ] Uzgodnić częstotliwość klatek / IMU i liczbę `substeps` na klatkę.

**6–9 h — Plan A**
- [ ] `AdaptiveDecoder`: nagroda = stabilność zawisu + kierunek na beacon. Pozycja świata tylko do nagrody i metryk, nie jako wejście sterujące.
- [ ] Plastyczność tylko w dekoderze, BANC/VNC bez zmian.
- [ ] Logować `motor_features` i `unmatched_ids`.

**~9 h → Plan B**, **~10 h → Plan C**, jeśli Plan A nie jest stabilny.

**10–12 h — demo:** nagranie zawisu i lotu do beacona, logi, slajd z wizualizacją BANC.

## Ryzyka

1. Niezgodna wersja root IDs między Osobą 1 a 2 → rośnie `unmatched_ids`.
2. Słaby sygnał na MN i sprzężenie halter → yaw (patrz wyniki wyżej).
3. Czas kroku na pełnym grafie.
4. Mapowanie skrzydło → quadcopter jest umowne. Plan C musi działać, zanim zaczniemy uczyć.

## Uruchomienie

```bash
pip install -e .[dev]
pytest
```

```python
from banc_control import BancController, Connectome
from banc_control.stubs import FakeVision, ToyDrone

c = Connectome.from_banc()            # oficjalny BANC v888 z data/banc_888/
ctrl = BancController(c)
vision, drone, imu = FakeVision(c), ToyDrone(), None
ctrl.calibrate_rest(30, visual=vision(0.0))
ctrl.calibrate_scale([vision(-1.0), vision(1.0)])
ctrl.calibrate_haltere_sign(vision(0.0))
for _ in range(500):
    cmd = ctrl.step(vision(bearing=0.3), imu)
    imu = drone.step(cmd)
```
