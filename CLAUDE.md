# NeuroFly — kontekst dla Claude

Hackathon (12 h): zamknięta pętla **wzrok → connectome Drosophila → sterowanie → symulowany dron → nowy obraz**.
Język zespołu: polski. Źródła planu: tablica Miro „Drosophila Connectome → Drone — Hackathon Plan”
i Google Doc „Drosophila Vision → BANC — plan integracji” (oba u właściciela repo).

| Osoba | Zakres | Stan w repo |
|---|---|---|
| 1 | Kamera RGB → FlyGym Retina → RetinaMapper → FlyVis → neurony BANC | brak kodu; zastępuje ją stub `FakeVision` |
| 2 | BANC/VNC → motoneurony skrzydeł → thrust/roll/pitch/yaw | `banc_control/` — ten kod |
| 3 | Gazebo Sim, quadcopter, kamera, IMU, epizody | brak kodu; zastępuje ją `ToyDrone` |

Szczegółowy plan Osoby 2 z checklistą: `docs/osoba2-plan.md`.

## Zasada nr 1: tylko oficjalny BANC

Użytkownik wprost zażądał trzymania się oficjalnego BANC. Nie wolno wracać do syntetycznych/wymyślonych
grafów ani zgadywać nazw typów neuronów. Wcześniejsza wersja miała syntetyczny stub i schematyczną
wizualizację — została odrzucona („pseudo sieć, nie przypomina BANC”) i usunięta.

- Wersja: **BANC v888** (materializacja z publikacji Bates et al. 2026).
- Źródło: publiczny bucket `https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/`
  (bez logowania), mirror z DOI: https://doi.org/10.7910/DVN/7WTH1N. Dokumentacja schematów: `documentation/*.md` w buckecie.
- `python scripts/download_banc.py` → `data/banc_888/` (`banc_888_meta.feather`, `banc_888_edgelist_simple_v2.feather`).
  `data/` jest w `.gitignore` (~360 MB; szkielety SWC kolejne ~540 MB).
- Root ID zmieniają się między wersjami. Wszyscy muszą używać v888; meta ma `root_626`, `root_850` do mapowania.

## Architektura `banc_control/`

- `contracts.py` — kontrakty: wejście od Osoby 1 `[{"banc_root_id", "cell_type", "activity"}]` (format z Google Doc),
  `ImuState` od Osoby 3, `FlightCommand` (thrust 0..1, roll/pitch/yaw −1..1) do Osoby 3.
- `connectome.py` — `Connectome.from_banc()` / `from_banc_tables()`. Odrzuca glia/trachea/not_a_neuron,
  krawędzie `count >= 5`. `W[post, pre] = znak(NT pre) * count / Σ|count| wejść post` (stabilność przy gain < 1).
  `flight_groups()` przypisuje grupy WYŁĄCZNIE z oficjalnych kolumn meta:
  `visual_projection`; DN z `super_cluster` „flight power” / „flight steering 1/2”; aferenty z `body_part_sensory` ~ haltere;
  MN z `cell_function` = wing_power (DLM/DVM) / wing_steering (b1, i1, iii3…) / wing_tension; każda grupa _L/_R.
- `dynamics.py` — model szybkości: `r += dt/tau * (-r + tanh(relu(gain*W r + I)))`, macierz rzadka scipy.
- `readout.py` — średnia aktywność 6 grup MN → komendy. Dekodery = Plany z Miro: C `ManualDecoder`, B `LinearDecoder` (LMS),
  A `AdaptiveDecoder` (node perturbation z nagrodą).
- `controller.py` — `BancController.step(visual, imu)`; kalibracje: `calibrate_rest(visual=scena neutralna)`,
  `calibrate_scale(bodźce)`, `calibrate_haltere_sign()`.
- `stubs.py` — `FakeVision` (pobudza `visual_projection` L/R wg kierunku beacona), `ToyDrone` (1-osiowa fizyka).
- `scripts/export_viz_data.py`, `scripts/export_anatomy.py` — dane do wizualizacji (`data/viz/`).
- Testy: `pytest` (mały graf w schemacie BANC + test pełnego v888, pomijany gdy brak danych).

## Co jest z BANC, a co jest naszym założeniem

Z BANC: neurony, połączenia, liczby synaps, przewidywany neuroprzekaźnik, wszystkie grupy, pozycje som, szkielety.

Nasze założenia (zawsze mów o nich wprost, nie przedstawiaj jako wyników BANC):
- model dynamiki i parametry (tau 20 ms, dt 5 ms, gain 0.9, 4 podkroki/klatkę),
- znaki NT: ACh +, GABA/glutaminian/histamina −, modulatory i brak predykcji +,
- wejście wzroku jako lewa/prawa strona `visual_projection` (do czasu danych od Osoby 1),
- kodowanie gyro → aferenty halter L/R (znak dobierany empirycznie przez `calibrate_haltere_sign`),
- przełożenie MN → dron: thrust ← średnia wing_power, roll ← wing_power L−R, yaw ← wing_steering L−R,
  pitch ← brak (trim w Planie C, uczony w A/B). Wzmocnienia Planu C: (0.2, 1, 1, 1).

## Wyniki na pełnym BANC v888 (175 401 neuronów, 1 613 322 krawędzi)

- Sygnał dochodzi od wzroku do MN skrzydeł, ale słaby (aktywność MN ~1e-3) → konieczne `calibrate_scale`.
- Beacon z boku → roll i yaw w jego stronę, stopniowane z kątem (np. −45°: roll −0.59, yaw −0.54; +45°: roll +0.61, yaw +1.0).
- Bez kalibracji baseline na scenie neutralnej thrust nasycał się do 1.0 (każde światło = pełny ciąg).
- Haltery: przy domyślnym kodowaniu pętla destabilizowała → znak odwrócony kalibracją (`haltere_sign = -1`).
  Po kalibracji roll jest korygowany, ale jest silne sprzężenie na yaw (±1) i spadek thrust.
  W symulacji podmuchu: szczyt przechyłu 41° → 30°, ale wolniejszy powrót — stabilizacja jeszcze niedobra.
- Wydajność: ~18 ms na podkrok na CPU (~70 ms/klatkę) — za wolno na 50 Hz.

## Następne kroki (Osoba 2)

1. Uzgodnić z Osobą 1: aktywność przypisana do root ID **v888**.
2. Przyspieszyć `RateDynamics` (torch.sparse na GPU — torch z CUDA jest zainstalowany; ewentualnie podgraf) do ≤ 5 ms/podkrok.
3. Uzgodnić z Osobą 3: `FlightCommand.to_json()` ↔ mixer, IMU, częstotliwość klatek.
4. Dostroić sprzężenie halter → yaw.
5. Plan A/B dopiero gdy pętla z Gazebo działa; Plan C musi działać zawsze jako fallback.

## Wizualizacja

Artifact (prywatny, właściciel musi udostępnić): https://claude.ai/artifact/Chj3SSu9eLrasT7RxNyRfa —
widok 3D: somy wszystkich neuronów z `position`, 863 szkielety SWC neuronów lotu, aktywność z symulacji
dla beacona lewo / prosto / prawo, tabela MN skrzydeł. Dane: `scripts/export_anatomy.py`, `scripts/export_viz_data.py`.

## Uwagi do repo

- `docs/prezentacja/` (NeuroFly.pptx + build.js) powstało w innej sesji; nie było weryfikowane względem wyników z v888.

- `neurofly/` w katalogu głównym to pusty, zagnieżdżony klon tego samego remote — nie commitować, do usunięcia przez właściciela.
- Uruchamianie: `pip install -e .[dev]`, `python scripts/download_banc.py`, `pytest`.
