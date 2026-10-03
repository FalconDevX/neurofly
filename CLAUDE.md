# NeuroFly — kontekst dla Claude

Hackathon (12 h): zamknięta pętla **wzrok → connectome Drosophila → sterowanie → symulowany dron → nowy obraz**.
Język zespołu: polski. Źródła planu: tablica Miro „Drosophila Connectome → Drone — Hackathon Plan”
i Google Doc „Drosophila Vision → BANC — plan integracji” (oba u właściciela repo).

| Osoba | Zakres | Stan w repo |
|---|---|---|
| 1 | Kamera RGB → FlyGym Retina → RetinaMapper → FlyVis → neurony BANC | `visual_pipeline/` gotowe (PR #1, #6, #7, #10): mapa FlyVis→v888, oczy MuJoCo `drone_eyes.py`, serwer ZMQ `scripts/vision_server.py` |
| 2 | BANC/VNC → motoneurony skrzydeł → thrust/roll/pitch/yaw | `banc_control/` gotowe (Plan C), GPU; kalibracja na scenach z symulatora przez ZMQ |
| 3 | Symulator drona w MuJoCo (zamiast Gazebo — WSL2 nie działał), kamera, IMU, epizody | tylko `environment.yml` + model X2 z menagerie; **brak `DroneEnv`, regulatora, mixera** — wąskie gardło |

Plany osób: `docs/osoba1-plan.md`, `docs/osoba2-plan.md`, `docs/osoba3-plan.md`. Plan domknięcia całości: sekcja „Plan domknięcia” niżej.
Środowisko wzroku (flyvis, flygym, Python 3.12) nie jest zainstalowane na każdej maszynie — testy wzroku są wtedy pomijane.

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
  krawędzie `count >= 5` bez autapsów (część autapsów jest w pliku v888 mimo dokumentacji; paper ich nie liczy). `W[post, pre] = znak(NT pre) * count / Σ|count| wejść post` (stabilność przy gain < 1).
  `flight_groups()` przypisuje grupy WYŁĄCZNIE z oficjalnych kolumn meta:
  `visual_projection`; DN z `super_cluster` „flight power” / „flight steering 1/2”; aferenty z `body_part_sensory` ~ haltere;
  MN z `cell_function` = wing_power (DLM/DVM) / wing_steering (b1, i1, iii3…) / wing_tension; każda grupa _L/_R.
- `dynamics.py` — model szybkości: `r += dt/tau * (-r + tanh(relu(gain*W r + I)))`. Backend `device="cuda"` (torch, CSR float32,
  domyślny gdy jest CUDA) lub `"cpu"` (scipy, float64); zgodność GPU/CPU ~1e-7, test `test_gpu_matches_cpu`.
  `rates_at(idx)` kopiuje z GPU tylko wybrane neurony; `BancController.motor_features()` czyta tylko MN.
- `readout.py` — średnia aktywność 6 grup MN → komendy. Dekodery = Plany z Miro: C `ManualDecoder`, B `LinearDecoder` (LMS),
  A `AdaptiveDecoder` (node perturbation z nagrodą).
- `controller.py` — `BancController.step(visual, imu)`; kalibracje w kolejności: `calibrate_rest(visual=scena neutralna)`,
  `calibrate_scale(bodźce)`, `calibrate_yaw_sign(turn_right, turn_left)` (znak osi yaw z odruchu optomotorycznego
  na sekwencjach obrotu; odwraca `decoder.M[3]`, `yaw_axis_sign`), `calibrate_haltere_sign()` (osobne znaki
  `haltere_sign` dla roll_rate i `haltere_yaw_sign` dla yaw_rate). Napęd halter = `haltere_roll_weight`·roll_rate +
  `haltere_yaw_weight`·yaw_rate, domyślnie **0 i 0.5** (roll nie idzie przez haltery, patrz wyniki).
- `stubs.py` — `FakeVision` (pobudza `visual_projection` L/R wg kierunku beacona), `ToyDrone` (1-osiowa fizyka).
- `scripts/export_viz_data.py`, `scripts/export_anatomy.py` — dane do wizualizacji (`data/viz/`).

Integracja (w `visual_pipeline/`, ale wspólna z Osobą 2 i 3):
- `zmq_protocol.py` — protokół symulator ↔ serwer (REQ/REP, `[nagłówek JSON, klatka L, klatka P]`), lekki (numpy + pyzmq).
  Nagłówek `calib`: sceny `neutral` / `left` / `right` (cel 0°, ∓60°), `turn_left` / `turn_right` (sekwencje obrotu),
  potem `finish`. `VisionClient.calibrate(render)` robi całość; Osoba 3 daje tylko `render(bearing)` → (L, P).
- `server.py` — `ControlServer(bridge, ctrl).handle(header, L, P)`: logika serwera bez gniazda (testowalna z fałszywym mostem),
  `calibrate_controller(ctrl, scenes)`. `scripts/vision_server.py` = gniazdo + kalibracja syntetyczna na starcie.
- `scripts/example_sim_client.py` — X2 z oczami w MuJoCo: kalibracja na scenach MuJoCo, potem pętla (bez mixera).
- Testy: `pytest` (mały graf w schemacie BANC + test pełnego v888, pomijany gdy brak danych).

## Co jest z BANC, a co jest naszym założeniem

Z BANC: neurony, połączenia, liczby synaps, przewidywany neuroprzekaźnik, wszystkie grupy, pozycje som, szkielety.

Nasze założenia (zawsze mów o nich wprost, nie przedstawiaj jako wyników BANC):
- model dynamiki i parametry (tau 20 ms, dt 5 ms, gain 0.9, 4 podkroki/klatkę),
- znaki NT: ACh +, GABA/glutaminian/histamina −, modulatory i brak predykcji +,
- wejście wzroku jako lewa/prawa strona `visual_projection` (do czasu danych od Osoby 1),
- kodowanie gyro → aferenty halter L/R (znaki dobierane empirycznie przez `calibrate_haltere_sign`; wagi roll 0 / yaw 0.5),
- znak osi yaw dekodera dobierany z odruchu optomotorycznego (`calibrate_yaw_sign`), nie z anatomii,
- przełożenie MN → dron: thrust ← średnia wing_power, roll ← wing_power L−R, yaw ← wing_steering L−R,
  pitch ← brak (trim w Planie C, uczony w A/B). Wzmocnienia Planu C: (0.2, 1, 1, 1).

## Wyniki na pełnym BANC v888 (175 401 neuronów, 1 534 828 krawędzi w grafie)

- Sygnał dochodzi od wzroku do MN skrzydeł, ale słaby (aktywność MN ~1e-3) → konieczne `calibrate_scale`.
- Beacon z boku → roll i yaw w jego stronę, stopniowane z kątem (np. −45°: roll −0.79, yaw −0.73; +45°: roll +0.87, yaw +1.0; thrust 0.49 → ~0.38).
- Bez kalibracji baseline na scenie neutralnej thrust nasycał się do 1.0 (każde światło = pełny ciąg).
- Haltery: pobudzenie aferentów halter L/R daje w v888 głównie yaw (±1), roll słabo (−0.14), do tego spadek thrust.
  Jeden kanał L/R nie stabilizuje więc dwóch osi: z roll_rate w napędzie (nawet z osobnymi znakami) `ToyDrone`
  po kopnięciu 2 rad/s rozkręca się do 6–11 rad/s. Stąd domyślnie haltery tylko z yaw_rate, a poziom (roll/pitch)
  ma trzymać regulator symulatora.
- Haltery tylko yaw (`haltere_yaw_weight` 0.5, `FakeVision`, `ToyDrone`, roll = 0): kopnięcie yaw 2 rad/s → −0.03 rad/s
  po 5 s; cel 29° w prawo → +1° po 5 s, thrust 0.49. Bez halter oscyluje (+26…+33° po 5 s). Waga 1.0/2.0: gorzej
  (resztkowy błąd 10–20°, thrust 0.45/0.35). Do sprawdzenia z prawdziwym FlyVis i w MuJoCo.
- Wydajność (RTX 4060 Laptop): GPU 0.3 ms/podkrok, ~2.6 ms/klatkę (4 podkroki + wejście); CPU ~24 ms/klatkę.
  Wejście wzrokowe jest zwektoryzowane (`Connectome.indices_of`), ~1.4 ms dla 7k rekordów.

## Plan domknięcia (ustalony 2026-10-03, idziemy według niego)

Zrobione wcześniej: ID v888 uzgodnione z Osobą 1 (`unmatched_ids = 0`), dynamika na GPU, oczy MuJoCo i most ZMQ (Osoba 1).

**Etap 1 — pętla z MuJoCo działa w ogóle** (ścieżka krytyczna: Osoba 3)
- [ ] O3: `DroneEnv` (`reset`/`step`) na X2 z oczami z `drone_eyes.py`, regulator prędkości kątowych + mixer
  `[thrust, roll, pitch, yaw]` → 4 silniki, `pyzmq` w `environment.yml`.
- [ ] O3: klient ZMQ w `DroneEnv.step()` na wzór `example_sim_client.py`, z `VisionClient.calibrate(render)` po starcie.
- [x] O1+O2: kalibracja na scenach z symulatora (`calib` w protokole) — naprawia thrust = 0 na scenie MuJoCo.
  Zrobione na gałęzi `osoba2-sim-calibration`, testowane fałszywym mostem; **niesprawdzone z prawdziwym FlyVis + MuJoCo**.

**Etap 2 — zachowanie**
- [x] O2: znak yaw z obrotu (`calibrate_yaw_sign`), wpięty w kalibrację serwera (syntetyczną i z symulatora).
- [x] O2: haltery: osobne znaki roll/yaw, domyślnie tylko yaw_rate.
- [ ] O2+O3: **wariant demo (Plan C, musi działać zawsze):** symulator trzyma poziom (angle mode), BANC daje yaw + thrust
  (roll z BANC ignorowany, jak `run_closed_loop.py --yaw-only`).
- [ ] O1/O2: potwierdzić z FlyVis: `scripts/check_optomotor.py` po kalibracji ma „hamuje obrót”, `run_closed_loop.py` bez ciągłego obrotu.

**Etap 3 — demo i prezentacja**
- [ ] O3: scenariusze zawis → skręt do celu z boku → podmuch; zapis wideo i metryk (błąd kursu w czasie).
- [ ] Wszyscy: nagranie demo — oczy, aktywność BANC (explorer), dron obok siebie.
- [ ] O2: `docs/prezentacja/` na liczby z v888, z rozdziałem „z BANC” / „nasze założenia”.
- [ ] Opcjonalnie: Plan A/B (`LinearDecoder`/`AdaptiveDecoder`) dopiero gdy Etap 2 działa w MuJoCo.

Do uzgodnienia z Osobą 3: angle mode czy tylko acro (czy roll z BANC w ogóle idzie do drona); 30 FPS kamer vs krok
fizyki; jeden serwer `--mode command` (domyślnie) czy osobny proces Osoby 2 (`--mode activity`).

## Wizualizacja

Artifact (prywatny, właściciel musi udostępnić): https://claude.ai/artifact/Chj3SSu9eLrasT7RxNyRfa —
ciemny dashboard (styl eksploratora connectomu, na życzenie użytkownika): 3D somy wszystkich neuronów z `position`
kolorowane `super_class`, 863 szkielety SWC neuronów lotu, tryby Eksploruj / Obwód lotu / Aktywność, zakres
cały / mózg / VNC, wykres klas i regionów, inspektor neuronu z prawdziwymi partnerami (top 8 z edgelisty),
wyszukiwarka, komendy drona. Wszystkie liczby prawdziwe — nie wstawiać danych z mockupów. Dane: `scripts/export_anatomy.py`, `scripts/export_viz_data.py`.

Eksplorator lokalnie: `python scripts/explorer.py` (http://localhost:8000) — serwer stdlib, strona `explorer/`
(HTML + JS, three.js z CDN, bez npm/builda), dane prosto z `data/viz/`. Zastąpił wersję Next.js (gałąź `osoba2-nextjs-explorer`).
Model na żywo (domyślnie; `--no-live` wyłącza): serwer trzyma `BancController` na pełnym v888 na CUDA (RTX 4060: ~6 ms/krok
z kopiowaniem), kalibracja jak w `export_viz_data.py`, wejście FakeVision + yaw_rate z suwaków strony. `/api/frame` zwraca
poziomy wszystkich som (uint8, kwantyzacja na GPU: `RateDynamics.levels_at`), aktywność neuronów lotu i komendy.
Komendy na żywo zgodne z wynikami wyżej (+45°: roll 0.85, yaw 1.0; −45°: −0.78 / −0.73).

## Uwagi do repo

- `docs/prezentacja/` (NeuroFly.pptx + build.js) powstało w innej sesji; nie było weryfikowane względem wyników z v888.

- `neurofly/` w katalogu głównym to pusty, zagnieżdżony klon tego samego remote — nie commitować, do usunięcia przez właściciela.
- Uruchamianie: `pip install -e .[dev]`, `python scripts/download_banc.py`, `pytest`.
