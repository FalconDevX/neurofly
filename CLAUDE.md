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
  A `AdaptiveDecoder` (node perturbation z nagrodą). `BancController(readout="dn")`: za 6 średnimi MN idą pojedyncze
  neurony DN lotu (375, `DN_GROUPS`), kolumny DN w macierzy ręcznej = 0, wagi tylko z treningu; `ensure_features`,
  `load_weights` sprawdza rozmiar. `fit_step(apply=False)` + `apply_pending()` = jeden krok LMS po epizodzie.
- `controller.py` — `BancController.step(visual, imu)`; kalibracje w kolejności: `calibrate_rest(visual=scena neutralna)`,
  `calibrate_scale(bodźce)`, `calibrate_yaw_sign(turn_right, turn_left)` (znak osi yaw z odruchu optomotorycznego
  na sekwencjach obrotu; odwraca `decoder.M[3]`, `yaw_axis_sign`), `calibrate_haltere_sign()` (osobne znaki
  `haltere_sign` dla roll_rate i `haltere_yaw_sign` dla yaw_rate), `calibrate_haltere_gain()` (`haltere_gain` tak,
  żeby 1 rad/s dawało surowy yaw 0.5). Napęd halter = `haltere_roll_weight`·roll_rate +
  `haltere_yaw_weight`·yaw_rate, domyślnie **0 i 0.5** (roll nie idzie przez haltery, patrz wyniki).
  `warm_start(visual)` po resecie (serwer robi to przy `reset`), inaczej pierwsze klatki dawały thrust 0 / yaw ±1.
  Dekoder: `save(path)` / `load_weights(path)` (po kalibracji), LMS Planu B jest znormalizowany (NLMS).
- `stubs.py` — `FakeVision` (pobudza `visual_projection` L/R wg kierunku beacona), `ToyDrone` (1-osiowa fizyka).
- `scripts/export_viz_data.py`, `scripts/export_anatomy.py` — dane do wizualizacji (`data/viz/`).
- `scripts/check_side_decoding.py` — czy z aktywności BANC da się odczytać stronę celu (statyczne sceny `DroneEnv`).
- `scripts/train_decoder.py` (`--readout dn` domyślnie): start wiersza yaw z regresji na statycznych scenach (`sweep_fit`),
  potem lot: wagi stałe w epizodzie, krok po epizodzie, cele w parach ±b, ewaluacja co `--eval-every`.
  `scripts/train_distributed.py` — to samo na kilku GPU w LAN (master + workerzy ZMQ; worker sprawdza CUDA).
  Master czeka na `--workers` (domyślnie 2) zgłoszeń `ping` przez `--wait-join` s i na ich gotowość przez `--wait-ready` s;
  jeśli slave nie odpowiada — przerwanie (workerzy dostają `stop`, nic nie zapisane, kod 1).
- `sim/brain_panel.py` — `python -m sim.viewer --brain [--brain-decoder …]` (.venv312): przy locie WASD prawy panel
  z BANC na żywo (somy z przodu kolorowane zmianą aktywności, grupy lotu L/P, komenda dekodera); sieć nie steruje.
  Rysowanie: `BrainView`, też w wideo: `fly_banc.py --local --brain --video …` (dron + oczy + panel BANC).

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
- odczyt yaw z pojedynczych DN lotu (`readout="dn"`) i jego wagi z treningu z nauczycielem (Plan B) — liniowy dekoder
  jest nasz, BANC tylko dostarcza aktywność DN.

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

## Wyniki w MuJoCo (prawdziwy FlyVis + v888, `fly_banc.py --local`, 2026-10-03)

- Kalibracja na scenach MuJoCo: wzrok zmienia aktywność MN tylko o ~1% spoczynku (`scale` ~1e-7…1e-6 przy
  `baseline` ~1e-5). Przy stałym `haltere_gain` 0.5 haltery dawały surowy yaw ~6000 na 1 rad/s → yaw ±1 co klatkę,
  thrust 0. Po `calibrate_haltere_gain` (gain ~7e-5) pętla jest spokojna.
- `--thrust hold`: zawis błąd < 1°, wysokość stała. Cel +60° → 62°, −60° → 61° po 8 s (brak skrętu do celu;
  w pętli otwartej cel z lewej yaw −0.39, z prawej −0.06). Podmuch: haltery tłumią obrót, kurs nie wraca (24°).
- `--thrust banc`: thrust 0.3–0.46 w locie, dron dotyka ziemi po ~4.5 s.

### Diagnoza oczu i odczytu (2026-10-03, oczy na nosie drona — commit Osoby 3)
- Kamery i FlyVis są w porządku: cel z lewej widzi lewe oko, z prawej prawe, zmiana wejścia głównie po stronie celu.
- **Ograniczenie danych v888:** lewy płat wzrokowy ma dużo mniej neuronów z `cell_type` (np. Tm1 L 2 / P 807,
  T4a 276 / 805), choć neuronów jest podobnie (optic_lobe_intrinsic 29 224 L / 36 405 P). Mapa FlyVis→BANC Osoby 1
  idzie po typach, więc lewe oko zasila 5 535 neuronów, prawe 16 927. Wyrównanie wejścia (×3) nie zmieniło wyniku.
- **Średnie 6 grup MN nie odróżniają strony celu** (L−R ±0.5%, ten sam znak dla celu z obu stron). Strona jest w
  pojedynczych neuronach: `check_side_decoding.py`, regresja z walidacją „bez jednej odległości”, trafność strony:
  6 średnich MN 68%, MN pojedynczo 84%, **DN lotu pojedynczo 99%** (korelacja kąta 0.88), VPN 94% (gain 1).
  `visual_gain` 30/100 poprawia 6 średnich MN do 91%, DN bez zmian → zostajemy przy gain 1.
- Dekoder `readout="dn"` (`data/decoders/planB_dn.npz`, 120 epizodów, `--thrust hold`): sam start z regresji
  23.2° / 4 z 4 w stronę celu, po treningu 17.9° / 4 z 4 (wcześniej 6 MN: 45–53°, 2–3 z 4).
  `fly_banc.py --local --decoder data/decoders/planB_dn.npz`: zawis 1.7°, cel +60° → 7° (ustalone po 6 s),
  −60° → 21°, podmuch maks. 35° → wraca do 1.5° po 3.9 s. Wideo: `data/videos/demo_planB_dn.mp4`.

## Plan domknięcia (ustalony 2026-10-03, idziemy według niego)

Zrobione wcześniej: ID v888 uzgodnione z Osobą 1 (`unmatched_ids = 0`), dynamika na GPU, oczy MuJoCo i most ZMQ (Osoba 1).

**Etap 1 — pętla z MuJoCo działa w ogóle** (ścieżka krytyczna: Osoba 3)
- [x] O3: `DroneEnv` (`sim/env.py`, `sim/world.py`): X2 Osoby 3 + oczy + cel mocap, `Pilot` angle/acro
  (`thrust_mode` banc/hold), scenariusze `hover`/`turn_right`/`turn_left`/`gust`, `summarize` (metryki).
  Mixer z desaturacją yaw (wcześniej yaw ±1 zerował parę silników i dron wznosił się ~2 m/s). Opis: `docs/osoba3-plan.md`.
- [x] O3: `scripts/fly_banc.py` (ZMQ albo `--local` w jednym procesie, `--video`, wyniki `data/runs/`),
  `LocalClient` w `visual_pipeline/server.py`, kalibracja `DroneEnv.calibration_render`.
- [x] O1+O2: kalibracja na scenach z symulatora (`calib` w protokole) — naprawia thrust = 0 na scenie MuJoCo.
  Sprawdzone z prawdziwym FlyVis + MuJoCo (`fly_banc.py --local`).

**Etap 2 — zachowanie**
- [x] O2: znak yaw z obrotu (`calibrate_yaw_sign`), wpięty w kalibrację serwera (syntetyczną i z symulatora).
- [x] O2: haltery: osobne znaki roll/yaw, domyślnie tylko yaw_rate.
- [x] O2+O3: **skręt do celu:** Plan C (6 średnich MN) nie skręca, Plan B z odczytem DN (`--readout dn`) skręca
  w obie strony i wraca po podmuchu (wyniki wyżej). Thrust z BANC dalej < 0.5 → demo na `--thrust hold`.
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

## Eksplorator Next.js (`explorer/`)

Następca artifactu: Next.js 16 + three.js / `@react-three/fiber`, render na GPU. Somy = jedna `THREE.Points`
z ShaderMaterial, szkielety = jedna `LineSegments`; filtry (klasy, grupy, zakres mózg/VNC, tryb, beacon) to uniformy
w `lib/shaders.ts`. Mieszanie alfa zwykłe z małym kryciem (nie addytywne) — użytkownik odrzucił przepalanie do bieli.
Etykiety regionów to divy nad canvasem pozycjonowane w `useFrame` (drei `Html` gubiło etykietę mózgu).
Dane: `npm run dev` / `build` uruchamia `scripts/sync-data.mjs`, który kopiuje `../data/viz/*.json` do `public/data`
(ignorowane w git). Statyczny eksport `out/`.

## Uwagi do repo

- `docs/prezentacja/` (NeuroFly.pptx + build.js) powstało w innej sesji; nie było weryfikowane względem wyników z v888.

- CI: `.github/workflows/tests.yml` — pytest (bez GPU/FlyVis/danych BANC, te testy się pomijają) i typy eksploratora.
- Uruchamianie: torch z CUDA, potem `pip install -e .[all]` (extras: `vision`, `sim`, `dev`), `python scripts/download_banc.py`, `flyvis download-pretrained`, `pytest`.
- `python scripts/doctor.py [--master IP]` sprawdza instalację (Python 3.12, pakiety, torch z CUDA, wagi FlyVis,
  dane BANC, dekodery, łącze z masterem) i podaje polecenia naprawy. Master odpowiada na `probe` bez liczenia workera.
- Lot do celu w świecie Osoby 3: `sim/banc_pilot.py` (BANC → thrust/roll/pitch/yaw, `WorldEnv` angle), trening
  `scripts/train_world.py` / `train_distributed.py --world` (master i workerzy muszą mieć ten sam tryb — inaczej
  master odrzuca), okno `python -m sim.run_env --banc <wagi>`. Założenia: czysty korytarz bez bloków, maszt ×4.
- Lot w świecie: `sim/world_decoder.py` (`WorldDecoder`). yaw TYLKO z BANC (6 MN + DN lotu + wyraz wolny; czujniki mają
  wagę 0 z konstrukcji), thrust/roll/pitch z BANC + czujników drona (wysokość z dalmierza, v_z, prędkość przód/bok
  z przepływu optycznego, żyroskop) — NASZE ZAŁOŻENIE: BANC = percepcja i kierunek, czujniki = stabilizacja.
  Bez czujników te osie były nieuczalne (nauczyciel steruje z wysokości/prędkości, których BANC nie dostaje).
  Uczenie: DAgger, statystyki XᵀX/Xᵀy ze wszystkich epizodów, wagi z regresji grzbietowej po każdej partii;
  w `train_distributed --world` workerzy wysyłają statystyki, master liczy regresję. Błąd osobno dla osi (`loss_axes`).
  Yaw na start z dekodera zawisu (`planB_distributed.npz`, 1000 epizodów na 2 GPU: 25.1° → 2.65°).
- Wykresy: `python scripts/plot_training.py <plik.json>` → `*_wykresy.png` (robione też automatycznie po treningu).
