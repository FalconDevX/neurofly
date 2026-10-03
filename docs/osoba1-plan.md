# Osoba 1: wzrok → BANC (`visual_pipeline/`)

START: klatki RGB z dwóch kamer drona. STOP: aktywność FlyVis przypisana do neuronów BANC
w kontrakcie Osoby 2 (`banc_control.contracts.BancActivation`).
Poza zakresem: propagacja BANC/VNC, motoneurony, sterowanie.

## Użycie

```python
from visual_pipeline import VisionBridge
from banc_control import BancController, Connectome

bridge = VisionBridge(fps=30)
ctrl = BancController(Connectome.from_banc())
cmd = ctrl.step(bridge.step_batch(frame_left, frame_right), imu)  # szybka ścieżka, ten sam proces
records = bridge.step(frame_left, frame_right)   # list[BancActivation], np. do JSON
```

## Instalacja (Python 3.12)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .[vision,dev]
flyvis download-pretrained            # wagi FlyVis (~kilkaset MB, Google Drive autorów)
pytest tests/test_visual_pipeline.py
```

Windows: `visual_pipeline/datamate_win_fix.py` łata błąd biblioteki `datamate` (usuwanie
otwartego pliku HDF5, WinError 32), bez którego FlyVis nie startuje. Ładuje się automatycznie.

## Wejście

| Parametr | Wartość |
|---|---|
| Kamery | 2 (stereo): `left`, `right` |
| Format | `uint8` RGB; inny rozmiar skalowany do `(512, 450, 3)` (H×W), potem zamiana na luminancję |
| FPS | 30 (budżet ~33 ms na klatkę) |
| Kąt widzenia | jak u muchy; kamery MuJoCo wg specyfikacji z Etapu 5 + `VisionBridge(fisheye=True)` |

## Wyjście (co klatkę)

`list[BancActivation]`: `banc_root_id` (= `banc_888_id`, jak w `banc_control`), `cell_type`
(typ w BANC), `activity` (surowa aktywność FlyVis, bez normalizacji). Neuron BANC
przypisany kilku komórkom FlyVis dostaje ich średnią. Neurony bez dopasowania są pomijane.
Pozycja retinotopowa każdego neuronu jest w `visual_pipeline/flyvis_banc_map.csv` (`u`, `v`).

## Etap 1: kamera → FlyVis

- Model FlyVis `flow/0000/000`: 45 669 komórek, 65 typów, dt = 1/90 s (3 kroki na klatkę 30 FPS).
- Wejście FlyVis: luminancja = suma dwóch kanałów Retina (każde omatidium ma aktywny tylko jeden: pale lub yellow).
- `RetinaMapper` (we flygym 2.1 go nie ma, napisany od nowa): środki omatidiów → współrzędne (u, v), 721/721 dopasowań 1:1.
- Siatka bez odbicia odtwarza renderer treningowy FlyVis (BoxEye): r = 0,996 wobec ≤ 0,16 dla pozostałych 11 obrotów/odbić (`scripts/check_retina_orientation.py`).
- **Prawe oko jest odbijane poziomo, lewe nie.** Siatka FlyVis to oko „widziane z zewnątrz" (dla prawego oka przód po prawej stronie kadru), a kamera patrzy od środka. Test kratką w 12 kierunkach (`scripts/check_motion_directions.py`): bez odbicia T4a/T5a preferują ruch tył→przód; z odbiciem T4a/b/c i T5a/b/c/d zgadzają się z anatomią (a przód→tył, b tył→przód, c w górę, d w dół). T4d tego modelu FlyVis preferuje ruch w górę (słaba selektywność 0,51).
- Czas: ~4 ms na klatkę stereo (RTX 3070 Ti), pierwsza klatka ~0,7 s rozgrzewki.

## Etap 2: FlyVis → BANC (`visual_pipeline/flyvis_banc_map.csv`)

Odtworzenie:

```bash
python scripts/download_banc.py
python scripts/fetch_skeletons.py --workers 64 <wszystkie typy z mapy>   # ~23 tys. plików
python scripts/build_flyvis_banc_map.py
python scripts/validate_flyvis_banc_map.py
python scripts/check_rotation_banc.py     # kierunek ruchu w BANC przy obrocie drona
python scripts/coverage_diagnostics.py    # co ogranicza pokrycie
python scripts/investigate_left_t4.py     # diagnostyka lewej strony
```

Metoda:
1. Arkusz referencyjny: rozgałęzienia Mi1 ze szkieletów SWC (ciała komórek leżą w kilku warstwach i dają ~2,7× gorszą pozycję) → Isomap → skala z powierzchni na komórkę (1 Mi1 = 1 kolumna).
2. Pozycje pozostałych neuronów:
   - z **własnego szkieletu** (węzły w medulli rzutowane na arkusz Mi1) dla wszystkich typów z rozgałęzieniami w medulli: 76% przypisanych neuronów po prawej, 39% po lewej;
   - ze średniej pozycji połączonych synaptycznie neuronów dla T5 (dendryty w lobuli), CT1 i neuronów bez szkieletu w zasobniku.
3. Orientacja (obrót siatki FlyVis względem arkusza BANC):
   - Procrustes przesunięć wejść T4 względem FlyVis, tylko wejścia wyznaczające kierunek T4: **Tm3, Mi4, Mi9, C3**. Mi1 leży prawie w osi kolumny T4 (cosinus +0,20 nawet po prawej), a drobne wejścia (TmY15, T4→T4) są zaszumione. Przedział z bootstrapu po komórkach T4.
   - **Prawa strona: obrót +100°, 95% CI +99°…+103°**, cosinus 0,89.
   - **Lewa strona: lustro prawej względem płaszczyzny środkowej (odbicie +88°)**, potwierdzone neuronami brzegu grzbietowego (DRA).
   - Kontrola grzbietu: „góra siatki" w 3D wskazuje na neurony DRA (cosinus +0,89 prawa, +0,81 lewa; po obrocie o 180°: −0,89 / −0,81).
4. Przypisanie 1:1 w obrębie typu (algorytm węgierski), próg 1 kolumna. Pary dalsze niż próg mają zaporowy koszt, więc algorytm najpierw maksymalizuje liczbę par w progu. Przy zwykłym koszcie odległości przesuwał całe łańcuchy par tuż za próg: dla Mi1 95% kolumn miało neuron w ≤ 1 kolumnie, a zostawało tylko 70%.

Wynik:

| | Prawe oko | Lewe oko |
|---|---|---|
| Komórki FlyVis z neuronem BANC | **16 927** / 45 669 (wcześniej 12 963) | **5 535** / 45 669 (wcześniej 4 623) |
| Typy z dopasowaniem | 48 / 65 | 48 / 65 |
| Mediana odległości dopasowania | 0,42 kolumny | 0,43 kolumny |

**Lewa strona: dlaczego dane T4 wskazywały zwrot przeciwny** (`scripts/investigate_left_t4.py`):
po lewej zostaje tylko kilka par wejście → T4 z ≥ 100 synapsami. Wcześniejsze dopasowanie
zdominowała para Mi1 → T4a (2 648 synaps, cosinus −0,86), która nie niesie informacji
o kierunku (po prawej +0,20). Po jej wykluczeniu dane lewej strony prawie nic nie mówią
(cosinus 0,21, wolne dopasowanie nie rozróżnia skrętności). Nie ma wzoru wskazującego na
zamianę etykiet (Mi4/Mi9 czy podtypów T4). Test kolejności warstw płytki lobuli okazał się
niemiarodajny (zawodzi także po prawej), więc nie jest podstawą wniosku. Orientacja lewej
strony pochodzi z symetrii i jest zgodna z DRA.

**Kierunek ruchu end-to-end** (`scripts/check_rotation_banc.py`, bez dekodera Osoby 2): przy
skręcie w prawo prawe oko aktywuje BANC T4b/T5b (tył→przód), lewe T4a/T5a (przód→tył),
przy skręcie w lewo odwrotnie. 4/4 przypadki zgodne z anatomią; przed poprawką lustra 4/4 odwrócone.

Braki wynikają głównie z BANC:
- Lamina nieobrazowana: R1–R6 i Am bez odpowiednika, L1–L5 częściowo.
- Typy nieobecne w BANC: Mi3, Mi11, Mi12, Tm28, Tm30, Tm5Y, TmY13, TmY18. CT1 (1 neuron) odpadł.
- Lewa strona: 72% neuronów płata wzrokowego bez typu.
- FlyVis ma 721 kolumn, oko ~800, więc brzeg BANC nie ma pary.

Walidacja: 65–92% silnie połączonych par kolumnowych (np. Mi1 → T4a 82%) trafia w tę samą lub
sąsiednią kolumnę, mediana 1 kolumna, losowo ~12. Po zwiększeniu pokrycia odsetek spadł o kilka
punktów (więcej par blisko progu), ale walidacja jest mniej cykliczna: większość pozycji
pochodzi ze szkieletów, a nie z partnerów.

## Etap 4: pętla zamknięta i czas rzeczywisty

```bash
python scripts/run_closed_loop.py                       # pełna pętla ze stubami
python scripts/run_closed_loop.py --yaw-only --no-imu   # sam wzrok steruje kursem
```

Kamera stereo → `VisionBridge` → `BancController` → `ToyDrone` → kamera. Gazebo (Osoba 3)
zastępuje `FakeStereoCamera` (niebo, ziemia, ciemny pionowy pas = cel) i `ToyDrone`.
Klatki z dowolnej kamery skaluje `visual_pipeline.frames.prepare_frame` (np. 640×480 RGBA).

**Szybka ścieżka `VisualBatch`** (`bridge.step_batch`, obsługiwana przez `BancController`):
te same dane co `list[BancActivation]`, ale jako tablice. Lista ~18k obiektów kosztowała
~25–30 ms na klatkę po stronie wzroku plus pętlę Pythona w kontrolerze. JSON i lista działają dalej.

Czas klatki (RTX 3070 Ti, pełny BANC 175k neuronów):

| Krok | przed | po |
|---|---|---|
| kamera (stub) | 7,5 | 7,5 |
| Retina + mapper | 1,0 | 1,0 |
| FlyVis | 16,6 | 6,1 |
| rekordy `BancActivation` | 30,4 | 0 |
| kontroler BANC (Osoba 2) | 44,9 | 34,9 |
| **razem** | **101 ms (9,9 FPS)** | **49,8 ms (20 FPS)** |

Po scaleniu z `main` (dynamika BANC Osoby 2 na GPU): kamera 9,0, Retina 1,2, FlyVis 3,6,
mapowanie 0,2, kontroler 2,5 → **razem 16,4 ms (61 FPS)**, budżet 30 FPS spełniony.

Zachowanie w pętli (po poprawce orientacji):
- ID zgodne: `unmatched_ids = 0` dla wszystkich neuronów wzroku.
- Wejście ruchu do BANC ma poprawny kierunek (`check_rotation_banc.py`, 4/4).
- Pętla otwarta: cel z lewej → `yaw +0,56`, z prawej → `yaw +0,19`. Statyczny cel nie rozróżnia już stron (przed poprawką: −1,00 / +0,61).
- Odruch optomotoryczny przez dekoder (`scripts/check_optomotor.py`): wymuszony obrót ±90°/s daje komendę zgodną z obrotem, czyli wzmacnia go.
- Przed poprawką obie rzeczy wyglądały dobrze, bo **dwa błędy się znosiły**: odwrócone wejście ruchu i znak po stronie BANC/dekodera. Znak yaw ustala `calibrate_scale` na statycznym celu, który teraz nie rozróżnia stron.
- Pełna pętla: dron obraca się bez końca przez sprzężenie gyro → haltery (`roll_rate + yaw_rate`, znak kalibrowany tylko na przechyle). Z `--no-imu` ciągły obrót znika.

Do Osoby 2:
- przy poprawnym wejściu ruchu odruch optomotoryczny ma zły znak: sprawdzić znaki NT / dynamikę na drodze płytka lobuli → DN → MN skrzydeł i kalibrację znaku yaw (np. na obrocie zamiast statycznego celu);
- rozdzielić yaw_rate od roll_rate w halterach, zmniejszyć wzmocnienie yaw, skok na starcie po `dyn.reset()`.

## Etap 5: przekazanie (MuJoCo Osoby 3, ZMQ, demo)

![Potok Osoby 1 na renderze MuJoCo](img/osoba1_pipeline.png)

`python scripts/demo_figure.py`: kamera MuJoCo → Retina → FlyVis T4a/T4b → aktywność neuronów BANC
na pozycjach ich ciał, przy obrocie drona w prawo.

### Kamery-oczy dla drona X2 (`visual_pipeline/drone_eyes.py`)

Konwencja FlyGym (Retina jest pod nią skalibrowana):

| Parametr | Wartość |
|---|---|
| Kamery | 2, na ciele `x2`, `pos` (0.12, ±0.03, 0.03) m |
| Kierunek | 70° w lewo / w prawo od +x, poziomo; ~14° widzenia obuocznego z przodu |
| Orientacja kadru | góra = +z drona; prawe oko: przód po lewej stronie kadru, lewe: po prawej |
| `fovy` | 157° (jak `flygym vision.yaml`) |
| Rozmiar | 512×450 (H×W); scena musi mieć `<global offheight="512">` |
| Ciało drona | niewidoczne dla oczu (grupy geometrii 2 i 3), jak głowa muchy w FlyGym |
| Korekcja | surowy kadr prostoliniowy; `correct_fisheye` robi serwer (`fisheye=True`) |

`x2_with_eyes(menagerie_dir)` zapisuje `x2_eyes.xml` i `scene_eyes.xml` obok modelu z menagerie,
`MujocoEyes(model).render(data)` zwraca (lewa, prawa) klatkę.

Kontrola na renderze MuJoCo (`scripts/check_mujoco_eyes.py --preview eyes.png`): przy wymuszonym
obrocie drona T4/T5 w BANC zmieniają się zgodnie z anatomią w obu oczach (a−b: −0,030 prawe, +0,032 lewe).

### Most ZMQ (`visual_pipeline/zmq_protocol.py`, `scripts/vision_server.py`)

```bash
python scripts/vision_server.py                  # środowisko Osoby 1: wzrok + BancController
python scripts/example_sim_client.py             # środowisko symulatora: mujoco + numpy + pyzmq
```

- Żądanie: `[nagłówek JSON, klatka lewa, klatka prawa]`, nagłówek `{"shape", "imu", "reset"}`.
- Odpowiedź (tryb `command`): `{"thrust", "roll", "pitch", "yaw", "unmatched_ids", "timing_ms"}`.
  Tryb `--mode activity` zwraca zamiast tego `root_ids` i `activity` (dla osobnego procesu Osoby 2).
- IMU w konwencji `ImuState`: yaw + = w prawo, więc z MuJoCo `yaw = −ω_z`; roll + = prawe skrzydło w dół.
- `VisionClient` wymaga tylko `numpy` i `pyzmq` (bez torch/FlyVis): Osoba 3 musi dodać `pyzmq` do `environment.yml`.

Pomiar (RTX 3070 Ti, ten sam komputer): serwer ~23 ms na klatkę (przygotowanie klatek z korekcją
„rybiego oka" ~11, FlyVis ~7, kontroler ~3), z przesyłem ~25 ms. Pierwsza klatka ~6 s (kompilacja numba).
Luminancja liczona całkowitoliczbowo: 1,5 ms zamiast 5,7 ms na oko.

**Kolor:** każde omatidium Retina czyta tylko kanał G albo B, więc w kolorowej scenie MuJoCo dawało
fałszywy kontrast w szachownicę. Most zamienia klatki na luminancję przed Retina.

**Kalibracja na scenach z symulatora** (`VisionClient.calibrate`, żądanie `calibrate`):
symulator renderuje 3 statyczne sceny stereo w zawisie, z celem na wprost, 60° w lewo i 60° w
prawo (`CALIBRATION_BEARING_DEG`), i wysyła je w jednej wiadomości. Serwer ustala na nich
odpowiedź FlyVis i robi `calibrate_rest` / `calibrate_scale` / `calibrate_haltere_sign`
kontrolera Osoby 2. Na starcie serwer kalibruje się na scenach syntetycznych (awaryjnie);
pole `calibration` w odpowiedzi mówi, która kalibracja obowiązuje (`synthetic` / `sim`).

Wynik w MuJoCo (`scripts/example_sim_client.py`, cel 30° w prawo, obrót 30°/s):

| | kalibracja syntetyczna | kalibracja z symulatora (2,2 s) |
|---|---|---|
| `thrust`, cel na wprost | 0,11 | **0,50** (zawis) |
| `thrust`, pozostałe klatki | 0,00 | 0,18–0,43 |
| `roll` | −0,13…+0,78 | −0,04…+0,15 |

`yaw` = +1 przy obrocie w prawo w obu wariantach: to znany zły znak odruchu optomotorycznego
po stronie kontrolera (Etap 4), nie wejście wzrokowe.

## Znane ograniczenia

- Lewa strona: orientacja z symetrii i DRA, bo dane T4 lewej strony są za słabe do samodzielnego dopasowania (za mało opisanych wejść T4).
- Pozycje T5, CT1 i neuronów bez szkieletu (24% po prawej, 61% po lewej) nadal liczone z partnerów.
- Precyzja pozycji ~1 kolumna (uproszczone szkielety, Isomap zakrzywionej medulli).
- Pokrycie lewego oka ograniczone adnotacjami BANC (72% neuronów lewego płata bez typu).

## Do zrobienia

1. Osoba 3: kalibracja na scenach z symulatora przez `VisionClient.calibrate` po starcie serwera.
2. Osoba 2: znak odruchu optomotorycznego i sprzężenie halter → yaw (Etap 4).
