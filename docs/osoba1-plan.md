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
| Format | `uint8`, `(512, 450, 3)` RGB (H×W), tyle oczekuje `flygym.vision.retina.Retina` |
| FPS | 30 (budżet ~33 ms na klatkę) |
| Kąt widzenia | jak u muchy |

## Wyjście (co klatkę)

`list[BancActivation]`: `banc_root_id` (= `banc_888_id`, jak w `banc_control`), `cell_type`
(typ w BANC), `activity` (surowa aktywność FlyVis, bez normalizacji). Neuron BANC
przypisany kilku komórkom FlyVis dostaje ich średnią. Neurony bez dopasowania są pomijane.
Pozycja retinotopowa każdego neuronu jest w `visual_pipeline/flyvis_banc_map.csv` (`u`, `v`).

## Etap 1: kamera → FlyVis

- Model FlyVis `flow/0000/000`: 45 669 komórek, 65 typów, dt = 1/90 s (3 kroki na klatkę 30 FPS).
- Wejście FlyVis: luminancja = suma dwóch kanałów Retina (każde omatidium ma aktywny tylko jeden: pale lub yellow).
- `RetinaMapper` (we flygym 2.1 go nie ma, napisany od nowa): środki omatidiów → współrzędne (u, v), 721/721 dopasowań 1:1.
- Orientacja potwierdzona dla prawego oka: zgodność z rendererem treningowym FlyVis (BoxEye), r = 0,996 wobec ≤ 0,16 dla pozostałych 11 obrotów/odbić siatki (`scripts/check_retina_orientation.py`).
- Lewe oko: lustro poziome prawego (symetria dwustronna). Założenie, poprawne przy kamerach ustawionych symetrycznie.
- Czas: ~4 ms na klatkę stereo (RTX 3070 Ti), pierwsza klatka ~0,7 s rozgrzewki.

## Etap 2: FlyVis → BANC (`visual_pipeline/flyvis_banc_map.csv`)

Odtworzenie:

```bash
python scripts/download_banc.py
python scripts/fetch_skeletons.py Mi1
python scripts/build_flyvis_banc_map.py
python scripts/validate_flyvis_banc_map.py
```

Metoda:
1. Arkusz referencyjny: rozgałęzienia Mi1 ze szkieletów SWC (ciała komórek leżą w kilku warstwach i dają ~2,7× gorszą pozycję) → Isomap → skala z powierzchni na komórkę (1 Mi1 = 1 kolumna).
2. Pozostałe typy: średnia pozycja połączonych synaptycznie, już umieszczonych neuronów (iteracyjnie). Omija problem skrzyżowania wzrokowego.
3. Orientacja: Procrustes między przesunięciami wejść T4a–d/T5a–d w FlyVis i w BANC.
4. Przypisanie 1:1 w obrębie typu (algorytm węgierski), próg 1 kolumna.

Wynik:

| | Prawe oko | Lewe oko |
|---|---|---|
| Komórki FlyVis z neuronem BANC | 13 177 / 45 669 | 4 632 / 45 669 |
| Typy z dopasowaniem | 48 / 65 | 48 / 65 |

Braki wynikają głównie z BANC:
- Lamina nieobrazowana: R1–R6 i Am bez odpowiednika, L1–L5 częściowo.
- Typy nieobecne w BANC: Mi3, Mi11, Mi12, Tm28, Tm30, Tm5Y, TmY13, TmY18. CT1 (1 neuron) odpadł.
- Lewa strona: 72% neuronów płata wzrokowego bez typu.
- FlyVis ma 721 kolumn, oko ~800, więc brzeg BANC nie ma pary.

Walidacja: 77–98% silnie połączonych par kolumnowych (np. Mi1 → T4a) trafia w tę samą lub
sąsiednią kolumnę, losowo mediana ~10 kolumn.

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

Część Osoby 1 to ~7 ms. Do 30 FPS brakuje po stronie kontrolera (Osoba 2: GPU / podgraf).

Zachowanie w pętli:
- ID zgodne: `unmatched_ids = 0` dla wszystkich 17 809 neuronów.
- Pętla otwarta: cel z lewej → `yaw −1,00`, z prawej → `yaw +0,61` (skręt do celu).
- Wymuszony obrót ±90°/s → komenda przeciwna (odruch optomotoryczny, stabilizuje kurs). Znak wzroku jest poprawny.
- Pełna pętla: dron obraca się bez końca. Przyczyna: sprzężenie gyro → haltery w kontrolerze (`roll_rate + yaw_rate`, znak kalibrowany tylko na przechyle). Z `--no-imu` obrót znika.
- `--yaw-only --no-imu`: kurs stabilny (kąt do celu 35–58°), ale dron nie skręca do celu. Komendy nasycają się ±1, a reakcja na ruch obrazu przeważa nad przyciąganiem do celu.

Do Osoby 2: rozdzielić yaw_rate od roll_rate w haltery, zmniejszyć wzmocnienie yaw, skok na starcie po `dyn.reset()`.

## Znane ograniczenia

- **Obrót mapy BANC niepewny.** Wynik wskazuje rotację, nie odbicie, ale dopasowanie jest umiarkowane (średni cosinus 0,52 prawa / 0,37 lewa). Typy są poprawne, ale który fragment obrazu trafia do którego neuronu, może być obrócony o kilkadziesiąt stopni.
- Walidacja jest częściowo cykliczna (pozycje typów innych niż Mi1 liczone z partnerów) i nie wykrywa globalnego obrotu.

## Do zrobienia

1. Niezależne ustalenie obrotu (osie grzbiet–brzuch / przód–tył w przestrzeni BANC).
2. Szkielety dla pozostałych typów: mniej cykliczności, większe pokrycie.
3. Ustalić z Osobą 3, jak symulacja dostarcza klatki (ten sam proces czy osobna aplikacja).
