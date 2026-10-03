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
- Siatka bez odbicia odtwarza renderer treningowy FlyVis (BoxEye): r = 0,996 wobec ≤ 0,16 dla pozostałych 11 obrotów/odbić (`scripts/check_retina_orientation.py`).
- **Prawe oko jest odbijane poziomo, lewe nie.** Siatka FlyVis to oko „widziane z zewnątrz" (dla prawego oka przód po prawej stronie kadru), a kamera patrzy od środka. Test kratką w 12 kierunkach (`scripts/check_motion_directions.py`): bez odbicia T4a/T5a preferują ruch tył→przód; z odbiciem T4a/b/c i T5a/b/c/d zgadzają się z anatomią (a przód→tył, b tył→przód, c w górę, d w dół). T4d tego modelu FlyVis preferuje ruch w górę (słaba selektywność 0,51).
- Czas: ~4 ms na klatkę stereo (RTX 3070 Ti), pierwsza klatka ~0,7 s rozgrzewki.

## Etap 2: FlyVis → BANC (`visual_pipeline/flyvis_banc_map.csv`)

Odtworzenie:

```bash
python scripts/download_banc.py
python scripts/fetch_skeletons.py Mi1 T4a T4b T4c T4d Mi4 Mi9 Tm3
python scripts/build_flyvis_banc_map.py
python scripts/validate_flyvis_banc_map.py
python scripts/check_rotation_banc.py     # kierunek ruchu w BANC przy obrocie drona
```

Metoda:
1. Arkusz referencyjny: rozgałęzienia Mi1 ze szkieletów SWC (ciała komórek leżą w kilku warstwach i dają ~2,7× gorszą pozycję) → Isomap → skala z powierzchni na komórkę (1 Mi1 = 1 kolumna).
2. Pozostałe typy: średnia pozycja połączonych synaptycznie, już umieszczonych neuronów (iteracyjnie). Omija problem skrzyżowania wzrokowego.
3. Orientacja (obrót siatki FlyVis względem arkusza BANC):
   - Pozycje T4a–d, Mi4, Mi9, Tm3 z ich własnych szkieletów, rzutowane na arkusz Mi1 (bez cykliczności kroku 2). Procrustes przesunięć wejść T4 względem FlyVis, przedział z bootstrapu po komórkach T4.
   - **Prawa strona: obrót +94°, 95% CI +92°…+96°**, cosinus 0,78 (stara metoda: +102°, cosinus 0,52).
   - **Lewa strona: lustro prawej względem płaszczyzny środkowej (odbicie +82°).** Dane T4 lewej strony (6× mniej opisanych T4) dają tę samą oś, ale przeciwny zwrot. O wyborze decydują dwa niezależne źródła, które się zgadzają: symetria dwustronna i neurony brzegu grzbietowego (DRA).
   - Kontrola grzbietu: kierunek „góra siatki" w 3D wskazuje na neurony DRA (cosinus +0,87 prawa, +0,84 lewa; po obrocie o 180°: −0,87 / −0,84).
4. Przypisanie 1:1 w obrębie typu (algorytm węgierski), próg 1 kolumna.

Wynik:

| | Prawe oko | Lewe oko |
|---|---|---|
| Komórki FlyVis z neuronem BANC | 13 048 / 45 669 | 4 652 / 45 669 |
| Typy z dopasowaniem | 48 / 65 | 48 / 65 |

Względem poprzedniej mapy neurony BANC przesunęły się o medianę 1 kolumny po prawej
(90%: ≤ 2) i 3 kolumny po lewej (90%: ≤ 4).

**Kierunek ruchu end-to-end** (`scripts/check_rotation_banc.py`, bez dekodera Osoby 2): przy
skręcie w prawo prawe oko aktywuje BANC T4b/T5b (tył→przód), lewe T4a/T5a (przód→tył),
przy skręcie w lewo odwrotnie. 4/4 przypadki zgodne z anatomią; przed poprawką lustra 4/4 odwrócone.

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

## Znane ograniczenia

- Lewa strona: orientacja z symetrii i DRA, a nie z danych T4 lewej strony (które wskazują zwrot przeciwny). Przyczyna rozbieżności niezbadana; podejrzenie: adnotacje podtypów T4 po lewej.
- Walidacja par kolumnowych jest częściowo cykliczna (pozycje typów innych niż Mi1 i T4/Mi4/Mi9/Tm3 liczone z partnerów).
- Precyzja pozycji ~1 kolumna (uproszczone szkielety, Isomap zakrzywionej medulli).

## Do zrobienia

1. Szkielety dla pozostałych typów: mniej cykliczności, większe pokrycie.
2. Ustalić z Osobą 3, jak symulacja dostarcza klatki (ten sam proces czy osobna aplikacja).
3. Sprawdzić adnotacje T4 po lewej stronie BANC (rozbieżność 180°).
