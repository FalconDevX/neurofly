"""Pobiera szkielety SWC neuronów BANC danych typów (publiczny bucket Lee Lab, bez logowania).

    python scripts/fetch_skeletons.py Mi1            # ~1200 plików, kilka MB

Zapisuje data/banc_888/skeletons/<root_id>.swc. Pliki są nazwane root ID, często
starszej wersji, więc dla każdego neuronu próbujemy kolejnych wersji ID.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banc_control.connectome import DEFAULT_DATA_DIR, META_FILE  # noqa: E402

BASE = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/neuron_skeletons/swcs-from-pcg-skel/"
ID_COLS = ["root_id", "banc_888_id", "root_890", "root_888", "root_850", "root_626"]
OUT_DIR = DEFAULT_DATA_DIR / "skeletons"


def main() -> None:
    args = sys.argv[1:]
    workers = 16
    if "--workers" in args:  # pliki są małe, ogranicza liczba równoległych połączeń
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_feather(DEFAULT_DATA_DIR / META_FILE, columns=ID_COLS + ["cell_type", "side"])
    meta = meta[meta.cell_type.isin(args) & meta.side.isin(["left", "right"])]

    def fetch(row):
        target = OUT_DIR / f"{row.root_id}.swc"
        if target.exists():
            return "cached"
        for c in ID_COLS:
            rid = getattr(row, c)
            if pd.isna(rid):
                continue
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(BASE + f"{rid}.swc", timeout=30) as r:
                        tmp = target.with_suffix(".tmp")
                        tmp.write_bytes(r.read())
                        tmp.replace(target)  # przerwane pobieranie nie zostawia uciętego .swc
                    return c
                except urllib.error.HTTPError:
                    break  # 404: ta wersja ID nie ma szkieletu, próbujemy następnej
                except (urllib.error.URLError, OSError):
                    time.sleep(1 + attempt)  # zerwane połączenie: ponów
            else:
                return "network_error"
        return "missing"

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(fetch, meta.itertuples()))
    print(pd.Series(results).value_counts().to_string())


if __name__ == "__main__":
    main()
