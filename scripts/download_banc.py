"""Pobiera oficjalne pliki BANC v888 z publicznego bucketu Lee Lab (bez logowania).

    python scripts/download_banc.py            # meta + edgelist v2 (~360 MB)
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banc_control.connectome import DEFAULT_DATA_DIR, EDGES_FILE, META_FILE  # noqa: E402

BUCKET = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome"
FILES = {
    META_FILE: f"compiled_data/banc_888/{META_FILE}",
    EDGES_FILE: f"compiled_data/banc_888/{EDGES_FILE}",
    "README.txt": "README.txt",
    "documentation/banc_888_meta.md": "documentation/banc_888_meta.md",
    "documentation/banc_888_edgelist_simple_v2.md": "documentation/banc_888_edgelist_simple_v2.md",
}


def main() -> None:
    for local, remote in FILES.items():
        dest = DEFAULT_DATA_DIR / local
        if dest.exists():
            print(f"jest   {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"pobieram {remote} …")
        urllib.request.urlretrieve(f"{BUCKET}/{remote}", dest)
    print("gotowe:", DEFAULT_DATA_DIR)


if __name__ == "__main__":
    main()
