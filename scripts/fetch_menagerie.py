"""Pobiera model drona Skydio X2 z MuJoCo Menagerie do third_party/ (katalog jest w .gitignore).

    python scripts/fetch_menagerie.py
    python -m mujoco.viewer --mjcf=third_party/mujoco_menagerie/skydio_x2/scene.xml

Sparse checkout — ściąga tylko skydio_x2, nie całe repo Menagerie.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party" / "mujoco_menagerie"
REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
MODELS = ["skydio_x2"]


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main() -> None:
    if not (DEST / ".git").exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(DEST))
    git("sparse-checkout", "set", *MODELS, cwd=DEST)
    scene = DEST / "skydio_x2" / "scene.xml"
    if not scene.exists():
        sys.exit(f"Brak {scene} po pobraniu")
    print("gotowe:", scene)


if __name__ == "__main__":
    main()
