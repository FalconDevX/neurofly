"""Windows fix for datamate's _write_h5 (used by FlyVis to cache the connectome).

The original opens the file with mode="w", fails to find "data", then unlinks the
file while it is still open. Windows forbids deleting an open file (WinError 32).
Import this module before using FlyVis.
"""
import h5py as h5
import numpy as np

import datamate.directory
import datamate.io


def _write_h5(path, val):
    val = np.asarray(val)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        path.rmdir()
    elif path.exists():
        path.unlink()
    with h5.File(path, libver="latest", mode="w") as f:
        f["data"] = val
        f.swmr_mode = True


datamate.io._write_h5 = _write_h5
datamate.directory._write_h5 = _write_h5
