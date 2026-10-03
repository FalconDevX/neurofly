"""Sprawdza instalację NeuroFly na tym komputerze i mówi, co doinstalować.

    python scripts/doctor.py                         # wszystko lokalnie
    python scripts/doctor.py --master 192.168.50.1   # do tego: czy master treningu odpowiada

Każdy punkt: OK / BRAK + polecenie naprawy. Kod wyjścia 1, gdy czegoś brakuje do treningu na GPU.
Nic nie instaluje i nie zmienia — tylko sprawdza.
"""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TORCH_CUDA = "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124"


class Report:
    def __init__(self) -> None:
        self.missing = 0

    def ok(self, what: str, detail: str = "") -> None:
        print(f"  OK    {what}{'  — ' + detail if detail else ''}")

    def bad(self, what: str, fix: str) -> None:
        self.missing += 1
        print(f"  BRAK  {what}\n        → {fix}")


def check_python(r: Report) -> None:
    v = sys.version_info
    if (v.major, v.minor) == (3, 12):
        r.ok("Python 3.12", sys.executable)
    else:
        r.bad(f"Python {v.major}.{v.minor} (FlyVis wymaga 3.12)", "py -3.12 -m venv .venv312 && .venv312\\Scripts\\activate")


def check_packages(r: Report) -> None:
    for module, pip in (("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
                        ("zmq", "pyzmq"), ("mujoco", "mujoco"), ("gymnasium", "gymnasium"),
                        ("cv2", "opencv-python-headless"), ("imageio", "imageio imageio-ffmpeg"),
                        ("flyvis", "flyvis==1.2.0"), ("flygym", "flygym==2.1.0")):
        try:
            m = importlib.import_module(module)
            r.ok(module, getattr(m, "__version__", ""))
        except Exception as e:  # ImportError, ale flyvis potrafi rzucić czymś innym przy złej instalacji
            r.bad(f"{module} ({type(e).__name__})", f"pip install {pip}   (albo: pip install -e .[vision,sim,dev])")


def check_torch(r: Report) -> None:
    try:
        import torch
    except ImportError:
        r.bad("torch", TORCH_CUDA)
        return
    if torch.cuda.is_available():
        r.ok("torch z CUDA", f"{torch.__version__}, {torch.cuda.get_device_name(0)}")
    else:
        r.bad(f"torch {torch.__version__} bez CUDA (BANC i FlyVis liczyłyby na CPU, ~10× wolniej)",
              f"pip uninstall -y torch torchvision && {TORCH_CUDA}   (sterownik NVIDIA: nvidia-smi)")


def check_flyvis_weights(r: Report) -> None:
    try:
        import flyvis

        net = Path(flyvis.results_dir) / "flow" / "0000" / "000"
    except Exception:
        return  # brak flyvis zgłoszony wyżej
    if net.exists():
        r.ok("wagi FlyVis", str(net))
    else:
        r.bad(f"wagi FlyVis ({net})",
              "flyvis download-pretrained   albo rozpakuj data/flyvis_results_flow.zip z laptopa do "
              f"{Path(flyvis.results_dir)}")


def check_banc(r: Report) -> None:
    from banc_control.connectome import DEFAULT_DATA_DIR, EDGES_FILE, META_FILE

    for f in (META_FILE, EDGES_FILE):
        p = DEFAULT_DATA_DIR / f
        if p.exists():
            r.ok(f"BANC v888: {f}", f"{p.stat().st_size / 1e6:.0f} MB")
        else:
            r.bad(f"BANC v888: {f}", "python scripts/download_banc.py")


def check_decoders(r: Report) -> None:
    d = ROOT / "data" / "decoders"
    files = sorted(p.name for p in d.glob("*.npz")) if d.exists() else []
    if files:
        r.ok("wagi dekoderów", ", ".join(files))
    else:  # nie blokuje treningu rozproszonego: master bierze wagi startowe od innego workera
        print("  INFO  brak data/decoders/*.npz — trening rozproszony weźmie wagi startowe od innego workera")


def check_master(r: Report, host: str, port: int) -> None:
    import zmq

    try:
        socket.getaddrinfo(host, port)
    except OSError:
        r.bad(f"adres mastera {host!r}", "popraw --master (np. 192.168.50.1)")
        return
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    # "probe": master tylko odpowiada, nie liczy tego jako workera
    sock.send(json.dumps({"type": "probe", "name": f"doctor@{socket.gethostname()}"}).encode())
    try:
        sock.recv()
        r.ok(f"master {host}:{port} odpowiada")
    except zmq.Again:
        r.bad(f"master {host}:{port} nie odpowiada w 5 s",
              "czy master działa? zapora na porcie 5555 (scripts/net_master.py), ta sama sieć / IP")
    finally:
        sock.close()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", help="IP mastera treningu do sprawdzenia połączenia")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    r = Report()
    print(f"NeuroFly doctor — {socket.gethostname()}, repo {ROOT}")
    for title, check in (("Python", check_python), ("Pakiety", check_packages), ("GPU", check_torch),
                         ("FlyVis", check_flyvis_weights), ("Dane BANC", check_banc), ("Dekodery", check_decoders)):
        print(f"\n{title}:")
        check(r)
    if args.master:
        print("\nSieć:")
        check_master(r, args.master, args.port)
    print(f"\n{'Wszystko gotowe.' if not r.missing else f'Do poprawy: {r.missing}.'}")
    sys.exit(1 if r.missing else 0)


if __name__ == "__main__":
    main()
