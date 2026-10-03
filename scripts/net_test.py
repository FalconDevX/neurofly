"""Test połączenia ZMQ między dwoma komputerami (LAN / Tailscale / Radmin) przed treningiem.

Ten sam wzorzec co protokół symulator ↔ serwer (REQ/REP, [nagłówek JSON, klatka L, klatka P]),
ale serwer tylko odsyła krótką odpowiedź, więc mierzy samą sieć. Tylko numpy + pyzmq.

    # komputer A (serwer, np. z GPU):
    python scripts/net_test.py server                      # nasłuch na tcp://*:5555
    # komputer B (symulator):
    python scripts/net_test.py client --host 10.29.188.153 # IP komputera A

Klient wysyła klatki o rozmiarze oczu MuJoCo (512×450×3, ×2) i podaje czas obiegu oraz
maksymalne FPS, a na końcu ocenia, czy łącze wystarczy na 30 FPS.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np
import zmq


def local_ips() -> list[str]:
    try:
        return sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    except OSError:
        return []


def server(port: int) -> None:
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    print(f"serwer testowy na tcp://*:{port}, adresy tego komputera: {', '.join(local_ips())}", flush=True)
    n, nbytes, t0 = 0, 0, time.perf_counter()
    while True:
        parts = sock.recv_multipart()
        header = json.loads(parts[0])
        size = sum(len(p) for p in parts)
        sock.send_multipart([json.dumps({"ok": True, "bytes": size, "host": socket.gethostname(),
                                         "seq": header.get("seq")}).encode()])
        if header.get("hello"):
            print(f"połączenie od {header['hello']}", flush=True)
        n, nbytes = n + 1, nbytes + size
        if n % 100 == 0:
            dt = time.perf_counter() - t0
            print(f"{n} wiadomości, {nbytes / dt / 1e6:.1f} MB/s", flush=True)
            n, nbytes, t0 = 0, 0, time.perf_counter()


def client(host: str, port: int, n: int, shape: tuple[int, ...], fps: float, timeout_ms: int) -> None:
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")

    try:
        t = time.perf_counter()
        sock.send_multipart([json.dumps({"hello": socket.gethostname()}).encode()])
        reply = json.loads(sock.recv_multipart()[0])
    except zmq.Again:
        raise SystemExit(f"brak odpowiedzi z tcp://{host}:{port} w {timeout_ms} ms — sprawdź IP, czy serwer "
                         f"działa i czy zapora Windows na serwerze przepuszcza TCP {port}")
    print(f"połączono z {reply['host']} ({host}:{port}), ping ZMQ {1e3 * (time.perf_counter() - t):.1f} ms")

    rng = np.random.default_rng(0)
    left, right = (rng.integers(0, 256, shape, np.uint8) for _ in range(2))
    payload = 2 * left.nbytes
    rtt = []
    for k in range(n):
        t = time.perf_counter()
        sock.send_multipart([json.dumps({"seq": k, "shape": list(shape)}).encode(), left.tobytes(), right.tobytes()])
        reply = json.loads(sock.recv_multipart()[0])
        rtt.append(time.perf_counter() - t)
        if reply["bytes"] < payload:
            raise SystemExit(f"serwer dostał {reply['bytes']} B zamiast {payload} B")
    rtt_ms = 1e3 * np.array(rtt)
    med, p95 = np.median(rtt_ms), np.percentile(rtt_ms, 95)
    print(f"{n} kroków po {payload / 1e6:.2f} MB (2 × {shape}): obieg mediana {med:.1f} ms, p95 {p95:.1f} ms, "
          f"max {rtt_ms.max():.1f} ms")
    print(f"przepustowość {payload / (med / 1e3) * 8 / 1e6:.0f} Mbit/s, maks. {1e3 / med:.0f} kroków/s "
          f"(sama sieć, bez FlyVis/BANC)")
    budget = 1e3 / fps
    verdict = "OK" if p95 < 0.5 * budget else "na granicy" if p95 < budget else "ZA WOLNO"
    print(f"budżet {fps:.0f} FPS = {budget:.0f} ms/krok, sieć zajmuje p95 {p95:.0f} ms → {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("server", "client"))
    ap.add_argument("--host", default="127.0.0.1", help="IP serwera (dla klienta)")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("-n", type=int, default=200, help="liczba kroków testu")
    ap.add_argument("--shape", type=int, nargs=3, default=(512, 450, 3), help="kształt jednej klatki H W C")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--timeout", type=int, default=5000, help="ms")
    args = ap.parse_args()
    if args.role == "server":
        server(args.port)
    else:
        client(args.host, args.port, args.n, tuple(args.shape), args.fps, args.timeout)


if __name__ == "__main__":
    main()
