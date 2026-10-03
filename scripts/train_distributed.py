"""Trening dekodera (Plan B / A) na kilku komputerach z GPU w sieci LAN: master + workerzy (ZMQ).

    # komputer A (master, port 5555 musi być otwarty w zaporze — patrz scripts/net_master.py):
    python scripts/train_distributed.py master --plan B --episodes 120
    # na KAŻDYM komputerze z GPU (też na A, w osobnym terminalu), w środowisku .venv312:
    python scripts/train_distributed.py worker --host 127.0.0.1        # na A
    python scripts/train_distributed.py worker --host 192.168.50.1     # na B (IP mastera)

Master (tylko numpy + pyzmq, bez GPU) trzyma macierz dekodera ``M`` i rozdaje zadania po ``--batch``
epizodów. Worker ma własny DroneEnv + FlyVis + BANC na swoim GPU, kalibruje się lokalnie na tych
samych scenach MuJoCo, leci epizody z uczeniem jak w ``train_decoder.py`` i odsyła tylko zmianę wag
ΔM (4 × 6) i metryki; master dodaje ΔM do ``M`` (asynchronicznie, bez czekania na wolniejszego).
Przez sieć nie idą klatki, więc łącze nie jest wąskim gardłem.

Workerzy muszą mieć ten sam znak osi yaw z kalibracji (``yaw_axis_sign``), inaczej ich ΔM by się
znosiły — master odrzuca niezgodnego workera. Ewaluację przed i po (stałe kąty celu, bez szumu)
robi pierwszy wolny worker. Wynik: ``--out`` (.npz jak ``LinearDecoder.save`` + .json z przebiegiem).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PORT = 5555


def beta_schedule(episode: int, total: int, plan: str) -> float:
    """Jak ``train_decoder.beta_schedule`` (kopia: master nie importuje MuJoCo/torch)."""
    return max(0.0, 1.0 - episode / max(1, total // 2)) if plan == "B" else 0.0


# --- master ---------------------------------------------------------------------------------------
class Master:
    def __init__(self, args) -> None:
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.M = None
        self.ref = None  # kalibracja pierwszego workera (znak osi yaw, trymy)
        if args.init:
            d = np.load(args.init)
            self.M = d["M"].copy()
        self.M0 = None
        self.next_episode = 0
        self.done_episodes = 0
        self.before = self.after = None
        self.eval_pending = {"before": False, "after": False}
        self.history, self.workers = [], {}
        self.finished = False

    def task(self, name: str) -> dict:
        a = self.args
        if self.finished:
            return {"kind": "stop"}
        if not self.eval_pending["before"]:  # ewaluacja wag startowych; inni trenują równolegle
            self.eval_pending["before"] = True
            return {"kind": "eval", "tag": "before", "M": self.M0.tolist()}
        if self.next_episode < a.episodes:
            eps = []
            for _ in range(min(a.batch, a.episodes - self.next_episode)):
                eps.append({"index": self.next_episode,
                            "bearing": float(self.rng.uniform(-a.max_bearing, a.max_bearing)),
                            "beta": beta_schedule(self.next_episode, a.episodes, a.plan),
                            "seed": int(self.rng.integers(1 << 31))})
                self.next_episode += 1
            return {"kind": "train", "M": self.M.tolist(), "episodes": eps}
        if self.done_episodes < a.episodes or self.before is None or self.eval_pending["after"]:
            return {"kind": "wait", "seconds": 5}  # inni workerzy jeszcze liczą
        self.eval_pending["after"] = True
        return {"kind": "eval", "tag": "after", "M": self.M.tolist()}

    def handle(self, msg: dict) -> dict:
        name = msg.get("name", "?")
        kind = msg["type"]
        if kind == "hello":
            calib = msg["calib"]
            if self.ref is None:
                self.ref = {k: calib[k] for k in ("yaw_axis_sign", "hover_thrust", "pitch_trim")}
                if self.M is None:
                    self.M = np.array(msg["M"])
                self.M0 = self.M.copy()
                print(f"[{name}] pierwszy worker, kalibracja: {calib}", flush=True)
            elif calib["yaw_axis_sign"] != self.ref["yaw_axis_sign"]:
                print(f"[{name}] ODRZUCONY: yaw_axis_sign {calib['yaw_axis_sign']} ≠ {self.ref['yaw_axis_sign']}",
                      flush=True)
                return {"kind": "stop", "error": "inny znak osi yaw z kalibracji niż u pierwszego workera"}
            else:
                print(f"[{name}] dołączył, kalibracja: {calib}", flush=True)
            self.workers[name] = {"episodes": 0, "since": time.time()}
        elif kind == "result" and msg["kind"] == "train":
            self.M += np.array(msg["dM"])
            self.done_episodes += len(msg["metrics"])
            self.workers.setdefault(name, {"episodes": 0})["episodes"] += len(msg["metrics"])
            for m in msg["metrics"]:
                self.history.append({**m, "worker": name})
                print(f"ep {m['index']:4d} [{name}]: cel {m['bearing']:+5.0f}° → {m['final_deg']:5.1f}°  "
                      f"beta {m['beta']:.2f}  loss {m['loss']:.4f}  ({self.done_episodes}/{self.args.episodes})",
                      flush=True)
        elif kind == "result" and msg["kind"] == "eval":
            setattr(self, msg["tag"], msg["eval"])
            show(f"[{name}] {'przed treningiem' if msg['tag'] == 'before' else 'po treningu    '}", msg["eval"])
            if msg["tag"] == "after":
                self.save()
                self.finished = True
        return self.task(name)

    def save(self) -> None:
        a = self.args
        out = a.out or ROOT / "data" / "decoders" / f"plan{a.plan}_distributed.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"plan": a.plan, "episodes": a.episodes, "workers": list(self.workers),
                "before": self.before["mean_final_deg"], "after": self.after["mean_final_deg"]}
        np.savez(out, M=self.M, hover_thrust=self.ref["hover_thrust"], pitch_trim=self.ref["pitch_trim"],
                 meta=np.array(repr(meta)))
        out.with_suffix(".json").write_text(json.dumps(
            {"args": vars(a), "workers": self.workers, "before": self.before, "after": self.after,
             "M_before": self.M0.tolist(), "M_after": self.M.tolist(), "history": self.history},
            default=str, indent=1))
        show("przed treningiem", self.before)
        show("po treningu    ", self.after)
        print(f"zapisano {out}; workerzy: {self.workers}", flush=True)


def run_master(args) -> None:
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    ips = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    print(f"master na tcp://*:{args.port}, adresy: {', '.join(ips)}; czekam na workerów", flush=True)
    master = Master(args)
    idle_since = None
    while True:
        if sock.poll(1000):
            msg = json.loads(sock.recv())
            try:
                reply = master.handle(msg)
            except Exception as e:  # REQ po drugiej stronie czeka na odpowiedź
                reply = {"kind": "stop", "error": f"{type(e).__name__}: {e}"}
                print("błąd mastera:", reply["error"], flush=True)
            sock.send(json.dumps(reply).encode())
            idle_since = time.time() if master.finished else None
        elif master.finished and idle_since and time.time() - idle_since > 15:
            break  # pozostali workerzy dostaną "stop" przy następnym żądaniu albo wyjdą po timeoucie


# --- worker ---------------------------------------------------------------------------------------
def run_worker(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from sim.env import summarize
    from train_decoder import Runner, build

    name = args.name or socket.gethostname()
    import torch

    if torch.cuda.is_available():
        print(f"[{name}] GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__})", flush=True)
    elif not args.cpu:  # BANC i FlyVis po cichu przeszłyby na CPU (~10× wolniej)
        sys.exit(f"[{name}] brak CUDA w torch {torch.__version__} (wersja CPU?). Zainstaluj wersję z CUDA:\n"
                 "    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124\n"
                 "albo uruchom z --cpu.")
    t0 = time.perf_counter()
    env, bridge, ctrl, calib = build(args.plan, args.lr, args.noise, args.reward_lr, args.thrust)
    dec = ctrl.decoder
    print(f"[{name}] gotowy po {time.perf_counter() - t0:.0f} s, kalibracja: "
          f"{ {k: v for k, v in calib.items() if k not in ('ok', 'scene')} }", flush=True)
    runner = Runner(env, bridge, ctrl)

    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60_000)
    sock.connect(f"tcp://{args.host}:{args.port}")

    def send(msg: dict) -> dict:
        sock.send(json.dumps(msg, default=float).encode())
        return json.loads(sock.recv())

    task = send({"type": "hello", "name": name, "M": dec.M.tolist(),
                 "calib": {"yaw_axis_sign": ctrl.yaw_axis_sign, "haltere_gain": ctrl.haltere_gain,
                           "hover_thrust": dec.hover_thrust, "pitch_trim": dec.pitch_trim}})
    while task["kind"] != "stop":
        if task["kind"] == "wait":
            time.sleep(task["seconds"])
            task = send({"type": "poll", "name": name})
            continue
        M = np.array(task["M"])
        dec.M = M.copy()
        if task["kind"] == "eval":
            print(f"[{name}] ewaluacja {task['tag']}…", flush=True)
            task = send({"type": "result", "kind": "eval", "name": name, "tag": task["tag"],
                         "eval": runner.evaluate(args.eval_duration)})
            continue
        metrics = []
        for ep in task["episodes"]:
            t = time.perf_counter()
            log, loss = runner.episode(ep["bearing"], args.duration, learn=args.plan, beta=ep["beta"],
                                       rng=np.random.default_rng(ep["seed"]))
            m = summarize(log, env.fps)
            metrics.append({**ep, "loss": loss, **m})
            print(f"[{name}] ep {ep['index']}: cel {ep['bearing']:+5.0f}° → {m['final_deg']:5.1f}° "
                  f"({time.perf_counter() - t:.0f} s)", flush=True)
        task = send({"type": "result", "kind": "train", "name": name,
                     "dM": (dec.M - M).tolist(), "metrics": metrics})
    print(f"[{name}] koniec{': ' + task['error'] if task.get('error') else ''}", flush=True)


# --- wspólne --------------------------------------------------------------------------------------
def show(tag: str, ev: dict) -> None:
    keys = [k for k in ev if k.startswith(("+", "-"))]
    per = "  ".join(f"{k}°→{ev[k]['final_deg']:.0f}°" for k in keys)
    print(f"{tag}: średni końcowy |kąt| {ev['mean_final_deg']:.1f}°, skręt w stronę celu "
          f"{ev['turns_toward']}/{len(keys)}  ({per})", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: przekierowane stdout jest w cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=("master", "worker"))
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--plan", choices=("A", "B"), default="B", help="musi być ten sam na masterze i workerach")
    # master
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--batch", type=int, default=2, help="epizodów na jedno zadanie")
    ap.add_argument("--max-bearing", type=float, default=90.0)
    ap.add_argument("--init", type=Path, help="wagi startowe (.npz); domyślnie z kalibracji pierwszego workera")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    # worker
    ap.add_argument("--host", default="127.0.0.1", help="IP mastera")
    ap.add_argument("--name", help="nazwa workera w logu (domyślnie nazwa komputera)")
    ap.add_argument("--cpu", action="store_true", help="pozwól liczyć bez GPU")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--eval-duration", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--reward-lr", type=float, default=0.05)
    ap.add_argument("--thrust", choices=("banc", "hold"), default="hold")
    args = ap.parse_args()
    run_master(args) if args.role == "master" else run_worker(args)


if __name__ == "__main__":
    main()
