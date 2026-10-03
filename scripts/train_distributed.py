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

``--world`` (master i workerzy): trening lotu do celu w świecie Osoby 3 (``scripts/train_world.py``):
BANC steruje thrust/roll/pitch/yaw, epizod = losowy świat, ewaluacja = ``EVAL_WORLDS`` (ile razy cel).

    python scripts/train_distributed.py master --world --episodes 200
    python scripts/train_distributed.py worker --world --host 127.0.0.1     # i na slave: --host 192.168.50.1

Kontrola przed treningiem: worker zaraz po starcie (po sprawdzeniu CUDA, przed wczytaniem BANC) wysyła
``ping``. Master czeka na ``--workers`` zgłoszeń przez ``--wait-join`` s, a potem na ich gotowość (``hello``
po kalibracji) przez ``--wait-ready`` s. Gdy ktoś się nie zgłosi (np. slave nie odpowiada), master
przerywa: workerzy dostają ``stop``, nic nie jest trenowane ani zapisywane, kod wyjścia 1.
Trening rusza dopiero, gdy są wszyscy.

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


def short_gpu(name: str) -> str:
    """„NVIDIA GeForce RTX 4060 Laptop GPU” → „RTX 4060” (etykieta workera w logach)."""
    for junk in ("NVIDIA ", "GeForce ", " Laptop GPU", " GPU"):
        name = name.replace(junk, "")
    return name.strip() or "?"


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
        self.best = None  # najlepsze wagi według ewaluacji na stałych światach (--world)
        self.evals = []  # ewaluacje: przed, w trakcie (co --eval-every epizodów), po
        self.mid_pending: dict[str, np.ndarray] = {}  # tag ewaluacji w trakcie → oceniane wagi
        self.next_eval = getattr(args, "eval_every", 0) or 0
        self.wdec = None  # --world: WorldDecoder (statystyki DAgger sumowane od workerów, wagi z regresji)
        if args.init:
            d = np.load(args.init)
            self.M = d["M"].copy() if "M" in d.files else None
            if "w_yaw" in d.files:  # wagi z train_world.py
                from sim.world_decoder import WorldDecoder

                self.M = WorldDecoder.load(args.init).to_matrix()
        self.M0 = None
        self.next_episode = 0
        self.last_bearing = 0.0
        self.done_episodes = 0
        self.before = self.after = None
        self.eval_pending = {"before": False, "after": False}
        self.history, self.workers = [], {}
        self.init_from_worker = False
        self.pinged: dict[str, str] = {}  # nazwa → GPU, zgłoszenia przed wczytaniem BANC
        self.finished = False
        self.error: str | None = None  # powód przerwania

    def label(self, name: str) -> str:
        """Etykieta w logu: karta graficzna workera; przy dwóch takich samych kartach + nazwa komputera."""
        gpu = self.pinged.get(name)
        if not gpu:
            return name
        same = [n for n, g in self.pinged.items() if g == gpu]
        return gpu if len(same) == 1 else f"{gpu} ({name})"

    @property
    def needed(self) -> int:
        return getattr(self.args, "workers", 1)

    def abort(self, reason: str) -> None:
        self.error, self.finished = reason, True
        print(f"PRZERWANO: {reason}", flush=True)

    def check_deadlines(self, elapsed: float) -> None:
        """Wywoływane co sekundę przez ``run_master``: brak workerów w czasie → przerwanie."""
        if self.finished:
            return
        a = self.args
        if len(self.pinged) < self.needed and elapsed > getattr(a, "wait_join", 120.0):
            self.abort(f"zgłosiło się {len(self.pinged)}/{self.needed} workerów w {a.wait_join:.0f} s "
                       f"({', '.join(self.pinged) or 'nikt'}) — sprawdź slave (IP, zapora, czy proces działa)")
        elif len(self.workers) < self.needed and elapsed > getattr(a, "wait_ready", 900.0):
            missing = sorted(set(self.pinged) - set(self.workers))
            self.abort(f"gotowych {len(self.workers)}/{self.needed} workerów po {a.wait_ready:.0f} s; "
                       f"nie skończyli startu: {', '.join(missing) or '?'}")

    def task(self, name: str) -> dict:
        a = self.args
        if self.finished:
            return {"kind": "stop", **({"error": self.error} if self.error else {})}
        if len(self.workers) < self.needed:  # start dopiero, gdy są wszyscy
            return {"kind": "wait", "seconds": 2}
        if not self.eval_pending["before"]:  # ewaluacja wag startowych; inni trenują równolegle
            self.eval_pending["before"] = True
            return {"kind": "eval", "tag": "before", "M": self.M0.tolist()}
        if (getattr(a, "world", False) and self.next_eval and self.done_episodes >= self.next_eval
                and self.next_episode < a.episodes):  # walidacja w trakcie: bieżące wagi na stałych światach
            tag = f"mid@{self.done_episodes}"
            self.next_eval += a.eval_every
            self.mid_pending[tag] = self.M.copy()
            return {"kind": "eval", "tag": tag, "M": self.M.tolist()}
        if self.next_episode < a.episodes:
            eps = []
            for _ in range(min(a.batch, a.episodes - self.next_episode)):
                # pary lustrzane ±b (jak train_decoder.py), żeby dekoder nie uczył się jednej strony
                b = (float(self.rng.uniform(-a.max_bearing, a.max_bearing)) if self.next_episode % 2 == 0
                     else -self.last_bearing)
                self.last_bearing = b
                eps.append({"index": self.next_episode,
                            "bearing": b,
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
        who = self.label(name)
        kind = msg["type"]
        if kind == "probe":  # scripts/doctor.py --master: test łącza, nie worker
            return {"kind": "pong", "workers": len(self.pinged), "needed": self.needed}
        if kind == "ping":
            mine, theirs = bool(getattr(self.args, "world", False)), bool(msg.get("world", False))
            if mine != theirs:  # np. master --world, a worker bez: macierze wag mają inny rozmiar
                flag = lambda w: "z --world" if w else "bez --world"  # noqa: E731
                err = (f"inny tryb: master {flag(mine)}, worker {flag(theirs)} — uruchom workera "
                       f"{'z' if mine else 'bez'} --world")
                print(f"[{short_gpu(msg.get('gpu', '?'))}] ODRZUCONY (komputer {name}): {err}", flush=True)
                return {"kind": "stop", "error": err}
            self.pinged[name] = short_gpu(msg.get("gpu", "?"))
            print(f"[{self.label(name)}] zgłosił się (komputer {name}), {len(self.pinged)}/{self.needed}", flush=True)
            if self.finished:
                return self.task(name)
            return {"kind": "pong"}
        if self.finished and self.error:
            return self.task(name)
        if kind == "hello":
            calib = msg["calib"]
            if self.M is not None and np.shape(msg["M"]) != self.M.shape:
                err = (f"wagi workera {np.shape(msg['M'])} ≠ mastera {self.M.shape} — inny --readout / --world "
                       "albo --init z innego treningu")
                print(f"[{who}] ODRZUCONY: {err}", flush=True)
                return {"kind": "stop", "error": err}
            if self.ref is None:
                self.ref = {k: calib[k] for k in ("yaw_axis_sign", "hover_thrust", "pitch_trim")}
                if self.M is None:
                    self.M = np.array(msg["M"])
                self.M0 = self.M.copy()
                print(f"[{who}] pierwszy worker, kalibracja: {calib}", flush=True)
            elif calib["yaw_axis_sign"] != self.ref["yaw_axis_sign"]:
                print(f"[{who}] ODRZUCONY: yaw_axis_sign {calib['yaw_axis_sign']} ≠ {self.ref['yaw_axis_sign']}",
                      flush=True)
                return {"kind": "stop", "error": "inny znak osi yaw z kalibracji niż u pierwszego workera"}
            else:
                print(f"[{who}] dołączył, kalibracja: {calib}", flush=True)
            if msg.get("init") and not self.init_from_worker and not self.args.init and not self.eval_pending["before"]:
                # wagi startowe od workera, który ma plik --world-init (np. slave go nie ma: data/ poza gitem)
                self.M, self.M0 = np.array(msg["M"]), np.array(msg["M"])
                print(f"[{who}] wagi startowe z jego pliku", flush=True)
            self.init_from_worker |= bool(msg.get("init"))
            self.workers[name] = {"episodes": 0, "since": time.time(), "gpu": self.pinged.get(name, "?")}
        elif kind == "result" and msg["kind"] == "train":
            if "stats" in msg:  # --world: DAgger, wagi od nowa z regresji na wszystkich danych
                from sim.world_decoder import WorldDecoder

                if self.wdec is None:
                    self.wdec = WorldDecoder.for_matrix(self.M)
                self.wdec.merge(msg["stats"])
                self.wdec.fit()
                self.M = self.wdec.to_matrix()
            else:
                self.M += np.array(msg["dM"])
            self.done_episodes += len(msg["metrics"])
            self.workers.setdefault(name, {"episodes": 0})["episodes"] += len(msg["metrics"])
            for m in msg["metrics"]:
                self.history.append({**m, "worker": name, "gpu": self.pinged.get(name, "?")})
                if "outcome" in m:  # --world
                    la = m.get("loss_axes") or {}
                    axes = "  ".join(f"{k} {v:.3f}" for k, v in la.items()) or f"loss {m['loss']:.4f}"
                    print(f"ep {m['index']:4d} [{who}]: świat {m['world']} → {m['outcome']:<12s} najbliżej "
                          f"{m['min_dist']:4.1f} m  beta {m['beta']:.2f}  {axes}  "
                          f"({self.done_episodes}/{self.args.episodes})", flush=True)
                    continue
                print(f"ep {m['index']:4d} [{who}]: cel {m['bearing']:+5.0f}° → {m['final_deg']:5.1f}°  "
                      f"beta {m['beta']:.2f}  loss {m['loss']:.4f}  ({self.done_episodes}/{self.args.episodes})",
                      flush=True)
        elif kind == "result" and msg["kind"] == "eval":
            tag, ev = msg["tag"], msg["eval"]
            M_eval = self.M0 if tag == "before" else self.mid_pending.pop(tag, None) if tag.startswith("mid") else self.M
            self.consider_best(tag, ev, M_eval)
            if tag.startswith("mid"):
                show(f"[{who}] po {tag[4:]:>4s} epizodach", ev)
                return self.task(name)
            setattr(self, tag, ev)
            show(f"[{who}] {'przed treningiem' if tag == 'before' else 'po treningu    '}", ev)
            if msg["tag"] == "after":
                self.save()
                self.finished = True
        return self.task(name)

    def consider_best(self, tag: str, ev: dict, M) -> None:
        """Zapamiętuje wagi z najlepszą ewaluacją: najpierw liczba celów, potem średnio najbliżej (--world)."""
        if "episodes" not in ev or M is None:
            return
        done = 0 if tag == "before" else int(tag[4:]) if tag.startswith("mid") else self.done_episodes
        self.evals.append({"tag": tag, "after_episodes": done, "reached": ev["reached"], "n": ev["n"],
                           "mean_min_dist": ev["mean_min_dist"]})
        score = (ev["reached"], -ev["mean_min_dist"])
        if self.best is None or score > self.best["score"]:
            self.best = {"score": score, "tag": tag, "M": np.array(M), "reached": ev["reached"],
                         "mean_min_dist": ev["mean_min_dist"]}
            print(f"  nowe najlepsze wagi ({tag}): cel {ev['reached']}/{ev['n']}, średnio najbliżej "
                  f"{ev['mean_min_dist']:.1f} m", flush=True)

    def save(self) -> None:
        a = self.args
        name = "world_distributed.npz" if getattr(a, "world", False) else f"plan{a.plan}_distributed.npz"
        out = a.out or ROOT / "data" / "decoders" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"plan": a.plan, "episodes": a.episodes, "workers": list(self.workers),
                "before": self.before.get("reached", self.before["mean_final_deg"]),
                "after": self.after.get("reached", self.after["mean_final_deg"])}
        if getattr(a, "world", False):  # plik WorldDecoder (sim.run_env --banc, sim.banc_pilot)
            from sim.world_decoder import WorldDecoder

            WorldDecoder.for_matrix(self.M).save(out, **meta, samples=self.wdec.n if self.wdec else 0)
            if self.best is not None:  # końcowe wagi mogą być gorsze niż któraś walidacja w trakcie
                best = out.with_name(out.stem + "_best.npz")
                WorldDecoder.for_matrix(self.best["M"]).save(best, **meta, best_from=self.best["tag"])
                print(f"najlepsze wagi ({self.best['tag']}: cel {self.best['reached']}, średnio najbliżej "
                      f"{self.best['mean_min_dist']:.1f} m) → {best}", flush=True)
        else:
            np.savez(out, M=self.M, hover_thrust=self.ref["hover_thrust"], pitch_trim=self.ref["pitch_trim"],
                     meta=np.array(repr(meta)))
        out.with_suffix(".json").write_text(json.dumps(
            {"args": vars(a), "workers": self.workers, "before": self.before, "after": self.after,
             "evals": self.evals, "best": {k: v for k, v in (self.best or {}).items() if k not in ("M", "score")},
             "M_before": self.M0.tolist(), "M_after": self.M.tolist(), "history": self.history},
            default=str, indent=1))
        show("przed treningiem", self.before)
        show("po treningu    ", self.after)
        import subprocess  # wykresy przebiegu (matplotlib jest w środowisku workerów; brak → tylko komunikat)

        subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_training.py"), str(out.with_suffix(".json"))],
                       check=False)
        print(f"zapisano {out}; workerzy: {self.workers}", flush=True)


def run_master(args) -> None:
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    ips = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    print(f"master na tcp://*:{args.port}, adresy: {', '.join(ips)}; czekam na workerów", flush=True)
    master = Master(args)
    print(f"potrzeba {master.needed} workerów: zgłoszenie w {args.wait_join:.0f} s, gotowość w {args.wait_ready:.0f} s",
          flush=True)
    idle_since = None
    t0 = time.time()
    while True:
        master.check_deadlines(time.time() - t0)
        if master.error and idle_since is None:
            idle_since = time.time()
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
    if master.error:
        sys.exit(1)


# --- worker ---------------------------------------------------------------------------------------
def run_worker(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from sim.env import summarize
    from train_decoder import Runner, build, sweep_fit
    from train_world import EVAL_WORLDS, TRAIN_SEED_OFFSET, default_yaw_init, setup

    host = args.name or socket.gethostname()  # identyfikator u mastera
    import torch

    name = short_gpu(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else f"CPU {host}"  # etykieta w logu

    if torch.cuda.is_available():
        print(f"[{name}] GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__})", flush=True)
    elif not args.cpu:  # BANC i FlyVis po cichu przeszłyby na CPU (~10× wolniej)
        sys.exit(f"[{name}] brak CUDA w torch {torch.__version__} (wersja CPU?). Zainstaluj wersję z CUDA:\n"
                 "    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124\n"
                 "albo uruchom z --cpu.")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    try:  # literówka typu "127.0.0." — ZMQ czekałby po cichu
        socket.getaddrinfo(args.host, args.port)
        if args.host.count(".") == 3 and not all(p.isdigit() for p in args.host.split(".")):
            raise OSError
    except OSError:
        sys.exit(f"[{name}] zły adres mastera --host {args.host!r} (np. 127.0.0.1 albo 192.168.50.1)")
    print(f"[{name}] łączę z masterem {args.host}:{args.port}…", flush=True)
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 15_000)  # zgłoszenie: master odpowiada od razu
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{args.host}:{args.port}")

    def send(msg: dict) -> dict:
        sock.send(json.dumps(msg, default=float).encode())
        try:
            return json.loads(sock.recv())
        except zmq.Again:
            sys.exit(f"[{name}] master {args.host}:{args.port} nie odpowiada — przerywam "
                     "(sprawdź adres, czy master działa i zaporę na porcie)")

    # zgłoszenie przed wczytaniem BANC: master od razu wie, czy ten komputer jest osiągalny
    reply = send({"type": "ping", "name": host, "gpu": gpu, "world": args.world})
    if reply["kind"] == "stop":
        sys.exit(f"[{name}] master przerwał: {reply.get('error', '')}")
    print(f"[{name}] połączony z masterem {args.host}:{args.port}, ładuję BANC…", flush=True)
    sock.setsockopt(zmq.RCVTIMEO, 60_000)

    t0 = time.perf_counter()
    init = None
    if args.world:  # lot do celu w WorldEnv (train_world.py); wagi startowe z --world-init
        init = args.world_init if args.world_init and args.world_init.exists() else default_yaw_init()
        print(f"[{name}] yaw na start: {init or 'brak pliku — od zera (master weźmie wagi od innego workera)'}",
              flush=True)
        env, pilot, runner, calib = setup(init, args.beacon_scale, args.max_time, args.beacon_alpha,
                                          args.sensors, args.motor_tau, args.vision_range, args.beacon_color)
        ctrl, dec = pilot.ctrl, pilot.ctrl.decoder
        wdec = pilot.world
    else:
        env, bridge, ctrl, calib = build(args.plan, args.lr, args.noise, args.reward_lr, args.thrust,
                                         args.readout, args.visual_gain)
        dec = ctrl.decoder
    if not args.world and not args.no_sweep:  # master bierze wagi startowe od pierwszego workera (chyba że --init)
        fit = sweep_fit(env, bridge, ctrl)
        print(f"[{name}] start z regresji: korelacja yaw {fit['r']:+.2f}, trafność strony {fit['side_acc']:.0%}",
              flush=True)
    print(f"[{name}] gotowy po {time.perf_counter() - t0:.0f} s, kalibracja: "
          f"{ {k: v for k, v in calib.items() if k not in ('ok', 'scene')} }", flush=True)
    if not args.world:
        runner = Runner(env, bridge, ctrl)

    M_start = wdec.to_matrix() if args.world else dec.M
    task = send({"type": "hello", "name": host, "M": M_start.tolist(), "init": init is not None,
                 "calib": {"yaw_axis_sign": ctrl.yaw_axis_sign, "haltere_gain": ctrl.haltere_gain,
                           "hover_thrust": dec.hover_thrust, "pitch_trim": dec.pitch_trim}})
    while task["kind"] != "stop":
        if task["kind"] == "wait":
            time.sleep(task["seconds"])
            task = send({"type": "poll", "name": host})
            continue
        M = np.array(task["M"])
        if M.shape != M_start.shape:  # też przy starym masterze bez kontroli trybu
            sys.exit(f"[{name}] wagi od mastera {M.shape} ≠ moje {M_start.shape}: inny tryb treningu — "
                     f"uruchom workera {'bez' if args.world else 'z'} --world (tak jak master)")
        if args.world:
            wdec.from_matrix(M)
            wdec.reset_stats()  # do mastera idą tylko dane z tego zadania
        else:
            dec.M = M.copy()
        if task["kind"] == "eval":
            print(f"[{name}] ewaluacja {task['tag']}…", flush=True)
            ev = (runner.evaluate(EVAL_WORLDS, log=lambda m: print(f"[{name}]{m}", flush=True)) if args.world
                  else runner.evaluate(args.eval_duration))
            task = send({"type": "result", "kind": "eval", "name": host, "tag": task["tag"], "eval": ev})
            continue
        metrics = []
        for ep in task["episodes"]:
            t = time.perf_counter()
            if args.world:
                m = runner.episode(TRAIN_SEED_OFFSET + ep["seed"] % 1_000_000, learn=True, beta=ep["beta"],
                                   rng=np.random.default_rng(ep["seed"]), start_noise=True)
                metrics.append({**ep, **m})
                print(f"[{name}] ep {ep['index']}: świat {m['world']} → {m['outcome']} najbliżej {m['min_dist']:.1f} m "
                      f"({time.perf_counter() - t:.0f} s)", flush=True)
                continue
            log, loss = runner.episode(ep["bearing"], args.duration, learn=args.plan, beta=ep["beta"],
                                       rng=np.random.default_rng(ep["seed"]))
            m = summarize(log, env.fps)
            metrics.append({**ep, "loss": loss, **m})
            print(f"[{name}] ep {ep['index']}: cel {ep['bearing']:+5.0f}° → {m['final_deg']:5.1f}° "
                  f"({time.perf_counter() - t:.0f} s)", flush=True)
        if args.world:  # statystyki DAgger zamiast zmiany wag; master liczy regresję
            task = send({"type": "result", "kind": "train", "name": host, "stats": wdec.stats(as_lists=True),
                         "metrics": metrics})
        else:
            task = send({"type": "result", "kind": "train", "name": host,
                         "dM": (dec.M - M).tolist(), "metrics": metrics})
    print(f"[{name}] koniec{': ' + task['error'] if task.get('error') else ''}", flush=True)
    if task.get("error"):
        sys.exit(1)


# --- wspólne --------------------------------------------------------------------------------------
def show(tag: str, ev: dict) -> None:
    if "episodes" in ev:  # --world (jak sim.banc_pilot.show_world; master nie importuje MuJoCo)
        per = "  ".join(f"{e['world']}:{'CEL' if e['reached'] else e['outcome'].split(':')[0]}({e['min_dist']:.0f}m)"
                        for e in ev["episodes"])
        print(f"{tag}: cel {ev['reached']}/{ev['n']}, średnio najbliżej {ev['mean_min_dist']:.1f} m  ({per})", flush=True)
        return
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
    ap.add_argument("--world", action="store_true", help="lot do celu w świecie Osoby 3 (master i workerzy)")
    ap.add_argument("--eval-every", type=int, default=50,
                    help="--world, master: walidacja co tyle epizodów i zapis najlepszych wag (0 = tylko przed/po)")
    ap.add_argument("--world-init", type=Path, default=None,
                    help="worker --world: dekoder zawisu do startu yaw (domyślnie planB_distributed / planB_dn)")
    ap.add_argument("--beacon-scale", type=float, default=1.0, help="worker --world: rozmiar znacznika celu")
    ap.add_argument("--beacon-alpha", type=float, default=None, help="worker --world: przezroczystość znacznika (1 = pełny)")
    ap.add_argument("--sensors", choices=("real", "ideal"), default="real", help="worker --world: czujniki drona")
    ap.add_argument("--beacon-color", choices=("scene", "dark-red"), default="dark-red",
                    help="kolor celu: ciemnoczerwony (kontrast jasności dla FlyVis) albo ze sceny (pomarańczowy)")
    ap.add_argument("--motor-tau", type=float, default=0.04, help="worker --world: opóźnienie silników [s]")
    ap.add_argument("--vision-range", type=float, default=float("inf"), help="worker --world: bliżej yaw z BANC, dalej z GPS [m]")
    ap.add_argument("--max-time", type=float, default=40.0, help="worker --world: s na epizod")
    ap.add_argument("--workers", type=int, default=2, help="ilu workerów musi się zgłosić (inaczej przerwanie)")
    ap.add_argument("--wait-join", type=float, default=120.0, help="s na zgłoszenie się wszystkich workerów")
    ap.add_argument("--wait-ready", type=float, default=900.0, help="s na wczytanie BANC i kalibrację u wszystkich")
    ap.add_argument("--out", type=Path)
    # worker
    ap.add_argument("--host", default="127.0.0.1", help="IP mastera")
    ap.add_argument("--name", help="nazwa workera w logu (domyślnie nazwa komputera)")
    ap.add_argument("--cpu", action="store_true", help="pozwól liczyć bez GPU")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--eval-duration", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=0.5, help="średni krok LMS po epizodzie")
    ap.add_argument("--readout", choices=("mn", "dn"), default="dn", help="musi być ten sam na wszystkich workerach")
    ap.add_argument("--visual-gain", type=float, default=1.0)
    ap.add_argument("--no-sweep", action="store_true", help="bez startu z regresji na statycznych scenach")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--reward-lr", type=float, default=0.05)
    ap.add_argument("--thrust", choices=("banc", "hold"), default="hold")
    args = ap.parse_args()
    run_master(args) if args.role == "master" else run_worker(args)


if __name__ == "__main__":
    main()
