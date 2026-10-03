"""Trening dekodera MN → komendy (Plan B / Plan A) w pętli zamkniętej z MuJoCo. BANC się nie zmienia.

    python scripts/train_decoder.py --plan B [--episodes 40] [--out data/decoders/planB.npz]
    python scripts/train_decoder.py --plan A --init data/decoders/planB.npz

Wszystko w jednym procesie (.venv312: torch, FlyVis, mujoco): DroneEnv → VisionBridge → BancController.
Uczy się wyłącznie macierz ``decoder.M`` (4 × 6: thrust/roll/pitch/yaw z 6 grup MN skrzydeł BANC).

- Plan B (``LinearDecoder``, znormalizowany LMS): cel = komenda „nauczyciela” z UPRZYWILEJOWANEGO
  stanu symulatora (yaw ∝ kąt do celu, thrust trzyma wysokość, roll/pitch 0). Na początku dron leci
  komendami nauczyciela, potem coraz częściej własnymi (DAgger), żeby uczyć się na stanach, które sam odwiedza.
- Plan A (``AdaptiveDecoder``, node perturbation): szum na wyjściu dekodera, nagroda = postęp w kierunku
  celu w tej klatce (spadek |kąta|) minus kara za wysokość.

Przed i po treningu ewaluacja na stałych kątach celu (bez szumu); wynik i wagi → ``--out``
(+ JSON z przebiegiem obok). Wczytanie wag: ``fly_banc.py --local --decoder``, ``vision_server.py --decoder``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from banc_control import BancController, Connectome, FlightCommand, ImuState  # noqa: E402
from banc_control.readout import AdaptiveDecoder, LinearDecoder  # noqa: E402
from sim.env import DroneEnv, Scenario, summarize  # noqa: E402
from visual_pipeline import VisionBridge  # noqa: E402
from visual_pipeline.server import ControlServer, LocalClient  # noqa: E402

EVAL_BEARINGS = (-60.0, -30.0, 30.0, 60.0)


def teacher(info: dict, start_z: float, k_yaw: float = 1.5, k_z: float = 1.0) -> FlightCommand:
    """Komenda wzorcowa z prawdziwego stanu (tylko do Planu B, sieć jej nie widzi)."""
    return FlightCommand(thrust=0.5 + float(np.clip(k_z * (start_z - info["z"]), -0.3, 0.3)),
                         yaw=float(np.clip(k_yaw * info["bearing"], -1.0, 1.0)))


def build(plan: str, lr: float = 0.05, noise: float = 0.1, reward_lr: float = 0.05, thrust: str = "hold"):
    """DroneEnv + VisionBridge + BancController z dekoderem Planu ``plan``, skalibrowane na scenach
    MuJoCo dokładnie jak przez ZMQ (te same sceny, ten sam ControlServer). → (env, bridge, ctrl, calib)."""
    decoder = LinearDecoder(lr=lr) if plan == "B" else AdaptiveDecoder(noise=noise, reward_lr=reward_lr)
    ctrl = BancController(Connectome.from_banc(), decoder=decoder)
    bridge = VisionBridge(fps=30, fisheye=True)
    env = DroneEnv(thrust_mode=thrust)
    if plan == "A":
        decoder.noise = 0.0  # kalibracja bez szumu eksploracji
    calib = LocalClient(ControlServer(bridge, ctrl)).calibrate(env.calibration_render)
    if plan == "A":
        decoder.noise = noise
    return env, bridge, ctrl, calib


def beta_schedule(episode: int, total: int, plan: str) -> float:
    """Udział nauczyciela (DAgger) w Planie B: od 1 do 0 w pierwszej połowie treningu."""
    return max(0.0, 1.0 - episode / max(1, total // 2)) if plan == "B" else 0.0


class Runner:
    def __init__(self, env: DroneEnv, bridge: VisionBridge, ctrl: BancController) -> None:
        self.env, self.bridge, self.ctrl = env, bridge, ctrl

    def episode(self, bearing_deg: float, duration: float, learn: str | None = None, beta: float = 0.0,
                rng: np.random.Generator | None = None) -> tuple[list[dict], float]:
        """Jeden epizod. ``learn``: None (ewaluacja) / "B" / "A". ``beta``: udział nauczyciela (Plan B)."""
        env, ctrl, dec = self.env, self.ctrl, self.ctrl.decoder
        obs, info = env.reset(Scenario("train", bearing_deg, duration), rng=rng)
        self.bridge.reset()
        ctrl.dyn.reset()
        ctrl.warm_start(self.bridge.settle(obs.left, obs.right))  # bez skoku szare tło → scena
        log, losses, done = [info], [], False
        max_turn = env.pilot.max_yaw_rate / env.fps
        while not done:
            visual = self.bridge.step_batch(obs.left, obs.right)
            cmd = ctrl.step(visual, ImuState(**{k: tuple(v) for k, v in obs.imu.items()}))
            cmd.roll = 0.0  # wariant angle: roll z BANC i tak nie idzie do drona
            act = cmd
            if learn == "B":
                target = teacher(info, env.start_z)
                losses.append(dec.fit_step(cmd.debug["motor_features"], target))
                if rng.random() < beta:
                    act = target
            prev = abs(info["bearing"])
            obs, _, terminated, truncated, info = env.step(act)
            if learn == "A":
                progress = np.clip((prev - abs(info["bearing"])) / max_turn, -1.0, 1.0)
                dec.reward(float(progress) - env.alt_weight * abs(info["z"] - env.start_z))
            log.append(info)
            done = terminated or truncated
        return log, float(np.mean(losses)) if losses else float("nan")

    def evaluate(self, duration: float) -> dict:
        dec = self.ctrl.decoder
        noise = getattr(dec, "noise", None)
        if noise is not None:
            dec.noise = 0.0
        out = {}
        for b in EVAL_BEARINGS:
            log, _ = self.episode(b, duration)
            m = summarize(log, self.env.fps)
            yaw0 = np.mean([i["cmd"]["yaw"] for i in log[1:16]])  # pierwsze 0.5 s
            m["turns_toward"] = bool(np.sign(yaw0) == np.sign(b))
            out[f"{b:+.0f}"] = m
        if noise is not None:
            dec.noise = noise
        out["mean_final_deg"] = float(np.mean([out[f"{b:+.0f}"]["final_deg"] for b in EVAL_BEARINGS]))
        out["turns_toward"] = int(sum(out[f"{b:+.0f}"]["turns_toward"] for b in EVAL_BEARINGS))
        return out


def show(tag: str, ev: dict) -> None:
    per = "  ".join(f"{b:+.0f}°→{ev[f'{b:+.0f}']['final_deg']:.0f}°" for b in EVAL_BEARINGS)
    print(f"{tag}: średni końcowy |kąt| {ev['mean_final_deg']:.1f}°, skręt w stronę celu "
          f"{ev['turns_toward']}/{len(EVAL_BEARINGS)}  ({per})", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: przekierowane stdout jest w cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=("A", "B"), default="B")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--duration", type=float, default=5.0, help="s na epizod treningowy")
    ap.add_argument("--eval-duration", type=float, default=6.0)
    ap.add_argument("--max-bearing", type=float, default=90.0, help="losowy cel w ±tyle stopni")
    ap.add_argument("--lr", type=float, default=0.05, help="Plan B: krok LMS")
    ap.add_argument("--noise", type=float, default=0.1, help="Plan A: szum na wyjściu")
    ap.add_argument("--reward-lr", type=float, default=0.05, help="Plan A")
    ap.add_argument("--init", type=Path, help="start z wag z wcześniejszego treningu (np. Plan B → A)")
    ap.add_argument("--thrust", choices=("banc", "hold"), default="hold",
                    help="hold (domyślnie): wysokość trzyma symulator, uczymy głównie yaw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or ROOT / "data" / "decoders" / f"plan{args.plan}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    env, bridge, ctrl, calib = build(args.plan, args.lr, args.noise, args.reward_lr, args.thrust)
    decoder = ctrl.decoder
    print(f"start + kalibracja {time.perf_counter() - t0:.0f} s:",
          {k: v for k, v in calib.items() if k not in ("ok", "scene")}, flush=True)
    if args.init:
        decoder.load_weights(args.init)
        print(f"wagi startowe: {args.init}")
    M0 = decoder.M.copy()

    runner = Runner(env, bridge, ctrl)
    before = runner.evaluate(args.eval_duration)
    show("przed treningiem", before)

    history = []
    for ep in range(args.episodes):
        t = time.perf_counter()
        bearing = float(rng.uniform(-args.max_bearing, args.max_bearing))
        beta = beta_schedule(ep, args.episodes, args.plan)
        log, loss = runner.episode(bearing, args.duration, learn=args.plan, beta=beta, rng=rng)
        m = summarize(log, env.fps)
        history.append({"episode": ep, "bearing": bearing, "beta": beta, "loss": loss, **m,
                        "M": decoder.M.tolist()})
        print(f"ep {ep:3d}: cel {bearing:+5.0f}° → {m['final_deg']:5.1f}°  beta {beta:.2f}  "
              f"loss {loss:.4f}  z {m['z_min']:.2f}–{m['z_max']:.2f}  ({time.perf_counter() - t:.0f} s)",
              flush=True)

    after = runner.evaluate(args.eval_duration)
    show("przed treningiem", before)
    show("po treningu    ", after)
    np.set_printoptions(precision=3, suppress=True)
    print("M przed:\n", M0, "\nM po:\n", decoder.M)

    meta = {"plan": args.plan, "episodes": args.episodes, "before": before["mean_final_deg"],
            "after": after["mean_final_deg"]}
    decoder.save(out, **meta)
    out.with_suffix(".json").write_text(json.dumps(
        {"args": vars(args), "calibration": calib, "before": before, "after": after,
         "M_before": M0.tolist(), "M_after": decoder.M.tolist(), "history": history},
        default=str, indent=1))
    print(f"zapisano {out} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
