"""Pętla zamknięta w MuJoCo: DroneEnv (Osoba 3) ↔ wzrok + BANC (Osoby 1 i 2), scenariusze z metrykami.

    python scripts/vision_server.py                       # środowisko wzroku (.venv312), osobny terminal
    python scripts/fly_banc.py [--scenario turn_right gust] [--video demo.mp4]

    python scripts/fly_banc.py --local                    # wszystko w jednym procesie (.venv312)

Na starcie kalibracja kontrolera na scenach z tego samego symulatora (``VisionClient.calibrate``),
potem każdy scenariusz z ``sim.env.SCENARIOS``. Wariant demo Planu C: ``--mode angle`` (domyślnie) —
symulator trzyma poziom, BANC daje yaw i thrust (jako wznoszenie), roll z BANC ignorowany.
Wyniki: metryki na ekranie i ``data/runs/<czas>/`` (JSON z przebiegiem, opcjonalnie MP4).
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
from sim.env import SCENARIOS, DroneEnv, summarize  # noqa: E402
from visual_pipeline.zmq_protocol import DEFAULT_ADDRESS, VisionClient  # noqa: E402


def local_client(decoder: Path | None = None):
    """Serwer wzroku + BancController w tym procesie (wymaga torch/FlyVis i danych BANC).
    Odczyt (6 MN / + pojedyncze DN) wynika z rozmiaru wag ``decoder`` (``train_decoder.py --readout``)."""
    from banc_control import BancController, Connectome
    from visual_pipeline import VisionBridge
    from visual_pipeline.server import ControlServer, LocalClient

    readout = "dn" if decoder and np.load(decoder)["M"].shape[1] > 6 else "mn"
    ctrl = BancController(Connectome.from_banc(), readout=readout)
    return LocalClient(ControlServer(VisionBridge(fps=30, fisheye=True), ctrl)), ctrl


def video_frame(env: DroneEnv, obs, info: dict) -> np.ndarray:
    """Kadr: widok zza drona + oba oczy (pomniejszone) pod spodem."""
    chase = env.render_chase()
    eyes = np.concatenate([obs.left[::2, ::2], obs.right[::2, ::2]], axis=1)  # 256 × 450
    h = chase.shape[1] * eyes.shape[0] // eyes.shape[1]
    idx = np.linspace(0, eyes.shape[0] - 1, h).astype(int)
    jdx = np.linspace(0, eyes.shape[1] - 1, chase.shape[1]).astype(int)
    return np.concatenate([chase, eyes[idx][:, jdx]], axis=0)


def brain_frame(frame: np.ndarray, view, ctrl, reply: dict, scenario: str, info: dict) -> np.ndarray:
    """Kadr wideo + panel BANC (``sim.brain_panel.BrainView``) po prawej, podpis scenariusza na górze."""
    import cv2
    from types import SimpleNamespace

    panel = view.render(ctrl.dyn.rates_at(np.arange(ctrl.c.n)), SimpleNamespace(**reply))
    h = frame.shape[0]
    panel = cv2.resize(panel, (round(panel.shape[1] * h / panel.shape[0]), h), interpolation=cv2.INTER_AREA)
    out = np.concatenate([frame, panel], axis=1)
    H, W = -(-out.shape[0] // 16) * 16, out.shape[1] // 16 * 16  # wymiary podzielne przez 16 dla H.264
    out = np.ascontiguousarray(np.pad(out, ((0, H - out.shape[0]), (0, 0), (0, 0)))[:, :W])
    text = f"{scenario}   t {info['t']:4.1f} s   cel {np.rad2deg(info['bearing']):+5.0f} st"
    cv2.rectangle(out, (0, 0), (360, 34), (20, 24, 30), -1)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_episode(env: DroneEnv, client: VisionClient, scenario: str, frames: list | None = None,
                yaw_only: bool = True, brain=None) -> tuple[list[dict], dict]:
    """``brain``: (BrainView, BancController) — panel sieci w kadrze (tylko ``--local``)."""
    obs, info = env.reset(scenario)
    if brain is not None:
        brain[0].reset()
    log, k, done = [info], 0, False
    while not done:
        reply = client.step(obs.left, obs.right, obs.imu, reset=(k == 0))
        if yaw_only:
            reply["roll"] = 0.0
        obs, reward, terminated, truncated, info = env.step(reply)
        info["reward"], info["server_ms"] = reward, reply.get("timing_ms", {}).get("total")
        log.append(info)
        if frames is not None:
            frame = video_frame(env, obs, info)
            frames.append(brain_frame(frame, *brain, reply, scenario, info) if brain is not None else frame)
        if k % 30 == 0:
            c = info["cmd"]
            print(f"  t={info['t']:4.1f} s  cel {np.rad2deg(info['bearing']):+6.1f}°  z {info['z']:.2f} m  "
                  f"thrust {c['thrust']:.2f} yaw {c['yaw']:+.2f}", flush=True)
        k += 1
        done = terminated or truncated
    if terminated:
        print(f"  przerwany: z {info['z']:.2f} m, roll {np.rad2deg(info['roll']):+.0f}°")
    return log, summarize(log, env.fps)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: przekierowane stdout jest w cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default=DEFAULT_ADDRESS)
    ap.add_argument("--local", action="store_true", help="wzrok + BANC w tym procesie zamiast ZMQ")
    ap.add_argument("--decoder", type=Path, help="wagi dekodera z train_decoder.py (tylko z --local)")
    ap.add_argument("--scenario", nargs="+", default=["hover", "turn_right", "turn_left", "gust"],
                    choices=sorted(SCENARIOS))
    ap.add_argument("--mode", choices=("angle", "acro"), default="angle")
    ap.add_argument("--thrust", choices=("banc", "hold"), default="banc",
                    help="hold: wysokość trzyma symulator (BANC steruje tylko yaw)")
    ap.add_argument("--use-roll", action="store_true", help="roll z BANC → ruch w bok (tylko angle)")
    ap.add_argument("--max-yaw-rate", type=float, default=1.0, help="rad/s dla yaw = ±1")
    ap.add_argument("--video", type=Path, help="zapisz MP4 (widok zza drona + oczy)")
    ap.add_argument("--brain", action="store_true", help="w wideo panel z aktywnością BANC (tylko z --local)")
    ap.add_argument("--no-calib", action="store_true", help="zostaw kalibrację, którą ma serwer")
    args = ap.parse_args()

    env = DroneEnv(mode=args.mode, use_roll=args.use_roll, max_yaw_rate=args.max_yaw_rate,
                   thrust_mode=args.thrust)
    if args.local:
        client, ctrl = local_client(args.decoder)
    else:
        if args.decoder:
            ap.error("--decoder działa z --local; serwer ZMQ: scripts/vision_server.py --decoder")
        client, ctrl = VisionClient(args.address, timeout_ms=60_000), None

    if not args.no_calib:
        t = time.perf_counter()
        print("kalibracja:", client.calibrate(env.calibration_render), f"({time.perf_counter() - t:.0f} s)",
              flush=True)
    if args.decoder:
        ctrl.decoder.load_weights(args.decoder)
        print(f"dekoder: {args.decoder}")

    out = ROOT / "data" / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    frames = [] if args.video else None
    brain = None
    if args.brain:
        if ctrl is None:
            ap.error("--brain działa z --local")
        from sim.brain_panel import BrainView

        brain = (BrainView(ctrl.c), ctrl)
    results = {}
    for name in args.scenario:
        print(f"scenariusz {name}:", flush=True)
        log, metrics = run_episode(env, client, name, frames, yaw_only=args.mode == "angle" and not args.use_roll,
                                   brain=brain)
        results[name] = metrics
        (out / f"{name}.json").write_text(json.dumps({"metrics": metrics, "log": log}, default=float))
        print("  " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in metrics.items()), flush=True)
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "results": results}, default=str, indent=2))
    if frames:
        import imageio.v3 as iio

        iio.imwrite(args.video, np.stack(frames), fps=env.fps, codec="libx264")
        print(f"wideo: {args.video}")
    print(f"wyniki: {out}")


if __name__ == "__main__":
    main()
