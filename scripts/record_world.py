"""Nagrywa lot drona z BANC w świecie Osoby 3 (WorldEnv): kamera za dronem, widok z góry, oczy, telemetria.

    .venv312\\Scripts\\python scripts/record_world.py data/decoders/world_nostab_best.npz --seed 101 \\
        --obstacles path --banc-axes thrust roll pitch --out data/videos/clips/nostab_101

Wynik: <out>_chase.mp4 (1280×720, kamera krąży za dronem), <out>_top.mp4 (widok z góry na start i cel),
<out>_eyes.mp4 (oba oczy), <out>.json (wynik epizodu i telemetria co klatkę). Ustawienia jak w treningu
(train_distributed --world): znacznik ×4, GPS dalej niż 4 m, czujniki z szumem, opóźnienie silników 40 ms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

W, H = 1280, 720


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decoder", type=Path)
    ap.add_argument("--seed", type=int, required=True, help="ziarno świata")
    ap.add_argument("--obstacles", choices=("clear", "path", "keep"), default="clear")
    ap.add_argument("--path-blocks", type=int, default=2)
    ap.add_argument("--banc-axes", nargs="*", choices=("thrust", "roll", "pitch"), default=[])
    ap.add_argument("--vision-range", type=float, default=4.0)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--out", type=Path, required=True, help="prefiks plików wyjściowych")
    a = ap.parse_args()

    from sim.banc_pilot import BancPilot, WorldRunner, bearing, ground_z
    from sim.propellers import PropellerVisuals
    from sim.trail import Trail
    from sim.world_env import WorldEnv

    pilot = BancPilot(a.decoder, brain=False, beacon_scale=4.0, vision_range=a.vision_range,
                      beacon_color="dark-red", banc_axes=a.banc_axes)
    env = WorldEnv(control="angle", start_noise=False, max_time=a.max_time, sensors="real", motor_tau=0.04)
    env.reset(seed=0)
    pilot.bind(env)
    runner = WorldRunner(env, pilot, obstacles=a.obstacles, path_blocks=a.path_blocks)
    obs, info = runner.reset(a.seed)

    m = env.model
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    rend = mujoco.Renderer(m, H, W)
    props, trail = PropellerVisuals(m), Trail()
    chase = mujoco.MjvCamera()
    chase.type, chase.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, env.drone_id
    chase.distance, chase.elevation = 2.2, -18.0
    top = mujoco.MjvCamera()
    top.type = mujoco.mjtCamera.mjCAMERA_FREE
    start = env.data.xpos[env.drone_id].copy()
    goal = np.asarray(info["target"], float) if "target" in info else env.target.position(env.data)
    top.lookat[:] = (start + goal) / 2
    top.lookat[2] = 0.0
    top.distance = float(np.linalg.norm(goal[:2] - start[:2])) * 1.25 + 6
    top.elevation = -58.0
    top.azimuth = float(np.degrees(np.arctan2(goal[1] - start[1], goal[0] - start[0]))) - 90

    a.out.parent.mkdir(parents=True, exist_ok=True)
    w_chase = imageio.get_writer(f"{a.out}_chase.mp4", fps=30, codec="libx264", quality=8, macro_block_size=8)
    w_top = imageio.get_writer(f"{a.out}_top.mp4", fps=30, codec="libx264", quality=8, macro_block_size=8)
    w_eyes = imageio.get_writer(f"{a.out}_eyes.mp4", fps=30, codec="libx264", quality=8, macro_block_size=8)
    every = max(1, round((1 / 30) / env.dt))  # klatka wideo co tyle kroków symulacji
    tel, step, done, hits, touching, dmin = [], 0, False, 0, False, info["distance"]
    heading0 = None
    while not done:
        cmd = pilot.decide(obs)
        obs, _, term, trunc, info = env.step(pilot.action(cmd))
        done = term or trunc
        now = runner.touching_block()
        hits += now and not touching
        touching = now
        dmin = min(dmin, info["distance"])
        props.advance(env.data, env.dt)
        trail.add(env.data.xpos[env.drone_id].copy())
        step += 1
        if step % every and not done:
            continue
        t = float(env.data.time)
        nose = env.data.xmat[env.drone_id].reshape(3, 3)[:, 0]
        heading = float(np.degrees(np.arctan2(nose[1], nose[0])))
        heading0 = heading if heading0 is None else heading0 + ((heading - heading0 + 180) % 360 - 180) * 0.08
        chase.azimuth = heading0 + 25 * np.sin(t * 0.25)  # za dronem, powoli krąży na boki
        rend.update_scene(env.data, chase)
        props.draw(rend.scene, env.data)
        trail.draw(rend.scene)
        w_chase.append_data(rend.render())
        rend.update_scene(env.data, top)
        trail.draw(rend.scene)
        w_top.append_data(rend.render())
        eyes = obs["eyes"]
        w_eyes.append_data(np.concatenate([eyes[0], eyes[1]], axis=1))
        dbg = getattr(cmd, "debug", {}) or {}
        tel.append({"t": t, "dist": float(info["distance"]), "bearing_deg": float(np.degrees(bearing(env))),
                    "height": float(env.data.xpos[env.drone_id][2] - ground_z(env)),
                    "thrust": float(cmd.thrust), "roll": float(cmd.roll), "pitch": float(cmd.pitch),
                    "yaw": float(cmd.yaw), "w_vis": float(dbg.get("w_vis", np.nan)) if "w_vis" in dbg else None,
                    "touching": bool(now)})
    for w in (w_chase, w_top, w_eyes):
        w.close()
    rend.close()
    res = {"decoder": str(a.decoder), "seed": a.seed, "obstacles": a.obstacles, "banc_axes": a.banc_axes,
           "outcome": info["outcome"], "reached": info["outcome"] == "cel", "time": float(info["time"]),
           "start_dist": float(tel[0]["dist"]) if tel else None, "min_dist": float(dmin), "block_hits": int(hits),
           "telemetry": tel}
    Path(f"{a.out}.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"{a.out.name}: {info['outcome']} po {info['time']:.1f} s, najbliżej {dmin:.1f} m, zderzenia {hits}")


if __name__ == "__main__":
    main()
