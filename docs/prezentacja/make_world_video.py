"""Film o symulacji 3D i treningu modeli (1920×1080, 30 fps) + scenariusz lektora (EN) i napisy SRT.

    .venv312\\Scripts\\python docs/prezentacja/make_world_video.py [--still 5 20 ...]

Nagrania lotów: scripts/record_world.py → data/videos/clips/<nazwa>_{chase,top,eyes}.mp4 + .json.
Klipy do scen wybiera CLIPS (niżej). Tekst lektora i długości scen: SCENES; z nich powstaje
skrypt_world_en.md i .srt (te same czasy co w filmie).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from make_pipeline_video import (BG, CARD, EDGE, F_BODY, F_CAP, F_EYEBROW, F_HEAD, F_HEAD_BIG, FPS, GREY, H,  # noqa: E402
                                 MUTED, RED, ROOT, VIDEOS, WHITE, W, fit, font, shadowed_paste, spaced, ts, wrap)

CLIPS_DIR = VIDEOS / "clips"
DECODERS = ROOT / "data" / "decoders"
F_CARD = font("Michroma-Regular.ttf", 26)
F_SMALL = font("Roboto-Regular.ttf", 22)
F_NUM = font("Roboto-Light.ttf", 64)

# nagrania wybrane do scen (data/videos/clips/<nazwa>_*.mp4)
CLIPS = {"homing": "homing_101", "before": "before_113", "freefly": "nostab_135"}

SCENES = [  # (rodzaj, nagłówek, linia na ekranie, lektor, czas [s])
    ("intro", "FLYING ON A FLY BRAIN", "Our 3D world, piloted by a real fruit fly connectome",
     "This is our 3D world, and this drone is piloted by a real fruit fly brain.", 6.0),
    ("world", "A new world every flight", "Hills, buildings, wind, and a dark red mast as the target",
     "Every flight happens in a new random world: hills, buildings, wind, and a dark red mast as the target. "
     "Nothing is scripted.", 10.0),
    ("ladder", "Step by step", "Each model takes over more of the flight",
     "We trained the brain's readout step by step. First it learned to turn toward a target while hovering, "
     "then to do it precisely, trained on two GPUs at once.", 12.0),
    ("homing", "HOMING", "Flies across the world to the target",
     "Next, it learned to fly across the world. A teacher pilot shows the way, then steps back. In the end, "
     "one hundred thirty-eight of one hundred fifty flights reached the target with no teacher at all.", 14.0),
    ("before", "Take the help away", "No drone sensors: height and speed only from the fly brain",
     "Then we took away the drone's own sensors. Height and speed had to come from the fly brain alone. "
     "At first, it got lost.", 10.0),
    ("training", "Train again", "Two laptops, two GPUs, learning from every flight",
     "So we trained again, on two laptops in parallel, learning from every single flight. After one hundred "
     "sixty flights, it reached the target in five of six test worlds, with buildings in the way.", 12.0),
    ("freefly", "FREEFLY", "Height, speed and the final approach from the fly brain",
     "Now it holds its height, steers past the buildings and finds the mast. GPS points the way from afar, "
     "the last metres come from what its eyes see and the fly's own wiring.", 16.0),
    ("algo", "Under the hood", "The brain is never trained. We only learn how to read it.",
     "Under the hood, the fly brain itself is never trained. We only learn how to read it: a simple decoder "
     "watches single flight neurons and learns from a teacher, with every flight added to one shared dataset.",
     13.0),
    ("outro", "FROM A FLY'S BRAIN", "to a flying drone.",
     "From a fly's brain to a flying drone. NeuroFly.", 6.0),
]

LADDER = [("HOVER", "turns toward the target while hovering"),
          ("COMPASS", "the same skill, sharpened on two GPUs"),
          ("HOMING", "flies across a random world to the target"),
          ("FREEFLY", "height and speed from the fly brain alone")]

ALGO = [("1", "Average the wing muscle neurons", "could not tell left from right"),
        ("2", "Read 375 single flight neurons", "the target side shows up clearly"),
        ("3", "Imitate a teacher pilot", "DAgger: the teacher steps back over time"),
        ("4", "Learn from every flight", "one shared dataset, ridge regression, two GPUs"),
        ("5", "Remove the drone's sensors", "height and speed from the fly brain alone")]


class Stream:
    """Nagranie czytane po kolei (bez trzymania wszystkich klatek w pamięci), z przyspieszeniem ``speed``."""

    def __init__(self, path, start=0.0, speed=1.0, crop=None):
        self.path, self.start, self.speed, self.crop = path, start, speed, crop
        self.rd = imageio.get_reader(str(path))
        self.fps = self.rd.get_meta_data()["fps"]
        self.n = self.rd.count_frames()
        self.rd = imageio.get_reader(str(path))
        self.idx, self.last = -1, None

    def length(self):
        return self.n / self.fps

    def at(self, t):
        want = min(self.n - 1, int((self.start + t * self.speed) * self.fps))
        if want < self.idx:  # cofnięcie (przenikanie): od nowa
            self.rd = imageio.get_reader(str(self.path))
            self.idx, self.last = -1, None
        while self.idx < want:
            try:
                fr = self.rd.get_next_data()
            except (IndexError, StopIteration):
                break
            self.idx += 1
            if self.idx >= want - 1 or self.last is None:
                self.last = fr
        im = Image.fromarray(self.last)
        return im.crop(self.crop) if self.crop else im


def clip_info(name):
    return json.loads((CLIPS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def text_column(c, eyebrow, title, line):
    d = ImageDraw.Draw(c)
    spaced(d, (100, 190), eyebrow.upper(), F_EYEBROW, RED, 7)
    y = 235
    for ln in wrap(d, title.upper(), F_HEAD, 600):
        d.text((100, y), ln, font=F_HEAD, fill=WHITE)
        y += 74
    y += 24
    for ln in wrap(d, line, F_BODY, 560):
        d.text((100, y), ln, font=F_BODY, fill=GREY)
        y += 48
    return y


def badge(c, x, y, text, color=RED):
    d = ImageDraw.Draw(c)
    w = d.textlength(text, font=F_SMALL) + 28
    d.rectangle((x, y, x + w, y + 38), fill=CARD, outline=color, width=2)
    d.text((x + 14, y + 6), text, font=F_SMALL, fill=WHITE)


def ladder_cards(c, active, y=860):
    d = ImageDraw.Draw(c)
    x0, gap = 100, 24
    cw = (W - 2 * x0 - gap * 3) / 4
    for i, (name, desc) in enumerate(LADDER):
        x = int(x0 + i * (cw + gap))
        on, done = i == active, active is not None and i < active
        d.rectangle((x, y, x + int(cw), y + 150), fill=CARD, outline=RED if on else EDGE, width=3 if on else 2)
        d.text((x + 24, y + 22), f"{i + 1}", font=F_CARD, fill=RED if (on or done) else MUTED)
        d.text((x + 64, y + 22), name, font=F_CARD, fill=WHITE if (on or done) else MUTED)
        yy = y + 76
        for ln in wrap(d, desc, F_SMALL, cw - 48):
            d.text((x + 24, yy), ln, font=F_SMALL, fill=GREY if (on or done) else MUTED)
            yy += 30


class Renderer:
    def __init__(self):
        cl = {k: clip_info(v) for k, v in CLIPS.items()}
        self.info = cl
        dur = {s[0]: s[4] for s in SCENES}

        def flight(key, scene, extra=("chase", "top", "eyes")):
            n = CLIPS[key]
            ln = cl[key]["time"]
            sp = max(1.0, ln / dur[scene])  # cały lot w czasie sceny (timelapse, jeśli dłuższy)
            return {k: Stream(CLIPS_DIR / f"{n}_{k}.mp4", speed=sp) for k in extra}

        self.intro = flight("freefly", "intro", ("chase",))
        self.intro["chase"].start = max(0.0, cl["freefly"]["time"] * 0.55)
        self.intro["chase"].speed = 1.0
        self.world = flight("homing", "world", ("top", "chase"))
        self.world["top"].speed = self.world["chase"].speed = 1.0
        self.hover = Stream(VIDEOS / "demo_banc_dn.mp4", start=6.0, crop=(0, 36, 470, 356))
        self.hover_eye = Stream(VIDEOS / "demo_banc_dn.mp4", start=6.0, crop=(0, 358, 470, 636))
        self.homing = flight("homing", "homing")
        self.before = flight("before", "before")
        self.freefly = flight("freefly", "freefly")
        self.outro = flight("freefly", "outro", ("chase",))
        self.outro["chase"].start = max(0.0, cl["freefly"]["time"] - 7.0)
        self.outro["chase"].speed = 1.0
        ev = json.loads((DECODERS / "world_nostab.json").read_text(encoding="utf-8").replace("NaN", "null"))["evals"]
        self.evals = [(e["after_episodes"], e["reached"], e["n"]) for e in ev]

    # --- sceny ---
    def flight_layout(self, c, streams, t, label, badges=()):
        main = fit(streams["chase"].at(t), 1060, 600)
        mx, my = 760, 150
        shadowed_paste(c, main, mx, my)
        if "top" in streams:
            top = fit(streams["top"].at(t), 420, 240)
            shadowed_paste(c, top, mx + main.width - top.width - 18, my + 18, RED)
            ImageDraw.Draw(c).text((mx + main.width - top.width - 18, my + 18 + top.height + 6), "from above",
                                   font=F_SMALL, fill=WHITE)
        if "eyes" in streams:
            eyes = fit(streams["eyes"].at(t), 470, 140)
            ex, ey = mx + 18, my + main.height - eyes.height - 18
            shadowed_paste(c, eyes, ex, ey, RED)
            ImageDraw.Draw(c).text((ex, ey - 30), "what its two eyes see", font=F_SMALL, fill=WHITE)
        d = ImageDraw.Draw(c)
        d.text((mx, my + main.height + 16), label, font=F_CAP, fill=GREY)
        for i, b in enumerate(badges):
            badge(c, 100, 560 + i * 54, b)

    def scene(self, i, t):
        kind, title, line, _, dur = SCENES[i]
        p = t / dur
        c = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(c)
        if kind in ("intro", "outro"):
            src = self.intro if kind == "intro" else self.outro
            bg = src["chase"].at(t).resize((W, H), Image.LANCZOS)
            c.paste(Image.blend(bg, Image.new("RGB", (W, H), BG), 0.45), (0, 0))
            a = min(1.0, max(0.0, (t - 0.8) / 0.8))
            col = lambda base: tuple(int(BG[k] + (base[k] - BG[k]) * a) for k in range(3))  # noqa: E731
            if kind == "intro":
                d.text((110, 760), title, font=F_HEAD, fill=col(WHITE))
                d.text((114, 850), line, font=F_BODY, fill=col(GREY))
            else:
                d.text((110, 700), title, font=F_HEAD, fill=col(WHITE))
                d.text((110, 780), line.upper(), font=F_HEAD, fill=col(RED))
                spaced(d, (114, 900), "GITHUB.COM/FALCONDEVX/NEUROFLY", F_EYEBROW, col(GREY), 7)
            return c
        if kind == "world":
            text_column(c, "the 3D world", title, line)
            top = fit(self.world["top"].at(t), 1060, 600)
            shadowed_paste(c, top, 760, 150)
            ch = fit(self.world["chase"].at(t), 420, 240)
            shadowed_paste(c, ch, 760 + top.width - ch.width - 18, 150 + top.height - ch.height - 18, RED)
            d.text((760, 150 + top.height + 16), "MuJoCo physics, Skydio X2 drone, a fresh world every episode",
                   font=F_CAP, fill=GREY)
            return c
        if kind == "ladder":
            text_column(c, "evolution of the models", title, line)
            hv = fit(self.hover.at(t), 680, 440)
            shadowed_paste(c, hv, 800, 150)
            eye = fit(self.hover_eye.at(t), 300, 190)
            ex, ey = 800 + hv.width + 30, 150 + hv.height - eye.height
            shadowed_paste(c, eye, ex, ey, RED)
            d.text((ex, ey - 32), "what it sees", font=F_SMALL, fill=WHITE)
            d.text((800, 150 + hv.height + 12), "HOVER: the nose turns toward the mast", font=F_CAP, fill=GREY)
            ladder_cards(c, 0 if p < 0.55 else 1, y=770)
            return c
        if kind == "homing":
            text_column(c, "model 3 of 4", title, line)
            self.flight_layout(c, self.homing, t, self.flight_label("homing"),
                               badges=("heading from the fly brain near the target", "drone sensors help with height"))
            return c
        if kind == "before":
            text_column(c, "model 4, before training", title, line)
            self.flight_layout(c, self.before, t, self.flight_label("before"), badges=("no drone sensors",))
            return c
        if kind == "training":
            text_column(c, "model 4, training", title, line)
            self.chart(c, p)
            return c
        if kind == "freefly":
            text_column(c, "model 4 of 4", title, line)
            self.flight_layout(c, self.freefly, t, self.flight_label("freefly"),
                               badges=("no drone sensors", "buildings on the route"))
            return c
        if kind == "algo":
            text_column(c, "the algorithm, in five steps", title, line)
            shown = min(len(ALGO), int(p * len(ALGO) * 1.4) + 1)
            for k, (n, h, s) in enumerate(ALGO[:shown]):
                y = 170 + k * 132
                d.rectangle((800, y, 1820, y + 112), fill=CARD, outline=RED if k == shown - 1 else EDGE, width=2)
                d.text((830, y + 30), n, font=F_CARD, fill=RED)
                d.text((890, y + 20), h, font=F_BODY, fill=WHITE)
                d.text((890, y + 66), s, font=F_SMALL, fill=GREY)
            return c
        return c

    def flight_label(self, key):
        inf = self.info[key]
        speed = max(1.0, inf["time"] / {s[0]: s[4] for s in SCENES}[key])
        res = "reaches the target" if inf["reached"] else "does not reach the target"
        return f"real flight, {res}" + (f", shown {speed:.1f}× faster" if speed > 1.05 else "")

    def chart(self, c, p):
        d = ImageDraw.Draw(c)
        x0, y0, w, h = 820, 200, 960, 560
        d.text((x0, y0 - 50), "targets reached in 6 test worlds with buildings on the route", font=F_CAP, fill=GREY)
        n = len(self.evals)
        bw = w / n * 0.55
        for k, (ep, reached, tot) in enumerate(self.evals):
            grow = min(1.0, max(0.0, p * n * 1.3 - k))
            x = x0 + (k + 0.5) * w / n - bw / 2
            hh = h * reached / tot * grow
            d.line((x0, y0 + h, x0 + w, y0 + h), fill=EDGE, width=2)
            if hh >= 2:
                d.rectangle((x, y0 + h - hh, x + bw, y0 + h), fill=RED)
            if grow > 0.95:
                d.text((x + bw / 2 - 18, y0 + h - hh - 80), str(reached), font=F_NUM, fill=WHITE)
            d.text((x + bw / 2 - 30, y0 + h + 16), "start" if ep == 0 else str(ep), font=F_SMALL, fill=GREY)
        d.text((x0, y0 + h + 60), "training flights", font=F_SMALL, fill=MUTED)
        badge(c, 100, 600, "RTX 4060 + RTX 3070 Ti, in parallel")

    def frame_at(self, t):
        starts = np.cumsum([0.0] + [s[4] for s in SCENES])
        total = starts[-1]
        i = min(int(np.searchsorted(starts, t, side="right") - 1), len(SCENES) - 1)
        local = t - starts[i]
        img = self.scene(i, local)
        if t >= total - 0.6:
            img = Image.blend(Image.new("RGB", (W, H), BG), img, max(0.0, (total - t) / 0.6))
        elif local < 0.4:  # krótkie wejście z czerni (strumienie czytane są tylko do przodu)
            img = Image.blend(Image.new("RGB", (W, H), BG), img, local / 0.4)
        return img


def write_script(md_path, srt_path, total_note):
    starts = np.cumsum([0.0] + [s[4] for s in SCENES])
    md = ["# NeuroFly: 3D simulation and training, voice-over script (EN)", "",
          f"Video: `data/videos/neurofly_world.mp4`, {starts[-1]:.0f} s, 1920×1080. {total_note}", ""]
    srt = []
    for k, (kind, title, line, voice, dur) in enumerate(SCENES):
        a, b = starts[k], starts[k + 1]
        md += [f"## {ts(a, '.')[3:8]} to {ts(b, '.')[3:8]}  {title.title() if title.isupper() else title}", "",
               f"> {voice}", "", f"On screen: *{line}*", ""]
        srt += [str(k + 1), f"{ts(a + 0.2)} --> {ts(b - 0.2)}", voice, ""]
    md += ["## Evolution of the algorithm (short)", ""]
    md += [f"{n}. **{h}**: {s}." for n, h, s in ALGO]
    Path(md_path).write_text("\n".join(md) + "\n", encoding="utf-8")
    Path(srt_path).write_text("\n".join(srt), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=VIDEOS / "neurofly_world.mp4")
    ap.add_argument("--still", type=float, nargs="*", default=None)
    a = ap.parse_args()
    write_script(Path(__file__).parent / "skrypt_world_en.md", a.out.with_suffix(".srt"),
                 "Read at a calm pace (about 2.5 words per second). Each block starts with its scene.")
    r = Renderer()
    if a.still is not None:
        for t in sorted(a.still):
            r.frame_at(t).save(a.out.with_name(f"{a.out.stem}_t{t:05.1f}.jpg"), quality=88)
        return
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = sum(s[4] for s in SCENES)
    for n in range(int(total * FPS)):
        proc.stdin.write(r.frame_at(n / FPS).tobytes())
        if n % (FPS * 10) == 0:
            print(f"  {n / FPS:4.0f} s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("zapisano", a.out)


if __name__ == "__main__":
    main()
