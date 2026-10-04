"""Film „pipeline kafelek po kafelku” (1920×1080, 30 fps) + scenariusz lektora (EN) i napisy SRT.

    .venv312\\Scripts\\python docs/prezentacja/make_pipeline_video.py [--out data/videos/neurofly_pipeline.mp4]

Materiał: rendery z docs/prezentacja/assets i nagrania z data/videos (intro, przelot przez connectome, lot drona
z panelem BANC). Tekst lektora jest w SCENES — z niego powstaje też skrypt_pipeline_en.md i .srt (te same czasy).
"""
import argparse
import os
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "prezentacja" / "assets"
VIDEOS = ROOT / "data" / "videos"
FONTS = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
W, H, FPS = 1920, 1080, 30
FADE = 0.5  # s, przenikanie między scenami

BG = (9, 9, 11)
WHITE, GREY, MUTED, RED = (250, 250, 250), (161, 161, 170), (82, 82, 91), (255, 45, 111)
CARD, EDGE = (24, 24, 27), (63, 63, 70)


def font(name, size):
    for d in (FONTS, Path("C:/Windows/Fonts")):
        if (d / name).exists():
            return ImageFont.truetype(str(d / name), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


F_HEAD = font("Michroma-Regular.ttf", 54)
F_HEAD_BIG = font("Michroma-Regular.ttf", 96)
F_EYEBROW = font("Roboto-Regular.ttf", 22)
F_BODY = font("Roboto-Light.ttf", 34)
F_TILE = font("Roboto-Medium.ttf", 24)
F_TILE_N = font("Michroma-Regular.ttf", 22)
F_CAP = font("Roboto-Regular.ttf", 24)

TILES = ["Camera", "Fly eye", "FlyVis", "Fly brain", "Decoder", "Drone"]

# (tile index or None, title, on-screen line, narration, duration [s], media)
SCENES = [
    (None, "NEUROFLY", "The brain of a fruit fly pilots a drone",
     "What if a real fruit fly brain could fly a drone? This is NeuroFly.", 6.0, "intro"),
    (0, "Camera", "Two cameras on the drone's nose, 30 frames a second",
     "It starts with a camera. Two small cameras on the drone's nose watch the world, thirty frames every second.",
     8.0, "camera"),
    (1, "Fly eye", "721 facets per eye, each sees only brightness",
     "Each frame becomes a fly's compound eye: seven hundred and twenty-one tiny facets per eye, "
     "and each one sees only brightness.", 8.5, "eye"),
    (2, "FlyVis", "A published model of the fly's visual system",
     "FlyVis, a published model of the fly's visual system, turns those facets into the signals of real fly "
     "visual neurons, including its motion detectors.", 10.0, "flyvis"),
    (3, "Fly brain", "175,401 real neurons from the BANC connectome",
     "Those signals enter the fly brain itself: the BANC connectome. One hundred seventy-five thousand real "
     "neurons and one and a half million connections, mapped synapse by synapse, simulated on a laptop GPU.",
     13.0, "brain"),
    (4, "Decoder", "375 flight neurons become four numbers",
     "Flight commands leave the brain through three hundred seventy-five descending neurons. Our decoder reads "
     "them one by one and turns them into four numbers: thrust, roll, pitch and yaw.", 11.0, "decoder"),
    (5, "Drone", "It moves, sees a new picture, and the loop starts again",
     "The drone moves, its eyes see a new picture, and the loop starts again. Near the target, every turn "
     "comes from the fly brain. Far away, GPS helps.", 12.0, "drone"),
    (None, "EVOLUTION WROTE THE WIRING", "We gave it wings.",
     "Evolution wrote the wiring. We gave it wings. NeuroFly.", 6.0, "outro"),
]


# ---------- media ----------
def load_image(name):
    return Image.open(ASSETS / name).convert("RGB")


class Clip:
    """Fragment nagrania wczytany do pamięci (klatki RGB), odtwarzany z prędkością ``speed``."""

    def __init__(self, name, start, length, crop=None, speed=1.0):
        rd = imageio.get_reader(str(VIDEOS / name))
        fps = rd.get_meta_data()["fps"]
        first, n = int(start * fps), int(length * fps * speed) + 2
        self.frames = []
        for i, fr in enumerate(rd):
            if i < first:
                continue
            if len(self.frames) >= n:
                break
            im = Image.fromarray(fr)
            self.frames.append(im.crop(crop) if crop else im)
        rd.close()
        self.fps, self.speed = fps, speed

    def at(self, t):
        return self.frames[min(len(self.frames) - 1, int(t * self.fps * self.speed))]


def fit(im, box_w, box_h, zoom=1.0):
    """Skaluje do ramki (contain), z lekkim zbliżeniem (Ken Burns) przyciętym do środka."""
    s = min(box_w / im.width, box_h / im.height) * zoom
    out = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    if zoom > 1.0:  # przytnij z powrotem do rozmiaru bez zbliżenia
        cw = min(out.width, int(im.width * s / zoom))
        ch = min(out.height, int(im.height * s / zoom))
        l, t = (out.width - cw) // 2, (out.height - ch) // 2
        out = out.crop((l, t, l + cw, t + ch))
    return out


def shadowed_paste(canvas, im, x, y, border=EDGE):
    sh = Image.new("RGBA", (im.width + 60, im.height + 60), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle((30, 40, im.width + 30, im.height + 40), fill=(0, 0, 0, 150))
    sh = sh.filter(ImageFilter.GaussianBlur(14))
    canvas.paste(sh, (x - 30, y - 30), sh)
    canvas.paste(im, (x, y))
    ImageDraw.Draw(canvas).rectangle((x - 1, y - 1, x + im.width, y + im.height), outline=border, width=2)


# ---------- warstwy tekstu ----------
def spaced(d, xy, text, f, fill, spacing):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + spacing


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if d.textlength(test, font=f) <= width:
            cur = test
        else:
            lines.append(cur)
            cur = w_
    lines.append(cur)
    return lines


def tiles(canvas, active, t):
    d = ImageDraw.Draw(canvas)
    x0, y0, gap = 100, 940, 18
    tw = (W - 2 * x0 - gap * (len(TILES) - 1)) / len(TILES)
    for i, name in enumerate(TILES):
        x = int(x0 + i * (tw + gap))
        on = i == active
        done = active is not None and i < active
        d.rectangle((x, y0, x + int(tw), y0 + 70), fill=CARD, outline=RED if on else EDGE, width=3 if on else 2)
        if on:  # pasek postępu sceny w aktywnym kafelku
            d.rectangle((x, y0 + 64, x + int(tw * min(1.0, t)), y0 + 70), fill=RED)
        col = WHITE if on else (GREY if done else MUTED)
        d.text((x + 22, y0 + 22), str(i + 1), font=F_TILE_N, fill=RED if (on or done) else MUTED)
        d.text((x + 60, y0 + 20), name, font=F_TILE, fill=col)
        if i < len(TILES) - 1:  # łącznik między kafelkami
            cx = x + int(tw) + gap // 2
            d.line((cx - 6, y0 + 35, cx + 6, y0 + 35), fill=RED if done or on else EDGE, width=3)


def text_column(canvas, scene_i, title, line, alpha=1.0):
    d = ImageDraw.Draw(canvas)
    spaced(d, (100, 190), f"STEP {SCENES[scene_i][0] + 1} OF {len(TILES)}", F_EYEBROW, RED, 7)
    y = 235
    for ln in wrap(d, title.upper(), F_HEAD, 600):
        d.text((100, y), ln, font=F_HEAD, fill=WHITE)
        y += 74
    y += 24
    for ln in wrap(d, line, F_BODY, 560):
        d.text((100, y), ln, font=F_BODY, fill=GREY)
        y += 48


# ---------- sceny ----------
MEDIA_X, MEDIA_Y, MEDIA_W, MEDIA_H = 760, 150, 1060, 720


def media_box(canvas, im, border=EDGE):
    x = MEDIA_X + (MEDIA_W - im.width) // 2
    y = MEDIA_Y + (MEDIA_H - im.height) // 2
    shadowed_paste(canvas, im, x, y, border)
    return x, y


def caption(canvas, text, x, y):
    ImageDraw.Draw(canvas).text((x, y), text, font=F_CAP, fill=GREY)


class Renderer:
    def __init__(self):
        print("wczytywanie nagrań…", flush=True)
        self.intro = Clip("intro_poppy.mp4", 0.0, 6.5)
        # demo_banc_dn.mp4 (816×640) ma polskie napisy: etykieta sceny u góry i opisy panelu — przycinamy je
        drone, eye, brain = (0, 36, 470, 356), (0, 358, 470, 636), (482, 24, 816, 392)
        self.cam = Clip("demo_banc_dn.mp4", 6.0, 8.5, crop=drone)
        self.cam_eye = Clip("demo_banc_dn.mp4", 6.0, 8.5, crop=eye)
        self.tour = Clip("connectome_tour_720p.mp4", 8.5, 13.5)
        self.demo = Clip("demo_banc_dn.mp4", 15.0, 12.5, crop=drone)
        self.demo_eye = Clip("demo_banc_dn.mp4", 15.0, 12.5, crop=eye)
        self.demo_brain = Clip("demo_banc_dn.mp4", 15.0, 12.5, crop=brain)
        self.eye = load_image("eye_right.jpg")
        self.retina = load_image("retina_right.jpg")
        self.flyvis = load_image("thumb_flyvis.jpg")
        self.circuit = load_image("circuit_3d.jpg")
        self.panel = load_image("panel_live.jpg")
        self.title_bg = load_image("fly_drone.jpg")

    def scene(self, i, t):
        """Klatka sceny i w chwili t [s] od jej początku."""
        tile, title, line, _, dur, kind = SCENES[i]
        p = t / dur
        zoom = 1.0 + 0.05 * p
        c = Image.new("RGB", (W, H), BG)
        if kind == "intro":
            c.paste(fit(self.intro.at(t), W, H), (0, 0))
            d = ImageDraw.Draw(c)
            a = min(1.0, max(0.0, (t - 1.0) / 0.8))
            col = tuple(int(BG[k] + (WHITE[k] - BG[k]) * a) for k in range(3))
            d.text((110, 760), title, font=F_HEAD_BIG, fill=col)
            d.text((116, 900), line, font=F_BODY, fill=tuple(int(BG[k] + (GREY[k] - BG[k]) * a) for k in range(3)))
            return c
        if kind == "outro":
            bg = fit(self.title_bg, W, H, zoom)
            bg = Image.blend(bg.resize((W, H)), Image.new("RGB", (W, H), BG), 0.35)
            c.paste(bg, (0, 0))
            d = ImageDraw.Draw(c)
            d.text((110, 640), "EVOLUTION WROTE THE WIRING.", font=F_HEAD, fill=WHITE)
            d.text((110, 720), "WE GAVE IT WINGS.", font=F_HEAD, fill=RED)
            spaced(d, (114, 860), "GITHUB.COM/FALCONDEVX/NEUROFLY", F_EYEBROW, GREY, 7)
            return c

        text_column(c, i, title, line)
        if kind == "camera":
            drone = fit(self.cam.at(t), MEDIA_W, MEDIA_H - 40)
            x, y = media_box(c, drone)
            eye = fit(self.cam_eye.at(t), 380, 230)
            ex, ey = x + drone.width - eye.width - 24, y + drone.height - eye.height - 24
            shadowed_paste(c, eye, ex, ey, RED)
            caption(c, "what the drone sees", ex, ey - 34)
        elif kind == "eye":
            a = min(1.0, max(0.0, (t - dur * 0.35) / 1.2))  # kamera → oko złożone
            cam = fit(self.eye, MEDIA_W, MEDIA_H - 40, zoom)
            im = Image.blend(cam, fit(self.retina, MEDIA_W, MEDIA_H - 40, zoom).resize(cam.size), a)
            x, y = media_box(c, im, RED if a > 0.5 else EDGE)
            caption(c, "camera frame" if a < 0.5 else "the same frame, as a fly's eye sees it", x, y + im.height + 16)
        elif kind == "flyvis":
            im = fit(self.flyvis, MEDIA_W, MEDIA_H - 40, zoom)
            x, y = media_box(c, im)
            caption(c, "activity of fly visual neurons, computed by FlyVis", x, y + im.height + 16)
        elif kind == "brain":
            im = fit(self.tour.at(t), MEDIA_W, MEDIA_H)
            media_box(c, im)
        elif kind == "decoder":
            cir = fit(self.circuit, 560, MEDIA_H, zoom)
            pan = fit(self.panel, 440, MEDIA_H - 60)
            x0 = MEDIA_X + (MEDIA_W - cir.width - pan.width - 40) // 2
            shadowed_paste(c, cir, x0, MEDIA_Y + (MEDIA_H - cir.height) // 2)
            px, py = x0 + cir.width + 40, MEDIA_Y + (MEDIA_H - pan.height) // 2
            shadowed_paste(c, pan, px, py, RED)
            caption(c, "live: who steers, and each neuron's vote", px, py + pan.height + 12)
        elif kind == "drone":
            drone, eye, brain = self.demo.at(t), self.demo_eye.at(t), self.demo_brain.at(t)
            left = Image.new("RGB", (drone.width, drone.height + eye.height), BG)
            left.paste(drone, (0, 0))
            left.paste(eye, (0, drone.height))
            s = (MEDIA_H - 50) / left.height
            left = left.resize((int(left.width * s), int(left.height * s)), Image.LANCZOS)
            right = fit(brain, MEDIA_W - left.width - 30, int(MEDIA_H - 50))
            x0 = MEDIA_X + (MEDIA_W - left.width - right.width - 30) // 2
            shadowed_paste(c, left, x0, MEDIA_Y)
            rx = x0 + left.width + 30
            shadowed_paste(c, right, rx, MEDIA_Y, RED)
            caption(c, "the drone, and below what its eyes see", x0, MEDIA_Y + left.height + 12)
            caption(c, "fly brain live: red up, blue down", rx, MEDIA_Y + right.height + 12)
        tiles(c, tile, p)
        return c

    def frame_at(self, t):
        starts = np.cumsum([0.0] + [s[4] for s in SCENES])
        total = starts[-1]
        i = min(int(np.searchsorted(starts, t, side="right") - 1), len(SCENES) - 1)
        local = t - starts[i]
        img = self.scene(i, local)
        if i > 0 and local < FADE:  # przenikanie z poprzedniej sceny
            prev = self.scene(i - 1, SCENES[i - 1][4] - FADE + local)
            img = Image.blend(prev, img, local / FADE)
        if t >= total - 0.6:  # wyjście w czerń
            img = Image.blend(Image.new("RGB", (W, H), BG), img, max(0.0, (total - t) / 0.6))
        return img

    def frames(self):
        total = sum(s[4] for s in SCENES)
        for n in range(int(total * FPS)):
            yield self.frame_at(n / FPS)


def ts(t, sep=","):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def write_script(md_path, srt_path):
    starts = np.cumsum([0.0] + [s[4] for s in SCENES])
    md = ["# NeuroFly: pipeline video, voice-over script (EN)", "",
          f"Video: `data/videos/neurofly_pipeline.mp4`, {starts[-1]:.0f} s, 1920×1080. "
          "Read at a calm pace (about 2.5 words per second). Each block starts when its tile lights up.", ""]
    srt = []
    for k, (tile, title, line, voice, dur, _) in enumerate(SCENES):
        a, b = starts[k], starts[k + 1]
        label = f"Tile {tile + 1}: {title}" if tile is not None else ("Intro" if k == 0 else "Outro")
        md += [f"## {ts(a, '.')[3:8]} to {ts(b, '.')[3:8]}  {label}", "", f"> {voice}", "",
               f"On screen: *{line}*", ""]
        srt += [str(k + 1), f"{ts(a + 0.2)} --> {ts(b - 0.2)}", voice, ""]
    Path(md_path).write_text("\n".join(md), encoding="utf-8")
    Path(srt_path).write_text("\n".join(srt), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=VIDEOS / "neurofly_pipeline.mp4")
    ap.add_argument("--still", type=float, nargs="*", default=None, help="tylko klatki w chwilach t [s] (podgląd)")
    a = ap.parse_args()
    here = Path(__file__).parent
    write_script(here / "skrypt_pipeline_en.md", a.out.with_suffix(".srt"))
    r = Renderer()
    if a.still is not None:
        for t in a.still:
            r.frame_at(t).save(a.out.with_name(f"{a.out.stem}_t{t:05.1f}.jpg"), quality=88)
        return
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n, img in enumerate(r.frames()):
        proc.stdin.write(img.tobytes())
        if n % (FPS * 5) == 0:
            print(f"  {n / FPS:4.0f} s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("zapisano", a.out)


if __name__ == "__main__":
    main()
