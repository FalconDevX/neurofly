// HackYeah 2026, 10 slajdów: node build_hackyeah.js [out.pptx], potem python merge_title.py NeuroFly_hackyeah_raw.pptx NeuroFly_hackyeah.pptx
// Nagłówek (motyw, układy, pomocnicze) jak w build_mvp.js.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || path.join(__dirname, "NeuroFly_hackyeah_raw.pptx");
const A = (f) => path.join(__dirname, "assets", f);
const DATA = JSON.parse(fs.readFileSync(A("data.json"), "utf8"));
const W = DATA.world; // lot do celu w świecie (train_distributed.py --world)

const THEME = {
  name: "NeuroFly",
  headFontFace: "Michroma", // szeroki, geometryczny (styl „Aquire”); OFL — osadzony w .pptx (embed_fonts.ps1)
  bodyFontFace: "Roboto",
  colors: {
    dk1: "09090B", // tło (zinc-950)
    lt1: "FAFAFA", // tekst
    dk2: "18181B", // karty (zinc-900)
    lt2: "A1A1AA", // tekst drugorzędny (zinc-400)
    accent1: "2FD3C4", // teal — wzrok
    accent2: "FFB547", // amber — neurony zstępujące (DN)
    accent3: "FF4FB0", // pink — motoneurony
    accent4: "4F8CFF", // blue — haltery / czujniki
    accent5: "52525B", // zinc-600
    accent6: "27272A", // zinc-800 — linie, ramki
    hlink: "2FD3C4",
    folHlink: "A1A1AA",
  },
};
const T = THEME.colors;
// akcent interfejsu w stylu plakatu: czerwono-różowy → pomarańczowy (kolory danych w accent1–4 zostają jak w renderach)
const RED = "FF2D6F", ORANGE = "FF6A2B", VIOLET = "A98BFF", MUTED = "71717A";
const HEAD = THEME.headFontFace;
const MONO = "Consolas";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 × 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "NeuroFly — the brain of a fruit fly pilots a drone";
pres.author = "NeuroFly team";

// ---------- układy (layouts) ----------
pres.defineSlideMaster({ title: "TITLE", background: { path: A("shards_title.jpg") }, objects: [] });
pres.defineSlideMaster({
  title: "CONTENT",
  background: { path: A("shards_corner.jpg") },
  objects: [
    { image: { path: A("accent_line.png"), x: 0.44, y: 0.2, w: 0.04, h: 0.82 } }, // czerwona linia przy tytule
    { image: { path: A("logo.png"), x: 0.5, y: 5.13, w: 0.34, h: 0.2 } },
    { text: { text: "NeuroFly", options: { x: 0.9, y: 5.08, w: 2, h: 0.3, fontSize: 10, color: T.lt2, margin: 0 } } },
    {
      placeholder: {
        options: { name: "title", type: "title", x: 0.62, y: 0.5, w: 8.3, h: 0.4, fontSize: 16, bold: false, color: T.lt1, margin: 0,
          valign: "top", align: "left", charSpacing: 1, fontFace: HEAD },
        text: "",
      },
    },
  ],
  slideNumber: { x: 9.0, y: 5.08, w: 0.5, h: 0.3, fontSize: 10, color: T.lt2, align: "right" },
});

// ---------- pomocnicze ----------
const shadow = () => ({ type: "outer", color: "000000", blur: 18, offset: 6, angle: 90, opacity: 0.55 });

function card(slide, x, y, w, h, name, line) {
  // ostre prostokąty (geometrycznie, jak plakat), lekko przezroczyste nad tłem z odłamkami
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: T.dk2, transparency: 12 }, line: { color: line || "3F3F46", width: 0.75 }, shadow: shadow(), objectName: name,
  });
}

function text(slide, t, opts) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 14, color: T.lt1, valign: "top", ...opts });
}

function stat(slide, x, y, w, value, label, color, name, size) {
  text(slide, value, { x, y, w, h: 0.6, fontSize: size || 32, fontFace: "Roboto Light", color: color || T.lt1, objectName: `${name}-value` });
  text(slide, label, { x, y: y + 0.62, w, h: 0.5, fontSize: 12, color: T.lt2, objectName: `${name}-label` });
}

function tag(slide, x, y, label, color, name, w = 3) {
  // etykieta w kolorze grupy (bez kropki)
  text(slide, label, { x, y, w, h: 0.26, fontSize: 11, bold: true, color, objectName: `${name}-label` });
}

function eyebrow(slide, label, x, y, w, color, name) {
  // rozstrzelone wersaliki (jak „FREE TYPEFACE” na plakacie)
  text(slide, label.toUpperCase(), { x, y, w, h: 0.22, fontSize: 8, color: color || RED, charSpacing: 5, objectName: name || "eyebrow" });
}

function gloss(slide, items, x = 0.62, w = 8.85, y = 4.6) {
  // pasek objaśnień nad stopką: skróty i metryki ze slajdu, każda pozycja we własnej kolumnie (bez separatorów),
  // szerokość kolumny wg długości tekstu, żeby nic nie łamało się w pół zdania
  const gap = 0.3, len = items.map(([k, v]) => k.length + v.length + 14);
  const sum = len.reduce((a, b) => a + b, 0), free = w - gap * (items.length - 1);
  let cx = x;
  items.forEach(([k, v], i) => {
    const cw = (free * len[i]) / sum;
    text(slide, [{ text: k, options: { bold: true, color: T.lt2 } }, { text: ` ${v}`, options: { color: MUTED } }],
      { x: cx, y, w: cw, h: 0.45, fontSize: 9, objectName: `glossary-${i}` });
    cx += cw + gap;
  });
}

function img(slide, file, x, y, w, h, name) {
  // obrazek przycięty do ramki (cover), z cieniem
  slide.addImage({ path: A(file), x, y, w, h, sizing: { type: "cover", w, h }, objectName: name });
}

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  eyebrow(s, section, 0.62, 0.24, 6);
  s.addText(title.toUpperCase(), { placeholder: "title" });
  return s;
}

// wspólny wygląd wykresów: te same fonty, kolory i etykiety na każdym wykresie
const CHART = {
  dataLabelColor: T.lt1, dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: T.lt2, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLineShow: false,
  showTitle: false, legendColor: T.lt2, legendFontSize: 11, legendFontFace: "+mn-lt",
};
const chartOpts = (o) => ({ ...CHART, ...o });


// =================================================================================================
// HackYeah 2026: 10 slajdów według konspektu zespołu (dlaczego mucha, problem, wzrok, pętla, przypadek użycia,
// rola AI, kontrola, co zbudowane, dlaczego ważne). Treść poprawiona do stanu repo: FlyVis widzi jasność (nie kolor),
// symulator to MuJoCo, dron nie zwalnia/nie zatrzymuje się przy niepewności (tego nie ma), panel nie ma „ostrzeżenia
// o przeszkodzie”. Slajd 1 to miejsce na tytuł; merge_title.py podmienia go na slajd 1 z NeuroFly.pptx.
// =================================================================================================
const NS = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "data", "decoders", "world_nostab.json"), "utf8").replace(/\bNaN\b/g, "null")); // trening bez wspomagania (Python zapisuje NaN)

pres.addSection({ title: "Start" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Start" });
  text(s, "NEUROFLY", { x: 0.6, y: 2.05, w: 5.2, h: 0.85, fontSize: 40, fontFace: HEAD, objectName: "title" });
}

pres.addSection({ title: "Idea" });

// 2. Dlaczego mózg muszki
{
  const s = content("Why a fly brain?", "Idea");
  text(s, "A fruit fly reacts to motion, dodges obstacles and homes in on targets with a brain that fits on the tip of a pin.", {
    x: 0.62, y: 1.1, w: 5.2, h: 0.75, fontSize: 14, objectName: "lead" });
  card(s, 0.55, 2.0, 5.35, 1.0, "q", RED);
  eyebrow(s, "our question", 0.78, 2.12, 3, RED, "q-k");
  text(s, "Can the real wiring of an insect brain pilot an autonomous drone?", {
    x: 0.78, y: 2.38, w: 4.95, h: 0.55, fontSize: 15, bold: true, objectName: "q-t" });
  stat(s, 0.62, 3.2, 2.5, "175,401", "neurons, mapped synapse by synapse", RED, "s-n");
  stat(s, 3.3, 3.2, 2.6, "2.6 ms", "to run all of them once on a laptop GPU", T.lt1, "s-t");
  s.addImage({ path: A("fly_xray.jpg"), x: 6.2, y: 1.0, w: 3.25, h: 3.5, sizing: { type: "cover", w: 3.25, h: 3.5 }, shadow: shadow(), objectName: "img-fly" });
  gloss(s, [["Not a chatbot:", "no language model flies this drone"],
    ["Right:", "the fly with its brain and nerve cord, neurons at their real positions"]]);
  s.addNotes("NeuroFly is not another chatbot connected to a drone. It is an experiment in biologically grounded visual " +
    "intelligence: the controller is the published connectome of a fruit fly (BANC), run as a neural network.");
}

// 3. Problem
{
  const s = content("The problem", "Idea");
  text(s, "An autonomous drone decides from imperfect pictures, many times a second.", {
    x: 0.62, y: 1.1, w: 8.8, h: 0.35, fontSize: 14, objectName: "lead" });
  const qs = [
    ["Where is the target?", "a mast somewhere ahead, GPS off by metres"],
    ["Is something in the way?", "blocks, walls and towers on the route"],
    ["Which way to turn?", "left, right, how hard"],
    ["How to stay in the air?", "height, tilt, wind gusts"],
  ];
  qs.forEach(([h, d], i) => {
    const x = 0.55 + (i % 2) * 2.75, y = 1.65 + Math.floor(i / 2) * 1.2;
    card(s, x, y, 2.6, 1.05, `q-${i}`);
    text(s, h, { x: x + 0.18, y: y + 0.15, w: 2.3, h: 0.3, fontSize: 13, bold: true, objectName: `q-${i}-h` });
    text(s, d, { x: x + 0.18, y: y + 0.5, w: 2.3, h: 0.45, fontSize: 11, color: T.lt2, objectName: `q-${i}-d` });
  });
  s.addImage({ path: A("world.jpg"), x: 6.2, y: 1.65, w: 3.25, h: 2.25, sizing: { type: "cover", w: 3.25, h: 2.25 }, shadow: shadow(), objectName: "img-world" });
  text(s, "Our test world: hills, blocks, wind, a dark target mast", { x: 6.2, y: 3.97, w: 3.25, h: 0.4, fontSize: 10, color: T.lt2, objectName: "cap-world" });
  gloss(s, [["Usual answer:", "big datasets, long training, a black box"],
    ["Our path:", "a small, inspectable circuit that evolution already tuned"]]);
  s.addNotes("Classic systems need large datasets, expensive training and opaque networks. We explore a different path: " +
    "the measured wiring of a fly, where every neuron has a name and a function label.");
}

pres.addSection({ title: "How it works" });

// 4. Od klatki kamery do widoku muchy
{
  const s = content("From a camera frame to a fly's view", "How it works");
  const steps = [
    ["eye_right.jpg", "Camera", "two eyes on the drone's nose, 157° each"],
    ["retina_right.jpg", "Compound eye", "721 ommatidia per eye, one brightness value each"],
    ["thumb_flyvis.jpg", "Motion vision", "FlyVis turns it into fly visual neuron activity"],
  ];
  steps.forEach(([f, h, d], i) => {
    const x = 0.55 + i * 3.05;
    card(s, x, 1.1, 2.85, 2.85, `st-${i}`, i === 1 ? RED : undefined);
    img(s, f, x + 0.1, 1.2, 2.65, 1.7, `st-${i}-img`);
    text(s, [{ text: `${i + 1}  `, options: { color: RED, fontFace: HEAD, fontSize: 12 } }, { text: h, options: { bold: true } }],
      { x: x + 0.2, y: 3.05, w: 2.5, h: 0.3, fontSize: 14, objectName: `st-${i}-h` });
    text(s, d, { x: x + 0.2, y: 3.4, w: 2.5, h: 0.5, fontSize: 11, color: T.lt2, objectName: `st-${i}-d` });
  });
  text(s, "Each ommatidium sees only a small patch of the scene. No full image ever reaches the brain.", {
    x: 0.62, y: 4.1, w: 8.8, h: 0.3, fontSize: 12, italic: true, objectName: "point" });
  gloss(s, [["Ommatidium:", "one facet of the compound eye"], ["FlyVis:", "published model of the fly eye and optic lobe"],
    ["Brightness only:", "like the fly photoreceptors FlyVis models, no colour"]]);
  s.addNotes("The camera image is resampled into the hexagonal lattice of a fly eye (FlyGym retina, 721 ommatidia per eye). " +
    "FlyVis (Lappalainen et al.) computes the activity of fly visual neuron types, including motion detectors T4 and T5. " +
    "Those activities are mapped by cell type onto 22,462 BANC neurons.");
}

// 5. Jak działa NeuroFly: pięć kroków
{
  const s = content("How NeuroFly works", "How it works");
  const nodes = [
    ["Camera", "sees the 3D world", "drone_chase.jpg", ""],
    ["Fly eye", "ommatidia + FlyVis", "retina_left.jpg", "TRAINED NET"],
    ["Fly brain", "175,401 BANC neurons", "thumb_wiring.jpg", "REAL CONNECTOME"],
    ["Decoder", "flight neurons to 4 numbers", "thumb_decoder.jpg", "LEARNED"],
    ["Drone", "thrust, roll, pitch, yaw", "drone_live.jpg", ""],
  ];
  const w = 1.66, gx = 0.13;
  nodes.forEach(([hd, d, f, ai], i) => {
    const x = 0.55 + i * (w + gx), y = 1.15;
    card(s, x, y, w, 2.6, `node-${i}`, ai ? RED : undefined);
    img(s, f, x + 0.07, y + 0.07, w - 0.14, 1.25, `node-${i}-img`);
    if (ai) {
      s.addShape(pres.shapes.RECTANGLE, { x: x + 0.07, y: y + 0.07, w: 1.3, h: 0.2, fill: { color: RED }, line: { color: RED, width: 0 }, objectName: `node-${i}-ai` });
      text(s, ai, { x: x + 0.12, y: y + 0.095, w: 1.25, h: 0.16, fontSize: 7, bold: true, charSpacing: 1, objectName: `node-${i}-ai-t` });
    }
    text(s, [{ text: `${i + 1}  `, options: { color: RED, fontFace: HEAD, fontSize: 11 } }, { text: hd, options: { bold: true } }],
      { x: x + 0.14, y: y + 1.45, w: w - 0.24, h: 0.3, fontSize: 13, objectName: `node-${i}-title` });
    text(s, d, { x: x + 0.14, y: y + 1.8, w: w - 0.24, h: 0.6, fontSize: 10.5, color: T.lt2, objectName: `node-${i}-desc` });
  });
  text(s, "Every frame goes once around: seeing, processing, deciding, moving, then a new picture.", {
    x: 0.62, y: 3.95, w: 8.8, h: 0.3, fontSize: 12, italic: true, objectName: "loop" });
  gloss(s, [["Red tag:", "where AI works in the loop"], ["Simulator:", "MuJoCo physics with a Skydio X2 drone model"],
    ["Speed:", "30 camera frames per second"]]);
  s.addNotes("Three AI parts: FlyVis (trained network of the fly visual system), the BANC connectome (biological network, " +
    "rate model on the GPU, never trained) and the decoder that reads flight neurons, learned by imitation (DAgger) " +
    "on two laptop GPUs at once.");
}

// 6. Przypadek użycia
{
  const s = content("Use case: reach a target safely", "How it works");
  s.addImage({ path: A("drone_live.jpg"), x: 0.55, y: 1.1, w: 4.2, h: 2.83, sizing: { type: "cover", w: 4.2, h: 2.83 }, shadow: shadow(), objectName: "img-drone" });
  text(s, "A drone in an unfamiliar area must find a marked target. No route is pre-written: it has to interpret what it sees.", {
    x: 5.05, y: 1.1, w: 4.4, h: 0.8, fontSize: 13, objectName: "story" });
  const x0 = 5.05, wB = 4.4, split = x0 + wB * 0.55;
  s.addShape(pres.shapes.RECTANGLE, { x: x0, y: 2.05, w: split - x0, h: 0.42, fill: { color: T.accent4, transparency: 55 }, line: { color: T.accent4, width: 0.75 }, objectName: "bar-gps" });
  s.addShape(pres.shapes.RECTANGLE, { x: split, y: 2.05, w: x0 + wB - split, h: 0.42, fill: { color: RED, transparency: 35 }, line: { color: RED, width: 0.75 }, objectName: "bar-banc" });
  text(s, "far: rough GPS", { x: x0 + 0.15, y: 2.15, w: 2.2, h: 0.25, fontSize: 12, objectName: "bar-gps-t" });
  text(s, "last 4 m: fly brain", { x: split + 0.12, y: 2.15, w: 1.9, h: 0.25, fontSize: 12, bold: true, objectName: "bar-banc-t" });
  const acts = ["turn towards the target it sees", "correct its heading after a gust", "steer past blocks on the route (in training)"];
  text(s, acts.map((a, i) => ({ text: a, options: { breakLine: i < acts.length - 1 } })), {
    x: 5.05, y: 2.65, w: 4.4, h: 1.0, fontSize: 12, paraSpaceAfter: 4, objectName: "acts" });
  stat(s, 5.05, 3.6, 2.1, `${W.model_only.reached} / ${W.model_only.n}`, "flights reached the target, teacher off", RED, "s-reach", 24);
  stat(s, 7.35, 3.6, 2.1, "35° to 1.5°", "back on course 3.9 s after a gust", T.lt1, "s-gust", 24);
  gloss(s, [["Target:", "a dark mast, easy to see for a brightness-only eye"], ["Status:", "simulation, not yet a real drone"]], 0.62, 8.85, 4.75);
  s.addNotes("Concrete use case: the final approach, where GPS drifts by metres, e.g. homing on a marked landing zone. " +
    "Far away a GPS heading is used; inside 4 m the turn comes only from BANC flight neurons. 138 of the last 150 " +
    "training flights reached the target with the teacher fully off (clear corridor). Obstacle avoidance is being trained.");
}

pres.addSection({ title: "AI and control" });

// 7. Rola AI
{
  const s = content("What makes the AI meaningful", "AI and control");
  s.addChart(pres.charts.BAR, [{ name: "Target-side accuracy", labels: ["muscle neurons, averaged", "muscle neurons, one by one", "vision neurons", "flight neurons, one by one"], values: [68, 84, 94, 99] }],
    chartOpts({
      x: 0.45, y: 1.1, w: 4.9, h: 2.95, barDir: "bar",
      chartColors: [T.accent5, T.accent5, T.accent5, RED],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisMaxVal: 115, valAxisMinVal: 0,
      showLegend: false, objectName: "chart-side",
    }));
  text(s, "How often fly neurons tell where the target is (50% = guessing)", { x: 0.62, y: 4.08, w: 4.7, h: 0.3, fontSize: 10, color: T.lt2, objectName: "chart-cap" });
  const rows = [
    ["Real biology as the network", "the brain is the measured connectome, not random weights"],
    ["Decisions from the current image", "no scripted path, the heading comes from what the eyes see now"],
    ["Closed loop", "every move changes the next picture"],
  ];
  rows.forEach(([h, d], i) => {
    const y = 1.1 + i * 0.88;
    card(s, 5.6, y, 3.85, 0.76, `r-${i}`);
    text(s, h, { x: 5.78, y: y + 0.1, w: 3.55, h: 0.28, fontSize: 12.5, bold: true, objectName: `r-${i}-h` });
    text(s, d, { x: 5.78, y: y + 0.38, w: 3.55, h: 0.36, fontSize: 10.5, color: T.lt2, objectName: `r-${i}-d` });
  });
  text(s, "We do not claim to recreate a fly. We use its wiring to build a new kind of controller.", {
    x: 5.6, y: 3.78, w: 3.85, h: 0.5, fontSize: 11, italic: true, objectName: "claim" });
  gloss(s, [["Flight neurons:", "375 descending neurons that carry flight commands from brain to body"]]);
  s.addNotes("Key finding: the target direction is readable from single descending flight neurons in 99% of scenes, " +
    "but averaging the wing-muscle neurons loses it (68%). So the decoder reads neurons one by one. Cross-validated " +
    "ridge regression, scripts/check_side_decoding.py.");
}

// 8. Kontrola i weryfikacja
{
  const s = content("You can check every decision", "AI and control");
  s.addImage({ path: A("panel_live.jpg"), x: 0.55, y: 1.05, w: 2.3, h: 3.4, shadow: shadow(), objectName: "img-panel" });
  const rows = [
    ["What it sees", "the camera view and both fly eyes, live next to the drone"],
    ["Who steers", "live share of the heading from the fly brain vs GPS, and the final command"],
    ["Why it turns", "each flight neuron's vote for left or right, from the real decoder weights"],
    ["Take over any time", "fly by keyboard while the brain only watches; 3D explorer shows every neuron"],
  ];
  rows.forEach(([h, d], i) => {
    const y = 1.05 + i * 0.86;
    card(s, 3.2, y, 6.25, 0.78, `k-${i}`);
    text(s, String(i + 1), { x: 3.4, y: y + 0.2, w: 0.35, h: 0.4, fontSize: 18, fontFace: HEAD, color: RED, objectName: `k-${i}-n` });
    text(s, h, { x: 3.85, y: y + 0.1, w: 5.4, h: 0.28, fontSize: 13, bold: true, objectName: `k-${i}-h` });
    text(s, d, { x: 3.85, y: y + 0.38, w: 5.45, h: 0.4, fontSize: 10.5, color: T.lt2, objectName: `k-${i}-d` });
  });
  gloss(s, [["Left:", "live BANC panel in flight"], ["Not a black box:", "every command traces back from pixel to neuron to wing"]]);
  s.addNotes("The live panel (sim/brain_panel.py) shows signal strength in vision, flight, muscle and haltere groups, " +
    "who decides the heading (BANC vs GPS) and the votes of individual neurons. python -m sim.viewer --brain lets a " +
    "person fly while the network only observes. The Next.js explorer shows every neuron with its real synaptic partners.");
}

pres.addSection({ title: "Results" });

// 9. Co zbudowaliśmy i ograniczenia
{
  const s = content("What we built, and what is still limited", "Results");
  eyebrow(s, "newest result: no flight assist", 0.62, 1.05, 5, RED, "new-k");
  const labels = NS.evals.map((e) => (e.after_episodes === 0 ? "start" : `${e.after_episodes}`));
  s.addChart(pres.charts.BAR, [{ name: "Reached", labels, values: NS.evals.map((e) => e.reached) }],
    chartOpts({
      x: 0.45, y: 1.25, w: 4.4, h: 2.3, barDir: "col", chartColors: [RED],
      showValue: true, dataLabelPosition: "outEnd", valAxisMaxVal: 6.8, valAxisMinVal: 0, showLegend: false, objectName: "chart-nostab",
    }));
  text(s, `Targets reached in ${NS.evals[0].n} test worlds with blocks on the route, after N training flights. ` +
    "Height, tilt and heading near the target come only from the fly brain.", {
    x: 0.62, y: 3.6, w: 4.2, h: 0.75, fontSize: 10.5, color: T.lt2, objectName: "chart-cap" });
  eyebrow(s, "built during hackyeah", 5.15, 1.05, 4, T.lt2, "built-k");
  const built = ["3D drone world with wind and random obstacles", "camera to compound eye to full BANC on a GPU",
    "decoders trained on 2 laptops at once", "live brain panel and 3D connectome explorer"];
  text(s, built.map((b, i) => ({ text: b, options: { breakLine: i < built.length - 1 } })), {
    x: 5.15, y: 1.32, w: 4.3, h: 1.3, fontSize: 11.5, paraSpaceAfter: 3, objectName: "built" });
  eyebrow(s, "still limited", 5.15, 2.75, 4, RED, "lim-k");
  const lim = ["simulation only, no physical drone yet", "far away GPS sets the course; a simulated autopilot holds the tilt the brain asks for",
    "left eye feeds ~3x fewer neurons (fewer labels in BANC)"];
  text(s, lim.map((b, i) => ({ text: b, options: { breakLine: i < lim.length - 1 } })), {
    x: 5.15, y: 3.02, w: 4.3, h: 1.35, fontSize: 11.5, paraSpaceAfter: 3, color: T.lt1, objectName: "lim" });
  gloss(s, [["From BANC:", "neurons, synapses, excite or inhibit, groups"],
    ["Our assumptions:", "the activity equation, how vision enters, how neurons become drone commands"]]);
  s.addNotes(`Newest run (world_nostab): thrust, roll and pitch only from BANC, drone sensors weighted 0. Evaluation on ` +
    `${NS.evals[0].n} fixed worlds with 2 blocks on the path: ${NS.evals.map((e) => e.reached).join(", ")} reached ` +
    `after ${NS.evals.map((e) => e.after_episodes).join(", ")} flights; mean closest approach ` +
    `${NS.evals[0].mean_min_dist.toFixed(1)} m to ${NS.evals[NS.evals.length - 1].mean_min_dist.toFixed(1)} m. ` +
    "Only 6 test worlds, so treat it as promising, not final. The attitude controller of the simulator still holds the " +
    "commanded tilt and GPS sets the heading beyond 4 m.");
}

// 10. Dlaczego to ważne + zespół i ujawnienie
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Results" });
  text(s, "FROM A FLY'S VISUAL WORLD", { x: 0.6, y: 0.55, w: 8.8, h: 0.45, fontSize: 20, fontFace: HEAD, charSpacing: 1, objectName: "close-1" });
  text(s, "TO AUTONOMOUS NAVIGATION.", { x: 0.6, y: 1.0, w: 8.8, h: 0.45, fontSize: 20, fontFace: HEAD, charSpacing: 1, color: RED, objectName: "close-2" });
  eyebrow(s, "why it matters", 0.6, 1.8, 4, RED, "why-k");
  const why = ["an original idea grounded in real neuroscience", "AI at the core: vision, connectome, decoder",
    "a working interactive demo", "transparent outputs, a human can take over", "grows into research, education, robotics"];
  text(s, why.map((b, i) => ({ text: b, options: { breakLine: i < why.length - 1 } })), {
    x: 0.6, y: 2.08, w: 4.5, h: 1.6, fontSize: 11.5, paraSpaceAfter: 2, objectName: "why" });
  text(s, "Team the roook: Dawid Wypych, Mateusz Nowaczek, Krzysztof Mazur", { x: 0.6, y: 3.85, w: 8.8, h: 0.3, fontSize: 11, objectName: "members" });
  eyebrow(s, "built on, with thanks", 5.3, 1.8, 4, RED, "src-k");
  text(s, [
    { text: "BANC v888 connectome", options: { bold: true, breakLine: true } },
    { text: "Bates et al. 2026, doi 10.7910/DVN/7WTH1N", options: { color: T.lt2, breakLine: true } },
    { text: "FlyVis, FlyGym, MuJoCo, Skydio X2 model", options: { bold: true, breakLine: true } },
    { text: "open models, simulator, MuJoCo Menagerie", options: { color: T.lt2, breakLine: true } },
    { text: "PyTorch, Next.js, three.js", options: { bold: true, breakLine: true } },
    { text: "GPU simulation and the 3D explorer", options: { color: T.lt2, breakLine: true } },
    { text: "Claude Code", options: { bold: true, breakLine: true } },
    { text: "AI assistant for code, docs and slides", options: { color: T.lt2 } },
  ], { x: 5.3, y: 2.08, w: 4.2, h: 2.2, fontSize: 11, paraSpaceAfter: 2, objectName: "sources" });
  text(s, "GITHUB.COM/FALCONDEVX/NEUROFLY", { x: 0.6, y: 4.85, w: 6, h: 0.3, fontSize: 10, color: RED, charSpacing: 4,
    hyperlink: { url: "https://github.com/FalconDevX/neurofly" }, objectName: "repo" });
  s.addNotes("Disclosure: BANC data, FlyVis, FlyGym, MuJoCo and the X2 model are open resources; PyTorch, Next.js and " +
    "three.js are libraries; Claude Code was used as an AI coding assistant. We did not use Godot or FlyWire data. " +
    "Everything else was built during HackYeah.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (process.env.APPLY_THEME) await require(process.env.APPLY_THEME).applyTheme(OUT, THEME);
  console.log("zapisano", OUT);
})();
