// Wersja MVP (10 slajdów pod kryteria HackYeah AI): node build_mvp.js [out.pptx], potem python merge_title.py out.pptx
// Nagłówek (motyw, układy, pomocnicze) skopiowany z build.js.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || path.join(__dirname, "NeuroFly_mvp_raw.pptx");
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
// MVP: 10 slajdów pod kryteria HackYeah AI (Idea 30, Kategoria 20, Użyteczność 20, Design 20, Kompletność 10).
// Slajd 1 to miejsce na tytuł; merge_title.py podmienia go na ręcznie poprawiony slajd 1 z NeuroFly.pptx.
// =================================================================================================
pres.addSection({ title: "Start" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Start" });
  text(s, "NEUROFLY", { x: 0.6, y: 2.05, w: 5.2, h: 0.85, fontSize: 40, fontFace: HEAD, objectName: "title" });
}

pres.addSection({ title: "Idea" });

// 2. Hak: wagi zmierzone, nie wytrenowane
{
  const s = content("Measured, not trained", "Idea");
  // lewa kolumna: porównanie i liczby, prawa: cały connectome BANC z podpisanymi częściami
  card(s, 0.55, 1.05, 5.4, 0.95, "c-ai");
  eyebrow(s, "typical AI", 0.78, 1.15, 3.5, T.lt2, "c-ai-k");
  text(s, "Random weights, learned from millions of examples. Nobody can say what one weight means.", {
    x: 0.78, y: 1.4, w: 5.0, h: 0.55, fontSize: 12, color: T.lt2, objectName: "c-ai-d" });
  card(s, 0.55, 2.12, 5.4, 1.08, "c-nf", RED);
  eyebrow(s, "neurofly", 0.78, 2.22, 3.5, RED, "c-nf-k");
  text(s, "The real wiring of a fruit fly, measured synapse by synapse. Tuned by evolution, every neuron has a name.", {
    x: 0.78, y: 2.47, w: 5.0, h: 0.7, fontSize: 13, objectName: "c-nf-d" });
  stat(s, 0.55, 3.35, 2.5, "18.6 M", "real synapses in the controller", RED, "s-syn");
  stat(s, 3.3, 3.35, 2.6, "0", "invented connections or weights", T.lt1, "s-zero");
  s.addImage({ path: A("connectome_3d.jpg"), x: 6.3, y: 1.0, w: 2.95, h: 3.58, shadow: shadow(), objectName: "img-connectome" });
  gloss(s, [["Connectome:", "the complete wiring diagram of a nervous system"],
    ["Right:", "the whole BANC connectome, 175,401 neurons at their real positions"]]);
  s.addNotes("Hook: AI usually means random weights trained on huge data. We do the opposite: the network is the real " +
    "fly connectome, mapped synapse by synapse. We run it as the brain of a drone in a closed loop. The brain itself is " +
    "never trained; we only learn how to read its flight neurons.");
}

// 3. Przypadek użycia: ostatnie metry
{
  const s = content("Use case: the last metres", "Idea");
  s.addImage({ path: A("world.jpg"), x: 0.55, y: 1.15, w: 4.2, h: 2.95, shadow: shadow(), objectName: "img-world" });
  text(s, "Simulated world: hills, blocks, wind, a target mast", { x: 0.55, y: 4.17, w: 4.2, h: 0.25, fontSize: 10, color: T.lt2, objectName: "cap-world" });
  text(s, "A rescue drone homes in on a beacon among rubble or trees, where GPS drifts by metres. " +
    "GPS brings it close. The last 4 m are steered by the fly brain, from a plain camera.", {
    x: 5.05, y: 1.15, w: 4.4, h: 1.2, fontSize: 13, objectName: "story" });
  const x0 = 5.05, wB = 4.4, split = x0 + wB * 0.55;
  s.addShape(pres.shapes.RECTANGLE, { x: x0, y: 2.55, w: split - x0, h: 0.45, fill: { color: T.accent4, transparency: 55 }, line: { color: T.accent4, width: 0.75 }, objectName: "bar-gps" });
  s.addShape(pres.shapes.RECTANGLE, { x: split, y: 2.55, w: x0 + wB - split, h: 0.45, fill: { color: RED, transparency: 35 }, line: { color: RED, width: 0.75 }, objectName: "bar-banc" });
  text(s, "far: GPS", { x: x0 + 0.15, y: 2.66, w: 2, h: 0.25, fontSize: 12, objectName: "bar-gps-t" });
  text(s, "near: fly brain", { x: split + 0.15, y: 2.66, w: 1.9, h: 0.25, fontSize: 12, bold: true, objectName: "bar-banc-t" });
  stat(s, 5.05, 3.15, 2.1, "2.5 m", "GPS error in our simulation", T.accent4, "s-gps");
  stat(s, 7.35, 3.15, 2.1, "100%", "of the final heading comes from fly neurons", RED, "s-banc");
  gloss(s, [["Beacon:", "the target the drone must reach, here a dark mast"],
    ["Heading:", "the direction the nose points"], ["Status:", "shown in simulation, not yet on a real drone"]]);
  s.addNotes("Concrete use case: the final approach, where GPS is not precise enough. Far away a GPS pilot flies; inside " +
    "4 m the turn command comes only from BANC flight neurons. Drone sensors keep it level and at height. Today this runs " +
    "in MuJoCo; the hardware step is on slide 9.");
}

pres.addSection({ title: "How it works" });

// 4. Pętla w 8 krokach, z zaznaczeniem, gdzie jest AI
{
  const s = content("How it works: one loop, 8 steps", "How it works");
  const nodes = [
    ["Camera", "2 eyes, 157° view each", "eye_right.jpg", T.accent1, ""],
    ["Retina", "721 facets per eye", "retina_right.jpg", T.accent1, ""],
    ["FlyVis", "fly vision model", "thumb_flyvis.jpg", T.accent1, "TRAINED NET"],
    ["Into the brain", "22,462 neurons get input", "thumb_map.jpg", T.accent1, ""],
    ["Fly brain", "175,401 neurons on GPU", "thumb_wiring.jpg", T.accent2, "REAL CONNECTOME"],
    ["Flight neurons", "commands to the wings", "thumb_circuit.jpg", T.accent3, "REAL CONNECTOME"],
    ["Decoder", "turns them into 4 numbers", "thumb_decoder.jpg", T.lt2, "IMITATION LEARNING"],
    ["Drone", "moves, then back to 1", "drone_chase.jpg", T.accent4, ""],
  ];
  const w = 1.95, h = 1.5, gx = 0.35, y1 = 1.1, y2 = 2.92;
  nodes.forEach(([hd, d, f, col, ai], i) => {
    const x = 0.55 + (i % 4) * (w + gx), y = i < 4 ? y1 : y2;
    card(s, x, y, w, h, `node-${i}`, ai ? RED : undefined);
    img(s, f, x + 0.07, y + 0.07, w - 0.14, 0.76, `node-${i}-img`);
    if (ai) {
      s.addShape(pres.shapes.RECTANGLE, { x: x + 0.07, y: y + 0.07, w: 1.42, h: 0.2, fill: { color: RED }, line: { color: RED, width: 0 }, objectName: `node-${i}-ai` });
      text(s, ai, { x: x + 0.12, y: y + 0.095, w: 1.35, h: 0.16, fontSize: 7, bold: true, charSpacing: 1, objectName: `node-${i}-ai-t` });
    }
    text(s, [{ text: `${i + 1}  `, options: { color: col, fontFace: HEAD, fontSize: 11 } }, { text: hd, options: { bold: true } }],
      { x: x + 0.14, y: y + 0.9, w: w - 0.24, h: 0.28, fontSize: 13, objectName: `node-${i}-title` });
    text(s, d, { x: x + 0.14, y: y + 1.18, w: w - 0.24, h: 0.28, fontSize: 10, color: T.lt2, objectName: `node-${i}-desc` });
  });
  gloss(s, [["Red tag:", "where AI works in the loop"], ["FlyVis:", "published model of the fly eye and visual brain"],
    ["Speed:", "2.6 ms per frame for the whole connectome on a laptop GPU"]]);
  s.addNotes("Every camera frame goes once around the loop. Three AI parts: FlyVis (a trained network of the fly visual " +
    "system), the BANC connectome itself (a biological neural network, simulated as a rate model on the GPU) and the " +
    "decoder, learned by imitation (DAgger) and trained on 2 laptop GPUs at once.");
}

// 5. Mózg BANC
{
  const s = content("The brain: real, not invented", "How it works");
  s.addImage({ path: A("circuit_3d.jpg"), x: 0.4, y: 1.0, w: 3.0, h: 3.64, objectName: "render-circuit" });
  stat(s, 3.9, 1.15, 2.7, "175,401", "neurons in the fly brain and nerve cord", T.lt1, "s-neurons");
  stat(s, 6.75, 1.15, 2.7, "1.53 M", "neuron-to-neuron connections we use", T.lt1, "s-edges");
  const groups = [["375", "flight neurons from the brain", T.accent2], ["60", "wing muscle neurons", T.accent3], ["428", "haltere neurons, the fly's gyroscope", T.accent4]];
  groups.forEach(([c, d, col], i) => {
    const x = 3.9 + i * 1.9;
    text(s, c, { x, y: 2.5, w: 1.75, h: 0.45, fontSize: 22, fontFace: "Roboto Light", color: col, objectName: `g-${i}-c` });
    text(s, d, { x, y: 2.97, w: 1.75, h: 0.5, fontSize: 10, color: T.lt2, objectName: `g-${i}-d` });
  });
  text(s, "Every group comes from official BANC labels. We never guess a neuron's name or add a connection.", {
    x: 3.9, y: 3.7, w: 5.5, h: 0.5, fontSize: 12, italic: true, color: T.lt1, objectName: "rule" });
  gloss(s, [["Left:", "863 flight neurons in full 3D shape, from BANC skeletons"],
    ["Haltere:", "tiny hind wing that senses rotation"]], 3.9, 5.55);
  s.addNotes("Official BANC v888 (Bates et al. 2026, public Lee Lab bucket, DOI 10.7910/DVN/7WTH1N). Groups from " +
    "super_cluster, cell_function and body_part_sensory columns. Edges with at least 5 synapses, sign from the predicted " +
    "neurotransmitter.");
}

pres.addSection({ title: "Results" });

// 6. Odkrycie: 99%
{
  const s = content("Single neurons know the target", "Results");
  s.addChart(pres.charts.BAR, [{ name: "Target-side accuracy", labels: ["wing muscle neurons, averaged", "wing muscle neurons, one by one", "vision neurons", "flight neurons, one by one"], values: [68, 84, 94, 99] }],
    chartOpts({
      x: 0.5, y: 1.15, w: 5.6, h: 3.35, barDir: "bar",
      chartColors: [T.accent5, T.accent5, T.accent5, RED],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisMaxVal: 110, valAxisMinVal: 0,
      showLegend: false, objectName: "chart-side",
    }));
  stat(s, 6.6, 1.3, 3, "99%", "of scenes where single flight neurons tell left from right correctly", RED, "s-dn");
  stat(s, 6.6, 2.6, 3, "68%", "if you average the muscle neurons: left and right cancel out", T.lt2, "s-mn");
  text(s, "So we read the brain neuron by neuron.", { x: 6.6, y: 3.85, w: 3, h: 0.5, fontSize: 12, italic: true, objectName: "so" });
  gloss(s, [["Accuracy:", "how often we can tell which side the target is on (50% = guessing)"]]);
  s.addNotes("Ridge regression angle ~ activity, target from −90° to +90° at 3 distances, validated on the distance left " +
    "out (scripts/check_side_decoding.py). Direction lives in individual descending neurons and is lost when averaged.");
}

// 7. Wyniki
{
  const s = content("Results: it learns to fly to the target", "Results");
  const rows = [
    ["25.1° to 2.65°", "average heading miss, before vs after 1,000 training flights"],
    ["138 / 150", "flights reached the target with no teacher at all (92%)"],
    ["18.2 m to 2.1 m", "closest approach in 6 new worlds never seen in training"],
    ["35° to 1.5°", "pushed off course by a gust, back on course in 3.9 s"],
  ];
  rows.forEach(([v, d], i) => {
    const y = 1.12 + i * 0.84;
    card(s, 0.55, y, 5.5, 0.72, `r-${i}`);
    text(s, v, { x: 0.78, y: y + 0.13, w: 2.6, h: 0.45, fontSize: 22, fontFace: "Roboto Light", color: RED, objectName: `r-${i}-v` });
    text(s, d, { x: 3.4, y: y + 0.12, w: 2.5, h: 0.5, fontSize: 11, color: T.lt2, valign: "middle", objectName: `r-${i}-d` });
  });
  s.addImage({ path: A("drone_live.jpg"), x: 6.35, y: 1.12, w: 3.1, h: 2.09, shadow: shadow(), objectName: "img-drone" });
  s.addImage({ path: A("eyes_live.jpg"), x: 6.35, y: 3.3, w: 3.1, h: 1.18, shadow: shadow(), objectName: "img-eyes" });
  gloss(s, [["Heading miss:", "angle between the nose and the target, 0° = straight at it"],
    ["Bottom right:", "what the drone's two eyes see, the target is the dark mast"]]);
  s.addNotes("Real FlyVis + full BANC v888 in MuJoCo. Heading: 1,000 flights on 2 GPUs (planB_distributed). Worlds: 300 " +
    "episodes with noisy sensors and wind; the last 150 with the teacher fully switched off reached the target 138 times. " +
    "New worlds: seeds 101–106, best checkpoint 5 of 6 reached. Gust: planB_dn decoder, thrust held by the simulator.");
}

// 8. Kontrola i weryfikacja
{
  const s = content("You stay in control", "Results");
  s.addImage({ path: A("panel_live.jpg"), x: 0.55, y: 1.05, w: 2.3, h: 3.4, shadow: shadow(), objectName: "img-panel" });
  const rows = [
    ["See who steers", "Live share of the heading from the fly brain vs GPS, plus the final command."],
    ["See why it turns", "Each flight neuron's vote for left or right, from the real decoder weights."],
    ["Take over any time", "Sensors keep the drone level, GPS takes over far away, a pilot can fly by keyboard while the brain only watches."],
    ["Check every neuron", "3D explorer: every neuron, its class and its real synaptic partners from BANC."],
  ];
  rows.forEach(([h, d], i) => {
    const y = 1.05 + i * 0.86;
    card(s, 3.2, y, 6.25, 0.78, `k-${i}`);
    text(s, String(i + 1), { x: 3.4, y: y + 0.2, w: 0.35, h: 0.4, fontSize: 18, fontFace: HEAD, color: RED, objectName: `k-${i}-n` });
    text(s, h, { x: 3.85, y: y + 0.1, w: 5.4, h: 0.28, fontSize: 13, bold: true, objectName: `k-${i}-h` });
    text(s, d, { x: 3.85, y: y + 0.38, w: 5.45, h: 0.4, fontSize: 10.5, color: T.lt2, objectName: `k-${i}-d` });
  });
  gloss(s, [["Left:", "live BANC panel in flight, a brain you can watch"],
    ["Not a black box:", "every command traces back from pixel to neuron to wing"]]);
  s.addNotes("How users verify outputs and stay in control: the live panel (sim/brain_panel.py) shows signal strength in " +
    "vision, DN, MN and haltere groups, who decides the heading (BANC vs GPS) and the votes of individual neurons. " +
    "python -m sim.viewer --brain lets a person fly with WASD while the network only observes. The Next.js explorer " +
    "shows every neuron with its top synaptic partners from the real edge list.");
}

pres.addSection({ title: "Honesty" });

// 9. Ograniczenia i plan
{
  const s = content("Honest limits and next steps", "Honesty");
  eyebrow(s, "what does not work yet", 0.62, 1.1, 4, RED, "lim-k");
  const lim = [
    ["Lift", "fly-brain thrust too weak, the simulator holds height"],
    ["Left eye", "fewer labelled neurons in BANC, ~3× less input"],
    ["Obstacles", "the fly brain does not avoid blocks yet"],
    ["Hardware", "simulation only, no real drone flown"],
  ];
  lim.forEach(([h, d], i) => {
    const y = 1.4 + i * 0.72;
    text(s, h, { x: 0.62, y, w: 1.3, h: 0.3, fontSize: 13, bold: true, objectName: `lim-${i}-h` });
    text(s, d, { x: 1.95, y: y + 0.02, w: 2.85, h: 0.55, fontSize: 11, color: T.lt2, objectName: `lim-${i}-d` });
  });
  const road = [
    ["TODAY", "Full BANC steers a simulated drone in random worlds", RED, "demonstrated"],
    ["NEXT", "Two cameras on a micro-drone, connectome on an onboard GPU", ORANGE, "planned"],
    ["LATER", "Search and rescue homing, low-power neuromorphic chips", T.lt2, "potential"],
  ];
  road.forEach(([k, d, col, st], i) => {
    const y = 1.1 + i * 1.12;
    card(s, 5.15, y, 4.3, 0.98, `road-${i}`, i === 0 ? RED : undefined);
    text(s, k, { x: 5.38, y: y + 0.14, w: 1.5, h: 0.3, fontSize: 13, fontFace: HEAD, color: col, objectName: `road-${i}-k` });
    text(s, st, { x: 7.7, y: y + 0.16, w: 1.55, h: 0.26, fontSize: 9, color: T.lt2, align: "right", objectName: `road-${i}-s` });
    text(s, d, { x: 5.38, y: y + 0.5, w: 3.9, h: 0.42, fontSize: 11, objectName: `road-${i}-d` });
  });
  gloss(s, [["From BANC:", "neurons, synapses, excite or inhibit, neuron groups, 3D shapes"],
    ["Our assumptions:", "the neuron activity equation, how vision enters, how flight neurons become drone commands"]]);
  s.addNotes("We would rather show the limits than hide them. Each is a concrete next step. The split between what is " +
    "BANC data and what is our modelling choice is stated on every result.");
}

// 10. Zespół, zasoby, zamknięcie
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Honesty" });
  text(s, "EVOLUTION WROTE THE WEIGHTS.", { x: 0.6, y: 0.55, w: 8.8, h: 0.45, fontSize: 20, fontFace: HEAD, charSpacing: 1, objectName: "close-1" });
  text(s, "WE GAVE THEM WINGS.", { x: 0.6, y: 1.0, w: 8.8, h: 0.45, fontSize: 20, fontFace: HEAD, charSpacing: 1, color: RED, objectName: "close-2" });
  eyebrow(s, "team the roook", 0.6, 1.8, 4, RED, "team-k");
  const team = [["The eyes", "camera, fly retina, FlyVis, input to BANC"], ["The brain", "full connectome on GPU, decoders, training"], ["Body and world", "drone, sensors, wind, random worlds"]];
  team.forEach(([h, d], i) => {
    const y = 2.1 + i * 0.5;
    text(s, h, { x: 0.6, y, w: 1.6, h: 0.3, fontSize: 13, bold: true, objectName: `team-${i}-h` });
    text(s, d, { x: 2.2, y: y + 0.02, w: 3.0, h: 0.3, fontSize: 11, color: T.lt2, objectName: `team-${i}-d` });
  });
  text(s, "Dawid Wypych      Mateusz Nowaczek      Krzysztof Mazur", { x: 0.6, y: 3.65, w: 4.6, h: 0.3, fontSize: 12, objectName: "members" });
  eyebrow(s, "built on, with thanks", 5.3, 1.8, 4, RED, "src-k");
  text(s, [
    { text: "BANC v888 connectome", options: { bold: true, breakLine: true } },
    { text: "Bates et al. 2026, doi 10.7910/DVN/7WTH1N", options: { color: T.lt2, breakLine: true } },
    { text: "FlyVis, FlyGym, MuJoCo, Skydio X2 model", options: { bold: true, breakLine: true } },
    { text: "open models and simulator (MuJoCo Menagerie)", options: { color: T.lt2, breakLine: true } },
    { text: "Claude Code", options: { bold: true, breakLine: true } },
    { text: "AI assistant for code, docs and slides", options: { color: T.lt2 } },
  ], { x: 5.3, y: 2.1, w: 4.2, h: 1.9, fontSize: 11, paraSpaceAfter: 2, objectName: "sources" });
  text(s, "Built during HackYeah: the whole loop, controller, decoders and training, drone simulator, live panel, 3D explorer.", {
    x: 0.6, y: 4.3, w: 8.8, h: 0.3, fontSize: 10, color: T.lt2, objectName: "built" });
  text(s, "GITHUB.COM/FALCONDEVX/NEUROFLY", { x: 0.6, y: 4.85, w: 6, h: 0.3, fontSize: 10, color: RED, charSpacing: 4,
    hyperlink: { url: "https://github.com/FalconDevX/neurofly" }, objectName: "repo" });
  s.addNotes("Close: evolution wrote the weights, we gave them wings. Team roles follow the three parts of the loop. " +
    "Disclosure as required: BANC data, FlyVis, FlyGym, MuJoCo and the X2 model are open resources; Claude Code was used " +
    "as a coding assistant. Everything else was built during HackYeah.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (process.env.APPLY_THEME) await require(process.env.APPLY_THEME).applyTheme(OUT, THEME);
  console.log("zapisano", OUT);
})();
