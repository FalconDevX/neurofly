// Prezentacja NeuroFly: minimalistyczna, ciemna, z renderami 3D z prawdziwych danych BANC v888 i MuJoCo.
//   node build.js [NeuroFly.pptx]   (npm i pptxgenjs; grafiki w assets/ robi render_assets.py, liczby z repo / CLAUDE.md)
//   APPLY_THEME=<skill pptx>/scripts/apply_theme.js — opcjonalnie wpisuje kolory motywu do pliku
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || path.join(__dirname, "NeuroFly.pptx");
const A = (f) => path.join(__dirname, "assets", f);
const DATA = JSON.parse(fs.readFileSync(A("data.json"), "utf8"));

const THEME = {
  name: "NeuroFly",
  headFontFace: "Michroma", // szeroki, geometryczny (styl „Aquire”); OFL — osadzony w .pptx (embed_fonts.ps1)
  bodyFontFace: "Roboto",
  colors: {
    dk1: "09090B", // tło (zinc-950)
    lt1: "FAFAFA", // tekst
    dk2: "18181B", // karty (zinc-900)
    lt2: "A1A1AA", // tekst drugorzędny (zinc-400)
    accent1: "2FD3C4", // teal — wzrok, jedyny akcent interfejsu
    accent2: "FFB547", // amber — neurony zstępujące (DN)
    accent3: "FF4FB0", // pink — motoneurony
    accent4: "4F8CFF", // blue — haltery / sensoryka
    accent5: "52525B", // zinc-600
    accent6: "27272A", // zinc-800 — linie, ramki
    hlink: "2FD3C4",
    folHlink: "A1A1AA",
  },
};
const T = THEME.colors;
// akcent interfejsu w stylu plakatu: czerwono-różowy → pomarańczowy (kolory danych w accent1–4 zostają jak w renderach)
const RED = "FF2D6F", ORANGE = "FF6A2B";
const HEAD = THEME.headFontFace;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 × 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "NeuroFly — a fruit-fly brain flies a drone";
pres.author = "NeuroFly team";
const C = pres.SchemeColor;

// ---------- układy (layouts) ----------
pres.defineSlideMaster({
  title: "TITLE",
  background: { path: A("shards_title.jpg") },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { path: A("shards_corner.jpg") },
  objects: [
    { image: { path: A("accent_line.png"), x: 0.44, y: 0.2, w: 0.04, h: 0.82 } },   // czerwona linia przy tytule
    { image: { path: A("chevron.png"), x: 9.05, y: 0.28, w: 0.45, h: 0.3 } },        // podwójne trójkąty
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

function card(slide, x, y, w, h, name) {
  // ostre prostokąty (geometrycznie, jak plakat), lekko przezroczyste nad tłem z odłamkami
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: T.dk2, transparency: 12 }, line: { color: "3F3F46", width: 0.75 }, shadow: shadow(), objectName: name,
  });
}

function text(slide, t, opts) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 14, color: T.lt1, valign: "top", ...opts });
}

function stat(slide, x, y, w, value, label, color, name) {
  text(slide, value, { x, y, w, h: 0.6, fontSize: 32, fontFace: "Roboto Light", color: color || T.lt1, objectName: `${name}-value` });
  text(slide, label, { x, y: y + 0.62, w, h: 0.5, fontSize: 12, color: T.lt2, objectName: `${name}-label` });
}

function tag(slide, x, y, label, color, name) {
  // mała pigułka: kropka koloru + etykieta (oznaczenia „z BANC” / „nasze założenie”)
  slide.addShape(pres.shapes.OVAL, { x, y: y + 0.07, w: 0.12, h: 0.12, fill: { color }, line: { color, width: 0 }, objectName: `${name}-dot` });
  text(slide, label, { x: x + 0.2, y, w: 3, h: 0.26, fontSize: 11, color: T.lt2, objectName: `${name}-label` });
}

function arrow(slide, x1, y1, x2, y2, color, name) {
  // szerokość i wysokość linii muszą być ≥ 0 (ujemne psują plik dla PowerPointa) — kierunek przez odbicie
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: color || T.accent5, width: 1.5, endArrowType: "triangle" }, objectName: name,
  });
}

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  // mała czerwona etykieta sekcji nad tytułem (rozstrzelone wersaliki, jak „FREE TYPEFACE” na plakacie)
  text(s, section.toUpperCase(), { x: 0.62, y: 0.24, w: 6, h: 0.22, fontSize: 8, color: RED, charSpacing: 6, objectName: "eyebrow" });
  s.addText(title.toUpperCase(), { placeholder: "title" });
  return s;
}

// =================================================================================================
// 1. Tytuł
pres.addSection({ title: "Start" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Start" });
  s.addImage({ path: A("fly_xray.jpg"), x: 3.55, y: 0.1, w: 6.45, h: 3.72, objectName: "render-fly-xray" });
  s.addImage({ path: A("logo.png"), x: 0.6, y: 0.95, w: 1.0, h: 0.59, objectName: "logo" });
  text(s, "BANC V888 CONNECTOME", { x: 0.6, y: 1.78, w: 5.2, h: 0.25, fontSize: 9, color: RED, charSpacing: 5, objectName: "eyebrow" });
  text(s, "NEUROFLY", { x: 0.6, y: 2.05, w: 5.2, h: 0.85, fontSize: 40, fontFace: HEAD, charSpacing: 2, objectName: "title" });
  s.addImage({ path: A("accent_line.png"), x: 0.6, y: 3.05, w: 0.05, h: 0.95, objectName: "accent-line" });
  s.addImage({ path: A("chevron.png"), x: 0.8, y: 3.1, w: 0.42, h: 0.28, objectName: "chevron" });
  text(s, "A fruit-fly brain\nflies a drone", { x: 1.35, y: 3.05, w: 3.0, h: 0.8, fontSize: 18, color: T.lt2, objectName: "subtitle" });
  text(s, "FULL BANC V888 CONNECTOME  ·  175,401 NEURONS  ·  CLOSED LOOP IN MUJOCO", {
    x: 0.6, y: 4.75, w: 8.8, h: 0.3, fontSize: 8, color: T.lt2, charSpacing: 4, objectName: "footer",
  });
  s.addNotes("NeuroFly: the drone camera image goes through a model of the fly eye (FlyVis), then through the real " +
    "Drosophila connectome (BANC v888, brain + ventral nerve cord), and flight-neuron activity steers the drone in simulation. " +
    "Render: semi-transparent NeuroMechFly body (FlyGym) with the BANC connectome inside — body alignment is illustrative.");
}

// 2. Idea: pętla zamknięta
pres.addSection({ title: "Idea" });
{
  const s = content("Closed loop: vision → flight", "Idea");
  const steps = [
    ["Image", "two eye cameras on the drone", T.accent1],
    ["Connectome", "BANC v888 on the GPU", T.accent2],
    ["Control", "flight neurons → commands", T.accent3],
    ["Motion", "drone in MuJoCo → new image", T.accent4],
  ];
  const cx = 5, cy = 3.0, R = 1.55;
  steps.forEach(([h, d, col], i) => {
    const a = -Math.PI / 2 + (i * Math.PI) / 2;
    const x = cx + R * 1.55 * Math.cos(a) - 1.15, y = cy + R * Math.sin(a) - 0.42;
    card(s, x, y, 2.3, 0.84, `step-${i}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: y + 0.18, w: 0.16, h: 0.16, fill: { color: col }, line: { color: col, width: 0 }, objectName: `step-${i}-dot` });
    text(s, h, { x: x + 0.45, y: y + 0.1, w: 1.75, h: 0.32, fontSize: 15, bold: true, objectName: `step-${i}-title` });
    text(s, d, { x: x + 0.45, y: y + 0.44, w: 1.8, h: 0.3, fontSize: 11, color: T.lt2, objectName: `step-${i}-desc` });
  });
  s.addImage({ path: A("logo.png"), x: 4.45, y: 2.68, w: 1.1, h: 0.65, objectName: "logo-center" });
  s.addNotes("Every camera frame goes through the whole loop ~30 times per second. There is no hand-coded heading " +
    "controller: the direction to the target is read out from descending-neuron (DN) activity in BANC.");
}

// 3. Pipeline
pres.addSection({ title: "Pipeline" });
{
  const s = content("Pipeline: from pixel to propeller", "Pipeline");
  const nodes = [
    ["Camera", "2 MuJoCo eyes, 157°, ±70°", T.accent1],
    ["Retina", "FlyGym: 721 ommatidia / eye", T.accent1],
    ["FlyVis", "optic lobe model", T.accent1],
    ["Map → BANC", "22,462 v888 neurons", T.accent1],
    ["BANC v888", "175,401 neurons, GPU", T.accent2],
    ["Readout", "6 MN groups + 375 flight DNs", T.accent3],
    ["Decoder", "linear, trained (DAgger)", T.lt2],
    ["Drone", "thrust · roll · pitch · yaw", T.accent4],
  ];
  const w = 1.92, h = 1.02, gx = 0.37, y1 = 1.45, y2 = 3.25;
  nodes.forEach(([hd, d, col], i) => {
    const row = i < 4 ? 0 : 1, k = i < 4 ? i : 7 - i; // drugi rząd od prawej do lewej (pętla)
    const x = 0.5 + k * (w + gx), y = row ? y2 : y1;
    card(s, x, y, w, h, `node-${i}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.16, y: y + 0.2, w: 0.14, h: 0.14, fill: { color: col }, line: { color: col, width: 0 }, objectName: `node-${i}-dot` });
    text(s, hd, { x: x + 0.38, y: y + 0.12, w: w - 0.5, h: 0.32, fontSize: 15, bold: true, objectName: `node-${i}-title` });
    text(s, d, { x: x + 0.16, y: y + 0.52, w: w - 0.3, h: 0.42, fontSize: 11, color: T.lt2, objectName: `node-${i}-desc` });
    if (row === 0 && k < 3) arrow(s, x + w + 0.04, y + h / 2, x + w + gx - 0.04, y + h / 2, T.accent5, `arrow-${i}`);
    if (row === 1 && k > 0) arrow(s, x - 0.04, y + h / 2, x - gx + 0.04, y + h / 2, T.accent5, `arrow-${i}`);
  });
  const xr = 0.5 + 3 * (w + gx) + w / 2;
  arrow(s, xr, y1 + h + 0.04, xr, y2 - 0.04, T.accent5, "arrow-down");
  text(s, "next frame: new camera image", { x: 0.5, y: 4.45, w: 4, h: 0.3, fontSize: 11, color: T.accent5, objectName: "loop-note" });
  s.addNotes("Top row: everything that happens before the connectome. Bottom row: connectome, flight-neuron readout, " +
    "decoder and drone. One frame (FlyVis + 4 BANC dynamics substeps on the GPU) takes ~15–20 ms.");
}

// 4. Przed connectomem: oczy → siatkówka
{
  const s = content("Before BANC: the fly eye", "Pipeline");
  const ims = [["eye_left.jpg", "Left camera"], ["retina_left.jpg", "Left retina"], ["eye_right.jpg", "Right camera"], ["retina_right.jpg", "Right retina"]];
  ims.forEach(([f, cap], i) => {
    const x = 0.5 + i * 2.27;
    s.addImage({ path: A(f), x, y: 1.3, w: 2.0, h: 2.28, shadow: shadow(), objectName: `img-${i}` });
    text(s, cap, { x, y: 3.66, w: 2.0, h: 0.28, fontSize: 11, color: T.lt2, objectName: `cap-${i}` });
  });
  text(s, "MuJoCo cameras with a fly-eye field of view → fisheye correction → 721 hexagonal ommatidia per eye (FlyGym) → " +
    "FlyVis: a trained, connectome-constrained optic lobe model → cell-type activity mapped onto BANC v888 neurons.", {
    x: 0.5, y: 4.08, w: 9, h: 0.8, fontSize: 13, color: T.lt1, objectName: "explain",
  });
  s.addNotes("The target (dark pole) is on the right: only the right eye sees it, as a dark column of ommatidia.");
}

// 5. Connectome BANC v888
pres.addSection({ title: "Connectome" });
{
  const s = content("BANC v888: brain + nerve cord", "Connectome");
  s.addImage({ path: A("connectome_3d.jpg"), x: 0.4, y: 1.05, w: 3.35, h: 4.07, objectName: "render-3d" });
  stat(s, 4.3, 1.25, 2.6, "175,401", "neurons (glia and trachea excluded)", T.lt1, "s-neurons");
  stat(s, 7.0, 1.25, 2.6, "18.6 M", "synapses in connections ≥ 5", T.lt1, "s-syn");
  stat(s, 4.3, 2.55, 2.6, "1.53 M", "connections in the model graph", T.lt1, "s-edges");
  stat(s, 7.0, 2.55, 2.6, "863", "flight-neuron skeletons (SWC)", T.lt1, "s-skel");
  tag(s, 4.3, 3.95, "optic lobes · 105,646", T.accent1, "t-ol");
  tag(s, 4.3, 4.27, "central brain · 42,620", "A98BFF", "t-cb");
  tag(s, 7.0, 3.95, "VNC · 26,769", T.accent3, "t-vnc");
  tag(s, 7.0, 4.27, "DN / AN · 3,165", T.accent2, "t-dn");
  s.addNotes("Official BANC v888 (Bates et al. 2026, public Lee Lab bucket). Somas from the position column; colors as in the explorer: " +
    "vision teal, interneurons violet, DNs amber, motor neurons rose. No synthetic graphs.");
}

// 6. Model dynamiki
{
  const s = content("Activity on the connectome", "Connectome");
  card(s, 0.5, 1.3, 5.3, 1.25, "eq-card");
  text(s, "r ← r + dt/τ · (−r + tanh(relu(g · W r + I)))", { x: 0.75, y: 1.62, w: 4.9, h: 0.5, fontSize: 20, fontFace: "Cambria", objectName: "eq" });
  const rows = [
    ["W", "synapse counts from BANC, sign from the predicted neurotransmitter, inputs normalized per neuron"],
    ["τ, dt, g", "20 ms, 5 ms, 0.9 — 4 substeps per camera frame"],
    ["I", "input: vision (FlyVis → BANC) + drone gyroscope onto haltere afferents"],
    ["GPU", "sparse CSR matrix in torch: ~2.6 ms per frame (RTX 4060)"],
  ];
  rows.forEach(([k, v], i) => {
    text(s, k, { x: 0.5, y: 2.85 + i * 0.5, w: 1.1, h: 0.4, fontSize: 14, bold: true, color: RED, objectName: `k-${i}` });
    text(s, v, { x: 1.65, y: 2.85 + i * 0.5, w: 4.2, h: 0.45, fontSize: 12, color: T.lt1, objectName: `v-${i}` });
  });
  card(s, 6.2, 1.3, 3.3, 3.45, "sign-card");
  text(s, "Synapse signs", { x: 6.45, y: 1.5, w: 2.8, h: 0.35, fontSize: 15, bold: true, objectName: "sign-title" });
  tag(s, 6.45, 2.0, "ACh — excitatory", T.accent1, "sg-ach");
  tag(s, 6.45, 2.35, "GABA — inhibitory", T.accent3, "sg-gaba");
  tag(s, 6.45, 2.7, "glutamate — inhibitory", T.accent3, "sg-glu");
  tag(s, 6.45, 3.05, "histamine — inhibitory", T.accent3, "sg-his");
  text(s, "The dynamics model and signs are our assumptions; connections and synapse counts come from BANC.", {
    x: 6.45, y: 3.55, w: 2.85, h: 0.9, fontSize: 11, color: T.lt2, objectName: "sign-note",
  });
}

// 7. Obwód lotu
{
  const s = content("Flight circuit: official annotations", "Connectome");
  s.addImage({ path: A("circuit_3d.jpg"), x: 0.4, y: 1.05, w: 3.35, h: 4.07, objectName: "render-circuit" });
  const groups = [
    ["DN flight power", "235", T.accent2, "super_cluster: flight power"],
    ["DN flight steering", "140", T.accent2, "super_cluster: flight steering"],
    ["MN wing power", "24", T.accent3, "DLM / DVM — power muscles"],
    ["MN wing steering", "24", T.accent3, "b1, i1, iii3… — steering muscles"],
    ["MN wing tension", "12", T.accent3, "wing tension"],
    ["Haltere afferents", "428", T.accent4, "rotation sensors ← drone gyroscope"],
  ];
  groups.forEach(([n, c, col, d], i) => {
    const y = 1.3 + i * 0.6;
    s.addShape(pres.shapes.OVAL, { x: 4.3, y: y + 0.1, w: 0.14, h: 0.14, fill: { color: col }, line: { color: col, width: 0 }, objectName: `g-${i}-dot` });
    text(s, n, { x: 4.6, y, w: 2.4, h: 0.32, fontSize: 14, bold: true, objectName: `g-${i}-name` });
    text(s, d, { x: 4.6, y: y + 0.3, w: 3.6, h: 0.26, fontSize: 11, color: T.lt2, objectName: `g-${i}-desc` });
    text(s, c, { x: 8.4, y, w: 1.1, h: 0.32, fontSize: 16, bold: true, align: "right", objectName: `g-${i}-count` });
  });
}

// 8. Odczyt: DN niosą stronę celu
pres.addSection({ title: "Readout" });
{
  const s = content("Target side lives in single DNs", "Readout");
  s.addChart(pres.charts.BAR, [{ name: "Target-side accuracy", labels: ["6 MN averages", "single MNs", "vision (VPN)", "single flight DNs"], values: [68, 84, 94, 99] }], {
    x: 0.5, y: 1.2, w: 5.6, h: 3.6, barDir: "bar",
    chartColors: [T.accent5, T.accent5, T.accent5, RED],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', dataLabelColor: T.lt1, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: T.lt2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valAxisMaxVal: 110, valAxisMinVal: 0,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLineShow: false,
    showLegend: false, showTitle: false, objectName: "chart-side",
  });
  stat(s, 6.6, 1.35, 3, "99%", "target-side accuracy from 375 single flight DNs (validated on an unseen distance)", RED, "s-dn");
  stat(s, 6.6, 2.85, 3, "68%", "from 6 MN group averages — averaging cancels the left / right difference", T.lt2, "s-mn");
  s.addNotes("Ridge regression angle ~ activity, target at −90…+90° at 3 distances, validated on the distance left out " +
    "of training (scripts/check_side_decoding.py).");
}

// 9. Ograniczenie danych
{
  const s = content("v888 limit: optic lobe asymmetry", "Readout");
  card(s, 0.5, 1.3, 4.3, 2.9, "left-card");
  card(s, 5.2, 1.3, 4.3, 2.9, "right-card");
  stat(s, 0.85, 1.55, 3.7, "36%", "of left-lobe neurons have a cell type", T.lt1, "s-l");
  stat(s, 0.85, 2.85, 3.7, "5,535", "neurons driven by the left eye", T.lt2, "s-l2");
  stat(s, 5.55, 1.55, 3.7, "80%", "of right-lobe neurons have a cell type", RED, "s-r");
  stat(s, 5.55, 2.85, 3.7, "16,927", "neurons driven by the right eye", RED, "s-r2");
  text(s, "The FlyVis → BANC map goes by cell type, so the left eye reaches ~3× fewer neurons. Equalizing the input did not change " +
    "the result; the single-DN readout still recognizes both sides.", { x: 0.5, y: 4.35, w: 9, h: 0.6, fontSize: 12, color: T.lt2, objectName: "note" });
}

// 10. Symulator
pres.addSection({ title: "Simulator" });
{
  const s = content("Simulator: X2 drone in MuJoCo", "Simulator");
  s.addImage({ path: A("drone_chase.jpg"), x: 0.5, y: 1.25, w: 4.3, h: 3.22, shadow: shadow(), objectName: "img-drone" });
  s.addImage({ path: A("world.jpg"), x: 5.2, y: 1.25, w: 4.3, h: 3.22, shadow: shadow(), objectName: "img-world" });
  text(s, "Skydio X2 (MuJoCo Menagerie) with eyes on the nose · hover and turn on a flat scene", { x: 0.5, y: 4.55, w: 4.3, h: 0.45, fontSize: 11, color: T.lt2, objectName: "cap-1" });
  text(s, "Random 60 × 60 m terrain, blocks, target · angle mode: the simulator holds the commanded tilt", { x: 5.2, y: 4.55, w: 4.3, h: 0.45, fontSize: 11, color: T.lt2, objectName: "cap-2" });
}

// 11. Trening
pres.addSection({ title: "Training" });
{
  const s = content("Training: only the decoder learns", "Training");
  const steps = [
    ["Teacher", "knows the true state: target bearing, height, speed"],
    ["Mixed flight", "the teacher flies with probability β, β: 1 → 0"],
    ["Data", "pairs (features, teacher command); hard samples ×3–7"],
    ["Regression", "weights refit on all data (DAgger + ridge)"],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.3 + i * 0.86;
    card(s, 0.5, y, 4.9, 0.72, `t-${i}`);
    text(s, String(i + 1), { x: 0.72, y: y + 0.16, w: 0.4, h: 0.4, fontSize: 18, fontFace: HEAD, color: RED, objectName: `t-${i}-n` });
    text(s, h, { x: 1.2, y: y + 0.1, w: 4, h: 0.3, fontSize: 14, bold: true, objectName: `t-${i}-h` });
    text(s, d, { x: 1.2, y: y + 0.39, w: 4.1, h: 0.3, fontSize: 11, color: T.lt2, objectName: `t-${i}-d` });
  });
  card(s, 5.8, 1.3, 3.7, 1.55, "split");
  text(s, "Axis split", { x: 6.05, y: 1.45, w: 3.2, h: 0.3, fontSize: 14, bold: true, objectName: "split-h" });
  tag(s, 6.05, 1.85, "yaw ← BANC only (DNs)", T.accent2, "sp-yaw");
  tag(s, 6.05, 2.2, "thrust/roll/pitch ← BANC + sensors", T.accent4, "sp-ctl");
  card(s, 5.8, 3.05, 3.7, 1.6, "dist");
  text(s, "2 GPUs over LAN", { x: 6.05, y: 3.2, w: 3.2, h: 0.3, fontSize: 14, bold: true, objectName: "dist-h" });
  text(s, "The master hands out worlds, workers (RTX 4060, RTX 3070 Ti) run episodes and send back only the XᵀX, Xᵀy " +
    "statistics — no video frames.", { x: 6.05, y: 3.55, w: 3.25, h: 1.0, fontSize: 11, color: T.lt2, objectName: "dist-d" });
}

// 12. Wyniki: zawis i skręt
pres.addSection({ title: "Results" });
{
  const s = content("Result: turning to the target", "Results");
  const labels = ["target −60°", "target −30°", "target +30°", "target +60°"];
  s.addChart(pres.charts.BAR, [
    { name: "before training", labels, values: ["-60", "-30", "+30", "+60"].map((k) => +DATA.before[k].toFixed(1)) },
    { name: "after training", labels, values: ["-60", "-30", "+30", "+60"].map((k) => +DATA.after[k].toFixed(1)) },
  ], {
    x: 0.5, y: 1.2, w: 5.8, h: 3.7, barDir: "col", barGapWidthPct: 60,
    chartColors: [T.accent5, RED],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"°"', dataLabelColor: T.lt1, dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: T.lt2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendColor: T.lt2, legendFontSize: 11, legendFontFace: "+mn-lt",
    showTitle: false, objectName: "chart-eval",
  });
  stat(s, 6.8, 1.35, 2.8, "2.65°", "mean final heading error without the teacher (25.1° before training)", RED, "s-err");
  stat(s, 6.8, 2.75, 2.8, "1,000", "episodes on 2 GPUs: 672 + 328", T.lt1, "s-ep");
  text(s, "Hover on a flat scene, height held by the simulator. Heading: single BANC DNs only.", {
    x: 6.8, y: 4.05, w: 2.8, h: 0.8, fontSize: 11, color: T.lt2, objectName: "s-note",
  });
}

// 13. Wyniki: lot w świecie (w toku)
{
  const s = content("World flight: in progress", "Results");
  card(s, 0.5, 1.3, 2.85, 1.75, "c-teacher");
  card(s, 3.58, 1.3, 2.85, 1.75, "c-before");
  card(s, 6.65, 1.3, 2.85, 1.75, "c-model");
  stat(s, 0.8, 1.55, 2.4, "6 / 6", "the teacher reaches the target — there is something to learn from", T.lt1, "w-t");
  stat(s, 3.88, 1.55, 2.4, "18.1 m", "before training: mean closest distance to the target", T.lt2, "w-b");
  stat(s, 6.95, 1.55, 2.4, "9.0 m", "after 192 episodes: 2× closer, 0 crashes, target 0 / 6", RED, "w-m");
  text(s, "Honestly: the model flies stably and gets closer to the target, but does not reach it yet. Next run: validation every " +
    "50 episodes keeping the best weights, harder samples weighted up, GPS for long-range heading.", { x: 0.5, y: 3.35, w: 9, h: 0.8, fontSize: 13, color: T.lt2, objectName: "w-note" });
}

// 14. Z BANC vs nasze założenia
pres.addSection({ title: "Honesty" });
{
  const s = content("From BANC vs. our assumptions", "Honesty");
  card(s, 0.5, 1.3, 4.3, 2.75, "bank");
  card(s, 5.2, 1.3, 4.3, 2.75, "ours");
  tag(s, 0.8, 1.5, "from BANC v888", T.accent1, "h-banc");
  tag(s, 5.5, 1.5, "our assumptions", T.accent2, "h-ours");
  const banc = ["neurons, connections, synapse counts", "predicted neurotransmitter", "flight groups from official annotations", "soma positions and SWC skeletons"];
  const ours = ["dynamics model and parameters", "neurotransmitter signs", "FlyVis → BANC map by cell type", "linear decoder and axis split", "drone sensors for thrust/roll/pitch"];
  text(s, banc.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < banc.length - 1 } })), {
    x: 0.8, y: 2.0, w: 3.8, h: 2.6, fontSize: 14, paraSpaceAfter: 6, objectName: "list-banc",
  });
  text(s, ours.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < ours.length - 1 } })), {
    x: 5.5, y: 2.0, w: 3.8, h: 2.6, fontSize: 14, paraSpaceAfter: 6, objectName: "list-ours",
  });
}

// 15. Zakończenie
pres.addSection({ title: "End" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "End" });
  s.addImage({ path: A("connectome_wide.jpg"), x: 4.1, y: 0.75, w: 5.9, h: 2.95, transparency: 35, objectName: "render-bg" });
  s.addImage({ path: A("logo.png"), x: 0.6, y: 1.1, w: 1.0, h: 0.59, objectName: "logo" });
  text(s, "NEUROFLY  ·  HACKYEAH", { x: 0.6, y: 1.85, w: 4.5, h: 0.25, fontSize: 9, color: RED, charSpacing: 6, objectName: "eyebrow" });
  text(s, "THANK YOU", { x: 0.6, y: 2.12, w: 5.2, h: 0.8, fontSize: 36, fontFace: HEAD, charSpacing: 2, objectName: "thanks" });
  s.addImage({ path: A("accent_line.png"), x: 0.6, y: 2.95, w: 0.05, h: 0.8, objectName: "accent-line" });
  text(s, "Live demo: BANC panel in flight\n3D connectome explorer", { x: 0.85, y: 2.98, w: 3.4, h: 0.75, fontSize: 15, color: T.lt2, objectName: "demo" });
  text(s, "GITHUB.COM/FALCONDEVX/NEUROFLY", { x: 0.6, y: 4.6, w: 6, h: 0.3, fontSize: 9, color: RED, charSpacing: 4, objectName: "repo" });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  const applyThemePath = process.env.APPLY_THEME;
  if (applyThemePath) {
    const { applyTheme } = require(applyThemePath);
    await applyTheme(OUT, THEME);
  }
  console.log("zapisano", OUT);
})();
