// Prezentacja NeuroFly: ciemna, w stylu plakatu (Michroma w tytułach, Roboto w treści, czerwono-pomarańczowy akcent),
// z renderami 3D z prawdziwych danych BANC v888 i MuJoCo.
//   node build.js [NeuroFly.pptx]   (npm i pptxgenjs; grafiki i data.json w assets/ robi render_assets.py, liczby z repo / CLAUDE.md)
//   APPLY_THEME=<skill pptx>/scripts/apply_theme.js — opcjonalnie wpisuje kolory motywu do pliku
//   embed_fonts.ps1 — osadza Michroma i Roboto (PowerPoint)
// Zasady treści: bez strzałek (ani w tekście, ani jako kształty), bez kropek, punktorów i separatorów „·”; na każdym
// slajdzie pasek objaśnień skrótów i metryk (gloss, kolumny), liczby tylko z repo, założenia nazwane wprost.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || path.join(__dirname, "NeuroFly.pptx");
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
pres.title = "NeuroFly — a fruit-fly brain flies a drone";
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
// START
// =================================================================================================
pres.addSection({ title: "Start" });

{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Start" });
  s.addImage({ path: A("fly_drone.jpg"), x: 3.6, y: 0.1, w: 6.4, h: 4.385, objectName: "render-fly-drone" });
  s.addImage({ path: A("logo.png"), x: 0.6, y: 0.95, w: 1.0, h: 0.59, objectName: "logo" });
  eyebrow(s, "BANC v888 connectome", 0.6, 1.78, 5.2, RED, "eyebrow");
  text(s, "NEUROFLY", { x: 0.6, y: 2.05, w: 5.2, h: 0.85, fontSize: 40, fontFace: HEAD, charSpacing: 2, objectName: "title" });
  s.addImage({ path: A("accent_line.png"), x: 0.6, y: 3.05, w: 0.05, h: 0.95, objectName: "accent-line" });
  text(s, "A fruit-fly brain\nflies a drone", { x: 0.85, y: 3.05, w: 3.0, h: 0.8, fontSize: 18, color: T.lt2, objectName: "subtitle" });
  text(s, "FULL BANC V888 CONNECTOME      175,401 NEURONS      CLOSED LOOP IN MUJOCO", {
    x: 0.6, y: 4.75, w: 8.8, h: 0.3, fontSize: 8, color: T.lt2, charSpacing: 4, objectName: "footer",
  });
  s.addNotes("NeuroFly: the drone camera image goes through a model of the fly eye (FlyVis), then through the real " +
    "Drosophila connectome (BANC v888, brain + ventral nerve cord), and flight-neuron activity steers the drone in simulation. " +
    "Render: semi-transparent NeuroMechFly body (FlyGym) with the BANC connectome inside — body alignment is illustrative — " +
    "next to the X2 drone from MuJoCo. The threads from flight neurons to the drone are an illustration of the loop, not data.");
}

pres.addSection({ title: "Idea" });

// 3. Pomysł — jedno zdanie
{
  const s = content("The idea", "Idea");
  text(s, [
    { text: "What if a ", options: { color: T.lt1 } },
    { text: "real brain wiring diagram", options: { color: RED } },
    { text: " — not an artificial network — decided where a drone flies?", options: { color: T.lt1 } },
  ], { x: 0.62, y: 1.15, w: 5.6, h: 1.35, fontSize: 24, fontFace: "Roboto Light", objectName: "big-idea" });
  const pillars = [
    ["Real wiring", "175,401 neurons and their synapses from the BANC v888 fly connectome. We invent no connections and no weights.", T.accent2],
    ["Real loop", "Camera, fly eye, brain, wing neurons, drone, new image: frame after frame.", T.accent1],
    ["Real test", "The connectome is judged by behaviour: does the drone turn to the target and reach it?", T.accent3],
  ];
  pillars.forEach(([h, d, col], i) => {
    const y = 2.65 + i * 0.62;
    text(s, h, { x: 0.62, y, w: 1.8, h: 0.3, fontSize: 14, bold: true, color: col, objectName: `p-${i}-h` });
    text(s, d, { x: 2.5, y, w: 3.9, h: 0.58, fontSize: 11, color: T.lt2, objectName: `p-${i}-d` });
  });
  s.addImage({ path: A("circuit_3d.jpg"), x: 6.75, y: 0.95, w: 3.0, h: 3.64, objectName: "render-circuit" });
  gloss(s, [["BANC:", "Brain And Nerve Cord, the first connectome of a whole fly brain plus nerve cord (Bates et al. 2026)"],
    ["v888:", "the public data release we use"]]);
  s.addNotes("This is the one sentence to remember. We take the actual wiring of a fruit fly — mapped synapse by synapse " +
    "in the BANC connectome — and put it in the loop. The only thing we train is a small linear readout from flight " +
    "neurons to drone commands. Render: 863 flight neurons from BANC skeletons.");
}

{
  const s = content("What makes NeuroFly different", "Idea");
  const rows = [
    ["Brain", "artificial network, trained end-to-end", "the real fly wiring diagram, kept frozen"],
    ["Scale", "a small, hand-picked circuit", "whole brain and nerve cord: 175,401 neurons"],
    ["Heading", "decided by the trained net or a PID loop", "decided by the fly's own flight neurons"],
    ["Learned", "everything", "only a simple readout of flight neurons"],
    ["Eyes", "a standard image network", "a fly-eye model: 721 facets per eye"],
  ];
  // najpierw definicje (co znaczy connectome i PID), potem porównanie
  gloss(s, [["Connectome:", "the complete wiring diagram of a nervous system, every neuron and synapse"],
    ["PID:", "the classic hand-tuned controller used in most drones"]], 0.62, 8.85, 1.08);
  const x0 = 0.62, cK = 1.05, cA = 2.35, cB = 2.55, y0 = 1.6, rh = 0.5;
  eyebrow(s, "typical approach", x0 + cK, y0, cA, MUTED, "th-a");
  eyebrow(s, "NeuroFly", x0 + cK + cA + 0.1, y0, cB, RED, "th-b");
  rows.forEach(([k, a, b], i) => {
    const y = y0 + 0.32 + i * rh;
    s.addShape(pres.shapes.LINE, { x: x0, y: y - 0.04, w: cK + cA + cB + 0.1, h: 0, line: { color: "3F3F46", width: 0.5 }, objectName: `row-${i}-line` });
    text(s, k, { x: x0, y: y + 0.06, w: cK, h: 0.4, fontSize: 12, bold: true, color: T.lt2, objectName: `row-${i}-k` });
    text(s, a, { x: x0 + cK, y: y + 0.06, w: cA - 0.15, h: 0.45, fontSize: 11, color: MUTED, objectName: `row-${i}-a` });
    text(s, b, { x: x0 + cK + cA + 0.1, y: y + 0.06, w: cB, h: 0.45, fontSize: 12, color: T.lt1, objectName: `row-${i}-b` });
  });
  const yEnd = y0 + 0.32 + rows.length * rh - 0.04; // linia zamykająca tabelę, karta kończy się na tej samej wysokości
  s.addShape(pres.shapes.LINE, { x: x0, y: yEnd, w: cK + cA + cB + 0.1, h: 0, line: { color: "3F3F46", width: 0.5 }, objectName: "row-end-line" });
  card(s, 6.85, y0, 2.65, yEnd - y0, "rule-card", RED);
  eyebrow(s, "our rule no. 1", 7.05, y0 + 0.16, 2.3, RED, "rule-k");
  text(s, "The connectome makes the key decision", { x: 7.05, y: y0 + 0.4, w: 2.3, h: 0.5, fontSize: 13, bold: true, objectName: "rule-h" });
  stat(s, 7.05, y0 + 0.9, 2.3, "100%", "of the heading near the target comes from BANC neurons", RED, "rule-s", 26);
  text(s, "We dropped an early synthetic network the moment it stopped looking like BANC.", {
    x: 7.05, y: y0 + 2.25, w: 2.3, h: 0.5, fontSize: 9, italic: true, color: T.lt2, objectName: "rule-note" });
  s.addNotes("Hundreds of projects call themselves brain-inspired. Almost all use an artificial network that borrows only " +
    "the word 'neuron', or a tiny hand-built circuit. We use the real, complete wiring of a fruit fly, we never train it, " +
    "and we built the system so that the most important decision — where to turn — is made by the fly's own descending " +
    "neurons. Drone sensors get zero weight on heading by construction. When an early prototype used a made-up network, " +
    "we threw it away.");
}

// 6. Pętla zamknięta: cały potok w 8 krokach na elipsie, z obrazkami (bez strzałek)
{
  const s = content("How it works: one loop, 8 steps", "Idea");
  // 8 kroków w dwóch rzędach, numer daje kolejność (bez strzałek); po kroku 8 nowy obraz wraca do kroku 1
  const nodes = [
    ["Camera", "2 eyes, 157° view each", "eye_right.jpg", T.accent1],
    ["Retina", "721 facets per eye", "retina_right.jpg", T.accent1],
    ["FlyVis", "fly vision model", "thumb_flyvis.jpg", T.accent1],
    ["Into the brain", "22,462 neurons get input", "thumb_map.jpg", T.accent1],
    ["Fly brain", "175,401 neurons", "thumb_wiring.jpg", T.accent2],
    ["Flight neurons", "the brain's flight commands", "thumb_circuit.jpg", T.accent3],
    ["Decoder", "turns them into 4 numbers", "thumb_decoder.jpg", T.lt2],
    ["Drone", "moves, then back to 1", "drone_chase.jpg", T.accent4],
  ];
  const w = 1.95, h = 1.5, gx = 0.35, y1 = 1.1, y2 = 2.92;
  nodes.forEach(([hd, d, f, col], i) => {
    const x = 0.55 + (i % 4) * (w + gx), y = i < 4 ? y1 : y2;
    card(s, x, y, w, h, `node-${i}`);
    img(s, f, x + 0.07, y + 0.07, w - 0.14, 0.76, `node-${i}-img`);
    text(s, [{ text: `${i + 1}  `, options: { color: col, fontFace: HEAD, fontSize: 11 } }, { text: hd, options: { bold: true } }],
      { x: x + 0.14, y: y + 0.9, w: w - 0.24, h: 0.28, fontSize: 13, objectName: `node-${i}-title` });
    text(s, d, { x: x + 0.14, y: y + 1.18, w: w - 0.24, h: 0.28, fontSize: 10, color: T.lt2, objectName: `node-${i}-desc` });
  });
  gloss(s, [["FlyVis:", "a published model of how the fly eye and visual brain process an image"],
    ["Flight neurons:", "neurons that carry flight commands from the brain to the wings"],
    ["Thrust, roll, pitch, yaw:", "total power, tilt sideways, nose up or down, turn"]]);
  s.addNotes("Every camera frame goes once around this loop, steps 1 to 8. Steps 1–4 happen before the connectome; " +
    "5–8 are the connectome, the flight-neuron readout, the decoder and the drone. One frame (FlyVis + 4 BANC substeps " +
    "on the GPU) takes ~15–20 ms. There is no hand-coded heading controller near the target: the direction is read out " +
    "from descending-neuron (DN) activity in BANC. Decoder thumbnail: the real yaw weights of the best world decoder, sorted.");
}

pres.addSection({ title: "How it works" });

// 9. Oko muchy
{
  const s = content("Steps 1–3: seeing like a fly", "How it works");
  const ims = [["eye_left.jpg", "Left camera"], ["retina_left.jpg", "Left retina"], ["eye_right.jpg", "Right camera"], ["retina_right.jpg", "Right retina"]];
  ims.forEach(([f, cap], i) => {
    const x = 0.55 + i * 2.27;
    s.addImage({ path: A(f), x, y: 1.2, w: 2.0, h: 2.28, shadow: shadow(), objectName: `img-${i}` });
    text(s, cap, { x, y: 3.55, w: 2.0, h: 0.28, fontSize: 11, color: T.lt2, objectName: `cap-${i}` });
  });
  text(s, "Each drone camera sees as wide as a fly eye. The picture is cut into 721 small hexagons, like the lenses of a fly eye, " +
    "then FlyVis works out how the fly's visual neurons react, and that signal goes into the matching BANC neurons.", {
    x: 0.55, y: 3.9, w: 8.9, h: 0.65, fontSize: 12, color: T.lt1, objectName: "explain",
  });
  gloss(s, [["Facet:", "one of the small lenses of a fly eye, one pixel of fly vision"],
    ["Optic lobe:", "the visual part of the brain behind each eye"]]);
  s.addNotes("The target (dark pole) is on the right: only the right eye sees it, as a dark column of facets.");
}

{
  const s = content("Steps 4–5: the fly brain", "How it works");
  s.addImage({ path: A("connectome_wiring.jpg"), x: 0.4, y: 1.0, w: 3.0, h: 3.64, objectName: "render-wiring" });
  stat(s, 3.9, 1.15, 2.7, "175,401", "neurons in the fly brain and nerve cord", T.lt1, "s-neurons");
  stat(s, 6.75, 1.15, 2.7, "18.6 M", "synapses, the contacts between neurons", T.lt1, "s-syn");
  stat(s, 3.9, 2.4, 2.7, "1.53 M", "neuron-to-neuron connections we use", T.lt1, "s-edges");
  stat(s, 6.75, 2.4, 2.7, "863", "flight neurons drawn in full 3D shape", T.lt1, "s-skel");
  tag(s, 3.9, 3.6, "optic lobes: 105,646", T.accent1, "t-ol");
  tag(s, 3.9, 3.9, "central brain: 42,620", VIOLET, "t-cb");
  tag(s, 6.75, 3.6, "VNC: 26,769", T.accent3, "t-vnc");
  tag(s, 6.75, 3.9, "DN / AN: 3,165", T.accent2, "t-dn");
  text(s, "Lines: 60,000 real connections from BANC, drawn as straight lines between neurons.", {
    x: 3.9, y: 4.25, w: 5.5, h: 0.3, fontSize: 10, italic: true, color: T.lt2, objectName: "wiring-note" });
  gloss(s, [["VNC:", "ventral nerve cord, the fly's spinal cord"], ["DN / AN:", "neurons linking brain and nerve cord, down / up"],
    ["Optic lobes:", "the visual part of the brain"]], 3.9, 5.55);
  s.addNotes("Official BANC v888 (Bates et al. 2026, public Lee Lab bucket). Dots are somas from the position column; " +
    "lines are real edges from the v888 edge list, sampled with probability proportional to synapse count and coloured " +
    "by the presynaptic class. The amber bundle through the neck is descending neurons. Lines are straight for drawing " +
    "only — real axons take other paths.");
}

// 12. Obwód lotu
{
  const s = content("Step 6: the flight neurons", "How it works");
  s.addImage({ path: A("circuit_3d.jpg"), x: 0.4, y: 1.0, w: 3.0, h: 3.64, objectName: "render-circuit" });
  const groups = [
    ["DN flight power", "235", T.accent2, "brain neurons that set wing power"],
    ["DN flight steering", "140", T.accent2, "brain neurons that steer"],
    ["MN wing power", "24", T.accent3, "drive the big muscles that beat the wings"],
    ["MN wing steering", "24", T.accent3, "drive small steering muscles at the wing base"],
    ["MN wing tension", "12", T.accent3, "keep the wing tight"],
    ["Haltere afferents", "428", T.accent4, "rotation sensors, here fed by the drone gyroscope"],
  ];
  groups.forEach(([n, c, col, d], i) => {
    const y = 1.15 + i * 0.55;
    text(s, n, { x: 3.9, y, w: 2.9, h: 0.3, fontSize: 13, bold: true, color: col, objectName: `g-${i}-name` });
    text(s, d, { x: 3.9, y: y + 0.27, w: 4.5, h: 0.26, fontSize: 10, color: T.lt2, objectName: `g-${i}-desc` });
    text(s, c, { x: 8.4, y, w: 1.05, h: 0.32, fontSize: 16, bold: true, align: "right", objectName: `g-${i}-count` });
  });
  gloss(s, [["Halteres:", "tiny club-shaped hind wings that sense body rotation, a natural gyroscope"],
    ["Labels:", "all groups come from official BANC annotations"]], 3.9, 5.55);
  s.addNotes("All groups come from official BANC v888 annotation columns (super_cluster, cell_function, body_part_sensory). " +
    "We never guess cell-type names.");
}

{
  const s = content("Step 6: where the brain knows the target", "How it works");
  s.addChart(pres.charts.BAR, [{ name: "Target-side accuracy", labels: ["wing muscle neurons, averaged", "wing muscle neurons, one by one", "vision neurons", "flight neurons, one by one"], values: [68, 84, 94, 99] }],
    chartOpts({
      x: 0.5, y: 1.15, w: 5.6, h: 3.35, barDir: "bar",
      chartColors: [T.accent5, T.accent5, T.accent5, RED],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisMaxVal: 110, valAxisMinVal: 0,
      showLegend: false, objectName: "chart-side",
    }));
  stat(s, 6.6, 1.3, 3, "99%", "of scenes where single flight neurons tell left from right correctly", RED, "s-dn");
  stat(s, 6.6, 2.8, 3, "68%", "if you average the muscle neurons: left and right cancel out", T.lt2, "s-mn");
  gloss(s, [["Accuracy:", "how often we can tell which side the target is on (50% = guessing)"]]);
  s.addNotes("Ridge regression angle ~ activity, target from −90° to +90° at 3 distances, validated on the distance left out " +
    "of training (scripts/check_side_decoding.py). A finding about the connectome: direction is carried by individual " +
    "descending neurons and is lost if you average motor neurons.");
}

{
  const s = content("Step 7: the decoder learns to read", "How it works");
  const steps = [
    ["Teacher", "a simple pilot that knows where the target is"],
    ["Taking turns", "the teacher flies less and less, then not at all"],
    ["Learning", "the readout copies what the teacher would do"],
    ["Repeat", "updated after every batch of flights"],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.15 + i * 0.84;
    card(s, 0.55, y, 4.9, 0.72, `t-${i}`);
    text(s, String(i + 1), { x: 0.77, y: y + 0.16, w: 0.4, h: 0.4, fontSize: 18, fontFace: HEAD, color: RED, objectName: `t-${i}-n` });
    text(s, h, { x: 1.25, y: y + 0.1, w: 4, h: 0.3, fontSize: 14, bold: true, objectName: `t-${i}-h` });
    text(s, d, { x: 1.25, y: y + 0.39, w: 4.1, h: 0.3, fontSize: 11, color: T.lt2, objectName: `t-${i}-d` });
  });
  card(s, 5.8, 1.15, 3.7, 1.5, "freeze", RED);
  text(s, "BANC stays frozen", { x: 6.05, y: 1.3, w: 3.2, h: 0.3, fontSize: 14, bold: true, objectName: "freeze-h" });
  text(s, "The fly brain itself never changes. We only learn how to read its flight neurons.", {
    x: 6.05, y: 1.67, w: 3.25, h: 0.9, fontSize: 11, color: T.lt2, objectName: "freeze-d" });
  card(s, 5.8, 2.85, 3.7, 1.65, "dist");
  text(s, "2 laptops working together", { x: 6.05, y: 3.0, w: 3.2, h: 0.3, fontSize: 14, bold: true, objectName: "dist-h" });
  text(s, "One computer hands out worlds, two laptop graphics cards fly them in parallel. " +
    "Every 50 flights we test and keep the best version.", { x: 6.05, y: 3.35, w: 3.25, h: 1.05, fontSize: 11, color: T.lt2, objectName: "dist-d" });
  gloss(s, [["Imitation learning:", "the model flies, the teacher says what it should have done"]]);
  s.addNotes("Hard moments: big angle to the target (> 20°), height error > 0.3 m, fast rotation (> 1 rad/s). Most frames " +
    "are 'target straight ahead, do nothing', so the hard ones are weighted up. Workers send XᵀX and Xᵀy, enough to " +
    "compute the regression on the master.");
}

{
  const s = content("Step 8: the drone and its world", "How it works");
  s.addImage({ path: A("drone_chase.jpg"), x: 0.62, y: 1.15, w: 3.95, h: 2.96, shadow: shadow(), objectName: "img-drone" });
  s.addImage({ path: A("world.jpg"), x: 5.5, y: 1.15, w: 3.95, h: 2.96, shadow: shadow(), objectName: "img-world" });
  text(s, "A Skydio X2 drone with two eyes on its nose", { x: 0.62, y: 4.2, w: 3.95, h: 0.4, fontSize: 11, color: T.lt2, objectName: "cap-1" });
  text(s, "Random worlds with hills, blocks, a target mast and wind", { x: 5.5, y: 4.2, w: 3.95, h: 0.4, fontSize: 11, color: T.lt2, objectName: "cap-2" });
  gloss(s, [["MuJoCo:", "free physics simulator used in robotics"],
    ["Episode:", "one flight from start until target, crash or time limit"]]);
}

// 19. Kto czym steruje
{
  const s = content("Who steers what: GPS far, fly brain near", "How it works");
  const x0 = 0.62, wBar = 8.85, y = 1.3, split = x0 + wBar * 0.62;
  s.addShape(pres.shapes.RECTANGLE, { x: x0, y, w: split - x0, h: 0.5, fill: { color: T.accent4, transparency: 55 }, line: { color: T.accent4, width: 0.75 }, objectName: "bar-gps" });
  s.addShape(pres.shapes.RECTANGLE, { x: split, y, w: x0 + wBar - split, h: 0.5, fill: { color: RED, transparency: 35 }, line: { color: RED, width: 0.75 }, objectName: "bar-banc" });
  text(s, "far from the target: heading from GPS + compass", { x: x0 + 0.2, y: y + 0.13, w: 4.8, h: 0.26, fontSize: 12, objectName: "bar-gps-t" });
  text(s, "last 4 m: heading from BANC", { x: split + 0.2, y: y + 0.13, w: 3.2, h: 0.26, fontSize: 12, bold: true, objectName: "bar-banc-t" });
  text(s, "drone", { x: x0, y: y + 0.58, w: 1, h: 0.24, fontSize: 10, color: T.lt2, objectName: "bar-l" });
  text(s, "target", { x: x0 + wBar - 1, y: y + 0.58, w: 1, h: 0.24, fontSize: 10, color: T.lt2, align: "right", objectName: "bar-r" });
  text(s, "smooth hand-over: up close GPS is not precise enough", {
    x: split - 2.4, y: y + 0.58, w: 4.8, h: 0.24, fontSize: 10, color: T.lt2, align: "center", objectName: "bar-mid",
  });
  card(s, 0.62, 2.45, 4.3, 1.95, "ax-yaw", RED);
  card(s, 5.17, 2.45, 4.3, 1.95, "ax-ctl");
  eyebrow(s, "where to turn", 0.9, 2.62, 3.5, RED, "ax-yaw-k");
  text(s, "BANC only", { x: 0.9, y: 2.9, w: 3.7, h: 0.4, fontSize: 18, bold: true, objectName: "ax-yaw-h" });
  text(s, "Only fly neurons decide the direction. Drone sensors are not allowed to help.", {
    x: 0.9, y: 3.38, w: 3.8, h: 0.9, fontSize: 12, color: T.lt2, objectName: "ax-yaw-d" });
  eyebrow(s, "staying level and at height", 5.45, 2.62, 3.8, T.accent4, "ax-ctl-k");
  text(s, "BANC + drone sensors", { x: 5.45, y: 2.9, w: 3.8, h: 0.4, fontSize: 18, bold: true, objectName: "ax-ctl-h" });
  text(s, "The fly brain never sees the drone's height or speed, so the sensors help here.", {
    x: 5.45, y: 3.38, w: 3.8, h: 0.9, fontSize: 12, color: T.lt2, objectName: "ax-ctl-d" });
  gloss(s, [["GPS:", "satellite position, here with a wandering 2.5 m error"], ["Heading:", "the direction the nose points"],
    ["Our choice:", "fly brain = where to go, sensors = stay stable"]]);
  s.addNotes("Our assumption, said out loud: BANC is perception and direction, the drone's sensors do stabilisation, much " +
    "like a fly whose wings have their own fast reflexes. GPS only brings the drone into view of the target; inside 4 m the " +
    "heading is decided by the connectome.");
}

pres.addSection({ title: "Results" });

{
  const s = content("Result 1: turning to the target", "Results");
  const labels = ["target −60°", "target −30°", "target +30°", "target +60°"];
  s.addChart(pres.charts.BAR, [
    { name: "before training", labels, values: ["-60", "-30", "+30", "+60"].map((k) => +DATA.before[k].toFixed(1)) },
    { name: "after training", labels, values: ["-60", "-30", "+30", "+60"].map((k) => +DATA.after[k].toFixed(1)) },
  ], chartOpts({
    x: 0.5, y: 1.1, w: 5.8, h: 3.4, barDir: "col", barGapWidthPct: 60,
    chartColors: [T.accent5, RED],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"°"',
    showLegend: true, legendPos: "t", objectName: "chart-eval",
  }));
  stat(s, 6.8, 1.25, 2.8, "2.65°", "average miss after turning, without the teacher (25.1° before training)", RED, "s-err");
  stat(s, 6.8, 2.65, 2.8, "1,000", "training flights", T.lt1, "s-ep");
  text(s, "Hovering drone, target to the side. Only fly neurons choose the turn.", {
    x: 6.8, y: 3.85, w: 2.8, h: 0.6, fontSize: 11, color: T.lt2, objectName: "s-note",
  });
  gloss(s, [["Miss:", "angle between the drone's nose and the target; 0° = pointing straight at it"]]);
  s.addNotes("Target at ±30° and ±60°, model flying alone. Bars: before vs after training.");
}

// 23. Podmuch
{
  const s = content("Result 2: hover, turn, gust", "Results");
  const rows = [
    ["Hover", "1.7°", "heading error while hovering"],
    ["Target +60°", "7°", "error after the turn settles (6 s)"],
    ["Target −60°", "21°", "turns, weaker on the left (the left-eye gap)"],
    ["Gust", "35° to 1.5°", "pushed off course, back on course in 3.9 s"],
  ];
  rows.forEach(([k, v, d], i) => {
    const y = 1.15 + i * 0.84;
    card(s, 0.55, y, 5.5, 0.72, `r-${i}`);
    text(s, k, { x: 0.78, y: y + 0.21, w: 1.5, h: 0.32, fontSize: 13, bold: true, objectName: `r-${i}-k` });
    text(s, v, { x: 2.25, y: y + 0.13, w: 1.75, h: 0.45, fontSize: 22, fontFace: "Roboto Light", color: RED, objectName: `r-${i}-v` });
    text(s, d, { x: 4.05, y: y + 0.13, w: 1.9, h: 0.5, fontSize: 10, color: T.lt2, objectName: `r-${i}-d` });
  });
  s.addImage({ path: A("drone_chase.jpg"), x: 6.35, y: 1.15, w: 3.15, h: 2.36, shadow: shadow(), objectName: "img-drone" });
  text(s, "Real fly vision and the full fly brain, flying the simulated drone.", {
    x: 6.35, y: 3.65, w: 3.15, h: 0.8, fontSize: 10, color: T.lt2, objectName: "r-cap" });
  gloss(s, [["Heading error:", "angle between the nose and the target"], ["Gust:", "a simulated sudden wind push"],
    ["Heading:", "where the nose points"]]);
  s.addNotes("fly_banc.py --local --decoder data/decoders/planB_dn.npz --thrust hold. Thrust is held by the simulator; " +
    "the connectome decides the heading and, through the halteres, damps the rotation after the gust.");
}

// 24. Lot w świecie
{
  const s = content("Result 3: flight in new worlds", "Results");
  const ev = W.evals;
  s.addChart(pres.charts.BAR, [{ name: "targets reached (of 6)", labels: ev.map((e) => String(e.after_episodes)), values: ev.map((e) => e.reached) }],
    chartOpts({
      x: 0.5, y: 1.1, w: 5.6, h: 2.95, barDir: "col", barGapWidthPct: 45,
      chartColors: ev.map((e) => (e.tag === W.best.tag ? RED : T.accent5)),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0" / 6"', valAxisMaxVal: 7, valAxisMinVal: 0,
      showLegend: false, objectName: "chart-world",
    }));
  text(s, "training flights done before each test; red: best weights, kept", {
    x: 0.7, y: 4.1, w: 5.4, h: 0.26, fontSize: 10, color: T.lt2, objectName: "world-axis" });
  stat(s, 6.6, 1.2, 3, `${W.best.reached} / 6`, "new worlds where the drone reached the target", RED, "s-best");
  stat(s, 6.6, 2.45, 3, `${ev[0].mean_min_dist.toFixed(1)} m to ${W.best.mean_min_dist.toFixed(1)} m`, "closest the drone got to the target, before vs after training", T.lt1, "s-dist", 26);
  text(s, "Start about 20 m away, 40 s to get there, imperfect sensors.", { x: 6.6, y: 3.7, w: 3, h: 0.6, fontSize: 11, color: T.lt2, objectName: "world-note" });
  gloss(s, [["Test worlds:", "6 fixed worlds the model never trained on"],
    ["Closest distance:", "how near the drone got to the target, averaged over the 6 worlds"]]);
  s.addNotes("Validation every 50 episodes on the same 6 worlds (seeds 101–106). Before training the drone did not even " +
    "get going (0/6, 18.2 m). The best checkpoint reaches 5 of 6 targets; the final one 4 of 6. " +
    "Run it: python -m sim.run_env --banc data/decoders/world_distributed_best.npz --beacon-scale 4 --vision-range 4 " +
    "--sensors real --motor-tau 0.04");
}

pres.addSection({ title: "Honesty" });

{
  const s = content("What does not work yet", "Honesty");
  const items = [
    ["Lift from the fly brain", "too weak: the drone slowly sinks, so the simulator holds the height"],
    ["Averaged muscle neurons", "do not turn the drone; the direction is only in single flight neurons"],
    ["Vision range", "the fly brain steers only the last 4 m; farther away GPS does"],
    ["Left eye", "fewer labelled neurons in the data, so the left eye reaches ~3× fewer"],
    ["Simulation only", "no real drone flown yet; sensors and wind are simulated"],
  ];
  items.forEach(([h, d], i) => {
    const y = 1.2 + i * 0.66;
    text(s, h, { x: 0.62, y, w: 2.7, h: 0.3, fontSize: 13, bold: true, color: i < 2 ? RED : ORANGE, objectName: `lim-${i}-h` });
    text(s, d, { x: 3.4, y, w: 6.0, h: 0.55, fontSize: 12, color: T.lt2, objectName: `lim-${i}-d` });
  });
  gloss(s, [["Lift:", "propeller power that keeps the drone in the air"]]);
  s.addNotes("We would rather show the limits than hide them. Each of these is a concrete next step.");
}

// 27. Z BANC vs nasze założenia
{
  const s = content("From BANC vs. our assumptions", "Honesty");
  card(s, 0.55, 1.15, 4.3, 2.85, "bank");
  card(s, 5.15, 1.15, 4.3, 2.85, "ours");
  tag(s, 0.85, 1.35, "from BANC v888", T.accent1, "h-banc");
  tag(s, 5.45, 1.35, "our assumptions", T.accent2, "h-ours");
  const banc = ["every neuron and every connection", "how strong each connection is", "which neurons excite or calm down", "which neurons are for flight", "the 3D shape of each neuron"];
  const ours = ["the equation for neuron activity", "how vision is plugged into the brain", "how flight neurons become drone commands",
    "the gyroscope feeds only turning", "GPS far away, fly brain in the last 4 m"];
  text(s, banc.map((t, i) => ({ text: t, options: { breakLine: i < banc.length - 1 } })), {
    x: 0.85, y: 1.8, w: 3.8, h: 2.1, fontSize: 13, paraSpaceAfter: 6, objectName: "list-banc",
  });
  text(s, ours.map((t, i) => ({ text: t, options: { breakLine: i < ours.length - 1 } })), {
    x: 5.45, y: 1.8, w: 3.8, h: 2.1, fontSize: 13, paraSpaceAfter: 6, objectName: "list-ours",
  });
  text(s, "No made-up wiring anywhere: every connection is real BANC data.", { x: 0.55, y: 4.15, w: 8.9, h: 0.3, fontSize: 12, color: RED, objectName: "rule" });
  gloss(s, [["BANC:", "the public map of a whole fly brain and nerve cord (Bates et al. 2026)"]]);
}

pres.addSection({ title: "Next" });

// 4. Wpływ — kto na tym zyskuje
{
  const s = content("Why it matters: who gains", "Next");
  const cards = [
    ["Neuroscientists", "Put real wiring in a body: which neurons matter?", "Found: single flight neurons know where the target is (99%)", T.accent2],
    ["Drone & robot builders", "Heading from a plain camera and a biological circuit", "Direction: small vision-guided drones", T.accent1],
    ["Students & educators", "Watch a real brain work: 3D explorer and live BANC panel", "Connectomics you can see", VIOLET],
    ["Anyone", "Open data and code, runs on a laptop GPU", "2.6 ms per frame for the full connectome", T.accent3],
  ];
  const cw = 4.35, ch = 1.45;
  cards.forEach(([h, d, e, col], i) => {
    const x = 0.62 + (i % 2) * (cw + 0.15), y = 1.2 + Math.floor(i / 2) * (ch + 0.18);
    card(s, x, y, cw, ch, `u-${i}`);
    text(s, h, { x: x + 0.25, y: y + 0.18, w: cw - 0.5, h: 0.34, fontSize: 15, bold: true, color: col, objectName: `u-${i}-h` });
    text(s, d, { x: x + 0.25, y: y + 0.56, w: cw - 0.5, h: 0.45, fontSize: 12, color: T.lt1, objectName: `u-${i}-d` });
    text(s, e, { x: x + 0.25, y: y + 1.06, w: cw - 0.5, h: 0.28, fontSize: 11, color: T.lt2, objectName: `u-${i}-e` });
  });
  gloss(s, [["DN:", "descending neuron, carries commands from the brain down to the body"],
    ["GPU:", "graphics card, updates all neurons in parallel"], ["ms:", "millisecond"]]);
  s.addNotes("Who is this for? Neuroscientists get a closed-loop test bench: the connectome has to produce behaviour, and " +
    "we can ask which neurons matter. Robot builders get a working example of insect-style visual homing; this is a " +
    "direction, not a product. Students get something they can see. Everything is open and runs on a gaming laptop.");
}

// 5. Od demo do użytkownika
{
  const s = content("From demo to users", "Next");
  const stages = [
    ["TODAY", "Simulation", ["Full BANC v888 steers the heading", "X2 drone in MuJoCo with noisy sensors", `Target reached in ${W.best.reached} of 6 new worlds`], RED, "demonstrated"],
    ["NEXT", "Real hardware", ["Two cameras on a real micro-drone", "Connectome on an onboard GPU", "Same decoder, same calibration"], ORANGE, "planned"],
    ["LATER", "Applications", ["Search & rescue: home in on a beacon", "Inspection where GPS is poor", "Low-power neuromorphic chips"], T.lt2, "potential"],
  ];
  const w = 2.75, gap = 0.3;
  stages.forEach(([k, h, items, col, status], i) => {
    const x = 0.62 + i * (w + gap), y = 1.2;
    card(s, x, y, w, 3.25, `st-${i}`, i === 0 ? RED : undefined);
    eyebrow(s, k, x + 0.25, y + 0.22, 2, col, `st-${i}-k`);
    text(s, h, { x: x + 0.25, y: y + 0.5, w: w - 0.5, h: 0.4, fontSize: 17, bold: true, objectName: `st-${i}-h` });
    text(s, items.map((t, j) => ({ text: t, options: { breakLine: j < items.length - 1 } })), {
      x: x + 0.25, y: y + 1.05, w: w - 0.45, h: 1.6, fontSize: 12, paraSpaceAfter: 8, color: T.lt1, objectName: `st-${i}-list`,
    });
    eyebrow(s, status, x + 0.25, y + 2.85, 2, col, `st-${i}-status`);
  });
  gloss(s, [["MuJoCo:", "physics simulator widely used in robotics research"],
    ["Neuromorphic chip:", "a brain-like chip that uses very little power"]]);
  s.addNotes("Be clear about what is shown and what is a direction. Today: simulation only, but with noisy sensors, wind " +
    "and motor lag. Next step is hardware. The applications column is potential, not something we tested.");
}

// 7. Zespół
{
  const s = content("How we split the work", "Next");
  const parts = [
    ["Person 1", "The eyes", "Takes the two drone camera images, turns them into what a fly eye sees and passes " +
      "that signal into the matching neurons of the fly brain.", "gives: activity of visual neurons", T.accent1],
    ["Person 2", "The brain", "Runs the whole fly connectome on a graphics card, listens to the flight neurons " +
      "and translates them into drone commands.", "gives: thrust, roll, pitch, yaw", T.accent2],
    ["Person 3", "Body and world", "Builds the drone, its sensors, wind and random worlds in a physics " +
      "simulator, flies the commands and renders the next camera image.", "gives: the next camera image", T.accent4],
  ];
  const w = 2.8, gap = 0.22;
  parts.forEach(([who, h, d, out, col], i) => {
    const x = 0.62 + i * (w + gap), y = 1.15;
    card(s, x, y, w, 2.85, `part-${i}`, col);
    eyebrow(s, who, x + 0.25, y + 0.22, w - 0.5, col, `part-${i}-who`);
    text(s, h, { x: x + 0.25, y: y + 0.5, w: w - 0.5, h: 0.4, fontSize: 17, bold: true, objectName: `part-${i}-h` });
    text(s, d, { x: x + 0.25, y: y + 1.0, w: w - 0.5, h: 1.2, fontSize: 11.5, color: T.lt1, objectName: `part-${i}-d` });
    text(s, out, { x: x + 0.25, y: y + 2.35, w: w - 0.5, h: 0.3, fontSize: 11, bold: true, color: col, objectName: `part-${i}-out` });
  });
  text(s, "12-hour hackathon. We first agreed what each part hands to the next, then everyone built and tested their part on their own.", {
    x: 0.62, y: 4.12, w: 8.85, h: 0.4, fontSize: 11, color: T.lt2, objectName: "team-note",
  });
  gloss(s, [["Physics simulator:", "MuJoCo, computes how the drone moves and what its cameras see"],
    ["Graphics card (GPU):", "updates all 175,401 neurons in parallel"]]);
  s.addNotes("Each person owned one part of the loop and one hand-over: eyes give the brain neuron activity, the brain " +
    "gives the body four numbers, the body gives the eyes a new image. In the code: visual_pipeline/, banc_control/, sim/; " +
    "linked by a small message protocol (ZMQ) and shared BANC v888 neuron IDs.");
}

{
  const s = content("Try it yourself", "Next");
  const cmds = [
    ["Explore the brain", "3D view of every neuron and the flight neurons", "cd explorer; npm run dev", T.accent1],
    ["Watch it fly", "the drone flies, the fly brain lights up next to it",
      "python -m sim.run_env --banc data/decoders/world_distributed_best.npz --beacon-scale 4 --vision-range 4 --sensors real --motor-tau 0.04", RED],
    ["Record a demo", "save a video of drone, eyes and brain",
      "python scripts/fly_banc.py --local --decoder data/decoders/planB_dn.npz --thrust hold --brain --video demo.mp4", T.accent2],
  ];
  cmds.forEach(([h, d, c, col], i) => {
    const y = 1.15 + i * 1.1;
    card(s, 0.55, y, 8.9, 0.95, `cmd-${i}`);
    text(s, h, { x: 0.78, y: y + 0.12, w: 2.9, h: 0.3, fontSize: 14, bold: true, color: col, objectName: `cmd-${i}-h` });
    text(s, d, { x: 3.7, y: y + 0.14, w: 5.6, h: 0.3, fontSize: 11, color: T.lt2, objectName: `cmd-${i}-d` });
    text(s, c, { x: 0.78, y: y + 0.48, w: 8.5, h: 0.42, fontSize: 9, fontFace: MONO, color: T.lt1, objectName: `cmd-${i}-c` });
  });
  text(s, "Setup:   pip install -e .[all]      python scripts/download_banc.py      flyvis download-pretrained      python scripts/doctor.py", {
    x: 0.55, y: 4.5, w: 8.9, h: 0.26, fontSize: 10, color: T.lt2, objectName: "setup" });
}

{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Next" });
  s.addImage({ path: A("connectome_wide.jpg"), x: 4.1, y: 0.75, w: 5.9, h: 2.95, transparency: 35, objectName: "render-bg" });
  s.addImage({ path: A("logo.png"), x: 0.6, y: 1.1, w: 1.0, h: 0.59, objectName: "logo" });
  eyebrow(s, "NeuroFly   HackYeah", 0.6, 1.85, 4.5, RED, "eyebrow");
  text(s, "THANK YOU", { x: 0.6, y: 2.12, w: 5.2, h: 0.8, fontSize: 36, fontFace: HEAD, charSpacing: 2, objectName: "thanks" });
  s.addImage({ path: A("accent_line.png"), x: 0.6, y: 2.95, w: 0.05, h: 0.8, objectName: "accent-line" });
  text(s, "A real fly brain chose the heading.\nLive demo: BANC panel in flight.", { x: 0.85, y: 2.98, w: 3.6, h: 0.75, fontSize: 15, color: T.lt2, objectName: "demo" });
  text(s, "GITHUB.COM/FALCONDEVX/NEUROFLY", { x: 0.6, y: 4.6, w: 6, h: 0.3, fontSize: 9, color: RED, charSpacing: 4, objectName: "repo" });
}

// =================================================================================================
// APPENDIX — szczegóły na pytania, po podziękowaniu
// =================================================================================================

pres.addSection({ title: "Appendix" });

// 11. Model dynamiki
{
  const s = content("Details: the brain equation", "Appendix");
  card(s, 0.55, 1.15, 5.3, 1.0, "eq-card");
  const v = (t) => ({ text: t, options: { italic: true } }), o = (t) => ({ text: t });
  text(s, [v("τ"), o(" "), v("dr"), o(" / "), v("dt"), o("  =  − "), v("r"), o("  +  tanh( relu( "), v("g W r"), o("  +  "), v("I"), o(" ) )")],
    { x: 0.55, y: 1.27, w: 5.3, h: 0.45, fontSize: 20, fontFace: "Cambria Math", align: "center", objectName: "eq" });
  text(s, "computed in steps of dt = 5 ms, 4 steps per camera frame", {
    x: 0.55, y: 1.78, w: 5.3, h: 0.25, fontSize: 10, color: T.lt2, align: "center", objectName: "eq-note" });
  const rows = [
    ["W", "how strongly each neuron talks to each other one: synapse counts from BANC, plus or minus"],
    ["τ, g", "how fast neurons react (20 ms) and overall strength (0.9)"],
    ["I", "what comes in: vision from the eyes and rotation from the drone gyroscope"],
    ["GPU", "the whole brain is updated in 2.6 ms on a laptop graphics card"],
  ];
  rows.forEach(([k, v], i) => {
    text(s, k, { x: 0.55, y: 2.4 + i * 0.5, w: 1.1, h: 0.4, fontSize: 14, bold: true, color: RED, objectName: `k-${i}` });
    text(s, v, { x: 1.7, y: 2.4 + i * 0.5, w: 4.15, h: 0.45, fontSize: 12, color: T.lt1, objectName: `v-${i}` });
  });
  card(s, 6.2, 1.15, 3.3, 3.3, "sign-card");
  text(s, "Excite or calm down", { x: 6.45, y: 1.35, w: 2.8, h: 0.35, fontSize: 15, bold: true, objectName: "sign-title" });
  tag(s, 6.45, 1.85, "ACh: excites the next neuron", T.accent1, "sg-ach");
  tag(s, 6.45, 2.2, "GABA: calms it down", T.accent3, "sg-gaba");
  tag(s, 6.45, 2.55, "glutamate: calms it down", T.accent3, "sg-glu");
  tag(s, 6.45, 2.9, "histamine: calms it down", T.accent3, "sg-his");
  text(s, "The connections come from BANC. The equation and the plus / minus signs are our own choice.", {
    x: 6.45, y: 3.4, w: 2.85, h: 0.9, fontSize: 11, color: T.lt2, objectName: "sign-note",
  });
  gloss(s, [["r:", "how active each neuron is, from 0 to 1"], ["relu, tanh:", "keep activity between 0 and 1"],
    ["ACh, GABA:", "chemicals a neuron releases to signal the next one"]]);
}

// 13. Kalibracja
{
  const s = content("Details: calibration", "Appendix");
  const steps = [
    ["Rest", "look at an empty scene to learn the calm level", "without it any light gave full power"],
    ["Scale", "show the target left and right to set the volume", "vision changes the signal by only ~1 %"],
    ["Turn direction", "spin the scene and watch which way the fly turns", "left and right taken from behaviour"],
    ["Gyro strength", "set how much rotation the brain feels", "too strong and the drone just spun"],
  ];
  const w = 2.05, gap = 0.22;
  steps.forEach(([h, d, why], i) => {
    const x = 0.62 + i * (w + gap), y = 1.2;
    card(s, x, y, w, 2.55, `cal-${i}`);
    text(s, String(i + 1), { x: x + 0.22, y: y + 0.2, w: 0.5, h: 0.45, fontSize: 22, fontFace: HEAD, color: RED, objectName: `cal-${i}-n` });
    text(s, h, { x: x + 0.22, y: y + 0.78, w: w - 0.4, h: 0.32, fontSize: 14, bold: true, objectName: `cal-${i}-h` });
    text(s, d, { x: x + 0.22, y: y + 1.15, w: w - 0.4, h: 0.55, fontSize: 11, color: T.lt1, objectName: `cal-${i}-d` });
    text(s, why, { x: x + 0.22, y: y + 1.8, w: w - 0.4, h: 0.6, fontSize: 10, italic: true, color: T.lt2, objectName: `cal-${i}-why` });
  });
  text(s, "Takes about 6 s at start-up, automatically, with no hand tuning.", {
    x: 0.62, y: 3.95, w: 8.85, h: 0.45, fontSize: 12, color: T.lt2, objectName: "cal-note",
  });
  gloss(s, [["Optomotor reflex:", "flies turn with a rotating scene to keep their view steady"]]);
  s.addNotes("Real FlyVis + MuJoCo scenes change motor-neuron activity by only about 1% of the resting level, so the " +
    "readout has to be calibrated against the network's own rest state. Without the haltere gain step the gyroscope " +
    "input dominated: raw yaw ~6000 per 1 rad/s.");
}

// 14. Haltery
{
  const s = content("Details: the fly's gyroscope", "Appendix");
  card(s, 0.62, 1.2, 4.25, 2.3, "h-bad");
  card(s, 5.22, 1.2, 4.25, 2.3, "h-good", RED);
  eyebrow(s, "turning and tilting", 0.9, 1.4, 3.5, T.lt2, "h-bad-k");
  stat(s, 0.9, 1.75, 3.7, "6–11 rad/s", "after a push the drone spins faster and faster", T.lt2, "h-bad-s");
  eyebrow(s, "turning only", 5.5, 1.4, 3.7, RED, "h-good-k");
  stat(s, 5.5, 1.75, 3.7, "0.03 rad/s", "after the same push: the fly brain stops the spin", RED, "h-good-s");
  text(s, "In BANC the fly's rotation sensors mostly control turning, hardly tilting. So we give them only the turning speed, " +
    "and the simulator keeps the drone level. Target 29° to the side: off by only 1° after 5 s " +
    "(without these sensors it keeps swinging between 26° and 33°).", { x: 0.62, y: 3.7, w: 8.85, h: 0.85, fontSize: 12, color: T.lt1, objectName: "h-note" });
  gloss(s, [["Kick:", "a sudden push that spins the drone"], ["rad/s:", "rotation speed, 1 rad/s ≈ 57°/s"],
    ["Yaw rate:", "how fast the drone turns left or right"]]);
}

// 16. Ograniczenie danych
{
  const s = content("Details: the left-eye gap", "Appendix");
  card(s, 0.55, 1.2, 4.3, 2.6, "left-card");
  card(s, 5.15, 1.2, 4.3, 2.6, "right-card");
  eyebrow(s, "left eye", 0.9, 1.38, 3, T.lt2, "l-k");
  eyebrow(s, "right eye", 5.5, 1.38, 3, RED, "r-k");
  stat(s, 0.9, 1.65, 3.7, "36%", "of visual neurons are labelled", T.lt1, "s-l");
  stat(s, 0.9, 2.75, 3.7, "5,535", "neurons the left eye can reach", T.lt2, "s-l2");
  stat(s, 5.5, 1.65, 3.7, "80%", "of visual neurons are labelled", RED, "s-r");
  stat(s, 5.5, 2.75, 3.7, "16,927", "neurons the right eye can reach", RED, "s-r2");
  text(s, "Vision can only reach labelled neurons, so the left eye reaches ~3× fewer. Boosting it did not change " +
    "the result, and single flight neurons still tell left from right.", { x: 0.55, y: 3.95, w: 8.9, h: 0.6, fontSize: 12, color: T.lt2, objectName: "note" });
  gloss(s, [["Labelled:", "the neuron's type is known, for example a motion detector"]]);
}

// 18. Czujniki
{
  const s = content("Details: imperfect sensors", "Appendix");
  const sens = [
    ["Gyroscope", "measures rotation, with noise and slow drift", T.accent4],
    ["Rangefinder", "height above ground, sometimes misses", T.accent4],
    ["Optical flow", "speed from a camera looking down", T.accent4],
    ["Barometer", "height from air pressure, drifts slowly", T.accent4],
    ["GPS + compass", "position off by about 2.5 m, arrives late", T.accent2],
    ["Motors", "need time to speed up, like real ones", T.accent3],
  ];
  sens.forEach(([h, d, col], i) => {
    const col2 = i % 2, row = Math.floor(i / 2);
    const x = 0.62 + col2 * 4.5, y = 1.15 + row * 0.9;
    card(s, x, y, 4.35, 0.78, `sn-${i}`);
    text(s, h, { x: x + 0.22, y: y + 0.1, w: 3.9, h: 0.3, fontSize: 14, bold: true, color: col, objectName: `sn-${i}-h` });
    text(s, d, { x: x + 0.22, y: y + 0.4, w: 4.0, h: 0.36, fontSize: 10, color: T.lt2, objectName: `sn-${i}-d` });
  });
  text(s, "The drone only gets these imperfect readings, never the true values.", {
    x: 0.62, y: 3.95, w: 8.8, h: 0.4, fontSize: 12, color: T.lt1, objectName: "sn-note",
  });
  gloss(s, [["Drift:", "an error that slowly grows over time"], ["Why:", "a model trained on perfect data breaks in the real world"]]);
  s.addNotes("A model trained on perfect values leans on them instead of on BANC and breaks in real conditions. So the " +
    "simulator adds noise, drift, dropouts and delay to every sensor, and the final world training ran with these real sensors.");
}

// 21. Teacher odchodzi, model zostaje
{
  const s = content("Details: the teacher steps back", "Appendix");
  const labels = W.blocks.map((b) => `${b.from + 1}–${b.from + b.n}`);
  s.addChart(pres.charts.BAR, [{ name: "target reached", labels, values: W.blocks.map((b) => Math.round((100 * b.reached) / b.n)) }],
    chartOpts({
      x: 0.5, y: 1.1, w: 5.9, h: 2.85, barDir: "col", barGapWidthPct: 45,
      chartColors: W.blocks.map((b) => (b.beta === 0 ? RED : T.accent5)),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', valAxisMaxVal: 115, valAxisMinVal: 0,
      showLegend: false, objectName: "chart-blocks",
    }));
  text(s, "teacher share:   " + W.blocks.map((b) => b.beta.toFixed(2)).join("        "), {
    x: 0.7, y: 4.0, w: 5.7, h: 0.26, fontSize: 10, color: T.lt2, objectName: "beta-row" });
  text(s, "training flights, 50 per bar; red: the model flies alone", { x: 0.7, y: 4.28, w: 5.7, h: 0.26, fontSize: 10, color: T.lt2, objectName: "chart-note" });
  stat(s, 6.8, 1.25, 2.8, `${W.model_only.reached} / ${W.model_only.n}`, "training flights flown by the model alone that reached the target", RED, "s-alone");
  stat(s, 6.8, 2.7, 2.8, `${W.flips} / ${W.episodes}`, "times the drone flipped over in all training flights", T.lt1, "s-flips");
  gloss(s, [["Target reached:", "the drone touches the target within 40 s"], ["Teacher share:", "how much of the time the teacher flies"]]);
  s.addNotes(`${W.episodes} episodes: ${Object.values(W.workers).map((w) => `${w.episodes} on ${w.gpu}`).join(", ")}. ` +
    "As the teacher's share β drops to zero, the success rate per 50 episodes stays at 88–96%.");
}

// 25. Szybkość
{
  const s = content("Details: speed", "Appendix");
  s.addChart(pres.charts.BAR, [{ name: "ms per camera frame", labels: ["processor", "graphics card"], values: [24, 2.6] }],
    chartOpts({
      x: 0.5, y: 1.2, w: 5.4, h: 2.4, barDir: "bar", barGapWidthPct: 50,
      chartColors: [T.accent5, RED],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0" ms"', valAxisMaxVal: 30, valAxisMinVal: 0,
      showLegend: false, objectName: "chart-speed",
    }));
  text(s, "time to update the whole fly brain once per camera image, on a laptop", { x: 0.7, y: 3.65, w: 5.2, h: 0.26, fontSize: 10, color: T.lt2, objectName: "speed-note" });
  stat(s, 6.4, 1.25, 3, "9×", "faster on the graphics card, with the same results", RED, "s-x");
  stat(s, 6.4, 2.65, 3, "0.3 ms", "for one update of all 1.53 M connections", T.lt1, "s-sub");
  text(s, "Fast enough to keep up with the camera in real time.", { x: 0.62, y: 4.1, w: 8.8, h: 0.35, fontSize: 12, color: T.lt2, objectName: "speed-d" });
  gloss(s, [["ms:", "millisecond, a thousandth of a second"]]);
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
