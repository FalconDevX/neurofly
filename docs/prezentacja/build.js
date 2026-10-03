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
  headFontFace: "Segoe UI",
  bodyFontFace: "Segoe UI",
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

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 × 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "NeuroFly — mózg muszki pilotuje drona";
pres.author = "Zespół NeuroFly";
const C = pres.SchemeColor;

// ---------- układy (layouts) ----------
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: T.dk1 },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: T.dk1 },
  objects: [
    { image: { path: A("logo.png"), x: 0.5, y: 5.13, w: 0.34, h: 0.2 } },
    { text: { text: "NeuroFly", options: { x: 0.9, y: 5.08, w: 2, h: 0.3, fontSize: 10, color: T.lt2, margin: 0 } } },
    {
      placeholder: {
        options: { name: "title", type: "title", x: 0.5, y: 0.35, w: 9, h: 0.65, fontSize: 28, bold: true, color: T.lt1, margin: 0, valign: "top", align: "left" },
        text: "",
      },
    },
  ],
  slideNumber: { x: 9.0, y: 5.08, w: 0.5, h: 0.3, fontSize: 10, color: T.lt2, align: "right" },
});

// ---------- pomocnicze ----------
const shadow = () => ({ type: "outer", color: "000000", blur: 18, offset: 6, angle: 90, opacity: 0.55 });

function card(slide, x, y, w, h, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: T.dk2 }, line: { color: T.accent6, width: 0.75 }, shadow: shadow(), objectName: name,
  });
}

function text(slide, t, opts) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 14, color: T.lt1, valign: "top", ...opts });
}

function stat(slide, x, y, w, value, label, color, name) {
  text(slide, value, { x, y, w, h: 0.6, fontSize: 36, bold: true, color: color || T.lt1, objectName: `${name}-value` });
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
  s.addText(title, { placeholder: "title" });
  return s;
}

// =================================================================================================
// 1. Tytuł
pres.addSection({ title: "Start" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Start" });
  s.addImage({ path: A("connectome_wide.jpg"), x: 4.1, y: 0.75, w: 5.9, h: 2.95, objectName: "render-connectome" });
  s.addImage({ path: A("logo.png"), x: 0.6, y: 1.25, w: 1.2, h: 0.71, objectName: "logo" });
  text(s, "NeuroFly", { x: 0.6, y: 2.1, w: 5, h: 0.9, fontSize: 54, bold: true, objectName: "title" });
  text(s, "Mózg muszki owocowej\npilotuje drona", { x: 0.6, y: 3.0, w: 3.5, h: 0.8, fontSize: 20, color: T.lt2, objectName: "subtitle" });
  text(s, "Pełny connectome BANC v888 · 175 401 neuronów · pętla zamknięta w MuJoCo", {
    x: 0.6, y: 4.55, w: 8.8, h: 0.35, fontSize: 12, color: T.accent5, objectName: "footer",
  });
  s.addNotes("NeuroFly: obraz z kamer drona przechodzi przez model oka muszki (FlyVis), potem przez prawdziwy connectome " +
    "Drosophila (BANC v888, mózg + brzuszny łańcuch nerwowy), a aktywność neuronów lotu steruje dronem w symulatorze.");
}

// 2. Idea: pętla zamknięta
pres.addSection({ title: "Idea" });
{
  const s = content("Zamknięta pętla: wzrok → connectome → lot", "Idea");
  const steps = [
    ["Obraz", "dwie kamery-oczy drona", T.accent1],
    ["Connectome", "BANC v888 na GPU", T.accent2],
    ["Sterowanie", "neurony lotu → komendy", T.accent3],
    ["Ruch", "dron w MuJoCo → nowy obraz", T.accent4],
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
  s.addNotes("Każda klatka obrazu przechodzi przez całą pętlę ~30 razy na sekundę. Nic nie jest ręcznie zaprogramowanym " +
    "regulatorem kursu: kierunek do celu odczytujemy z aktywności neuronów zstępujących (DN) w BANC.");
}

// 3. Pipeline
pres.addSection({ title: "Pipeline" });
{
  const s = content("Pipeline: od piksela do śmigła", "Pipeline");
  const nodes = [
    ["Kamera", "2 oczy MuJoCo, 157°, ±70°", T.accent1],
    ["Siatkówka", "FlyGym: 721 ommatidiów / oko", T.accent1],
    ["FlyVis", "model płata wzrokowego", T.accent1],
    ["Mapa → BANC", "22 462 neuronów v888", T.accent1],
    ["BANC v888", "175 401 neuronów, GPU", T.accent2],
    ["Odczyt", "6 grup MN + 375 DN lotu", T.accent3],
    ["Dekoder", "liniowy, uczony (DAgger)", T.lt2],
    ["Dron", "thrust · roll · pitch · yaw", T.accent4],
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
  text(s, "następna klatka: nowy obraz z kamer", { x: 0.5, y: 4.45, w: 4, h: 0.3, fontSize: 11, color: T.accent5, objectName: "loop-note" });
  s.addNotes("Górny rząd: wszystko, co dzieje się przed connectomem. Dolny: connectome, odczyt neuronów lotu, dekoder " +
    "i dron. Jedna klatka (FlyVis + 4 podkroki dynamiki BANC na GPU) to ~15–20 ms.");
}

// 4. Przed connectomem: oczy → siatkówka
{
  const s = content("Zanim sygnał trafi do BANC: oko muszki", "Pipeline");
  const ims = [["eye_left.jpg", "Kamera lewa"], ["retina_left.jpg", "Siatkówka lewa"], ["eye_right.jpg", "Kamera prawa"], ["retina_right.jpg", "Siatkówka prawa"]];
  ims.forEach(([f, cap], i) => {
    const x = 0.5 + i * 2.27;
    s.addImage({ path: A(f), x, y: 1.3, w: 2.0, h: 2.28, shadow: shadow(), objectName: `img-${i}` });
    text(s, cap, { x, y: 3.66, w: 2.0, h: 0.28, fontSize: 11, color: T.lt2, objectName: `cap-${i}` });
  });
  text(s, "Kamery MuJoCo z polem widzenia oka muchy → korekcja rybiego oka → 721 heksagonalnych ommatidiów na oko (FlyGym) → " +
    "FlyVis: wytrenowany model płata wzrokowego oparty na connectomie → aktywność typów komórek mapowana na neurony BANC v888.", {
    x: 0.5, y: 4.08, w: 9, h: 0.8, fontSize: 13, color: T.lt1, objectName: "explain",
  });
  s.addNotes("Cel (czarny słup) jest po prawej stronie: widać go tylko w prawym oku, jako ciemną kolumnę ommatidiów.");
}

// 5. Connectome BANC v888
pres.addSection({ title: "Connectome" });
{
  const s = content("BANC v888: mózg i brzuszny łańcuch nerwowy", "Connectome");
  s.addImage({ path: A("connectome_3d.jpg"), x: 0.4, y: 1.05, w: 3.35, h: 4.07, objectName: "render-3d" });
  stat(s, 4.3, 1.25, 2.6, "175 401", "neuronów (bez glejów i tchawek)", T.lt1, "s-neurons");
  stat(s, 7.0, 1.25, 2.6, "18,6 mln", "synaps w połączeniach ≥ 5", T.lt1, "s-syn");
  stat(s, 4.3, 2.55, 2.6, "1,53 mln", "połączeń w grafie modelu", T.lt1, "s-edges");
  stat(s, 7.0, 2.55, 2.6, "863", "szkielety neuronów lotu (SWC)", T.lt1, "s-skel");
  tag(s, 4.3, 3.95, "płaty wzrokowe · 105 646", T.accent1, "t-ol");
  tag(s, 4.3, 4.27, "mózg centralny · 42 620", "A98BFF", "t-cb");
  tag(s, 7.0, 3.95, "VNC · 26 769", T.accent3, "t-vnc");
  tag(s, 7.0, 4.27, "DN / AN · 3 165", T.accent2, "t-dn");
  s.addNotes("Oficjalny BANC v888 (Bates et al. 2026, publiczny bucket Lee Lab). Somy z kolumny position; kolory jak w eksploratorze: wzrok teal, interneurony fiolet, " +
    "DN amber, motoneurony rose. Żadnych syntetycznych grafów.");
}

// 6. Model dynamiki
{
  const s = content("Jak liczymy aktywność na connectomie", "Connectome");
  card(s, 0.5, 1.3, 5.3, 1.25, "eq-card");
  text(s, "r ← r + dt/τ · (−r + tanh(relu(g · W r + I)))", { x: 0.75, y: 1.62, w: 4.9, h: 0.5, fontSize: 20, fontFace: "Cambria", objectName: "eq" });
  const rows = [
    ["W", "liczby synaps z BANC, znak z przewidywanego neuroprzekaźnika, normalizacja wejść każdego neuronu"],
    ["τ, dt, g", "20 ms, 5 ms, 0,9 — 4 podkroki na klatkę obrazu"],
    ["I", "wejście: wzrok (FlyVis → BANC) + żyroskop drona na aferenty halter"],
    ["GPU", "macierz rzadka CSR w torch: ~2,6 ms na klatkę (RTX 4060)"],
  ];
  rows.forEach(([k, v], i) => {
    text(s, k, { x: 0.5, y: 2.85 + i * 0.5, w: 1.1, h: 0.4, fontSize: 14, bold: true, color: T.accent1, objectName: `k-${i}` });
    text(s, v, { x: 1.65, y: 2.85 + i * 0.5, w: 4.2, h: 0.45, fontSize: 12, color: T.lt1, objectName: `v-${i}` });
  });
  card(s, 6.2, 1.3, 3.3, 3.45, "sign-card");
  text(s, "Znaki synaps", { x: 6.45, y: 1.5, w: 2.8, h: 0.35, fontSize: 15, bold: true, objectName: "sign-title" });
  tag(s, 6.45, 2.0, "ACh — pobudzenie", T.accent1, "sg-ach");
  tag(s, 6.45, 2.35, "GABA — hamowanie", T.accent3, "sg-gaba");
  tag(s, 6.45, 2.7, "glutaminian — hamowanie", T.accent3, "sg-glu");
  tag(s, 6.45, 3.05, "histamina — hamowanie", T.accent3, "sg-his");
  text(s, "Model dynamiki i znaki to nasze założenia; połączenia i liczby synaps są z BANC.", {
    x: 6.45, y: 3.55, w: 2.85, h: 0.9, fontSize: 11, color: T.lt2, objectName: "sign-note",
  });
}

// 7. Obwód lotu
{
  const s = content("Obwód lotu: tylko oficjalne adnotacje BANC", "Connectome");
  s.addImage({ path: A("circuit_3d.jpg"), x: 0.4, y: 1.05, w: 3.35, h: 4.07, objectName: "render-circuit" });
  const groups = [
    ["DN flight power", "235", T.accent2, "super_cluster: flight power"],
    ["DN flight steering", "140", T.accent2, "super_cluster: flight steering"],
    ["MN wing power", "24", T.accent3, "DLM / DVM — mięśnie mocy"],
    ["MN wing steering", "24", T.accent3, "b1, i1, iii3… — mięśnie sterujące"],
    ["MN wing tension", "12", T.accent3, "napięcie skrzydła"],
    ["Aferenty halter", "428", T.accent4, "czujniki obrotu → żyroskop drona"],
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
pres.addSection({ title: "Odczyt" });
{
  const s = content("Strona celu jest w pojedynczych DN, nie w średnich", "Odczyt");
  s.addChart(pres.charts.BAR, [{ name: "Trafność strony celu", labels: ["6 średnich MN", "MN pojedynczo", "wzrok (VPN)", "DN lotu pojedynczo"], values: [68, 84, 94, 99] }], {
    x: 0.5, y: 1.2, w: 5.6, h: 3.6, barDir: "bar",
    chartColors: [T.accent5, T.accent5, T.accent5, T.accent1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', dataLabelColor: T.lt1, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: T.lt2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valAxisMaxVal: 110, valAxisMinVal: 0,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLineShow: false,
    showLegend: false, showTitle: false, objectName: "chart-side",
  });
  stat(s, 6.6, 1.35, 3, "99%", "trafność strony celu z 375 pojedynczych DN lotu (walidacja: nieznana odległość)", T.accent1, "s-dn");
  stat(s, 6.6, 2.85, 3, "68%", "z 6 średnich grup MN — uśrednianie kasuje różnicę lewo / prawo", T.lt2, "s-mn");
  s.addNotes("Regresja grzbietowa kąt ~ aktywność, cel pod kątami −90…+90° w 3 odległościach, walidacja na odległości " +
    "pominiętej w uczeniu (scripts/check_side_decoding.py).");
}

// 9. Ograniczenie danych
{
  const s = content("Ograniczenie v888: asymetria płatów wzrokowych", "Odczyt");
  card(s, 0.5, 1.3, 4.3, 2.9, "left-card");
  card(s, 5.2, 1.3, 4.3, 2.9, "right-card");
  stat(s, 0.85, 1.55, 3.7, "36%", "neuronów lewego płata ma typ komórki", T.lt1, "s-l");
  stat(s, 0.85, 2.85, 3.7, "5 535", "neuronów zasila lewe oko", T.lt2, "s-l2");
  stat(s, 5.55, 1.55, 3.7, "80%", "neuronów prawego płata ma typ komórki", T.accent1, "s-r");
  stat(s, 5.55, 2.85, 3.7, "16 927", "neuronów zasila prawe oko", T.accent1, "s-r2");
  text(s, "Mapa FlyVis → BANC idzie po typach komórek, więc lewe oko dociera do ~3× mniej neuronów. Wyrównanie wejścia nie zmieniło " +
    "wyniku; odczyt z pojedynczych DN i tak rozpoznaje obie strony.", { x: 0.5, y: 4.35, w: 9, h: 0.6, fontSize: 12, color: T.lt2, objectName: "note" });
}

// 10. Symulator
pres.addSection({ title: "Symulator" });
{
  const s = content("Symulator: dron X2 w MuJoCo", "Symulator");
  s.addImage({ path: A("drone_chase.jpg"), x: 0.5, y: 1.25, w: 4.3, h: 3.22, shadow: shadow(), objectName: "img-drone" });
  s.addImage({ path: A("world.jpg"), x: 5.2, y: 1.25, w: 4.3, h: 3.22, shadow: shadow(), objectName: "img-world" });
  text(s, "Skydio X2 (MuJoCo Menagerie) z oczami na nosie · zawis i obrót na płaskiej scenie", { x: 0.5, y: 4.55, w: 4.3, h: 0.45, fontSize: 11, color: T.lt2, objectName: "cap-1" });
  text(s, "Losowy teren 60 × 60 m, bloki, cel z masztem · tryb angle: symulator utrzymuje zadany przechył", { x: 5.2, y: 4.55, w: 4.3, h: 0.45, fontSize: 11, color: T.lt2, objectName: "cap-2" });
}

// 11. Trening
pres.addSection({ title: "Trening" });
{
  const s = content("Trening: uczy się tylko dekoder, BANC się nie zmienia", "Trening");
  const steps = [
    ["Nauczyciel", "zna prawdziwy stan: kąt do celu, wysokość, prędkość"],
    ["Lot mieszany", "z prawdopodobieństwem β steruje nauczyciel, β: 1 → 0"],
    ["Dane", "pary (cechy, komenda nauczyciela); trudne próbki ×3–7"],
    ["Regresja", "wagi od nowa na wszystkich danych (DAgger + ridge)"],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.3 + i * 0.86;
    card(s, 0.5, y, 4.9, 0.72, `t-${i}`);
    text(s, String(i + 1), { x: 0.72, y: y + 0.16, w: 0.4, h: 0.4, fontSize: 20, bold: true, color: T.accent1, objectName: `t-${i}-n` });
    text(s, h, { x: 1.2, y: y + 0.1, w: 4, h: 0.3, fontSize: 14, bold: true, objectName: `t-${i}-h` });
    text(s, d, { x: 1.2, y: y + 0.39, w: 4.1, h: 0.3, fontSize: 11, color: T.lt2, objectName: `t-${i}-d` });
  });
  card(s, 5.8, 1.3, 3.7, 1.55, "split");
  text(s, "Podział osi", { x: 6.05, y: 1.45, w: 3.2, h: 0.3, fontSize: 14, bold: true, objectName: "split-h" });
  tag(s, 6.05, 1.85, "yaw ← tylko BANC (DN)", T.accent2, "sp-yaw");
  tag(s, 6.05, 2.2, "thrust/roll/pitch ← BANC + czujniki", T.accent4, "sp-ctl");
  card(s, 5.8, 3.05, 3.7, 1.6, "dist");
  text(s, "2 GPU w LAN", { x: 6.05, y: 3.2, w: 3.2, h: 0.3, fontSize: 14, bold: true, objectName: "dist-h" });
  text(s, "Master rozdaje światy, workerzy (RTX 4060, RTX 3070 Ti) liczą epizody i odsyłają tylko statystyki XᵀX, Xᵀy — " +
    "bez klatek wideo.", { x: 6.05, y: 3.55, w: 3.25, h: 1.0, fontSize: 11, color: T.lt2, objectName: "dist-d" });
}

// 12. Wyniki: zawis i skręt
pres.addSection({ title: "Wyniki" });
{
  const s = content("Wynik: skręt do celu z BANC", "Wyniki");
  const labels = ["cel −60°", "cel −30°", "cel +30°", "cel +60°"];
  s.addChart(pres.charts.BAR, [
    { name: "przed treningiem", labels, values: ["-60", "-30", "+30", "+60"].map((k) => +DATA.before[k].toFixed(1)) },
    { name: "po treningu", labels, values: ["-60", "-30", "+30", "+60"].map((k) => +DATA.after[k].toFixed(1)) },
  ], {
    x: 0.5, y: 1.2, w: 5.8, h: 3.7, barDir: "col", barGapWidthPct: 60,
    chartColors: [T.accent5, T.accent1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"°"', dataLabelColor: T.lt1, dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: T.lt2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendColor: T.lt2, legendFontSize: 11, legendFontFace: "+mn-lt",
    showTitle: false, objectName: "chart-eval",
  });
  stat(s, 6.8, 1.35, 2.8, "2,65°", "średni końcowy błąd kursu bez nauczyciela (przed treningiem 25,1°)", T.accent1, "s-err");
  stat(s, 6.8, 2.75, 2.8, "1000", "epizodów na 2 GPU: 672 + 328", T.lt1, "s-ep");
  text(s, "Zawis na płaskiej scenie, wysokość trzyma symulator. Kierunek: wyłącznie pojedyncze DN z BANC.", {
    x: 6.8, y: 4.05, w: 2.8, h: 0.8, fontSize: 11, color: T.lt2, objectName: "s-note",
  });
}

// 13. Wyniki: lot w świecie (w toku)
{
  const s = content("Lot do celu w świecie: w toku", "Wyniki");
  card(s, 0.5, 1.3, 2.85, 1.75, "c-teacher");
  card(s, 3.58, 1.3, 2.85, 1.75, "c-before");
  card(s, 6.65, 1.3, 2.85, 1.75, "c-model");
  stat(s, 0.8, 1.55, 2.4, "6 / 6", "nauczyciel dolatuje do celu — jest od kogo się uczyć", T.lt1, "w-t");
  stat(s, 3.88, 1.55, 2.4, "18,1 m", "przed treningiem: średnio najbliżej celu", T.lt2, "w-b");
  stat(s, 6.95, 1.55, 2.4, "9,0 m", "po 192 epizodach: 2× bliżej, 0 wywrotek, cel 0 / 6", T.accent1, "w-m");
  text(s, "Uczciwie: model leci stabilnie i zbliża się do celu, ale jeszcze nie ląduje na polu. Kolejny trening: walidacja co 50 " +
    "epizodów z zapisem najlepszych wag i doważenie trudnych próbek.", { x: 0.5, y: 3.35, w: 9, h: 0.8, fontSize: 13, color: T.lt2, objectName: "w-note" });
}

// 14. Z BANC vs nasze założenia
pres.addSection({ title: "Uczciwość" });
{
  const s = content("Co jest z BANC, a co jest naszym założeniem", "Uczciwość");
  card(s, 0.5, 1.3, 4.3, 2.75, "bank");
  card(s, 5.2, 1.3, 4.3, 2.75, "ours");
  tag(s, 0.8, 1.5, "z BANC v888", T.accent1, "h-banc");
  tag(s, 5.5, 1.5, "nasze założenia", T.accent2, "h-ours");
  const banc = ["neurony, połączenia, liczby synaps", "przewidywany neuroprzekaźnik", "grupy lotu z oficjalnych adnotacji", "pozycje som i szkielety SWC"];
  const ours = ["model dynamiki i parametry", "znaki neuroprzekaźników", "mapa FlyVis → BANC po typach komórek", "liniowy dekoder i podział osi", "czujniki drona dla thrust/roll/pitch"];
  text(s, banc.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < banc.length - 1 } })), {
    x: 0.8, y: 2.0, w: 3.8, h: 2.6, fontSize: 14, paraSpaceAfter: 6, objectName: "list-banc",
  });
  text(s, ours.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < ours.length - 1 } })), {
    x: 5.5, y: 2.0, w: 3.8, h: 2.6, fontSize: 14, paraSpaceAfter: 6, objectName: "list-ours",
  });
}

// 15. Zakończenie
pres.addSection({ title: "Koniec" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Koniec" });
  s.addImage({ path: A("connectome_wide.jpg"), x: 4.1, y: 0.75, w: 5.9, h: 2.95, transparency: 35, objectName: "render-bg" });
  s.addImage({ path: A("logo.png"), x: 0.6, y: 1.35, w: 1.0, h: 0.59, objectName: "logo" });
  text(s, "Dziękujemy", { x: 0.6, y: 2.1, w: 5, h: 0.8, fontSize: 44, bold: true, objectName: "thanks" });
  text(s, "Demo: lot z panelem BANC na żywo\ni eksplorator 3D connectomu", { x: 0.6, y: 2.95, w: 3.5, h: 0.75, fontSize: 16, color: T.lt2, objectName: "demo" });
  text(s, "github.com/FalconDevX/neurofly", { x: 0.6, y: 4.55, w: 6, h: 0.35, fontSize: 12, color: T.accent1, objectName: "repo" });
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
