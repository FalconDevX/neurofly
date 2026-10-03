const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const gi = require("react-icons/gi");
const fa = require("react-icons/fa6");
const { applyTheme } = require("C:/Users/mateu/.claude/skills/synced/26456d5b-782a-4a8b-a5a9-dd8dcdd3f668_eac86e09-a636-449d-ad84-8f3300b72176/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "NeuroFly.pptx";

const THEME = {
  name: "NeuroFly",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "2B2116", // ciemny bursztyn tułowia
    lt1: "FFFFFF",
    dk2: "6B5844", // przygaszony brąz, tekst drugorzędny
    lt2: "F1EFEC", // neutralne tło kart
    accent1: "B3122E", // czerwień oka Drosophila
    accent2: "D9A441", // miód
    accent3: "8A6A44",
    accent4: "4E6E58",
    accent5: "E3C27E",
    accent6: "9C8E80",
    hlink: "B3122E",
    folHlink: "8A6A44",
  },
};
const H = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "NeuroFly";
pres.author = "Zespół NeuroFly";
const C = pres.SchemeColor;

// ---------- ikony ----------
async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// heks z ikoną — motyw omatidium
function hexIcon(slide, img, x, y, d, fill, name) {
  slide.addShape(pres.shapes.HEXAGON, {
    x, y, w: d, h: d * 0.88, fill: { color: fill }, line: { type: "none" }, objectName: name + " hex",
  });
  const s = d * 0.5;
  slide.addImage({ data: img, x: x + (d - s) / 2, y: y + (d * 0.88 - s) / 2, w: s, h: s, objectName: name + " ikona" });
}

// ---------- layouty ----------
pres.defineSlideMaster({
  title: "DARK",
  background: { color: H.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 5.4, h: 1.2, fontFace: THEME.headFontFace, fontSize: 48, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "Tytuł" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.85, w: 5.2, h: 1.0, fontSize: 20, color: C.accent2, valign: "top", align: "left", margin: 0 }, text: "Podtytuł" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.lt1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontFace: THEME.headFontFace, fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "Tytuł" } },
    { text: { text: "NeuroFly", options: { x: 0.5, y: 5.2, w: 3, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: H.dk2, align: "right" },
});

const T = (slide, text) => slide.addText(text, { placeholder: "title" });
const txt = (slide, text, o) => slide.addText(text, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...o });

(async () => {
  const I = {
    flyW: await icon(gi.GiFly, H.lt1),
    camW: await icon(fa.FaCamera, H.lt1),
    eyeW: await icon(fa.FaEye, H.lt1),
    brainW: await icon(fa.FaBrain, H.lt1),
    wingW: await icon(fa.FaWaveSquare, H.lt1),
    droneW: await icon(gi.GiDeliveryDrone, H.lt1),
    droneD: await icon(gi.GiDeliveryDrone, H.dk1),
    slidersW: await icon(fa.FaSliders, H.lt1),
    targetW: await icon(fa.FaCrosshairs, H.lt1),
    branchW: await icon(fa.FaCodeBranch, H.lt1),
    clockW: await icon(fa.FaClock, H.lt1),
    gaugeW: await icon(fa.FaGaugeHigh, H.lt1),
    warnW: await icon(fa.FaTriangleExclamation, H.lt1),
    netW: await icon(fa.FaNetworkWired, H.lt1),
    playW: await icon(fa.FaPlay, H.dk1),
  };

  // ================= 1. Tytuł =================
  pres.addSection({ title: "Wstęp" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Wstęp" });
  T(s, "NeuroFly");
  s.addText("Connectome muszki owocowej pilotuje drona", { placeholder: "body" });
  txt(s, "Hackathon, zespół 3 osób", { x: 0.6, y: 4.6, w: 5, h: 0.35, fontSize: 14, color: C.accent6 });
  // plaster omatidiów
  {
    const d = 0.78, hh = d * 0.88, dx = d * 0.76, dy = hh;
    const red = new Set(["2,1", "3,2", "1,3", "4,1", "3,0"]);
    for (let c = 0; c < 5; c++) {
      for (let r = 0; r < 5; r++) {
        const x = 6.0 + c * dx;
        const y = 0.35 + r * dy + (c % 2 ? hh / 2 : 0);
        if (y + hh > 5.35) continue;
        const key = `${c},${r}`;
        const center = key === "2,2";
        const fill = center ? H.accent1 : red.has(key) ? "7A1A22" : "3E3122";
        s.addShape(pres.shapes.HEXAGON, { x, y, w: d - 0.06, h: hh - 0.05, fill: { color: fill }, line: { type: "none" }, objectName: `omatidium ${key}` });
        if (center) s.addImage({ data: I.flyW, x: x + 0.16, y: y + 0.12, w: 0.44, h: 0.44, objectName: "muszka" });
      }
    }
  }
  s.addNotes("NeuroFly: bierzemy prawdziwy connectome muszki owocowej (BANC) i każemy mu sterować symulowanym dronem. Obraz z kamery drona trafia do modelu wzroku muszki, sygnał płynie przez jej mózg i rdzeń do motoneuronów skrzydeł, a stamtąd do silników drona.");

  // ================= 2. Pętla =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wstęp" });
  T(s, "Jedna zamknięta pętla: od piksela do śmigła");
  {
    const steps = [
      [I.camW, "Kamera RGB", "obraz z drona w Gazebo", H.dk2],
      [I.eyeW, "Retina + FlyVis", "model oka i płatów wzrokowych", H.dk2],
      [I.brainW, "Connectome BANC", "mózg + VNC, dynamika rate", H.accent1],
      [I.wingW, "Motoneurony skrzydeł", "6 grup → 4 komendy lotu", H.dk2],
      [I.droneW, "Dron w Gazebo", "fizyka, IMU, nowa klatka", H.dk2],
    ];
    const w = 1.62, gap = 0.22, x0 = 0.5, d = 0.95;
    steps.forEach(([img, head, sub, fill], i) => {
      const x = x0 + i * (w + gap);
      hexIcon(s, img, x + (w - d) / 2, 1.7, d, fill, `krok ${i + 1}`);
      txt(s, head, { x, y: 2.65, w, h: 0.55, fontSize: 15, bold: true, align: "center", valign: "top" });
      txt(s, sub, { x, y: 3.42, w, h: 0.5, fontSize: 12, color: C.text2, align: "center" });
      if (i < steps.length - 1)
        s.addShape(pres.shapes.LINE, { x: x + w - 0.12, y: 2.12, w: gap + 0.24, h: 0, line: { color: H.accent6, width: 1.5, endArrowType: "triangle" }, objectName: `strzałka ${i + 1}` });
    });
    // powrót
    const yb = 4.15;
    s.addShape(pres.shapes.LINE, { x: 0.5 + 4 * (w + gap) + w / 2, y: 3.9, w: 0, h: yb - 3.9, line: { color: H.accent1, width: 1.5 }, objectName: "powrót 1" });
    s.addShape(pres.shapes.LINE, { x: 0.5 + w / 2, y: yb, w: 4 * (w + gap), h: 0, line: { color: H.accent1, width: 1.5 }, objectName: "powrót 2" });
    s.addShape(pres.shapes.LINE, { x: 0.5 + w / 2, y: 3.9, w: 0, h: yb - 3.9, line: { color: H.accent1, width: 1.5, beginArrowType: "triangle" }, objectName: "powrót 3" });
    txt(s, "lot zmienia to, co widzi kamera, więc pętla domyka się w każdej klatce", { x: 2.0, y: 4.3, w: 6.0, h: 0.35, fontSize: 13, italic: true, color: C.accent1, align: "center" });
  }
  s.addNotes("Pętla jest zamknięta: to, jak dron poleci, zmienia następną klatkę z kamery. Nie ma tu ręcznie napisanego autopilota w środku. Decyzje przechodzą przez graf neuronów muszki.");

  // ================= 3. BANC w liczbach =================
  pres.addSection({ title: "Connectome" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Connectome" });
  T(s, "BANC: cały układ nerwowy w jednym grafie");
  txt(s, "Mózg i brzuszny łańcuch nerwowy (VNC) w jednym connectome. Droga od oka do skrzydła nie ma przerw, więc możemy ją symulować wprost.", { x: 0.5, y: 1.3, w: 3.3, h: 1.6, fontSize: 16 });
  txt(s, "Dane: BANC, materializacja 888", { x: 0.5, y: 4.6, w: 3.3, h: 0.3, fontSize: 11, color: C.text2 });
  {
    const stats = [
      ["188 508", "neuronów", "z czego 805 to motoneurony"],
      ["35,7 mln", "synaps", "zliczonych w krawędziach grafu"],
      ["1,6 mln", "silnych połączeń", "≥ 5 synaps, z 11,75 mln wszystkich"],
    ];
    stats.forEach(([big, label, sub], i) => {
      const y = 1.2 + i * 1.25;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.3, y, w: 5.2, h: 1.08, rectRadius: 0.08, fill: { color: H.lt2 }, line: { type: "none" }, objectName: `stat ${i + 1} tło` });
      txt(s, big, { x: 4.5, y: y + 0.1, w: 2.5, h: 0.88, fontFace: THEME.headFontFace, fontSize: 36, bold: true, color: i === 0 ? C.accent1 : C.text1, valign: "middle" });
      txt(s, label, { x: 7.05, y: y + 0.17, w: 2.3, h: 0.36, fontSize: 16, bold: true, valign: "middle" });
      txt(s, sub, { x: 7.05, y: y + 0.53, w: 2.3, h: 0.45, fontSize: 11, color: C.text2 });
    });
  }
  s.addNotes("Liczby pochodzą prosto z plików BANC 888, które mamy w repo. Do symulacji odcinamy słabe połączenia: zostawiamy krawędzie z co najmniej 5 synapsami, czyli około 1,6 miliona.");

  // ================= 4. Lejek =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Connectome" });
  T(s, "Od oka do skrzydła: 62 neurony na końcu");
  s.addChart(pres.charts.BAR, [{
    name: "Neurony",
    labels: ["Płaty wzrokowe", "Projekcje wzrokowe", "Mózg centralny", "Neurony zstępujące (DN)", "VNC", "Motoneurony skrzydeł"],
    values: [72947, 7316, 31879, 1316, 12866, 62],
  }], {
    x: 0.4, y: 1.15, w: 5.9, h: 3.9, barDir: "bar", catAxisOrientation: "maxMin",
    chartColors: [H.dk2], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "# ##0",
    dataLabelColor: H.dk1, dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: H.dk1, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLineShow: false, showLegend: false, showTitle: false, barGapWidthPct: 45,
  });
  {
    const items = [
      ["24", "power", "amplituda uderzenia, czyli ciąg"],
      ["24", "steering", "płaszczyzna uderzenia, czyli kierunek"],
      ["12", "tension", "napięcie mięśni skrzydła"],
    ];
    txt(s, "Wszystko, co muszka widzi, kończy się na 62 motoneuronach skrzydeł:", { x: 6.6, y: 1.25, w: 2.9, h: 0.8, fontSize: 14 });
    items.forEach(([n, k, d], i) => {
      const y = 2.15 + i * 0.85;
      txt(s, n, { x: 6.6, y, w: 0.75, h: 0.6, fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.accent1, valign: "middle" });
      txt(s, k, { x: 7.4, y: y + 0.02, w: 2.1, h: 0.3, fontSize: 14, bold: true });
      txt(s, d, { x: 7.4, y: y + 0.3, w: 2.1, h: 0.4, fontSize: 11, color: C.text2 });
    });
  }
  s.addNotes("Słupki idą w kolejności przepływu sygnału. Dziesiątki tysięcy neuronów wzrokowych, potem około 1300 neuronów zstępujących, które łączą mózg z VNC, a na końcu tylko 62 motoneurony skrzydeł. Z tych 62 odczytujemy sterowanie.");

  // ================= 5. Zespół =================
  pres.addSection({ title: "Architektura" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architektura" });
  T(s, "Trzy osoby, trzy moduły, dwa kontrakty");
  {
    const cols = [
      [I.eyeW, "Osoba 1", "Wzrok", "Kamera RGB → FlyGym Retina → RetinaMapper → FlyVis → aktywność neuronów BANC"],
      [I.brainW, "Osoba 2", "Sterowanie", "BANC/VNC → motoneurony skrzydeł → thrust, roll, pitch, yaw (banc_control/)"],
      [I.droneW, "Osoba 3", "Symulacja", "Gazebo Sim, quadcopter z kamerą i IMU, epizody treningowe"],
    ];
    const w = 2.6, gap = 0.6;
    cols.forEach(([img, who, role, body], i) => {
      const x = 0.5 + i * (w + gap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w, h: 2.45, rectRadius: 0.08, fill: { color: H.lt2 }, line: { type: "none" }, objectName: `${who} karta` });
      hexIcon(s, img, x + 0.25, 1.45, 0.72, i === 1 ? H.accent1 : H.dk2, who);
      txt(s, who, { x: x + 1.12, y: 1.5, w: 1.35, h: 0.28, fontSize: 12, color: C.text2 });
      txt(s, role, { x: x + 1.12, y: 1.78, w: 1.4, h: 0.38, fontSize: 18, bold: true });
      txt(s, body, { x: x + 0.25, y: 2.4, w: w - 0.5, h: 1.2, fontSize: 13 });
    });
    const contracts = [
      ["JSON z aktywnością", "banc_root_id, cell_type, activity"],
      ["FlightCommand ↔ IMU", "komenda lotu w jedną stronę, żyroskop w drugą"],
    ];
    contracts.forEach(([h, d], i) => {
      const cx = 0.5 + (i + 1) * w + i * gap + gap / 2;
      s.addShape(pres.shapes.LINE, { x: cx - 0.22, y: 2.5, w: 0.44, h: 0, line: { color: H.accent2, width: 2, beginArrowType: "triangle", endArrowType: "triangle" }, objectName: `kontrakt ${i + 1} strzałka` });
      s.addShape(pres.shapes.LINE, { x: cx, y: 2.62, w: 0, h: 1.18, line: { color: H.accent2, width: 1 }, objectName: `kontrakt ${i + 1} linia` });
      txt(s, h, { x: cx - 1.3, y: 3.9, w: 2.6, h: 0.3, fontSize: 13, bold: true, color: C.accent3, align: "center" });
      txt(s, d, { x: cx - 1.3, y: 4.2, w: 2.6, h: 0.5, fontSize: 11, color: C.text2, align: "center" });
    });
  }
  s.addNotes("Podział jest czysty: każda osoba ma swój moduł i komunikuje się z sąsiadem przez jeden kontrakt danych. Dzięki stubom (FakeVision, ToyDrone) każdy moduł da się testować samodzielnie, zanim połączymy całość.");

  // ================= 6. Mapowanie =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architektura" });
  T(s, "Skrzydła muszki jako komendy quadcoptera");
  {
    const rows = [
      ["thrust", "średnia amplituda uderzenia obu skrzydeł"],
      ["roll", "asymetria amplitudy lewe − prawe"],
      ["pitch", "przesunięcie płaszczyzny uderzenia przód − tył"],
      ["yaw", "przeciwne przesunięcie płaszczyzny na L i P"],
    ];
    rows.forEach(([k, d], i) => {
      const y = 1.3 + i * 0.88;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 5.6, h: 0.72, rectRadius: 0.06, fill: { color: H.lt2 }, line: { type: "none" }, objectName: `${k} tło` });
      txt(s, k, { x: 0.75, y, w: 1.2, h: 0.72, fontFace: THEME.headFontFace, fontSize: 22, bold: true, color: C.accent1, valign: "middle" });
      txt(s, d, { x: 2.0, y, w: 3.95, h: 0.72, fontSize: 14, valign: "middle" });
    });
    // panel halter
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.5, y: 1.3, w: 3.0, h: 3.36, rectRadius: 0.08, fill: { color: H.dk1 }, line: { type: "none" }, objectName: "panel halter" });
    hexIcon(s, I.gaugeW, 6.75, 1.55, 0.7, H.accent1, "halter");
    txt(s, "Halter = IMU", { x: 6.75, y: 2.35, w: 2.5, h: 0.4, fontSize: 18, bold: true, color: C.background1 });
    txt(s, "Muszka czuje obroty narządami zwanymi halterami. Żyroskop drona podajemy jako ich aferenty w VNC, więc stabilizacja też przechodzi przez connectome.", { x: 6.75, y: 2.8, w: 2.55, h: 1.75, fontSize: 13, color: C.background1 });
  }
  s.addNotes("To mapowanie jest umowne, ale oparte na tym, jak muszka naprawdę steruje lotem: amplituda daje siłę, asymetria daje przechył, a przesunięcie płaszczyzny uderzenia daje pochylenie i obrót. Haltery to biologiczny żyroskop, a IMU drona wchodzi dokładnie w to miejsce.");

  // ================= 7. Plany dekodera =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architektura" });
  T(s, "Trzy dekodery, a demo jest zawsze");
  {
    const plans = [
      [I.targetW, "Plan A", "Adaptive", "Uczenie z nagrody: stabilny zawis i kierunek na beacon. Perturbacja wag dekodera.", H.accent1],
      [I.branchW, "Plan B", "Linear", "Mała warstwa liniowa uczona regułą LMS. Szybka i przewidywalna.", H.dk2],
      [I.slidersW, "Plan C", "Manual", "Ręcznie skalibrowane wzmocnienia, zero uczenia. Gwarantowane demo.", H.dk2],
    ];
    const w = 2.8, gap = 0.3;
    plans.forEach(([img, p, name, body, fill], i) => {
      const x = 0.5 + i * (w + gap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w, h: 2.2, rectRadius: 0.08, fill: { color: H.lt2 }, line: { type: "none" }, objectName: `${p} karta` });
      hexIcon(s, img, x + 0.25, 1.45, 0.72, fill, p);
      txt(s, p, { x: x + 1.12, y: 1.5, w: 1.5, h: 0.28, fontSize: 12, color: C.text2 });
      txt(s, name, { x: x + 1.12, y: 1.78, w: 1.55, h: 0.38, fontSize: 18, bold: true });
      txt(s, body, { x: x + 0.25, y: 2.4, w: w - 0.5, h: 0.95, fontSize: 13 });
    });
    txt(s, "Connectome zostaje nietknięty. Uczy się wyłącznie dekoder na wyjściu z motoneuronów.", { x: 0.5, y: 3.8, w: 9.0, h: 0.5, fontSize: 15, italic: true, color: C.accent1 });
  }
  s.addNotes("Mamy trzy poziomy ambicji. Plan A uczy się z nagrody, Plan B to prosta regresja, Plan C to ręczne strojenie, które działa zawsze. Ważne: nie zmieniamy wag w connectome. Cała plastyczność jest w małym dekoderze, więc zachowanie wynika z prawdziwej sieci muszki.");

  // ================= 8. Harmonogram =================
  pres.addSection({ title: "Realizacja" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Realizacja" });
  T(s, "12 godzin: najpierw pętla, potem uczenie");
  {
    const phases = [
      [0, 4, "Moduły równolegle", "dane BANC, grupy neuronów, Plan C"],
      [4, 6, "Pełna pętla", "stuby → prawdziwy FlyVis i Gazebo"],
      [6, 9, "Plan A", "uczenie z nagrody"],
      [9, 10, "Fallback", "Plan B lub C"],
      [10, 12, "Demo", "nagranie, logi, wizualizacja"],
    ];
    const x0 = 0.5, W = 9.0, u = W / 12, y = 2.0;
    phases.forEach(([a, b, h, d], i) => {
      const x = x0 + a * u, w = (b - a) * u;
      const fill = i === 4 ? H.accent1 : i % 2 ? H.accent2 : H.dk2;
      s.addShape(pres.shapes.RECTANGLE, { x: x + 0.03, y, w: w - 0.06, h: 0.5, fill: { color: fill }, line: { type: "none" }, objectName: `faza ${i + 1}` });
      txt(s, `${a}–${b} h`, { x, y: y - 0.4, w, h: 0.3, fontSize: 12, color: C.text2, align: "center" });
      const tw = Math.max(w, 1.5), tx = Math.min(Math.max(x + w / 2 - tw / 2, x0), x0 + W - tw);
      const ty = i % 2 ? 3.55 : 2.7;
      txt(s, h, { x: tx, y: ty, w: tw, h: 0.32, fontSize: 14, bold: true, align: "center" });
      txt(s, d, { x: tx, y: ty + 0.32, w: tw, h: 0.5, fontSize: 11, color: C.text2, align: "center" });
      s.addShape(pres.shapes.LINE, { x: x + w / 2, y: y + 0.5, w: 0, h: ty - y - 0.55, line: { color: H.accent6, width: 0.75 }, objectName: `faza ${i + 1} łącznik` });
    });
  }
  s.addNotes("Kolejność jest celowa: najpierw cała pętla działa na stubach i Planie C, dopiero potem bierzemy się za uczenie. Jeśli Plan A nie ustabilizuje się do 9. godziny, przechodzimy na B albo C i spokojnie przygotowujemy demo.");

  // ================= 9. Ryzyka =================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Realizacja" });
  T(s, "Ryzyka znamy i każde ma plan awaryjny");
  {
    const risks = [
      ["Różne wersje root ID", "Jedna materializacja BANC dla Osoby 1 i 2. Logujemy unmatched_ids."],
      ["Które neurony sterują lotem?", "Grupy bierzemy z adnotacji BANC (power, steering), nie zgadujemy nazw."],
      ["Czas kroku symulacji", "Mierzymy na pełnym grafie, w razie potrzeby tniemy do podgrafu wzrok → DN → VNC."],
      ["Mapowanie jest umowne", "Plan C musi latać, zanim zaczniemy cokolwiek uczyć."],
    ];
    risks.forEach(([h, d], i) => {
      const x = 0.5 + (i % 2) * 4.6, y = 1.25 + Math.floor(i / 2) * 1.85;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.4, h: 1.65, rectRadius: 0.08, fill: { color: H.lt2 }, line: { type: "none" }, objectName: `ryzyko ${i + 1} tło` });
      hexIcon(s, I.warnW, x + 0.22, y + 0.25, 0.62, H.accent1, `ryzyko ${i + 1}`);
      txt(s, h, { x: x + 1.05, y: y + 0.22, w: 3.15, h: 0.4, fontSize: 16, bold: true, valign: "middle" });
      txt(s, d, { x: x + 1.05, y: y + 0.66, w: 3.15, h: 0.85, fontSize: 13, color: C.text2 });
    });
  }
  s.addNotes("Największe ryzyko to niezgodne identyfikatory neuronów między modułami, bo root ID zmieniają się po proofreadingu. Dlatego wszyscy pracujemy na jednej wersji BANC i liczymy, ile neuronów się nie dopasowało.");

  // ================= 10. Demo =================
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Realizacja" });
  T(s, "Demo");
  s.addText("Zawis i lot do beacona, sterowane przez connectome muszki", { placeholder: "body" });
  {
    const items = ["Nagranie zawisu i lotu do beacona", "Logi aktywności motoneuronów na żywo", "Wizualizacja ścieżki w BANC", "Kod: banc_control/ z testami pytest"];
    items.forEach((t, i) => {
      const y = 1.3 + i * 0.85;
      s.addShape(pres.shapes.HEXAGON, { x: 6.3, y, w: 0.5, h: 0.44, fill: { color: i === 0 ? H.accent1 : "3E3122" }, line: { type: "none" }, objectName: `punkt ${i + 1}` });
      txt(s, t, { x: 7.0, y: y - 0.05, w: 2.6, h: 0.6, fontSize: 14, color: C.background1, valign: "middle" });
    });
  }
  s.addNotes("Na koniec pokazujemy nagranie: dron zawisa i leci do beacona, a obok widać na żywo aktywność motoneuronów skrzydeł, które tym sterują.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("ok", OUT);
})();
