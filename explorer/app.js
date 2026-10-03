// Eksplorator BANC v888: three.js bez frameworka. Dane z serwera Pythona (scripts/explorer.py).
// Cała selekcja (kategorie, grupy, zakres mózg/VNC, tryb, beacon) to uniformy shaderów — bez przebudowy buforów.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ViewHelper } from "three/addons/helpers/ViewHelper.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

// ---------- dane ----------

const CATS = [
  { key: "sens", name: "Sensoryczne", short: "Sensoryczne", color: "#4f8cff", classes: ["sensory", "sensory_ascending", "sensory_descending"] },
  { key: "motor", name: "Motoryczne", short: "Motoryczne", color: "#ff4fb0", classes: ["motor", "visceral_circulatory", "ascending_visceral_circulatory"] },
  { key: "dnan", name: "Zstępujące / wstępujące", short: "DN / AN", color: "#ffb547", classes: ["descending", "ascending"] },
  { key: "vis", name: "Wzrokowe", short: "Wzrokowe", color: "#2fd3c4", classes: ["optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"] },
  { key: "inter", name: "Interneurony", short: "Interneurony", color: "#a98bff", classes: ["central_brain_intrinsic", "ventral_nerve_cord_intrinsic"] },
  { key: "unk", name: "Nieoznaczone", short: "Nieoznaczone", color: "#6f7896", classes: ["unknown"] },
];
const GROUPS = [
  { key: "dn_flight_power", name: "DN flight power", color: "#ffb547" },
  { key: "dn_flight_steering", name: "DN flight steering", color: "#ff8a3d" },
  { key: "wing_power", name: "MN wing power", color: "#ff4fb0" },
  { key: "wing_steering", name: "MN wing steering", color: "#ff7ad9" },
  { key: "wing_tension", name: "MN wing tension", color: "#d65cff" },
  { key: "haltere_aff", name: "Aferenty halter", color: "#4f8cff" },
];
const groupKey = (n) => n.group.replace(/_[LR]$/, "");
const GROUP_INDEX = Object.fromEntries(GROUPS.map((g, i) => [g.key, i]));

const SCALE = 0.01; // 1 jednostka sceny = 100 µm; oś y odwrócona (w BANC y rośnie w stronę VNC)
const actLevel = (a) => Math.min(1, Math.max(0, (Math.log10(Math.max(a, 1e-6)) + 6) / 6)); // log 10⁻⁶..1 → 0..1
const ALPH = "0123456789abcdefghijklmnopqrstuvwxyz";

function prepare(raw, cmds) {
  const S = raw.somas.xyz, n = S.length / 3;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) { const v = S[3 * i + d]; if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v; }
  // środek bryły, nie średnia (większość som leży w płatach wzrokowych)
  const center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const toScene = (x, y, z) => [(x - center[0]) * SCALE, -(y - center[1]) * SCALE, (z - center[2]) * SCALE];

  const classToCat = raw.classes.map((c) => Math.max(0, CATS.findIndex((k) => k.classes.includes(c))));
  const somaPos = new Float32Array(n * 3), somaCat = new Float32Array(n), somaAct = new Float32Array(n * 3);
  const reg = Object.fromEntries(raw.regions.map((r, i) => [r, i]));
  const sum = {};
  let brainMax = -Infinity, vncMin = Infinity;
  for (let i = 0; i < n; i++) {
    const x = S[3 * i], y = S[3 * i + 1], z = S[3 * i + 2];
    somaPos.set(toScene(x, y, z), 3 * i);
    somaCat[i] = classToCat[raw.somas.super_class[i]];
    for (let b = 0; b < 3; b++) somaAct[3 * i + b] = ALPH.indexOf(raw.somas.act[b][i]) / 35;
    const r = raw.somas.region[i];
    if (r === reg.central_brain && y > brainMax) brainMax = y;
    if (r === reg.ventral_nerve_cord && y < vncMin) vncMin = y;
    const key = r === reg.optic_lobe ? (x < center[0] ? "optic_lobe" : "") : raw.regions[r];
    if (key) { const s = (sum[key] ??= [0, 0, 0, 0]); s[0] += x; s[1] += y; s[2] += z; s[3]++; }
  }
  const neckUm = (Math.min(brainMax, 420) + Math.max(vncMin, 420)) / 2;
  const centroid = (k) => { const s = sum[k]; return toScene(s[0] / s[3], s[1] / s[3], s[2] / s[3]); };
  const brainCenter = centroid("central_brain"), vncCenter = centroid("ventral_nerve_cord");

  // szkielety → jedna geometria LineSegments (pary wierzchołków)
  let segs = 0;
  for (const nr of raw.neurons) for (const l of nr.lines) segs += l.length - 1;
  const pos = new Float32Array(segs * 6), group = new Float32Array(segs * 2), neuron = new Float32Array(segs * 2), act = new Float32Array(segs * 6);
  const pick = [];
  let k = 0;
  raw.neurons.forEach((nr, ni) => {
    const g = GROUP_INDEX[groupKey(nr)], lv = nr.act.map(actLevel);
    for (const l of nr.lines) {
      for (let j = 0; j < l.length; j++) {
        const p = toScene(...l[j]);
        if (j % 3 === 0) pick.push(p[0], p[1], p[2], ni);
        if (j === 0) continue;
        pos.set(toScene(...l[j - 1]), 6 * k); pos.set(p, 6 * k + 3);
        group[2 * k] = group[2 * k + 1] = g;
        neuron[2 * k] = neuron[2 * k + 1] = ni;
        act.set(lv, 6 * k); act.set(lv, 6 * k + 3);
        k++;
      }
    }
  });

  return {
    raw, cmds, toScene, somaPos, somaCat, somaAct,
    neckY: -(neckUm - center[1]) * SCALE,
    brainCenter, vncCenter,
    labels: [
      { name: "Mózg centralny", pos: brainCenter },
      { name: "Płat wzrokowy", pos: centroid("optic_lobe") },
      { name: "VNC", pos: vncCenter },
    ],
    line: { pos, group, neuron, act },
    pick: new Float32Array(pick),
    groupOf: raw.neurons.map((nr) => GROUP_INDEX[groupKey(nr)]),
    byId: new Map(raw.neurons.map((x, i) => [x.id, i])),
  };
}

// ---------- shadery ----------

const somaVertex = `
  attribute float aCat;
  attribute vec3 aAct;
  attribute float aLive;  // aktywność z modelu na żywo (uint8 znormalizowany → 0..1)
  uniform int uLive;
  uniform float uCatOn[6];
  uniform vec3 uCatColor[6];
  uniform int uBearing;
  uniform int uMode;      // 0 eksploruj, 1 obwód lotu, 2 aktywność
  uniform int uScope;     // 0 całość, 1 mózg, 2 VNC
  uniform float uNeckY;
  uniform float uSize;
  uniform float uPR;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    int c = int(aCat + 0.5);
    bool inScope = uScope == 0 || (uScope == 1 ? position.y > uNeckY : position.y <= uNeckY);
    if (uCatOn[c] < 0.5 || !inScope) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
    float a = uLive == 1 ? aLive : (uBearing == 0 ? aAct.x : (uBearing == 1 ? aAct.y : aAct.z));
    if (uMode == 2) {
      float hot = clamp((a - 0.4) / 0.6, 0.0, 1.0);
      vColor = mix(vec3(0.30, 0.36, 0.72), vec3(1.0, 0.82, 0.32), hot);
      vAlpha = 0.04 + 0.55 * hot;
    } else {
      vColor = uCatColor[c];
      vAlpha = uMode == 1 ? 0.05 : 0.22;
    }
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * uPR * (10.0 / -mv.z);
  }`;

// zwykłe mieszanie alfa z małym kryciem: gęste miejsca dążą do koloru klasy, nie do bieli
const somaFragment = `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.5, 0.1, d));
  }`;

const lineVertex = `
  attribute float aGroup;
  attribute float aNeuron;
  attribute vec3 aAct;
  attribute float aLive;
  uniform int uLive;
  uniform float uGroupOn[6];
  uniform vec3 uGroupColor[6];
  uniform float uSel;
  uniform int uBearing;
  uniform int uMode;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vY;
  void main() {
    int g = int(aGroup + 0.5);
    float on = uGroupOn[g] * (abs(aNeuron - uSel) < 0.5 ? 0.0 : 1.0);  // wybrany rysowany osobno
    float a = uLive == 1 ? aLive : (uBearing == 0 ? aAct.x : (uBearing == 1 ? aAct.y : aAct.z));
    if (uMode == 2) {
      vColor = a > 0.55 ? vec3(1.0, 0.82, 0.35) : vec3(0.36, 0.42, 0.66);
      vAlpha = (0.05 + 0.45 * a) * on;
    } else {
      vColor = uGroupColor[g];
      vAlpha = (uMode == 1 ? 0.5 : 0.22) * on;
    }
    vY = position.y;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const lineFragment = `
  uniform int uScope;
  uniform float uNeckY;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vY;
  void main() {
    if (vAlpha < 0.005) discard;
    if (uScope == 1 && vY <= uNeckY) discard;
    if (uScope == 2 && vY > uNeckY) discard;
    gl_FragColor = vec4(vColor, vAlpha);
  }`;

// ---------- stan i interfejs ----------

const MODES = [
  { id: "explore", name: "Eksploruj", note: "Somy kolorowane klasą (super_class), szkielety neuronów lotu na wierzchu." },
  { id: "flight", name: "Obwód lotu", note: "Wygaszone tło, wyróżniony obwód lotu: DN → VNC → motoneurony skrzydeł, haltery." },
  { id: "activity", name: "Aktywność", note: "Jasność = aktywność z modelu dla wybranego beacona (skala log 10⁻⁶–1)." },
];
const SCOPES = [{ id: "all", name: "Cały BANC" }, { id: "brain", name: "Tylko mózg" }, { id: "vnc", name: "Tylko VNC" }];
const MODE_ID = { explore: 0, flight: 1, activity: 2 };
const SCOPE_ID = { all: 0, brain: 1, vnc: 2 };
const AXES = ["thrust", "roll", "pitch", "yaw"];
const BEARINGS = ["lewo", "prosto", "prawo"];
const CLASS_PL = {
  descending: "DN", motor: "motoneuron", sensory: "sensoryczny", sensory_ascending: "sens. wstęp.", sensory_descending: "sens. zstęp.",
  central_brain_intrinsic: "interneuron", ventral_nerve_cord_intrinsic: "interneuron VNC", optic_lobe_intrinsic: "płat wzr.",
  visual_projection: "proj. wzrokowa", visual_centrifugal: "wzr. odśrodk.", ascending: "AN",
};
const REGION_SHORT = { optic_lobe: "płat wzr.", central_brain: "mózg", ventral_nerve_cord: "VNC" };
const REGION_NAMES = { optic_lobe: "Płat wzrokowy", central_brain: "Mózg centralny", ventral_nerve_cord: "VNC", "ascending/descending/inne": "Inne / szyja" };
const REGION_COLORS = ["#2fd3c4", "#a98bff", "#ff4fb0", "#6f7896"];
const HOME_DISTANCE = 15.5;

const fmt = (n) => n.toLocaleString("pl-PL");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const $ = (id) => document.getElementById(id);

const view = { bearing: 1, mode: "explore", scope: "all", cats: CATS.map(() => true), groups: GROUPS.map(() => true), selected: -1, spin: false };
const ui = { by: "class", dir: "in", copied: false };
// model na żywo: BANC v888 liczony przez serwer na CUDA, sterowany suwakami
const live = { available: false, on: true, info: null, bearing: 0, yaw: 0, state: null, flight: null };
const liveActive = () => live.available && live.on;
const signed = (v, d) => `${v > 0 ? "+" : ""}${(+v).toFixed(d)}`;
let data = null, gfx = null;

function set(patch) {
  const scopeChanged = "scope" in patch && patch.scope !== view.scope;
  const selChanged = "selected" in patch && patch.selected !== view.selected;
  Object.assign(view, patch);
  if (selChanged) ui.copied = false;
  render();
  if (gfx) {
    gfx.update();
    if (scopeChanged) gfx.reset();
    if (selChanged) gfx.select(view.selected);
  }
}

function render() {
  $("tabs").innerHTML = MODES.map((m) =>
    `<button class="tab" role="tab" aria-selected="${view.mode === m.id}" data-act="mode" data-arg="${m.id}">${m.name}</button>`).join("");
  $("pills").innerHTML = SCOPES.map((s) =>
    `<button aria-pressed="${view.scope === s.id}" data-act="scope" data-arg="${s.id}">${s.name}</button>`).join("");
  $("mode-note").textContent = view.mode === "activity" && liveActive()
    ? "Jasność = aktywność modelu liczonego teraz na GPU dla ustawień z suwaków (skala log 10⁻⁶–1)."
    : MODES.find((m) => m.id === view.mode).note;
  $("spin").setAttribute("aria-pressed", view.spin);
  const all = view.cats.every(Boolean);
  $("legend").innerHTML = `<button class="all" aria-pressed="${all}" data-act="cats"><i></i>Wszystkie</button>` +
    CATS.map((c, i) => `<button aria-pressed="${view.cats[i]}" data-act="cat" data-arg="${i}"><i style="background:${c.color}"></i>${c.name}</button>`).join("");
  if (!data) return;
  renderSide();
  renderRight();
}

function renderSide() {
  const counts = GROUPS.map((g) => data.raw.neurons.filter((n) => groupKey(n) === g.key).length);
  const s = data.raw.stats, on = liveActive();
  const bars = AXES.map((a) =>
    `<div class="cmd"><span>${a}</span><div class="bar ${a === "thrust" ? "" : "center"}"><i id="bar-${a}"></i></div><output id="out-${a}"></output></div>`).join("");
  const liveCard = `
    <div class="card">
      <div class="label">Model na żywo</div>
      <div class="status"><i${live.available ? "" : ' style="background:var(--muted);box-shadow:none"'}></i><span id="live-status">${esc(liveStatus())}</span></div>
      ${live.available ? `<label class="check"><input type="checkbox" id="live-on"${live.on ? " checked" : ""}> Licz na żywo (pełny v888)</label>` : ""}
    </div>`;
  const inputs = on ? `
      <label class="range"><span>Kąt</span><input type="range" id="live-bearing" min="-90" max="90" step="1" value="${live.bearing}"><output id="live-bearing-out">${signed(live.bearing, 0)}°</output></label>
      <label class="range"><span>Yaw rate</span><input type="range" id="live-yaw" min="-3" max="3" step="0.1" value="${live.yaw}"><output id="live-yaw-out">${signed(live.yaw, 1)}</output></label>
      <p class="note">Yaw rate (rad/s) z IMU → aferenty halter. Wzrok: FakeVision (lewa/prawa strona visual_projection), nie FlyVis.</p>`
    : `<div class="seg" role="group" aria-label="Kierunek beacona">${BEARINGS.map((b, i) => {
        const deg = data.raw.bearings[i];
        return `<button aria-pressed="${view.bearing === i}" data-act="bearing" data-arg="${i}">${b} ${deg !== 0 ? `${deg > 0 ? "+" : ""}${deg}°` : ""}</button>`;
      }).join("")}</div>`;
  const kv = [
    ["Neurony", fmt(s.neurons)], ["Połączenia ≥ 5", fmt(s.edges)], ["Synapsy", fmt(s.synapses)],
    ["Szkielety lotu", fmt(data.raw.neurons.length)], ["Wersja", "v888"], ["Źródło", "Lee Lab / Dataverse"],
  ];
  $("side").innerHTML = `
    <div class="card">
      <div class="label">Zbiór danych</div>
      <div class="select">BANC v888 <span class="label" style="letter-spacing:0">publikacja</span></div>
      <div class="status"><i></i>Dane oficjalne · Lee Lab</div>
    </div>
    <div class="card">
      <div class="label">Neurony lotu</div>
      <div class="nav">${GROUPS.map((g, i) =>
        `<button aria-pressed="${view.groups[i]}" data-act="group" data-arg="${i}"><i style="background:${g.color}"></i>${g.name}<span>${counts[i]}</span></button>`).join("")}</div>
    </div>
    ${liveCard}
    <div class="card">
      <div class="label">Beacon (wejście wzrokowe)</div>
      ${inputs}
      <div class="label">Komendy → dron (Plan C${on ? ", na żywo" : ""})</div>
      <div class="cmds">${bars}</div>
    </div>
    <div class="card">
      <h2>Przegląd zbioru</h2>
      <dl class="kv">${kv.map(([k, v]) => `<div class="kvrow"><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
      <a class="btn" href="https://doi.org/10.7910/DVN/7WTH1N" target="_blank" rel="noopener noreferrer">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>Pobierz dane BANC</a>
    </div>`;
  updateCmds();
}

function liveStatus() {
  const i = live.info;
  if (!i) return "łączenie z serwerem…";
  if (!live.available) return `Wyłączony: ${i.status}`;
  const st = live.state;
  return `${i.device === "cpu" ? "CPU" : `CUDA · ${i.gpu}`} · ${fmt(i.n)} neuronów` + (st ? ` · ${st.ms.toFixed(1)} ms/krok · t ${st.t.toFixed(1)} s` : "");
}

/** Paski komend i wartości na żywo bez przebudowy panelu (suwaki zostają nietknięte). */
function updateCmds() {
  if (!data) return;
  const cmd = liveActive() && live.state ? live.state.cmd : data.cmds[view.bearing];
  for (const a of AXES) {
    const bar = $(`bar-${a}`), out = $(`out-${a}`);
    if (!bar) continue;
    const v = cmd[a], w = a === "thrust" ? v * 100 : Math.abs(v) * 50;
    bar.style.left = `${a === "thrust" ? 0 : v >= 0 ? 50 : 50 - w}%`;
    bar.style.width = `${w}%`;
    out.textContent = v.toFixed(2);
  }
  const st = $("live-status");
  if (st) st.textContent = liveStatus();
  const av = $("act-val");
  if (av && view.selected >= 0) av.textContent = fmtAct(selectedAct());
}

const selectedAct = () => liveActive() && live.flight ? live.flight[view.selected] : data.raw.neurons[view.selected].act[view.bearing];
const fmtAct = (a) => (a < 0.01 ? a.toExponential(1) : a.toFixed(3));

function regionCard() {
  const s = data.raw.stats;
  let items = ui.by === "class"
    ? CATS.map((c) => ({ name: c.short, color: c.color, n: c.classes.reduce((t, k) => t + (s.by_class[k] ?? 0), 0) }))
    : Object.entries(s.by_region).map(([k, n], i) => ({ name: REGION_NAMES[k] ?? k, color: REGION_COLORS[i % REGION_COLORS.length], n }));
  items = items.filter((i) => i.n > 0).sort((a, b) => b.n - a.n);
  const total = items.reduce((t, i) => t + i.n, 0), R = 46, C = 2 * Math.PI * R;
  let off = 0;
  const arcs = items.map((i) => {
    const len = (i.n / total) * C;
    const el = `<circle r="${R}" fill="none" stroke="${i.color}" stroke-width="20" stroke-dasharray="${Math.max(0, len - 1.2)} ${C}" stroke-dashoffset="${-off}" transform="rotate(-90)"/>`;
    off += len;
    return el;
  }).join("");
  return `
    <div class="card">
      <h2>Przegląd regionów
        <select class="mini-select" id="by" aria-label="Grupowanie">
          <option value="class"${ui.by === "class" ? " selected" : ""}>Według klasy</option>
          <option value="region"${ui.by === "region" ? " selected" : ""}>Według regionu</option>
        </select>
      </h2>
      <div class="donut">
        <svg viewBox="-60 -60 120 120" role="img" aria-label="Wykres pierścieniowy">
          <circle r="${R}" fill="none" stroke="#1b2140" stroke-width="20"/>${arcs}
          <text y="-2" text-anchor="middle" fill="#e8ebf7" font-size="13" font-weight="600" style="font-family:var(--display)">${fmt(total)}</text>
          <text y="12" text-anchor="middle" fill="#8a93b5" font-size="7.5">neuronów</text>
        </svg>
        <div class="rows">${items.map((i) =>
          `<div><i style="background:${i.color}"></i>${esc(i.name)}<b>${fmt(i.n)}</b><span>${((i.n / total) * 100).toFixed(1)}%</span></div>`).join("")}</div>
      </div>
    </div>`;
}

function inspectorCards() {
  const n = data.raw.neurons[view.selected];
  const g = GROUPS[data.groupOf[view.selected]];
  const kv = [
    ["Root ID", `<span class="mono small">${esc(n.id)}</span>`],
    ["Klasa", esc(n.super_class === "descending" ? "zstępujący (DN)" : CLASS_PL[n.super_class] ?? n.super_class)],
    ["Funkcja", esc(n.function || "—")],
    ["Strona", n.side === "left" ? "lewa" : n.side === "right" ? "prawa" : "—"],
    ["NT", n.nt ? esc(`${n.nt}${n.nt_score != null ? ` (${n.nt_score})` : ""}`) : "—"],
    ["Syn. wej.", fmt(n.syn_in)], ["Syn. wyj.", fmt(n.syn_out)],
    ["Aktywność", `<span id="act-val">${fmtAct(selectedAct())}</span>`],
  ];
  const rows = ui.dir === "in" ? n.inputs : n.outputs;
  const body = rows.length ? rows.map(([id, type, cls, cnt, reg], k) => {
    const target = data.byId.get(id);
    const link = target != null ? ` class="link" tabindex="0" data-act="pick" data-arg="${target}"` : "";
    return `<tr${link} title="${esc(id)}"><td>${k + 1}</td><td>${esc(type)}</td><td>${esc(CLASS_PL[cls] ?? (cls || "—"))}</td><td class="n">${cnt}</td><td>${esc(REGION_SHORT[reg] ?? (reg || "—"))}</td></tr>`;
  }).join("") : `<tr><td colspan="5" class="note">Brak połączeń ≥ 5 synaps.</td></tr>`;
  return `
    <div class="card">
      <h2>Wybrany neuron <button class="ghost" data-act="copy">${ui.copied ? "Skopiowano" : "Kopiuj root ID"}</button></h2>
      <div class="neuron-head"><h3>${esc(n.cell_type || "bez typu")}</h3><span class="badge" style="color:${g.color}">${g.name}</span></div>
      <div class="neuron-body">
        <dl class="kv">${kv.map(([k, v]) => `<div class="kvrow"><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
        <canvas id="thumb" width="256" height="340" aria-label="Szkielet wybranego neuronu"></canvas>
      </div>
    </div>
    <div class="card">
      <h2>Połączenia synaptyczne</h2>
      <div class="subtabs" role="group" aria-label="Kierunek">
        <button aria-pressed="${ui.dir === "in"}" data-act="dir" data-arg="in">Wejścia (${fmt(n.syn_in)} syn.)</button>
        <button aria-pressed="${ui.dir === "out"}" data-act="dir" data-arg="out">Wyjścia (${fmt(n.syn_out)} syn.)</button>
      </div>
      <div class="tbl"><table>
        <thead><tr><th>#</th><th>Partner</th><th>Klasa</th><th class="n">Synapsy</th><th>Region</th></tr></thead>
        <tbody>${body}</tbody>
      </table></div>
      <p class="note">Top 8 partnerów z edgelisty v2 (≥ 5 synaps, bez autapsów). Kliknij partnera z obwodu lotu, żeby go wybrać.</p>
    </div>`;
}

function renderRight() {
  $("right").innerHTML = regionCard() + (view.selected >= 0 ? inspectorCards() : "");
  if (view.selected >= 0) drawThumb();
}

/** Miniatura szkieletu na tle sylwetki BANC (2D). */
function drawThumb() {
  const c = $("thumb"), t = c.getContext("2d"), W = c.width, H = c.height, k = 30;
  const P = (x, y) => [W / 2 + x * k, H / 2 - y * k];
  t.fillStyle = "rgba(140,150,200,.12)";
  const S = data.somaPos;
  for (let i = 0; i < S.length; i += 36) { const [a, b] = P(S[i], S[i + 1]); t.fillRect(a, b, 1, 1); }
  t.strokeStyle = GROUPS[data.groupOf[view.selected]].color; t.lineWidth = 1.2; t.beginPath();
  for (const l of data.raw.neurons[view.selected].lines) l.forEach((p, j) => {
    const s = data.toScene(...p), [a, b] = P(s[0], s[1]);
    if (j) t.lineTo(a, b); else t.moveTo(a, b);
  });
  t.stroke();
}

// ---------- akcje ----------

const ACTIONS = {
  mode: (a) => set({ mode: a }),
  scope: (a) => set({ scope: a }),
  cat: (a) => set({ cats: view.cats.map((v, j) => (j === +a ? !v : v)) }),
  cats: () => { const on = !view.cats.every(Boolean); set({ cats: view.cats.map(() => on) }); },
  group: (a) => set({ groups: view.groups.map((v, j) => (j === +a ? !v : v)) }),
  bearing: (a) => set({ bearing: +a }),
  pick: (a) => set({ selected: +a }),
  dir: (a) => { ui.dir = a; renderRight(); },
  copy: async () => {
    try { await navigator.clipboard.writeText(data.raw.neurons[view.selected].id); } catch { return; } // brak dostępu do schowka
    ui.copied = true; renderRight();
    setTimeout(() => { ui.copied = false; renderRight(); }, 1500);
  },
  spin: () => set({ spin: !view.spin }),
  zoom: (a) => gfx?.zoom(+a),
  reset: () => gfx?.reset(),
  side: () => gfx?.side(),
};

document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-act]");
  if (el) ACTIONS[el.dataset.act](el.dataset.arg);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.matches("tr[data-act]")) ACTIONS.pick(e.target.dataset.arg);
});
document.addEventListener("change", (e) => {
  if (e.target.id === "by") { ui.by = e.target.value; renderRight(); }
});

function setupSearch() {
  const q = $("q"), box = $("results");
  const close = () => { box.hidden = true; };
  const update = () => {
    const s = q.value.trim().toLowerCase();
    if (!s || !data) return close();
    const hits = data.raw.neurons.map((n, i) => [n, i]).filter(([n]) => n.cell_type.toLowerCase().includes(s) || n.id.includes(s)).slice(0, 8);
    box.innerHTML = hits.length
      ? hits.map(([n, i]) => `<button data-hit="${i}">${esc(n.cell_type || "bez typu")} · ${n.side === "left" ? "L" : "R"}<span>${GROUPS[data.groupOf[i]].name}</span></button>`).join("")
      : `<button disabled>Brak w obwodzie lotu<span>${data.raw.neurons.length} neuronów</span></button>`;
    box.hidden = false;
  };
  q.addEventListener("input", update);
  q.addEventListener("focus", update);
  box.addEventListener("click", (e) => {
    const b = e.target.closest("[data-hit]");
    if (!b) return;
    set({ selected: +b.dataset.hit });
    q.value = ""; close();
  });
  $("search").addEventListener("focusout", (e) => { if (!$("search").contains(e.relatedTarget)) close(); });
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); q.focus(); }
    if (e.key === "Escape") close();
  });
}

// ---------- scena ----------

function createScene(host) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.autoClear = false;
  host.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
  camera.position.set(0, 0, HOME_DISTANCE);
  const controls = new OrbitControls(camera, renderer.domElement);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.12, autoRotateSpeed: 0.8, minDistance: 2, maxDistance: 40 });
  const helper = new ViewHelper(camera, renderer.domElement);
  helper.setLabels?.("L", "Y", "Z");

  const somaGeo = new THREE.BufferGeometry();
  somaGeo.setAttribute("position", new THREE.BufferAttribute(data.somaPos, 3));
  somaGeo.setAttribute("aCat", new THREE.BufferAttribute(data.somaCat, 1));
  somaGeo.setAttribute("aAct", new THREE.BufferAttribute(data.somaAct, 3));
  const somaLive = new THREE.BufferAttribute(new Uint8Array(data.somaCat.length), 1, true);
  somaLive.setUsage(THREE.DynamicDrawUsage);
  somaGeo.setAttribute("aLive", somaLive);
  const somaMat = new THREE.ShaderMaterial({
    vertexShader: somaVertex, fragmentShader: somaFragment, transparent: true, depthWrite: false,
    uniforms: {
      uCatOn: { value: CATS.map(() => 1) }, uCatColor: { value: CATS.map((c) => new THREE.Color(c.color)) },
      uBearing: { value: 1 }, uLive: { value: 0 }, uMode: { value: 0 }, uScope: { value: 0 }, uNeckY: { value: data.neckY },
      uSize: { value: 4.2 }, uPR: { value: renderer.getPixelRatio() },
    },
  });
  const somas = new THREE.Points(somaGeo, somaMat);
  somas.frustumCulled = false;

  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.BufferAttribute(data.line.pos, 3));
  lineGeo.setAttribute("aGroup", new THREE.BufferAttribute(data.line.group, 1));
  lineGeo.setAttribute("aNeuron", new THREE.BufferAttribute(data.line.neuron, 1));
  lineGeo.setAttribute("aAct", new THREE.BufferAttribute(data.line.act, 3));
  const lineNeuron = Int32Array.from(data.line.neuron);
  const lineLive = new THREE.BufferAttribute(new Float32Array(lineNeuron.length), 1);
  lineLive.setUsage(THREE.DynamicDrawUsage);
  lineGeo.setAttribute("aLive", lineLive);
  const lineMat = new THREE.ShaderMaterial({
    vertexShader: lineVertex, fragmentShader: lineFragment, transparent: true, depthWrite: false,
    uniforms: {
      uGroupOn: { value: GROUPS.map(() => 1) }, uGroupColor: { value: GROUPS.map((g) => new THREE.Color(g.color)) },
      uSel: { value: -1 }, uBearing: { value: 1 }, uLive: { value: 0 }, uMode: { value: 0 }, uScope: { value: 0 }, uNeckY: { value: data.neckY },
    },
  });
  const lines = new THREE.LineSegments(lineGeo, lineMat);
  lines.frustumCulled = false;

  const selMat = new LineMaterial({ color: 0xffffff, linewidth: 2, transparent: true, opacity: 0.95, depthTest: false });
  let selected = null;
  scene.add(somas, lines);

  // etykiety regionów: zwykłe divy przesuwane co klatkę
  const labelsHost = $("labels");
  labelsHost.innerHTML = data.labels.map((l) => `<div class="label3d"><i></i>${l.name}</div>`).join("");
  const labelEls = [...labelsHost.children];
  const v = new THREE.Vector3();
  let W = 1, H = 1;

  const resize = () => {
    W = host.clientWidth; H = host.clientHeight;
    renderer.setSize(W, H, false);
    camera.aspect = W / H; camera.updateProjectionMatrix();
    selMat.resolution.set(W, H);
  };
  new ResizeObserver(resize).observe(host);
  resize();

  const focus = (target, distance, dir = new THREE.Vector3(0, 0, 1)) => {
    controls.target.set(...target);
    camera.position.copy(controls.target).addScaledVector(dir, distance);
    controls.update();
  };
  const home = () => view.scope === "brain" ? [data.brainCenter, 8.5] : view.scope === "vnc" ? [data.vncCenter, 9] : [[0, 0, 0], HOME_DISTANCE];

  const api = {
    update() {
      const s = somaMat.uniforms, l = lineMat.uniforms;
      s.uCatOn.value = view.cats.map((c) => (c ? 1 : 0));
      l.uGroupOn.value = view.groups.map((g) => (g ? 1 : 0));
      l.uSel.value = view.selected;
      for (const u of [s, l]) {
        u.uBearing.value = view.bearing;
        u.uLive.value = liveActive() ? 1 : 0;
        u.uMode.value = MODE_ID[view.mode];
        u.uScope.value = SCOPE_ID[view.scope];
      }
      controls.autoRotate = view.spin;
    },
    /** Nowa klatka z serwera: poziomy som (uint8) idą prosto do bufora GPU, szkielety per neuron. */
    setLive(somas, flight) {
      somaLive.array.set(somas);
      somaLive.needsUpdate = true;
      const lv = Float32Array.from(flight, actLevel), a = lineLive.array;
      for (let i = 0; i < a.length; i++) a[i] = lv[lineNeuron[i]];
      lineLive.needsUpdate = true;
    },
    reset: () => focus(...home()),
    zoom(f) { camera.position.sub(controls.target).multiplyScalar(f).add(controls.target); controls.update(); },
    side() {
      const d = camera.position.distanceTo(controls.target), dir = camera.position.clone().sub(controls.target).normalize();
      focus(controls.target.toArray(), d, Math.abs(dir.x) > 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0));
    },
    select(i) {
      if (selected) { scene.remove(selected); selected.geometry.dispose(); selected = null; }
      const ls = data.raw.neurons[i]?.lines ?? [], pts = [];
      for (const l of ls) for (let j = 1; j < l.length; j++) pts.push(...data.toScene(...l[j - 1]), ...data.toScene(...l[j]));
      if (!pts.length) return;
      const g = new LineSegmentsGeometry();
      g.setPositions(pts);
      selected = new LineSegments2(g, selMat);
      selected.renderOrder = 10;
      scene.add(selected);
    },
  };

  // wybór neuronu kliknięciem (rzutowanie punktów szkieletów tylko przy kliknięciu)
  const el = renderer.domElement;
  let down = null;
  el.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  el.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) { down = null; return; }
    down = null;
    const r = el.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, P = data.pick;
    let best = -1, bd = 14 * 14;
    for (let i = 0; i < P.length; i += 4) {
      const ni = P[i + 3];
      if (!view.groups[data.groupOf[ni]]) continue;
      v.set(P[i], P[i + 1], P[i + 2]).project(camera);
      if (v.z > 1) continue;
      const sx = (v.x + 1) / 2 * r.width, sy = (1 - v.y) / 2 * r.height, d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bd) { bd = d; best = ni; }
    }
    if (best >= 0) set({ selected: best });
  });

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.clear();
    renderer.render(scene, camera);
    helper.render(renderer);
    data.labels.forEach((l, i) => {
      const inScope = view.scope === "all" || (view.scope === "brain" ? l.pos[1] > data.neckY : l.pos[1] <= data.neckY);
      v.set(...l.pos).project(camera);
      const visible = inScope && v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
      labelEls[i].style.display = visible ? "flex" : "none";
      if (visible) labelEls[i].style.transform = `translate(${((v.x + 1) / 2) * W}px, ${((1 - v.y) / 2) * H}px) translate(-6px, -50%)`;
    });
  });

  return api;
}

// ---------- start ----------

render();
setupSearch();
try {
  const [raw, cmds] = await Promise.all(["data/banc_anatomy.json", "data/cmds.json"].map((u) =>
    fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.json(); })));
  data = prepare(raw, cmds);
  $("loading").remove();
  gfx = createScene($("view"));
  // start: najaktywniejszy DN flight power przy beaconie na wprost
  let best = 0;
  data.raw.neurons.forEach((n, i) => {
    if (groupKey(n) === "dn_flight_power" && n.act[1] > data.raw.neurons[best].act[1]) best = i;
  });
  gfx.update();
  set({ selected: best });
  pollLive();
} catch (e) {
  $("loading").textContent = `Nie udało się wczytać danych: ${e}`;
}

// ---------- model na żywo ----------

async function pollLive() {
  for (;;) {
    try {
      if (!live.available) {
        live.info = await fetch("api/live").then((r) => r.json());
        const was = live.available;
        live.available = !!live.info.ready;
        if (live.available !== was) set({});
        else updateCmds();
        if (!live.available) { await new Promise((r) => setTimeout(r, 1000)); continue; }
      }
      if (!live.on) { await new Promise((r) => setTimeout(r, 200)); continue; }
      const t0 = performance.now();
      const r = await fetch(`api/frame?bearing=${live.bearing}&yaw_rate=${live.yaw}`);
      if (r.ok) {
        live.state = JSON.parse(r.headers.get("X-State"));
        const buf = await r.arrayBuffer(), n = data.somaCat.length, off = Math.ceil(n / 4) * 4;
        live.flight = new Float32Array(buf, off, data.raw.neurons.length);
        gfx.setLive(new Uint8Array(buf, 0, n), live.flight);
        updateCmds();
      }
      await new Promise((res) => setTimeout(res, Math.max(0, 33 - (performance.now() - t0)))); // ~30 klatek/s
    } catch {
      live.available = false; live.info = { status: "brak połączenia z serwerem" };
      set({});
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
}

document.addEventListener("input", (e) => {
  if (e.target.id === "live-bearing") { live.bearing = +e.target.value; $("live-bearing-out").textContent = `${signed(live.bearing, 0)}°`; }
  if (e.target.id === "live-yaw") { live.yaw = +e.target.value; $("live-yaw-out").textContent = signed(live.yaw, 1); }
});
document.addEventListener("change", (e) => {
  if (e.target.id === "live-on") { live.on = e.target.checked; set({}); }
});
