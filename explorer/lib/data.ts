// Dane BANC v888 z eksportu Pythona (scripts/export_anatomy.py) → bufory gotowe do wysłania na GPU.

export type Partner = [id: string, cellType: string, superClass: string, synapses: number, region: string];

export interface NeuronRec {
  id: string;
  group: string; // np. dn_flight_power_L
  cell_type: string;
  super_class: string;
  cell_class: string;
  side: string;
  nt: string;
  nt_score: number | null;
  function: string;
  syn_in: number;
  syn_out: number;
  inputs: Partner[];
  outputs: Partner[];
  act: number[]; // aktywność z modelu dla beacona lewo / prosto / prawo
  lines: [number, number, number][][]; // polilinie szkieletu w µm
}

export interface RawData {
  bearings: number[];
  classes: string[];
  regions: string[];
  stats: {
    neurons: number;
    edges: number;
    synapses: number;
    by_class: Record<string, number>;
    by_region: Record<string, number>;
  };
  somas: { xyz: number[]; act: string[]; super_class: number[]; region: number[] };
  neurons: NeuronRec[];
  n_somas: number;
  n_total: number;
}

export interface Cmd {
  thrust: number;
  roll: number;
  pitch: number;
  yaw: number;
}

export const CATS = [
  { key: "sens", name: "Sensory", short: "Sensory", color: "#60a5fa", classes: ["sensory", "sensory_ascending", "sensory_descending"] },
  { key: "motor", name: "Motor", short: "Motor", color: "#fb7185", classes: ["motor", "visceral_circulatory", "ascending_visceral_circulatory"] },
  { key: "dnan", name: "Descending / ascending", short: "DN / AN", color: "#fbbf24", classes: ["descending", "ascending"] },
  { key: "vis", name: "Visual", short: "Visual", color: "#2dd4bf", classes: ["optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"] },
  { key: "inter", name: "Interneurons", short: "Interneurons", color: "#a1a1aa", classes: ["central_brain_intrinsic", "ventral_nerve_cord_intrinsic"] },
  { key: "unk", name: "Unannotated", short: "Unannotated", color: "#52525b", classes: ["unknown"] },
] as const;

export const GROUPS = [
  { key: "dn_flight_power", name: "DN flight power", color: "#fbbf24" },
  { key: "dn_flight_steering", name: "DN flight steering", color: "#f59e0b" },
  { key: "wing_power", name: "MN wing power", color: "#fb7185" },
  { key: "wing_steering", name: "MN wing steering", color: "#f43f5e" },
  { key: "wing_tension", name: "MN wing tension", color: "#e11d48" },
  { key: "haltere_aff", name: "Haltere afferents", color: "#60a5fa" },
] as const;

export const groupKey = (n: NeuronRec) => n.group.replace(/_[LR]$/, "");
export const GROUP_INDEX: Record<string, number> = Object.fromEntries(GROUPS.map((g, i) => [g.key, i]));

/** 1 jednostka sceny = 100 µm; oś y odwrócona (w BANC y rośnie w stronę VNC). */
export const SCALE = 0.01;

/** aktywność → poziom 0..1 w skali log 10⁻⁶..1 */
export const actLevel = (a: number) => Math.min(1, Math.max(0, (Math.log10(Math.max(a, 1e-6)) + 6) / 6));

/** Etykieta nad sceną: region (z pozycji som) albo grupa obwodu lotu (z punktów szkieletów). */
export interface Label {
  name: string;
  desc: string; // krótko: co to jest i jaką rolę ma w pętli NeuroFly
  pos: [number, number, number];
  color?: string; // kropka w kolorze grupy (regiony: biała)
  left?: boolean; // tekst na lewo od punktu (grupy lotu), żeby nie nachodził na etykiety regionów
  groups?: string[]; // klucze GROUPS — etykieta znika, gdy wszystkie te grupy są wyłączone
}

export interface Prepared {
  raw: RawData;
  cmds: Cmd[];
  center: [number, number, number]; // µm
  toScene: (x: number, y: number, z: number) => [number, number, number];
  somaPos: Float32Array;
  somaCat: Float32Array;
  somaAct: Float32Array; // n × 3 poziomów 0..1
  neckY: number; // granica mózg / VNC we współrzędnych sceny
  brainCenter: [number, number, number];
  vncCenter: [number, number, number];
  labels: Label[];
  line: { pos: Float32Array; group: Float32Array; neuron: Float32Array; act: Float32Array };
  pick: Float32Array; // x, y, z, indeks neuronu
  body: { pos: Float32Array; index: Uint32Array; part: Float32Array } | null; // ciało muszki (ilustracja), brak pliku → null
}

const ALPH = "0123456789abcdefghijklmnopqrstuvwxyz";

export async function loadData(): Promise<Prepared> {
  const [raw, cmds, body] = await Promise.all([
    fetch("data/banc_anatomy.json").then((r) => r.json() as Promise<RawData>),
    fetch("data/cmds.json").then((r) => r.json() as Promise<Cmd[]>),
    loadBody(),
  ]);
  const p = prepare(raw, cmds);
  if (body) {  // pozycje w µm BANC → scena (ten sam środek i skala co somy)
    const pos = new Float32Array(body.pos.length);
    for (let i = 0; i < body.pos.length; i += 3) pos.set(p.toScene(body.pos[i], body.pos[i + 1], body.pos[i + 2]), i);
    p.body = { pos, index: body.index, part: body.part };
  }
  return p;
}

/** Ciało muszki z scripts/export_fly_body.py: float32 pozycje (µm BANC), uint32 indeksy, uint8 części. */
async function loadBody(): Promise<{ pos: Float32Array; index: Uint32Array; part: Float32Array } | null> {
  try {
    const [meta, buf] = await Promise.all([
      fetch("data/fly_body.json").then((r) => (r.ok ? r.json() : null)),
      fetch("data/fly_body.bin").then((r) => (r.ok ? r.arrayBuffer() : null)),
    ]);
    if (!meta || !buf) return null;
    const n = meta.vertices as number, m = meta.triangles as number;
    return {
      pos: new Float32Array(buf, 0, n * 3),
      index: new Uint32Array(buf, n * 12, m * 3),
      part: Float32Array.from(new Uint8Array(buf, n * 12 + m * 12, n)),
    };
  } catch {
    return null;
  }
}

export function prepare(raw: RawData, cmds: Cmd[]): Prepared {
  const S = raw.somas.xyz, n = S.length / 3;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) { const v = S[3 * i + d]; if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v; }
  // środek bryły, nie średnia (większość som leży w płatach wzrokowych)
  const center: [number, number, number] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const toScene = (x: number, y: number, z: number): [number, number, number] =>
    [(x - center[0]) * SCALE, -(y - center[1]) * SCALE, (z - center[2]) * SCALE];

  const classToCat = raw.classes.map((c) => Math.max(0, CATS.findIndex((k) => (k.classes as readonly string[]).includes(c))));
  const somaPos = new Float32Array(n * 3), somaCat = new Float32Array(n), somaAct = new Float32Array(n * 3);
  const reg = Object.fromEntries(raw.regions.map((r, i) => [r, i]));
  const sum: Record<string, [number, number, number, number]> = {};
  let brainMax = -Infinity, vncMin = Infinity;
  for (let i = 0; i < n; i++) {
    const x = S[3 * i], y = S[3 * i + 1], z = S[3 * i + 2];
    somaPos.set(toScene(x, y, z), 3 * i);
    somaCat[i] = classToCat[raw.somas.super_class[i]];
    for (let b = 0; b < 3; b++) somaAct[3 * i + b] = ALPH.indexOf(raw.somas.act[b][i]) / 35;
    const r = raw.somas.region[i];
    if (r === reg.central_brain && y > brainMax) brainMax = y;
    if (r === reg.ventral_nerve_cord && y < vncMin) vncMin = y;
    const key = r === reg.optic_lobe ? (x < center[0] ? "optic_lobe_a" : "optic_lobe_b") : raw.regions[r];
    if (key) { const s = (sum[key] ??= [0, 0, 0, 0]); s[0] += x; s[1] += y; s[2] += z; s[3]++; }
  }
  const neckUm = (Math.min(brainMax, 420) + Math.max(vncMin, 420)) / 2;
  const centroid = (k: string): [number, number, number] => { const s = sum[k]; return toScene(s[0] / s[3], s[1] / s[3], s[2] / s[3]); };
  const brainCenter = centroid("central_brain"), vncCenter = centroid("ventral_nerve_cord");

  // szkielety → jedna geometria LineSegments (pary wierzchołków)
  let segs = 0;
  for (const nr of raw.neurons) for (const l of nr.lines) segs += l.length - 1;
  const pos = new Float32Array(segs * 6), group = new Float32Array(segs * 2), neuron = new Float32Array(segs * 2), act = new Float32Array(segs * 6);
  const pick: number[] = [];
  let k = 0;
  raw.neurons.forEach((nr, ni) => {
    const g = GROUP_INDEX[groupKey(nr)], lv = nr.act.map(actLevel);
    for (const l of nr.lines) {
      for (let j = 0; j < l.length; j++) {
        const p = toScene(...l[j]);
        if (j % 3 === 0) pick.push(p[0], p[1], p[2], ni);
        if (j === 0) continue;
        const q = toScene(...l[j - 1]);
        pos.set(q, 6 * k); pos.set(p, 6 * k + 3);
        group[2 * k] = group[2 * k + 1] = g;
        neuron[2 * k] = neuron[2 * k + 1] = ni;
        act.set(lv, 6 * k); act.set(lv, 6 * k + 3);
        k++;
      }
    }
  });

  const neckY = -(neckUm - center[1]) * SCALE;
  // kotwice grup lotu: środek punktów szkieletów (DN — tylko część w mózgu, bo aksony biegną do VNC)
  const gsum = GROUPS.map(() => [0, 0, 0, 0]);
  for (let i = 0; i < k * 2; i++) {
    const gi = group[i], y = pos[3 * i + 1];
    if (GROUPS[gi]?.key.startsWith("dn_") && y < neckY) continue;
    const s = gsum[gi];
    if (!s) continue;
    s[0] += pos[3 * i]; s[1] += y; s[2] += pos[3 * i + 2]; s[3]++;
  }
  const gcent = (keys: string[]): [number, number, number] => {
    const t = [0, 0, 0, 0];
    for (const key of keys) { const s = gsum[GROUP_INDEX[key]]; for (let d = 0; d < 4; d++) t[d] += s[d]; }
    return t[3] ? [t[0] / t[3], t[1] / t[3], t[2] / t[3]] : [0, 0, 0];
  };
  // w BANC prawa strona muchy ma mniejsze x (sprawdzone na meta v888: side=right ~48k, left ~194k voxeli)
  const [olRight, olLeft] = [centroid("optic_lobe_a"), centroid("optic_lobe_b")];
  const labels: Label[] = [
    { name: "Optic lobe R", desc: "Processes the right eye's image. The drone camera signal enters here (FlyVis → BANC); 80% of neurons are typed in v888.",
      pos: olRight, left: true },
    { name: "Optic lobe L", desc: "Left eye. Only 36% of this lobe's neurons are typed in v888, so the FlyVis map drives ~3× fewer of them.",
      pos: olLeft },
    { name: "Central brain", desc: "Integrates the senses and decides on movement; DN somas are here.",
      pos: [brainCenter[0], brainCenter[1] + 1.2, brainCenter[2]] }, // nad etykietami płatów, żeby opisy się nie nakładały
    { name: "Neck connective", desc: "DN axons run through here from the brain to the VNC — the only path for commands to the wings.",
      pos: [brainCenter[0], neckY, brainCenter[2]] },
    { name: "VNC", desc: "Ventral nerve cord (like the spinal cord): motor neurons of the wings, legs and halteres.",
      pos: [vncCenter[0], vncCenter[1] - 2.2, vncCenter[2]] },
    { name: "Flight DNs", desc: "Descending neurons, flight power / steering. The decoder reads heading (yaw) from individual DNs.",
      pos: (([x, y, z]) => [x, y - 0.6, z] as [number, number, number])(gcent(["dn_flight_power", "dn_flight_steering"])), color: GROUPS[0].color, groups: ["dn_flight_power", "dn_flight_steering"], left: true },
    { name: "Wing motor neurons", desc: "Power (DLM/DVM), steering and tension wing muscles — BANC's output to the drone.",
      pos: gcent(["wing_power", "wing_steering", "wing_tension"]), color: GROUPS[2].color, groups: ["wing_power", "wing_steering", "wing_tension"], left: true },
    { name: "Haltere afferents", desc: "The fly's rotation sensors (halteres). Here they receive the drone gyroscope signal.",
      pos: gcent(["haltere_aff"]), color: GROUPS[5].color, groups: ["haltere_aff"] },
  ];

  return {
    raw, cmds, center, toScene, somaPos, somaCat, somaAct,
    neckY,
    brainCenter, vncCenter,
    labels,
    line: { pos, group, neuron, act },
    pick: new Float32Array(pick),
    body: null,
  };
}
