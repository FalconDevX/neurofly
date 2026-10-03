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
  { key: "sens", name: "Sensoryczne", short: "Sensoryczne", color: "#4f8cff", classes: ["sensory", "sensory_ascending", "sensory_descending"] },
  { key: "motor", name: "Motoryczne", short: "Motoryczne", color: "#ff4fb0", classes: ["motor", "visceral_circulatory", "ascending_visceral_circulatory"] },
  { key: "dnan", name: "Zstępujące / wstępujące", short: "DN / AN", color: "#ffb547", classes: ["descending", "ascending"] },
  { key: "vis", name: "Wzrokowe", short: "Wzrokowe", color: "#2fd3c4", classes: ["optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"] },
  { key: "inter", name: "Interneurony", short: "Interneurony", color: "#a98bff", classes: ["central_brain_intrinsic", "ventral_nerve_cord_intrinsic"] },
  { key: "unk", name: "Nieoznaczone", short: "Nieoznaczone", color: "#6f7896", classes: ["unknown"] },
] as const;

export const GROUPS = [
  { key: "dn_flight_power", name: "DN flight power", color: "#ffb547" },
  { key: "dn_flight_steering", name: "DN flight steering", color: "#ff8a3d" },
  { key: "wing_power", name: "MN wing power", color: "#ff4fb0" },
  { key: "wing_steering", name: "MN wing steering", color: "#ff7ad9" },
  { key: "wing_tension", name: "MN wing tension", color: "#d65cff" },
  { key: "haltere_aff", name: "Aferenty halter", color: "#4f8cff" },
] as const;

export const groupKey = (n: NeuronRec) => n.group.replace(/_[LR]$/, "");
export const GROUP_INDEX: Record<string, number> = Object.fromEntries(GROUPS.map((g, i) => [g.key, i]));

/** 1 jednostka sceny = 100 µm; oś y odwrócona (w BANC y rośnie w stronę VNC). */
export const SCALE = 0.01;

/** aktywność → poziom 0..1 w skali log 10⁻⁶..1 */
export const actLevel = (a: number) => Math.min(1, Math.max(0, (Math.log10(Math.max(a, 1e-6)) + 6) / 6));

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
  labels: { name: string; pos: [number, number, number] }[];
  line: { pos: Float32Array; group: Float32Array; neuron: Float32Array; act: Float32Array };
  pick: Float32Array; // x, y, z, indeks neuronu
}

const ALPH = "0123456789abcdefghijklmnopqrstuvwxyz";

export async function loadData(): Promise<Prepared> {
  const [raw, cmds] = await Promise.all([
    fetch("data/banc_anatomy.json").then((r) => r.json() as Promise<RawData>),
    fetch("data/cmds.json").then((r) => r.json() as Promise<Cmd[]>),
  ]);
  return prepare(raw, cmds);
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
    const key = r === reg.optic_lobe ? (x < center[0] ? "optic_lobe" : "") : raw.regions[r];
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

  return {
    raw, cmds, center, toScene, somaPos, somaCat, somaAct,
    neckY: -(neckUm - center[1]) * SCALE,
    brainCenter, vncCenter,
    labels: [
      { name: "Mózg centralny", pos: brainCenter },
      { name: "Płat wzrokowy", pos: centroid("optic_lobe") },
      { name: "VNC", pos: vncCenter },
    ],
    line: { pos, group, neuron, act },
    pick: new Float32Array(pick),
  };
}
