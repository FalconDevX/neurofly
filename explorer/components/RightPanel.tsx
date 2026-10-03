"use client";

import { useEffect, useRef, useState } from "react";

import { CATS, GROUPS, groupKey, type Prepared } from "@/lib/data";

const fmt = (n: number) => n.toLocaleString("pl-PL");
const CLASS_PL: Record<string, string> = {
  descending: "DN", motor: "motoneuron", sensory: "sensoryczny", sensory_ascending: "sens. wstęp.", sensory_descending: "sens. zstęp.",
  central_brain_intrinsic: "interneuron", ventral_nerve_cord_intrinsic: "interneuron VNC", optic_lobe_intrinsic: "płat wzr.",
  visual_projection: "proj. wzrokowa", visual_centrifugal: "wzr. odśrodk.", ascending: "AN",
};
const REGION_SHORT: Record<string, string> = { optic_lobe: "płat wzr.", central_brain: "mózg", ventral_nerve_cord: "VNC" };
const REGION_NAMES: Record<string, string> = { optic_lobe: "Płat wzrokowy", central_brain: "Mózg centralny", ventral_nerve_cord: "VNC", "ascending/descending/inne": "Inne / szyja" };
const REGION_COLORS = ["#2dd4bf", "#a1a1aa", "#fb7185", "#52525b"];

export function RegionOverview({ data }: { data: Prepared }) {
  const [by, setBy] = useState<"class" | "region">("class");
  const s = data.raw.stats;
  let items = by === "class"
    ? CATS.map((c) => ({ name: c.short, color: c.color, n: c.classes.reduce((t, k) => t + (s.by_class[k] ?? 0), 0) }))
    : Object.entries(s.by_region).map(([k, n], i) => ({ name: REGION_NAMES[k] ?? k, color: REGION_COLORS[i % REGION_COLORS.length], n }));
  items = items.filter((i) => i.n > 0).sort((a, b) => b.n - a.n);
  const total = items.reduce((t, i) => t + i.n, 0), R = 46, C = 2 * Math.PI * R;
  let off = 0;
  return (
    <div className="card">
      <h2>Przegląd regionów
        <select className="mini-select" value={by} onChange={(e) => setBy(e.target.value as "class" | "region")} aria-label="Grupowanie">
          <option value="class">Według klasy</option>
          <option value="region">Według regionu</option>
        </select>
      </h2>
      <div className="donut">
        <svg viewBox="-60 -60 120 120" role="img" aria-label="Wykres pierścieniowy">
          <circle r={R} fill="none" stroke="#27272a" strokeWidth={20} />
          {items.map((i) => {
            const len = (i.n / total) * C, el = (
              <circle key={i.name} r={R} fill="none" stroke={i.color} strokeWidth={20}
                strokeDasharray={`${Math.max(0, len - 1.2)} ${C}`} strokeDashoffset={-off} transform="rotate(-90)" />
            );
            off += len;
            return el;
          })}
          <text y={-2} textAnchor="middle" fill="#fafafa" fontSize={13} fontWeight={600} fontFamily="var(--font-display)">{fmt(total)}</text>
          <text y={12} textAnchor="middle" fill="#a1a1aa" fontSize={7.5}>neuronów</text>
        </svg>
        <div className="rows">
          {items.map((i) => (
            <div key={i.name}><i style={{ background: i.color }} />{i.name}<b>{fmt(i.n)}</b><span>{((i.n / total) * 100).toFixed(1)}%</span></div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function Inspector({ data, index, bearing, onPick }: { data: Prepared; index: number; bearing: number; onPick: (i: number) => void }) {
  const [dir, setDir] = useState<"in" | "out">("in");
  const [copied, setCopied] = useState(false);
  const n = data.raw.neurons[index];
  const g = GROUPS.find((x) => x.key === groupKey(n))!;
  const byId = useRef<Map<string, number>>(null);
  if (!byId.current) byId.current = new Map(data.raw.neurons.map((x, i) => [x.id, i]));
  const act = n.act[bearing];
  const rows = dir === "in" ? n.inputs : n.outputs;

  const copy = async () => {
    try { await navigator.clipboard.writeText(n.id); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { /* brak dostępu do schowka */ }
  };

  return (
    <>
      <div className="card">
        <h2>Wybrany neuron <button className="ghost" onClick={copy}>{copied ? "Skopiowano" : "Kopiuj root ID"}</button></h2>
        <div className="neuron-head"><h3>{n.cell_type || "bez typu"}</h3><span className="badge" style={{ color: g.color }}>{g.name}</span></div>
        <div className="neuron-body">
          <dl className="kv">
            {([
              ["Root ID", <span key="id" className="mono small">{n.id}</span>],
              ["Klasa", n.super_class === "descending" ? "zstępujący (DN)" : CLASS_PL[n.super_class] ?? n.super_class],
              ["Funkcja", n.function || "—"],
              ["Strona", n.side === "left" ? "lewa" : n.side === "right" ? "prawa" : "—"],
              ["NT", n.nt ? `${n.nt}${n.nt_score != null ? ` (${n.nt_score})` : ""}` : "—"],
              ["Syn. wej.", fmt(n.syn_in)], ["Syn. wyj.", fmt(n.syn_out)],
              ["Aktywność", act < 0.01 ? act.toExponential(1) : act.toFixed(3)],
            ] as const).map(([k, v]) => <div key={k} className="kvrow"><dt>{k}</dt><dd>{v}</dd></div>)}
          </dl>
          <Thumb data={data} index={index} color={g.color} />
        </div>
      </div>

      <div className="card">
        <h2>Połączenia synaptyczne</h2>
        <div className="subtabs" role="group" aria-label="Kierunek">
          <button aria-pressed={dir === "in"} onClick={() => setDir("in")}>Wejścia ({fmt(n.syn_in)} syn.)</button>
          <button aria-pressed={dir === "out"} onClick={() => setDir("out")}>Wyjścia ({fmt(n.syn_out)} syn.)</button>
        </div>
        <div className="tbl">
          <table>
            <thead><tr><th>#</th><th>Partner</th><th>Klasa</th><th className="n">Synapsy</th><th>Region</th></tr></thead>
            <tbody>
              {rows.length ? rows.map(([id, type, cls, cnt, reg], k) => {
                const target = byId.current!.get(id);
                return (
                  <tr key={id} className={target != null ? "link" : ""} title={id} tabIndex={target != null ? 0 : undefined}
                    onClick={() => target != null && onPick(target)} onKeyDown={(e) => e.key === "Enter" && target != null && onPick(target)}>
                    <td>{k + 1}</td><td>{type}</td><td>{CLASS_PL[cls] ?? (cls || "—")}</td><td className="n">{cnt}</td><td>{REGION_SHORT[reg] ?? (reg || "—")}</td>
                  </tr>
                );
              }) : <tr><td colSpan={5} className="note">Brak połączeń ≥ 5 synaps.</td></tr>}
            </tbody>
          </table>
        </div>
        <p className="note">Top 8 partnerów z edgelisty v2 (≥ 5 synaps, bez autapsów). Kliknij partnera z obwodu lotu, żeby go wybrać.</p>
      </div>
    </>
  );
}

/** Miniatura szkieletu na tle sylwetki BANC (2D, rysowana raz na wybór). */
function Thumb({ data, index, color }: { data: Prepared; index: number; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const t = c.getContext("2d")!, W = c.width, H = c.height, k = 30;
    const P = (x: number, y: number) => [W / 2 + x * k, H / 2 - y * k];
    t.clearRect(0, 0, W, H);
    t.fillStyle = "rgba(140,150,200,.12)";
    const S = data.somaPos;
    for (let i = 0; i < S.length; i += 36) { const [a, b] = P(S[i], S[i + 1]); t.fillRect(a, b, 1, 1); }
    t.strokeStyle = color; t.lineWidth = 1.2; t.beginPath();
    for (const l of data.raw.neurons[index].lines) l.forEach((p, j) => {
      const s = data.toScene(...p), [a, b] = P(s[0], s[1]);
      if (j) t.lineTo(a, b); else t.moveTo(a, b);
    });
    t.stroke();
  }, [data, index, color]);
  return <canvas ref={ref} width={256} height={340} aria-label="Szkielet wybranego neuronu" />;
}
