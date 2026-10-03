"use client";

import { GROUPS, groupKey, type Prepared } from "@/lib/data";
import type { ViewState } from "./Scene";

const fmt = (n: number) => n.toLocaleString("pl-PL");
const AXES = ["thrust", "roll", "pitch", "yaw"] as const;
const BEARINGS = ["lewo", "prosto", "prawo"];

export function Sidebar({ data, view, set }: { data: Prepared; view: ViewState; set: (p: Partial<ViewState>) => void }) {
  const counts = GROUPS.map((g) => data.raw.neurons.filter((n) => groupKey(n) === g.key).length);
  const cmd = data.cmds[view.bearing];
  const s = data.raw.stats;
  return (
    <aside className="side">
      <div className="card">
        <div className="label">Zbiór danych</div>
        <div className="select">BANC v888 <span className="label" style={{ letterSpacing: 0 }}>publikacja</span></div>
        <div className="status"><i />Dane oficjalne · Lee Lab</div>
      </div>

      <div className="card">
        <div className="label">Neurony lotu</div>
        <div className="nav">
          {GROUPS.map((g, i) => (
            <button key={g.key} aria-pressed={view.groups[i]} onClick={() => set({ groups: view.groups.map((v, j) => (j === i ? !v : v)) })}>
              <i style={{ background: g.color }} />{g.name}<span>{counts[i]}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="label">Beacon (wejście wzrokowe)</div>
        <div className="seg" role="group" aria-label="Kierunek beacona">
          {BEARINGS.map((b, i) => (
            <button key={b} aria-pressed={view.bearing === i} onClick={() => set({ bearing: i })}>{b} {data.raw.bearings[i] !== 0 && `${data.raw.bearings[i] > 0 ? "+" : ""}${data.raw.bearings[i]}°`}</button>
          ))}
        </div>
        <div className="label">Komendy → dron (Plan C)</div>
        <div className="cmds">
          {AXES.map((a) => {
            const v = cmd[a], w = a === "thrust" ? v * 100 : Math.abs(v) * 50;
            const left = a === "thrust" ? 0 : v >= 0 ? 50 : 50 - w;
            return (
              <div className="cmd" key={a}>
                <span>{a}</span>
                <div className={`bar ${a === "thrust" ? "" : "center"}`}><i style={{ left: `${left}%`, width: `${w}%` }} /></div>
                <output>{v.toFixed(2)}</output>
              </div>
            );
          })}
        </div>
      </div>

      <div className="card">
        <h2>Przegląd zbioru</h2>
        <dl className="kv">
          {([
            ["Neurony", fmt(s.neurons)], ["Połączenia ≥ 5", fmt(s.edges)], ["Synapsy", fmt(s.synapses)],
            ["Szkielety lotu", fmt(data.raw.neurons.length)], ["Wersja", "v888"], ["Źródło", "Lee Lab / Dataverse"],
          ] as const).map(([k, v]) => <div key={k} className="kvrow"><dt>{k}</dt><dd>{v}</dd></div>)}
        </dl>
        <a className="btn" href="https://doi.org/10.7910/DVN/7WTH1N" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></svg>Pobierz dane BANC
        </a>
      </div>
    </aside>
  );
}
