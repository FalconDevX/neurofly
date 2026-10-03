"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CATS, GROUPS, groupKey, loadData, type Prepared } from "@/lib/data";
import { Inspector, RegionOverview } from "./RightPanel";
import type { Mode, Scope, ViewApi, ViewState } from "./Scene";
import { Sidebar } from "./Sidebar";

const Scene = dynamic(() => import("./Scene"), { ssr: false });

const MODES: { id: Mode; name: string; note: string }[] = [
  { id: "explore", name: "Explore", note: "Somas colored by class (super_class), flight-neuron skeletons on top." },
  { id: "flight", name: "Flight circuit", note: "Dimmed background, flight circuit highlighted: DN → VNC → wing motor neurons, halteres." },
  { id: "activity", name: "Activity", note: "Brightness = model activity for the selected beacon (log scale 10⁻⁶–1)." },
];
const SCOPES: { id: Scope; name: string }[] = [
  { id: "all", name: "Whole BANC" },
  { id: "brain", name: "Brain only" },
  { id: "vnc", name: "VNC only" },
];

export default function Explorer() {
  const [data, setData] = useState<Prepared | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ViewState>({
    bearing: 1, mode: "explore", scope: "all",
    cats: CATS.map(() => true), groups: GROUPS.map(() => true), selected: -1, spin: false,
    labels: true, descs: true, body: true,
  });
  const api = useRef<ViewApi | null>(null);
  const set = useCallback((patch: Partial<ViewState>) => setView((v) => ({ ...v, ...patch })), []);

  // etykiety / opisy: zapamiętane w przeglądarce (wygoda widza; bez dostępu do storage — domyślnie włączone)
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("neurofly.labels") ?? "null");
      if (saved) set({ labels: !!saved.labels, descs: !!saved.descs, body: saved.body ?? true });
    } catch { /* prywatne okno / zablokowany storage */ }
  }, [set]);
  useEffect(() => {
    try { localStorage.setItem("neurofly.labels", JSON.stringify({ labels: view.labels, descs: view.descs, body: view.body })); } catch { /* j.w. */ }
  }, [view.labels, view.descs, view.body]);

  useEffect(() => {
    loadData()
      .then((d) => {
        setData(d);
        // start: najaktywniejszy DN flight power przy beaconie na wprost
        let best = 0;
        d.raw.neurons.forEach((n, i) => {
          if (groupKey(n) === "dn_flight_power" && n.act[1] > d.raw.neurons[best].act[1]) best = i;
        });
        set({ selected: best });
      })
      .catch((e) => setError(String(e)));
  }, [set]);

  const pick = useCallback((i: number) => set({ selected: i }), [set]);
  const mode = MODES.find((m) => m.id === view.mode)!;

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="logo-fly.png" srcSet="logo-fly.png 1x, logo-fly@2x.png 2x" alt="NeuroFly logo" width={54} height={32} />
          <div><b>Neuro<span>Fly</span></b><small>BANC v888 · connectome explorer · GPU render</small></div>
        </div>
        <nav className="tabs" role="tablist" aria-label="View mode">
          {MODES.map((m) => (
            <button key={m.id} className="tab" role="tab" aria-selected={view.mode === m.id} onClick={() => set({ mode: m.id })}>{m.name}</button>
          ))}
        </nav>
        {data && <Search data={data} onPick={pick} />}
      </header>

      <div className="grid">
        {data ? <Sidebar data={data} view={view} set={set} /> : <aside className="side" />}

        <main className="stage">
          <div className="stage-top">
            <div className="pills" role="group" aria-label="Scope">
              {SCOPES.map((s) => (
                <button key={s.id} aria-pressed={view.scope === s.id} onClick={() => set({ scope: s.id })}>{s.name}</button>
              ))}
            </div>
            <span className="note">{mode.note}</span>
          </div>
          <div className="view">
            {data ? <Scene data={data} view={view} api={api} onPick={pick} /> : (
              <div className="loading">{error ? `Could not load data: ${error}` : "Loading BANC v888…"}</div>
            )}
            <div className="tools">
              <Tool title="Center view" onClick={() => api.current?.reset()} d="M12 5a7 7 0 1 0 0 14a7 7 0 1 0 0-14M12 10a2 2 0 1 0 0 4a2 2 0 1 0 0-4M12 2v3M12 19v3M2 12h3M19 12h3" />
              <Tool title="Zoom in" onClick={() => api.current?.zoom(0.8)} d="M12 5v14M5 12h14" />
              <Tool title="Zoom out" onClick={() => api.current?.zoom(1.25)} d="M5 12h14" />
              <Tool title="Side / front view" onClick={() => api.current?.side()} d="M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5" />
              <Tool title="Rotate" pressed={view.spin} onClick={() => set({ spin: !view.spin })} d="M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5" />
              {data?.body && (
                <Tool title={view.body ? "Hide fly body" : "Show fly body"} pressed={view.body} onClick={() => set({ body: !view.body })}
                  d="M12 4a2 2 0 1 0 0 4a2 2 0 1 0 0-4M12 8v12M12 10c-3-3-8-3-9 1 1 4 6 4 9 1M12 10c3-3 8-3 9 1-1 4-6 4-9 1M10 20h4" />
              )}
              <Tool title={view.labels ? "Hide labels" : "Show labels"} pressed={view.labels} onClick={() => set({ labels: !view.labels })}
                d="M4 7V4h16v3M9 20h6M12 4v16" />
              <Tool title={view.descs ? "Hide label descriptions" : "Show label descriptions"} pressed={view.labels && view.descs}
                onClick={() => set({ descs: !view.descs, labels: true })} d="M4 6h16M4 10h16M4 14h10M4 18h7" />
            </div>
          </div>
          <div className="legend">
            <button className="all" aria-pressed={view.cats.every(Boolean)} onClick={() => { const on = !view.cats.every(Boolean); set({ cats: view.cats.map(() => on) }); }}>
              <i />All
            </button>
            {CATS.map((c, i) => (
              <button key={c.key} aria-pressed={view.cats[i]} onClick={() => set({ cats: view.cats.map((v, j) => (j === i ? !v : v)) })}>
                <i style={{ background: c.color }} />{c.name}
              </button>
            ))}
          </div>
        </main>

        <section className="right">
          {data && (
            <>
              <RegionOverview data={data} />
              {view.selected >= 0 && <Inspector data={data} index={view.selected} bearing={view.bearing} onPick={pick} />}
            </>
          )}
        </section>
      </div>

      <p className="note foot">
        Source: BANC v888 (<a href="https://doi.org/10.7910/DVN/7WTH1N">Harvard Dataverse</a>). Somas from the <code>position</code> column,
        skeletons from the official SWC files (twigs &lt; 20 µm removed), partners and synapse counts from <code>banc_888_edgelist_simple_v2</code>.
        Activity and commands come from our model (dynamics, neurotransmitter signs, visual input as left/right side), so they are not BANC measurements.
        Render: WebGL (three.js) — all somas in a single draw call, filters in shaders.
      </p>
    </div>
  );
}

function Tool({ title, d, onClick, pressed }: { title: string; d: string; onClick: () => void; pressed?: boolean }) {
  return (
    <button title={title} aria-label={title} aria-pressed={pressed} onClick={onClick}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg>
    </button>
  );
}

function Search({ data, onPick }: { data: Prepared; onPick: (i: number) => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); input.current?.focus(); }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const hits = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return [];
    return data.raw.neurons.map((n, i) => [n, i] as const).filter(([n]) => n.cell_type.toLowerCase().includes(s) || n.id.includes(s)).slice(0, 8);
  }, [q, data]);
  return (
    <div className="search" onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setOpen(false); }}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="7" /><path d="M20 20l-4-4" /></svg>
      <input ref={input} type="search" placeholder="Search flight neurons: type or root ID…" aria-label="Search neuron"
        value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }} onFocus={() => setOpen(true)} />
      <kbd>Ctrl K</kbd>
      {open && q.trim() && (
        <div className="results">
          {hits.length ? hits.map(([n, i]) => (
            <button key={n.id} onClick={() => { onPick(i); setOpen(false); setQ(""); }}>
              {n.cell_type || "untyped"} · {n.side === "left" ? "L" : "R"}<span>{GROUPS.find((g) => g.key === groupKey(n))?.name}</span>
            </button>
          )) : <button disabled>Not in the flight circuit<span>{data.raw.neurons.length} neurons</span></button>}
        </div>
      )}
    </div>
  );
}
