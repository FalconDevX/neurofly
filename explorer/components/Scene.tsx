"use client";

import { GizmoHelper, GizmoViewport, Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type MutableRefObject } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import { CATS, GROUPS, type Prepared } from "@/lib/data";
import { bodyFragment, bodyVertex, lineFragment, lineVertex, somaFragment, somaVertex } from "@/lib/shaders";

export type Mode = "explore" | "flight" | "activity";
export type Scope = "all" | "brain" | "vnc";

export interface ViewState {
  bearing: number;
  mode: Mode;
  scope: Scope;
  cats: boolean[];
  groups: boolean[];
  selected: number;
  spin: boolean;
  body: boolean; // półprzezroczyste ciało muszki (NeuroMechFly) wokół connectomu
  labels: boolean; // etykiety części układu nerwowego nad sceną
  descs: boolean; // krótkie opisy pod etykietami
}

export interface ViewApi {
  zoom: (factor: number) => void;
  reset: () => void;
  side: () => void;
}

const MODE_ID: Record<Mode, number> = { explore: 0, flight: 1, activity: 2 };
const SCOPE_ID: Record<Scope, number> = { all: 0, brain: 1, vnc: 2 };
const HOME_DISTANCE = 15.5;

function Somas({ data, view }: { data: Prepared; view: ViewState }) {
  const { gl } = useThree();
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data.somaPos, 3));
    g.setAttribute("aCat", new THREE.BufferAttribute(data.somaCat, 1));
    g.setAttribute("aAct", new THREE.BufferAttribute(data.somaAct, 3));
    g.computeBoundingSphere();
    return g;
  }, [data]);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: somaVertex,
        fragmentShader: somaFragment,
        transparent: true,
        depthWrite: false,
        uniforms: {
          uCatOn: { value: CATS.map(() => 1) },
          uCatColor: { value: CATS.map((c) => new THREE.Color(c.color)) },
          uBearing: { value: 1 },
          uMode: { value: 0 },
          uScope: { value: 0 },
          uNeckY: { value: data.neckY },
          uSize: { value: 4.2 },
          uPR: { value: 1 },
        },
      }),
    [data],
  );
  const u = material.uniforms;
  u.uCatOn.value = view.cats.map((c) => (c ? 1 : 0));
  u.uBearing.value = view.bearing;
  u.uMode.value = MODE_ID[view.mode];
  u.uScope.value = SCOPE_ID[view.scope];
  u.uPR.value = gl.getPixelRatio();
  return <points geometry={geometry} material={material} frustumCulled={false} />;
}

function Skeletons({ data, view }: { data: Prepared; view: ViewState }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data.line.pos, 3));
    g.setAttribute("aGroup", new THREE.BufferAttribute(data.line.group, 1));
    g.setAttribute("aNeuron", new THREE.BufferAttribute(data.line.neuron, 1));
    g.setAttribute("aAct", new THREE.BufferAttribute(data.line.act, 3));
    return g;
  }, [data]);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: lineVertex,
        fragmentShader: lineFragment,
        transparent: true,
        depthWrite: false,
        uniforms: {
          uGroupOn: { value: GROUPS.map(() => 1) },
          uGroupColor: { value: GROUPS.map((g) => new THREE.Color(g.color)) },
          uSel: { value: -1 },
          uBoost: { value: 1 },
          uBearing: { value: 1 },
          uMode: { value: 0 },
          uScope: { value: 0 },
          uNeckY: { value: data.neckY },
        },
      }),
    [data],
  );
  const u = material.uniforms;
  u.uGroupOn.value = view.groups.map((g) => (g ? 1 : 0));
  u.uSel.value = view.selected;
  u.uBoost.value = view.body && data.body ? 1.8 : 1;
  u.uBearing.value = view.bearing;
  u.uMode.value = MODE_ID[view.mode];
  u.uScope.value = SCOPE_ID[view.scope];
  return <lineSegments geometry={geometry} material={material} frustumCulled={false} />;
}

/** Ciało muszki — ilustracja (NeuroMechFly dopasowane do BANC: głowa → mózg, tułów → VNC), nie dane BANC. */
function FlyBody({ data, view }: { data: Prepared; view: ViewState }) {
  const geometry = useMemo(() => {
    if (!data.body) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data.body.pos, 3));
    g.setAttribute("aPart", new THREE.BufferAttribute(data.body.part, 1));
    g.setIndex(new THREE.BufferAttribute(data.body.index, 1));
    g.computeVertexNormals();
    return g;
  }, [data]);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: bodyVertex,
        fragmentShader: bodyFragment,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        uniforms: { uColor: { value: new THREE.Color("#a1a1aa") }, uOpacity: { value: 0.9 } },
      }),
    [],
  );
  if (!geometry || !view.body) return null;
  return <mesh geometry={geometry} material={material} renderOrder={-1} frustumCulled={false} />;
}

function SelectedNeuron({ data, index }: { data: Prepared; index: number }) {
  const segments = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (const l of data.raw.neurons[index]?.lines ?? [])
      for (let j = 1; j < l.length; j++) pts.push(new THREE.Vector3(...data.toScene(...l[j - 1])), new THREE.Vector3(...data.toScene(...l[j])));
    return pts;
  }, [data, index]);
  if (!segments.length) return null;
  return <Line points={segments} segments color="#ffffff" lineWidth={2} transparent opacity={0.95} depthTest={false} />;
}

/** Rzutuje kotwice etykiet co klatkę i przesuwa zwykłe divy nakładki (bez przebudowy Reacta). */
function LabelProjector({ data, view, refs }: { data: Prepared; view: ViewState; refs: MutableRefObject<(HTMLDivElement | null)[]> }) {
  const { camera, size } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    const scope = view.scope;
    data.labels.forEach((l, i) => {
      const el = refs.current[i];
      if (!el) return;
      const groupOn = !l.groups || l.groups.some((g) => view.groups[GROUPS.findIndex((x) => x.key === g)]);
      const inScope = view.labels && groupOn &&
        (scope === "all" || (scope === "brain" ? l.pos[1] > data.neckY : l.pos[1] <= data.neckY));
      v.set(...l.pos).project(camera);
      const visible = inScope && v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
      el.style.display = visible ? "flex" : "none";
      // kropka zawsze w punkcie kotwicy; tekst w prawo albo (l.left) w lewo od niej
      if (visible) el.style.transform = `translate(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px) ` +
        (l.left ? "translate(calc(-100% + 6px), -9px)" : "translate(-6px, -9px)");
    });
  });
  return null;
}

/** Kamera, sterowanie i wybór neuronu kliknięciem (rzutowanie punktów szkieletów tylko przy kliknięciu). */
function Rig({ data, view, api, onPick }: { data: Prepared; view: ViewState; api: MutableRefObject<ViewApi | null>; onPick: (i: number) => void }) {
  const { camera, gl } = useThree();
  const controls = useRef<OrbitControlsImpl>(null);

  const focus = (target: [number, number, number], distance: number, dir = new THREE.Vector3(0, 0, 1)) => {
    const c = controls.current;
    if (!c) return;
    c.target.set(...target);
    camera.position.copy(c.target).addScaledVector(dir, distance);
    c.update();
  };

  useEffect(() => {
    if (view.scope === "brain") focus(data.brainCenter, 8.5);
    else if (view.scope === "vnc") focus(data.vncCenter, 9);
    else focus([0, 0, 0], HOME_DISTANCE);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view.scope, data]);

  useEffect(() => {
    api.current = {
      zoom: (f) => {
        const c = controls.current;
        if (!c) return;
        camera.position.sub(c.target).multiplyScalar(f).add(c.target);
        c.update();
      },
      reset: () => focus(view.scope === "brain" ? data.brainCenter : view.scope === "vnc" ? data.vncCenter : [0, 0, 0],
        view.scope === "all" ? HOME_DISTANCE : 9),
      side: () => {
        const c = controls.current;
        if (!c) return;
        const d = camera.position.distanceTo(c.target), dir = camera.position.clone().sub(c.target).normalize();
        focus([c.target.x, c.target.y, c.target.z], d, Math.abs(dir.x) > 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0));
      },
    };
  });

  useEffect(() => {
    const el = gl.domElement;
    let down: [number, number] | null = null;
    const onDown = (e: PointerEvent) => { down = [e.clientX, e.clientY]; };
    const onUp = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) { down = null; return; }
      down = null;
      const r = el.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const v = new THREE.Vector3(), P = data.pick;
      let best = -1, bd = 14 * 14;
      for (let i = 0; i < P.length; i += 4) {
        const ni = P[i + 3];
        if (!view.groups[groupOfIndex(data, ni)]) continue;
        v.set(P[i], P[i + 1], P[i + 2]).project(camera);
        if (v.z > 1) continue;
        const sx = (v.x + 1) / 2 * r.width, sy = (1 - v.y) / 2 * r.height, d = (sx - mx) ** 2 + (sy - my) ** 2;
        if (d < bd) { bd = d; best = ni; }
      }
      if (best >= 0) onPick(best);
    };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointerup", onUp);
    return () => { el.removeEventListener("pointerdown", onDown); el.removeEventListener("pointerup", onUp); };
  }, [gl, camera, data, view.groups, onPick]);

  return <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.12} autoRotate={view.spin} autoRotateSpeed={0.8} minDistance={2} maxDistance={40} />;
}

const groupCache = new WeakMap<Prepared, number[]>();
function groupOfIndex(data: Prepared, ni: number) {
  let g = groupCache.get(data);
  if (!g) {
    g = data.raw.neurons.map((n) => GROUPS.findIndex((x) => n.group.startsWith(x.key)));
    groupCache.set(data, g);
  }
  return g[ni];
}

export default function Scene({ data, view, api, onPick }: { data: Prepared; view: ViewState; api: MutableRefObject<ViewApi | null>; onPick: (i: number) => void }) {
  const labelRefs = useRef<(HTMLDivElement | null)[]>([]);
  return (
    <>
    <Canvas
      camera={{ position: [0, 0, HOME_DISTANCE], fov: 40, near: 0.1, far: 200 }}
      dpr={[1, 2]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      style={{ background: "transparent" }}
    >
      <FlyBody data={data} view={view} />
      <Somas data={data} view={view} />
      <Skeletons data={data} view={view} />
      <SelectedNeuron data={data} index={view.selected} />
      <LabelProjector data={data} view={view} refs={labelRefs} />
      <Rig data={data} view={view} api={api} onPick={onPick} />
      <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
        <GizmoViewport axisColors={["#fb7185", "#60a5fa", "#4ade80"]} labels={["L", "Y", "Z"]} labelColor="#09090b" />
      </GizmoHelper>
    </Canvas>
    <div className="labels" aria-hidden="true">
      {data.labels.map((l, i) => (
        <div key={`${l.name}-${i}`} className={`label3d${l.left ? " left" : ""}`} ref={(el) => { labelRefs.current[i] = el; }}>
          <i style={l.color ? { background: l.color } : undefined} />
          <span>{l.name}{view.descs && <small>{l.desc}</small>}</span>
        </div>
      ))}
    </div>
    </>
  );
}
