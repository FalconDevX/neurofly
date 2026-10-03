// Shadery WebGL: cała selekcja (kategorie, grupy, zakres mózg/VNC, tryb, beacon) dzieje się na GPU,
// więc zmiana filtra to tylko zmiana uniformu — bez przebudowy buforów.

export const somaVertex = /* glsl */ `
  attribute float aCat;
  attribute vec3 aAct;
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
    float a = uBearing == 0 ? aAct.x : (uBearing == 1 ? aAct.y : aAct.z);
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
  }
`;

// Zwykłe mieszanie alfa z małą kryciem: gęste miejsca dążą do koloru klasy, nie do bieli.
export const somaFragment = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.5, 0.1, d));
  }
`;

export const lineVertex = /* glsl */ `
  attribute float aGroup;
  attribute float aNeuron;
  attribute vec3 aAct;
  uniform float uGroupOn[6];
  uniform vec3 uGroupColor[6];
  uniform float uSel;
  uniform int uBearing;
  uniform int uMode;
  uniform float uBoost;   // > 1: szkielety jaśniejsze (np. na tle półprzezroczystego ciała muszki)
  varying vec3 vColor;
  varying float vAlpha;
  varying float vY;

  void main() {
    int g = int(aGroup + 0.5);
    float on = uGroupOn[g] * (abs(aNeuron - uSel) < 0.5 ? 0.0 : 1.0);  // wybrany rysowany osobno
    float a = uBearing == 0 ? aAct.x : (uBearing == 1 ? aAct.y : aAct.z);
    if (uMode == 2) {
      vColor = a > 0.55 ? vec3(1.0, 0.82, 0.35) : vec3(0.36, 0.42, 0.66);
      vAlpha = (0.05 + 0.45 * a) * on;
    } else {
      vColor = uGroupColor[g];
      vAlpha = min(1.0, (uMode == 1 ? 0.5 : 0.22) * uBoost) * on;
    }
    vY = position.y;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/** Ciało muszki: szare „szkło” — krawędzie (Fresnel) jaśniejsze, środek prawie przezroczysty, żeby nerwy
 *  w środku były widoczne. aPart: 0 tułów, 1 oczy, 2 nogi, 3 skrzydła, 4 czułki / aparat gębowy. */
export const bodyVertex = /* glsl */ `
  attribute float aPart;
  varying vec3 vN;
  varying vec3 vV;
  varying float vPart;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    vPart = aPart;
    gl_Position = projectionMatrix * mv;
  }
`;

export const bodyFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;
  varying vec3 vN;
  varying vec3 vV;
  varying float vPart;
  void main() {
    float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
    float part = vPart > 2.5 && vPart < 3.5 ? 0.55 : (vPart > 0.5 && vPart < 1.5 ? 1.25 : 1.0);  // skrzydła cieniej, oczy mocniej
    float a = uOpacity * part * (0.03 + 0.55 * fres);
    gl_FragColor = vec4(uColor * (0.75 + 0.5 * fres), a);
  }
`;

export const lineFragment = /* glsl */ `
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
  }
`;
