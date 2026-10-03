# NeuroFly BANC Explorer

Eksplorator connectomu BANC v888 w Next.js z renderem na GPU (WebGL przez three.js / `@react-three/fiber`).

- ~175 tys. som jako jedna chmura punktów z własnym shaderem (jedno wywołanie rysowania),
- 863 szkielety neuronów lotu (DN flight power / steering, MN skrzydeł, aferenty halter) jako jedna geometria linii,
- filtry klas i grup, zakres mózg / VNC, tryby Eksploruj / Obwód lotu / Aktywność i wybór beacona to uniformy shaderów,
- inspektor neuronu z prawdziwymi partnerami synaptycznymi, wykres klas i regionów, komendy drona (Plan C).

## Uruchomienie

Dane powstają w Pythonie (katalog repo):

```bash
python scripts/download_banc.py
python scripts/export_viz_data.py
python scripts/export_anatomy.py      # wymaga też szkieletów SWC w data/banc_888/swc
```

Potem w `explorer/`:

```bash
npm install
npm run dev        # http://localhost:3000 (predev kopiuje dane z ../data/viz do public/data)
npm run build      # statyczny eksport do out/
```

## Struktura

- `lib/data.ts` — typy, kategorie, przygotowanie buforów GPU z eksportu Pythona.
- `lib/shaders.ts` — shadery som i szkieletów.
- `components/Scene.tsx` — scena three.js, kamera, wybór neuronu kliknięciem, etykiety regionów.
- `components/Explorer.tsx`, `Sidebar.tsx`, `RightPanel.tsx` — interfejs.

Aktywność i komendy liczy model z `banc_control` (dynamika, znaki neuroprzekaźników, wejście wzroku jako
lewa/prawa strona) — to nie są pomiary z BANC. Geometria, typy i połączenia są z oficjalnego BANC v888.
