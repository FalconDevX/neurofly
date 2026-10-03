// Kopiuje dane wizualizacji z ../data/viz (wynik skryptów Pythona) do public/data.
//   python scripts/export_viz_data.py && python scripts/export_anatomy.py   (w katalogu repo)
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "..", "data", "viz");
const dst = join(here, "..", "public", "data");
const anatomy = join(src, "banc_anatomy.json");
const viz = join(src, "banc_viz.json");

if (!existsSync(anatomy) || !existsSync(viz)) {
  console.error(
    `Brak danych w ${src}.\nW katalogu repo uruchom:\n` +
      "  python scripts/download_banc.py\n  python scripts/export_viz_data.py\n  python scripts/export_anatomy.py",
  );
  process.exit(1);
}

mkdirSync(dst, { recursive: true });
copyFileSync(anatomy, join(dst, "banc_anatomy.json"));
// ciało muszki (NeuroMechFly) do półprzezroczystej nakładki — opcjonalne: python scripts/export_fly_body.py
for (const f of ["fly_body.bin", "fly_body.json"]) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(dst, f));

// komendy Planu C dla tych samych kierunków beacona co aktywność w anatomii, bez obrotu (roll_rate = 0)
const { bearings } = JSON.parse(readFileSync(anatomy, "utf8"));
const grid = JSON.parse(readFileSync(viz, "utf8")).grid;
const cmds = bearings.map((b) => grid.find((g) => g.bearing === b && g.roll_rate === 0).cmd);
writeFileSync(join(dst, "cmds.json"), JSON.stringify(cmds));
console.log(`dane → ${dst} (beacon ${bearings.join(", ")}°)`);
