// Build pod Artifact na claude.ai: względne ścieżki zasobów, katalog next/ zamiast _next/
// (serwis nie przyjmuje ścieżek zaczynających się od "_"), bez polyfilla noModule (ma znak U+FFFD).
//   npm run build:artifact   → out-artifact/ (index.html + pliki do opublikowania)
import { execSync } from "node:child_process";
import { cpSync, existsSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "out");
const dst = join(root, "out-artifact");

execSync("npm run build", { cwd: root, stdio: "inherit", env: { ...process.env, ASSET_PREFIX: "." } });

rmSync(dst, { recursive: true, force: true });
cpSync(out, dst, { recursive: true });
for (const f of readdirSync(dst)) {
  if (f.endsWith(".txt") || f === "404.html" || f.startsWith("_not-found")) rmSync(join(dst, f), { recursive: true });
}
renameSync(join(dst, "_next"), join(dst, "next"));

const walk = (d) => readdirSync(d).flatMap((f) => (statSync(join(d, f)).isDirectory() ? walk(join(d, f)) : [join(d, f)]));
for (const f of walk(dst).filter((f) => /\.(html|js|css)$/.test(f))) {
  const s = readFileSync(f, "utf8");
  const t = s.replaceAll("/_next/", "/next/").replaceAll('"_next/', '"next/');
  if (t !== s) writeFileSync(f, t);
}

const index = join(dst, "index.html");
const html = readFileSync(index, "utf8");
const nomodule = /<script src="\.\/next\/static\/chunks\/([^"]+)" noModule=""><\/script>/;
const m = html.match(nomodule);
if (m) {
  writeFileSync(index, html.replace(nomodule, ""));
  const polyfill = join(dst, "next", "static", "chunks", m[1]);
  if (existsSync(polyfill)) rmSync(polyfill);
}
console.log(`artifact → ${dst}`);
