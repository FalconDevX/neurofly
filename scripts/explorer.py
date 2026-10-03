"""Eksplorator BANC v888 w przeglądarce, serwowany z Pythona (bez npm i builda).

    python scripts/download_banc.py
    python scripts/export_viz_data.py
    python scripts/export_anatomy.py      # wymaga szkieletów SWC w data/banc_888/swc
    python scripts/explorer.py            # http://localhost:8000

Strona to `explorer/` (HTML + JS, three.js z CDN). Dane idą prosto z `data/viz/`:
`/data/banc_anatomy.json` bez zmian, `/data/cmds.json` liczone tu z siatki w `banc_viz.json`
(komendy Planu C dla kierunków beacona z anatomii, bez obrotu: roll_rate = 0).
"""

from __future__ import annotations

import argparse
import gzip
import json
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "explorer"
VIZ = ROOT / "data" / "viz"


def load_data(viz: Path) -> dict[str, bytes]:
    anatomy, grid_file = viz / "banc_anatomy.json", viz / "banc_viz.json"
    if not anatomy.exists() or not grid_file.exists():
        raise SystemExit(
            f"Brak danych w {viz}. W katalogu repo uruchom:\n"
            "  python scripts/download_banc.py\n  python scripts/export_viz_data.py\n  python scripts/export_anatomy.py"
        )
    raw = anatomy.read_bytes()
    bearings = json.loads(raw)["bearings"]
    grid = json.loads(grid_file.read_text(encoding="utf-8"))["grid"]
    cmds = [next(g["cmd"] for g in grid if g["bearing"] == b and g["roll_rate"] == 0) for b in bearings]
    print(f"dane z {viz} (beacon {', '.join(map(str, bearings))}°)")
    return {"/data/banc_anatomy.json": raw, "/data/cmds.json": json.dumps(cmds).encode()}


class Handler(SimpleHTTPRequestHandler):
    data: dict[str, bytes] = {}

    def do_GET(self) -> None:
        body = self.data.get(self.path.split("?", 1)[0])
        if body is None:
            return super().do_GET()
        gz = "gzip" in self.headers.get("Accept-Encoding", "")
        if gz:
            body = gzip.compress(body, compresslevel=5)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # bez logu każdego pliku
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--viz", type=Path, default=VIZ, help="katalog z banc_anatomy.json i banc_viz.json")
    p.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    a = p.parse_args()

    Handler.data = load_data(a.viz)
    srv = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(STATIC)))
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}/"
    print(f"eksplorator: {url}  (Ctrl+C kończy)")
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
