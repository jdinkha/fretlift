"""``fretlift-view``: browse and play the tabs in a folder (default: ``out``).

Serves the viewer page plus the folder's tab files, on this computer only
(127.0.0.1). The page remembers the last folder you picked.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = Path(__file__).with_name("viewer") / "index.html"
TABS = {".musicxml", ".xml", ".mxl", ".gp", ".gp3", ".gp4", ".gp5", ".gpx", ".atex"}  # what alphaTab opens


class Handler(BaseHTTPRequestHandler):
    default_dir = Path("out")

    def do_GET(self):
        # Only pages from this server may read files: a site that points its own
        # domain at 127.0.0.1 (DNS rebinding) arrives with a foreign Host header.
        if self.headers.get("Host", "").rsplit(":", 1)[0] not in ("127.0.0.1", "localhost"):
            return self._send(403, b"Forbidden", "text/plain")
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/list":
            folder = Path(q.get("dir") or self.default_dir).expanduser().resolve()
            if not folder.is_dir():
                return self._json(404, {"error": f"There's no folder at {folder}"})
            tabs = [p for p in folder.iterdir() if p.suffix.lower() in TABS and p.is_file()]
            tabs.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest first
            return self._json(200, {"dir": str(folder), "default": str(self.default_dir.resolve()),
                                    "files": [p.name for p in tabs]})
        if url.path == "/api/file":
            name = q.get("name", "")
            path = Path(q.get("dir", "")).expanduser() / name
            if Path(name).name != name or path.suffix.lower() not in TABS or not path.is_file():
                return self._json(404, {"error": f"Can't open {name}"})
            return self._send(200, path.read_bytes(), "application/octet-stream")
        self._json(404, {"error": "Not found"})

    def _json(self, code: int, data: dict):
        self._send(code, json.dumps(data).encode(), "application/json")

    def _send(self, code: int, body: bytes, kind: str):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fretlift-view", description=__doc__.splitlines()[0].strip("`"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = p.parse_args(argv)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        print(f"Port {args.port} is already in use. Try another: fretlift-view --port 8001", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Viewer running at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
