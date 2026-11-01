#!/usr/bin/env python3
"""Local static server for the config-search webpage (LOCAL mode only).

    cd web && SITE_MODE=local npm run build            # build the local-mode SPA (web/dist)
    python3 -I scripts/serve_local.py                  # serve web/dist

Serves the built SPA in `web/dist/`. In local mode the Submit tab shows the
`run_experiment.py` command to run yourself; nothing here runs BLIS or an LLM.

Endpoints (local only):
  GET  /                      -> the static site (web/dist/index.html, assets, reports)
  GET  /api/health            -> {"ok": true, "mode": "local"}

Stdlib only; no third-party deps. Binds localhost by default.

NOTE on local port binding: the dev sandbox may block listening sockets (EPERM). If so, run
this outside the sandbox or with sandbox.network.allowLocalBinding enabled.
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SITE = os.environ.get("CS_SITE_DIR") or os.path.join(ROOT, "web", "dist")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj=None, ctype="application/json"):
        body = (json.dumps(obj) if obj is not None else "").encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, {"ok": True, "mode": "local"})
        return self._serve_static()

    def _serve_static(self):
        # Minimal, path-traversal-safe static file server rooted at site/.
        rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        # realpath resolves symlinks too, so a symlink inside site/ can't escape the root.
        site_root = os.path.realpath(SITE)
        full = os.path.realpath(os.path.join(site_root, rel))
        if not (full == site_root or full.startswith(site_root + os.sep)):
            return self._send(403, {"error": "forbidden"})
        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        if not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = {
            ".html": "text/html", ".js": "application/javascript", ".css": "text/css",
            ".json": "application/json", ".svg": "image/svg+xml",
        }.get(os.path.splitext(full)[1], "application/octet-stream")
        data = open(full, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") or ctype.endswith(("javascript", "json")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass  # quiet


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local companion server for the config-search site.")
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    if not os.path.isdir(SITE):
        raise SystemExit(f"{SITE} not found — build the local-mode SPA first:\n"
                         "  cd web && SITE_MODE=local npm run build")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"config-search local server on http://{args.host}:{args.port}  (serving {os.path.relpath(SITE, ROOT)}/)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
