#!/usr/bin/env python3
"""Local companion server for the config-search webpage (LOCAL mode only).

    python3 -I scripts/build_gallery.py --mode local   # build the site in local mode
    python3 -I scripts/serve_local.py                  # serve it + accept submissions

Serves the built `site/` AND exposes a tiny submission API the Submit tab calls when the
site was built with `--mode local`. This is the "run the search yourself" path from the
spec (§12): it runs ONLY on a user's own machine, against their own LLM, and is never part
of the GitHub Pages deployment (the Pages build is `--mode pages` and has no server).

Endpoints (local only):
  GET  /                      -> the static site (site/index.html, assets, reports)
  GET  /api/health            -> {"ok": true, "mode": "local"}
  POST /api/search            -> validate a submitted Scenario, write submissions/<name>.yaml,
                                 and return the command to run the config-search skill against
                                 it. (The search BRAIN is an LLM; this server does not embed an
                                 LLM — it prepares the submission and hands back the exact,
                                 reviewable command, so the user stays in control of what runs.)

Stdlib only; no third-party deps. Binds localhost by default.

NOTE on local port binding: the dev sandbox may block listening sockets (EPERM). If so, run
this outside the sandbox or with sandbox.network.allowLocalBinding enabled.
"""
import argparse
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from functools import partial

try:
    import yaml
except ImportError:
    yaml = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SITE = os.path.join(ROOT, "site")
SUBMISSIONS = os.path.join(ROOT, "submissions")

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
KNOWN_HARDWARE = {"h100", "h200", "gb200-nvl72", "gb300", "a100-80", "a100-sxm", "b200", "b300", "l40s"}
KNOWN_WORKLOADS = {"chatbot", "summarization", "contentgen", "multidoc"}


def _catalog_models():
    """Model short-names the site knows about, read from the built catalog.js (best effort)."""
    path = os.path.join(SITE, "catalog.js")
    try:
        txt = open(path, encoding="utf-8").read()
        obj = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        return {m["name"] for m in obj.get("models", [])}
    except Exception:
        return set()


def validate_scenario(body):
    """Validate a submitted Scenario dict. Returns (scenario_dict, errors[])."""
    errors = []
    model = str(body.get("model", "")).strip()
    hardware = str(body.get("hardware", "")).strip().lower()
    workload = str(body.get("workload", "")).strip().lower()
    try:
        gpu_budget = int(body.get("gpu_budget"))
    except (TypeError, ValueError):
        gpu_budget = None

    models = _catalog_models()
    if models and model not in models:
        errors.append(f"unknown model {model!r}")
    elif not model:
        errors.append("model is required")
    if hardware not in KNOWN_HARDWARE:
        errors.append(f"unknown hardware {hardware!r}")
    if workload not in KNOWN_WORKLOADS:
        errors.append(f"unknown workload {workload!r}")
    if not gpu_budget or not (1 <= gpu_budget <= 100000):
        errors.append("gpu_budget must be an integer in [1, 100000]")

    floor = body.get("interactivity_floor") or {}
    rate = floor.get("tok_per_s_per_user", 50)
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        errors.append("interactivity_floor.tok_per_s_per_user must be numeric")
        rate = 50.0

    scenario = {
        "model": model, "hardware": hardware, "gpu_budget": gpu_budget,
        "workload": {"preset": workload},
        "objective": "dual-pareto",
        "interactivity_floor": {"tok_per_s_per_user": rate},
        "headroom": 0.8, "seed": 0,
    }
    return scenario, errors


def scenario_yaml(scenario):
    if yaml:
        return yaml.safe_dump(scenario, sort_keys=False)
    return json.dumps(scenario, indent=2)  # fallback: valid YAML is a superset of JSON


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

    def do_POST(self):
        if self.path != "/api/search":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, TypeError):
            return self._send(400, {"error": "invalid JSON body"})

        scenario, errors = validate_scenario(body)
        if errors:
            return self._send(400, {"error": "validation failed", "details": errors})

        # Derive a safe submission name and write the scenario for the user to run/commit.
        name = body.get("name") or f"{scenario['model']}-{scenario['hardware']}-{scenario['workload']['preset']}"
        name = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")[:64] or "submission"
        if not NAME_RE.match(name):
            name = "submission"
        os.makedirs(SUBMISSIONS, exist_ok=True)
        out = os.path.join(SUBMISSIONS, f"{name}.yaml")
        with open(out, "w", encoding="utf-8") as f:
            f.write(scenario_yaml(scenario))

        # The search brain is an LLM driving the skill; hand back the exact command rather
        # than silently spawning anything. The user runs it in their own Claude session.
        cmd = f"/config-search --scenario {os.path.relpath(out, ROOT)}"
        return self._send(200, {
            "ok": True,
            "submission": os.path.relpath(out, ROOT),
            "scenario_yaml": scenario_yaml(scenario),
            "next": "Run the config-search skill against this scenario in your local Claude "
                    "session, then rebuild the site to see the new report.",
            "command": cmd,
        })

    def _serve_static(self):
        # Minimal, path-traversal-safe static file server rooted at site/.
        rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(SITE, rel))
        if not full.startswith(SITE + os.sep) and full != SITE:
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
        raise SystemExit("site/ not found — run: python3 -I scripts/build_gallery.py --mode local")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"config-search local server on http://{args.host}:{args.port}  (serving {os.path.relpath(SITE, ROOT)}/)")
    print("  POST /api/search accepts a Scenario; the search runs via the config-search skill in your Claude session.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
