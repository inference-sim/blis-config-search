#!/usr/bin/env python3
"""Write the engine field lists of the pinned simulator to files.

TEMPORARY until every caller asks the binary (`blis fields --scenario`): config search reads the
`fields.json` of the engines the pinned inference-sim commit carries, through --fields. Since M3.2
the vLLM and SGLang engines are directories of inference-sim (engines/vllm, engines/sglang), so both
lists are read at pins.env's BLIS_SHA, the commit the binary is built from; there is no engine pin of
its own. Each file is fetched once per job and checked to be a FieldList of its engine, so a run uses
a reviewed file and not ambient state.

The vLLM list is written to OUT, and the SGLang list beside it as OUT with `.json` replaced by
`.sglang.json`, which blis_cli.fields reads for a spec that names the sglang engine.

  python3 -I scripts/fetch_fields.py OUT
"""
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINES = (("vllm", "engines/vllm/fields.json", ""), ("sglang", "engines/sglang/fields.json", ".sglang"))


def pinned_sha():
    with open(os.path.join(ROOT, "pins.env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("BLIS_SHA="):
                sha = line.split("=", 1)[1].strip()
                if re.fullmatch(r"[0-9a-f]{40}", sha):
                    return sha
                raise SystemExit(f"pins.env BLIS_SHA={sha!r} is not a 40-hex commit")
    raise SystemExit("pins.env has no BLIS_SHA")


def fields_url(sha, path):
    return f"https://raw.githubusercontent.com/inference-sim/inference-sim/{sha}/{path}"


def fetch(sha, path, kind):
    """(raw bytes, document) of the pinned commit's `path`, checked to be `kind`'s FieldList."""
    url = fields_url(sha, path)
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (a fixed https URL)
        raw = r.read()
    doc = json.loads(raw)
    if doc.get("kind") != "FieldList" or (doc.get("engine") or {}).get("kind") != kind or not doc.get("fields"):
        raise SystemExit(f"{url} is not a {kind} FieldList")
    return raw, doc


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    out = argv[0]
    sha = pinned_sha()
    for kind, path, infix in ENGINES:
        dest = out if not infix else re.sub(r"\.json$", "", out) + infix + ".json"
        raw, doc = fetch(sha, path, kind)
        with open(dest, "wb") as f:
            f.write(raw)
        print(f"{dest}: inference-sim {sha[:12]} {path}, {kind} {doc['engine']['version']}, {len(doc['fields'])} fields")


if __name__ == "__main__":
    main(sys.argv[1:])
