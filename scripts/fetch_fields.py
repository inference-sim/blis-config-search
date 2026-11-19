#!/usr/bin/env python3
"""Write the engine field lists for the pinned blis-vllm and blis-sglang releases to files.

TEMPORARY until the M2 substrate release lets the simulator print its engine's field list
(`blis fields`): the pinned simulator has no such command, so config search reads the
`fields.json` of the blis-vllm release pins.env names (VLLM_REF), through --fields. The file is
fetched once per job and checked to be a vLLM field list, so a run uses a reviewed file and not
ambient state.

The vLLM list is written to OUT. When pins.env names a blis-sglang release (SGLANG_REF), its list is
written beside it as OUT with `.json` replaced by `.sglang.json`, which blis_cli.fields reads for a
spec that names the sglang engine.

  python3 -I scripts/fetch_fields.py OUT
"""
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pinned_release(key="VLLM_REF", required=True):
    with open(os.path.join(ROOT, "pins.env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    if required:
        raise SystemExit(f"pins.env has no {key}")
    return None


def fields_ref(ref):
    """The git ref to fetch fields.json at. A tag is itself; a Go pseudo-version (`v0.32.1-0.20261111132033-
    96ea750e3a62`, what go.mod says of an unreleased commit) names its commit by the last twelve hex digits,
    which raw.githubusercontent.com resolves."""
    m = re.fullmatch(r"v\d+\.\d+\.\d+-(?:[0-9A-Za-z.]+\.)?\d{14}-([0-9a-f]{12})", ref)
    return m.group(1) if m else ref


def fetch(repo, ref, kind):
    """(raw bytes, document) of `repo`'s fields.json at `ref`, checked to be `kind`'s FieldList."""
    url = f"https://raw.githubusercontent.com/inference-sim/{repo}/{fields_ref(ref)}/fields.json"
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
    jobs = [("blis-vllm", pinned_release(), "vllm", out)]
    sglang = pinned_release("SGLANG_REF", required=False)
    if sglang:
        jobs.append(("blis-sglang", sglang, "sglang", re.sub(r"\.json$", "", out) + ".sglang.json"))
    for repo, ref, kind, path in jobs:
        raw, doc = fetch(repo, ref, kind)
        with open(path, "wb") as f:
            f.write(raw)
        print(f"{path}: {repo} {ref}, {kind} {doc['engine']['version']}, {len(doc['fields'])} fields")


if __name__ == "__main__":
    main(sys.argv[1:])
