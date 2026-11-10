#!/usr/bin/env python3
"""Write the engine's field list for the pinned blis-vllm release to a file.

TEMPORARY until the M2 substrate release lets the simulator print its engine's field list
(`blis fields`): the pinned simulator has no such command, so config search reads the
`fields.json` of the blis-vllm release pins.env names (VLLM_REF), through --fields. The file is
fetched once per job and checked to be a vLLM field list, so a run uses a reviewed file and not
ambient state.

  python3 -I scripts/fetch_fields.py OUT
"""
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pinned_release():
    with open(os.path.join(ROOT, "pins.env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("VLLM_REF="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("pins.env has no VLLM_REF")


def fields_ref(ref):
    """The git ref to fetch fields.json at. A tag is itself; a Go pseudo-version (`v0.32.1-0.20261111132033-
    96ea750e3a62`, what go.mod says of an unreleased commit) names its commit by the last twelve hex digits,
    which raw.githubusercontent.com resolves."""
    m = re.fullmatch(r"v\d+\.\d+\.\d+-(?:[0-9A-Za-z.]+\.)?\d{14}-([0-9a-f]{12})", ref)
    return m.group(1) if m else ref


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    ref = pinned_release()
    url = f"https://raw.githubusercontent.com/inference-sim/blis-vllm/{fields_ref(ref)}/fields.json"
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (a fixed https URL)
        raw = r.read()
    doc = json.loads(raw)
    if doc.get("kind") != "FieldList" or (doc.get("engine") or {}).get("kind") != "vllm" or not doc.get("fields"):
        raise SystemExit(f"{url} is not a vLLM FieldList")
    with open(argv[0], "wb") as f:
        f.write(raw)
    print(f"{argv[0]}: blis-vllm {ref}, vLLM {doc['engine']['version']}, {len(doc['fields'])} fields")


if __name__ == "__main__":
    main(sys.argv[1:])
