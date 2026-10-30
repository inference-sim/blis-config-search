#!/usr/bin/env python3
"""Fetch the replay corpus a spec needs, and verify it against the catalog's digest.

A trace REPLAY workload is fully described by `blis-catalog/workloads/<name>.yaml`, which names
the Hugging Face dataset, the PINNED revision, the file and the expected `sha256` -- but the
corpus itself is deliberately not vendored (it is 568 MB, and the catalog says so outright:
"the corpus is not vendored: a consumer fetches traces.jsonl ... and checks it against sha256
before replaying it").

So the corpus is the one input a replay spec cannot carry and CI cannot clone. This script is
the missing provisioning step: given a spec, it reads the workload's descriptor, downloads the
file named there at the revision named there, and refuses anything whose bytes do not match.

Why a script rather than a few lines of `curl` in the workflow: the dataset, revision and digest
belong to the catalog pin, so hard-coding them in YAML would create a second source of truth that
could drift from `CATALOG_REF`. Reading them back out of the descriptor cannot drift. It also
makes the step testable without GitHub.

Download-then-verify-then-rename, so an interrupted transfer can never masquerade as the corpus.
A spec whose workload is a generated shape needs nothing and exits 0 without touching the network.

  python3 -I scripts/fetch_corpus.py experiments/<name>.yaml --catalog ../blis-catalog
"""
import argparse, hashlib, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import yaml

HF_URL = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"
CHUNK = 1 << 20


def _descriptor(workload, catalog):
    """The workload's replay descriptor, or None when the workload is a generated shape."""
    path = os.path.join(catalog, "workloads", workload + ".yaml")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        entry = yaml.safe_load(f) or {}
    return entry.get("replay")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest, url, want_sha, token=None):
    """Download url to dest and verify its sha256. Returns False if the corpus was already there."""
    if os.path.isfile(dest) and digest(dest) == want_sha:
        print(f"corpus already present and verified: {dest}")
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    print(f"fetching {url}")
    with urllib.request.urlopen(req) as r, open(part, "wb") as out:
        total = 0
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    got = digest(part)
    if got != want_sha:
        os.remove(part)
        raise SystemExit(
            f"corpus digest mismatch -- refusing to replay it.\n"
            f"  expected {want_sha}\n  got      {got}\n"
            f"The catalog pins the revision, so a mismatch means a truncated transfer or a moved "
            f"dataset, not a new corpus. BLIS would refuse this file too.")
    os.rename(part, dest)
    print(f"corpus verified ({total} bytes): {dest}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="experiments/<name>.yaml")
    ap.add_argument("--catalog", default=os.environ.get("BLIS_CATALOG"),
                    help="blis-catalog checkout (default: $BLIS_CATALOG)")
    ap.add_argument("--dest", help="override the destination path (default: the spec's corpus path)")
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = yaml.safe_load(f) or {}
    sc = spec.get("scenario") or {}
    workload = sc.get("workload")

    if not args.catalog:
        raise SystemExit("--catalog (or $BLIS_CATALOG) is required to read the workload descriptor")

    desc = _descriptor(workload, args.catalog)
    if not desc:
        print(f"workload {workload!r} is a generated shape -- no corpus to fetch")
        return 0

    dest = args.dest or ((sc.get("replay") or {}).get("corpus"))
    if not dest:
        raise SystemExit(
            f"workload {workload!r} is a trace REPLAY, so the spec must state where the corpus "
            f"sits on this machine:\n  scenario:\n    replay:\n      corpus: .corpus/traces.jsonl")

    corpus = desc.get("corpus") or {}
    source = corpus.get("source") or {}
    dataset, revision = source.get("huggingface"), source.get("revision")
    want_sha = corpus.get("sha256")
    if not (dataset and revision and want_sha):
        raise SystemExit(
            f"the descriptor for {workload!r} does not name a dataset, a pinned revision and a "
            f"sha256, so the corpus cannot be fetched reproducibly")

    url = HF_URL.format(dataset=dataset, revision=revision, path=source.get("file", "traces.jsonl"))
    fetch(dest, url, want_sha, token=os.environ.get("HF_TOKEN"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
