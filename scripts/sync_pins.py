#!/usr/bin/env python3
"""Derive the pins that inference-sim's own go.mod already states, and write them to pins.env.

    python3 -I scripts/sync_pins.py --check      # CI/test mode: exit 1 on drift, write nothing
    python3 -I scripts/sync_pins.py              # rewrite the derived block in pins.env

WHY THIS EXISTS. pins.env had KERNEL_REF as a hand-typed line, and it drifted: it read v0.3.0
while inference-sim v0.15.0's go.mod requires blis-latency-kernel v0.5.0. That is a COST MODEL
version, so the stale pin misreported what priced a latency number rather than merely mislabeling
a pill. The fix is to stop typing what go.mod already knows.

WHAT IS DERIVED vs PINNED BY HAND. Only the Go dependencies of the simulator are derivable --
they are compiled into the binary, so BLIS_REF determines them:

    blis-schemas          direct require
    blis-vllm             direct require (absent before v0.14.0; the engine surface moved there)
    blis-latency-kernel   at v0.15.0 an INDIRECT require, pulled in via blis-schemas

The indirect case is why this parses the whole require set rather than the direct block: a reader
that only took direct requires would silently drop the kernel at v0.15.0 and publish a footer with
no cost-model version at all.

blis-catalog and blis-registry are NOT derivable at any BLIS version, and never will be: they are
not Go modules (no go.mod -- verified, zero hits in go.mod and go.sum alike), they are data
directories handed to the binary as arguments. One BLIS build runs against any catalog version, so
their refs are a genuinely independent axis and stay hand-pinned as CATALOG_REF / REGISTRY_REF.

WHY THE VALUES ARE COMMITTED RATHER THAN READ AT BUILD TIME. The site build (pages.yml) runs
actions/checkout on this repo alone and never clones inference-sim, so go.mod is unreadable when
the footer is generated -- the same absence that made the catalog/registry pills read "absent" in
production. So the derivation happens here, at pin-bump time, and the result is committed.
test_build.py re-derives and fails on mismatch when ../inference-sim is present, and skips when it
is not, so a contributor without the sibling is not blocked while CI still catches drift.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PINS = os.path.join(ROOT, "pins.env")

BEGIN = "# >>> derived from inference-sim's go.mod at BLIS_REF -- DO NOT EDIT BY HAND >>>"
END = "# <<< end derived block <<<"

# go.mod module path -> the pins.env key it becomes.
DERIVED = {
    "github.com/inference-sim/blis-schemas": "SCHEMAS_REF",
    "github.com/inference-sim/blis-vllm": "VLLM_REF",
    "github.com/inference-sim/blis-latency-kernel": "KERNEL_REF",
}

# `  github.com/x/y v1.2.3 // indirect` -> (module, version). The trailing comment is optional,
# so this matches direct and indirect requires alike.
_REQUIRE = re.compile(r"^\s*(\S+)\s+(v\S+?)\s*(?://.*)?$")


def read_pins(path=PINS):
    """pins.env as KEY -> value. Mirrors build_catalog.read_pins (plain KEY=VALUE lines)."""
    pins = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                pins[k.strip()] = v.strip()
    except OSError:
        return {}
    return pins


def gomod_at(ref, repo):
    """inference-sim's go.mod at `ref`, read from a git object -- never from the worktree.

    `git show <ref>:go.mod` is deliberate: the checkout may sit on any branch, and the pin is a
    statement about a TAG. Reading the working tree would publish whatever a developer left lying
    around, which is the mistake the inference-sim version pill already made once.
    """
    out = subprocess.run(["git", "-C", repo, "show", f"{ref}:go.mod"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise LookupError(f"cannot read go.mod at {ref} in {repo}: {out.stderr.strip()}")
    return out.stdout


def derive(ref, repo):
    """{pins.env key: version} for the Go deps of inference-sim at `ref`.

    Raises LookupError if the ref is unreadable. A module absent from go.mod is simply absent from
    the result -- blis-vllm is not a dependency before v0.14.0, and inventing a version for it
    would be worse than omitting the row.
    """
    found = {}
    for line in gomod_at(ref, repo).splitlines():
        m = _REQUIRE.match(line)
        if not m:
            continue
        mod, ver = m.group(1), m.group(2)
        if mod in DERIVED:
            found[DERIVED[mod]] = ver
    return found


def render(derived):
    lines = [BEGIN,
             "# Regenerate with: python3 -I scripts/sync_pins.py  (after changing BLIS_REF above)",
             "# Guarded by test_build.py::test_derived_pins_match_gomod."]
    for key in DERIVED.values():            # stable order, not dict-insertion order
        if key in derived:
            lines.append(f"{key}={derived[key]}")
    lines.append(END)
    return "\n".join(lines)


def splice(text, block):
    """Replace the derived block in pins.env, or append it when there is none yet."""
    if BEGIN in text and END in text:
        head, _, rest = text.partition(BEGIN)
        _, _, tail = rest.partition(END)
        return head + block + tail
    return text.rstrip("\n") + "\n\n" + block + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Derive go.mod-visible pins into pins.env")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if pins.env disagrees with go.mod; write nothing")
    ap.add_argument("--repo", default=os.path.join(ROOT, os.pardir, "inference-sim"),
                    help="inference-sim checkout to read go.mod from (read-only)")
    args = ap.parse_args(argv)

    pins = read_pins()
    ref = pins.get("BLIS_REF")
    if not ref:
        print("pins.env has no BLIS_REF; nothing to derive from", file=sys.stderr)
        return 2
    repo = os.path.abspath(args.repo)
    if not os.path.isdir(os.path.join(repo, ".git")):
        print(f"no inference-sim checkout at {repo}; skipping (nothing to compare against)")
        return 0
    try:
        derived = derive(ref, repo)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return 2
    if not derived:
        print(f"go.mod at {ref} names no blis Go dependencies -- refusing to write an empty block",
              file=sys.stderr)
        return 2

    drift = {k: (pins.get(k), v) for k, v in derived.items() if pins.get(k) != v}
    # A key pins.env states but go.mod no longer does is also drift: it would publish a dependency
    # the pinned binary does not have.
    for key in DERIVED.values():
        if key in pins and key not in derived:
            drift[key] = (pins[key], None)

    if args.check:
        if drift:
            print(f"pins.env disagrees with inference-sim {ref}'s go.mod:", file=sys.stderr)
            for k, (have, want) in sorted(drift.items()):
                print(f"  {k}: pins.env has {have!r}, go.mod says {want!r}", file=sys.stderr)
            print("run: python3 -I scripts/sync_pins.py", file=sys.stderr)
            return 1
        print(f"pins.env agrees with inference-sim {ref}: "
              + ", ".join(f"{k}={v}" for k, v in sorted(derived.items())))
        return 0

    with open(PINS, encoding="utf-8") as f:
        text = f.read()
    with open(PINS, "w", encoding="utf-8") as f:
        f.write(splice(text, render(derived)))
    verb = "updated" if drift else "unchanged"
    print(f"pins.env {verb} from inference-sim {ref}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(derived.items())))
    for k, (have, want) in sorted(drift.items()):
        print(f"  {k}: {have!r} -> {want!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
