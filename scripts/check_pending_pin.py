#!/usr/bin/env python3
"""Fail when a spec waiting for the pin can run on it: move it to experiments/ (CFG-6).

A spec in experiments/pending-pin/ names an engine kind the pinned simulator does not register
yet, so the validator and the dry run cannot read its field list. This check states that pending
condition: the spec must parse, name an engine kind, and the pinned build must not register that
kind. The day it does, this fails and the spec joins experiments/, where every check runs it.

  check_pending_pin.py --blis BLIS [DIR]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_cli  # noqa: E402
import settings  # noqa: E402
import validate_experiment as validator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def problems(blis, directory):
    registered = blis_cli.registered_kinds(blis)
    out = []
    for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        if not name.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(directory, name)
        spec, err = validator.load(path)
        if err:
            out.append(f"{name}: {err}")
            continue
        kind = settings.spec_engine_kind(spec)
        if kind not in settings.ENGINE_KINDS:
            out.append(f"{name}: scenario.engine_kind {kind!r} is none of {', '.join(settings.ENGINE_KINDS)}")
        elif kind == "vllm":
            out.append(f"{name}: a vllm spec is never pending on the pin; it belongs in experiments/")
        elif kind in registered:
            out.append(f"{name}: the pinned build registers the {kind} engine ({', '.join(registered)}), "
                       f"so this spec is no longer pending: move it from pending-pin/ to experiments/")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", default=os.path.join(ROOT, "experiments", "pending-pin"))
    ap.add_argument("--blis", default=os.environ.get("BLIS"))
    args = ap.parse_args(argv)
    try:
        found = problems(args.blis, args.directory)
    except blis_cli.BlisCommandError as e:
        print(f"check_pending_pin: {e}", file=sys.stderr)
        return 2
    for p in found:
        print(f"check_pending_pin: {p}", file=sys.stderr)
    if not found:
        n = len([f for f in os.listdir(args.directory) if f.endswith((".yaml", ".yml"))]) \
            if os.path.isdir(args.directory) else 0
        print(f"check_pending_pin: {n} spec(s) pending the pin, none runnable on the pinned build")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
