#!/usr/bin/env python3
"""Fail unless every run of a report was priced by a tagged release's build.

  check_released_build.py REPORT_DIR

Reads REPORT_DIR/runs.jsonl. A row records `released_build` from its Result's data block: false
means the binary was not stamped with a release tag (see build_blis.sh), true that it was, and a
missing value that the Result carried no data block. Only true passes, so a CI sweep that must
produce a committable report proves the build it ran on.
"""
import json
import os
import sys


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    path = os.path.join(argv[0], "runs.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    bad = [r for r in rows if r.get("released_build") is not True or r.get("data") != "embedded"]
    if not rows:
        print(f"check_released_build: {path} has no runs", file=sys.stderr)
        return 1
    if bad:
        print(f"check_released_build: {len(bad)} of {len(rows)} runs were not priced by a tagged release's "
              f"embedded data (released_build={sorted({str(r.get('released_build')) for r in bad})}): the "
              f"build is not a tagged release", file=sys.stderr)
        return 1
    print(f"check_released_build: all {len(rows)} runs priced by a tagged release")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
