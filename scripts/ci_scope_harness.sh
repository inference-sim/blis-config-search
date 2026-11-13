#!/bin/sh
# Print which CI jobs run for sample change lists, and fail if one differs from the expectation.
#   sh scripts/ci_scope_harness.sh
set -eu
cd "$(dirname "$0")/.."
one=$(ls reports | head -1)
bad=0
# row LABEL PINS_MOVED EXPECT FILE...   EXPECT = "python experiments web pages smoke-reports"
row() {
  label=$1; moved=$2; want=$3; shift 3
  got=$(python3 -I scripts/ci_scope.py --pins-moved "$moved" --files "$@" | python3 -I -c '
import json, sys
r = json.load(sys.stdin)
print(" ".join(["Y" if r[j] else "-" for j in ("python", "experiments", "web", "pages")]), len(r["smoke_reports"]))')
  [ "$want" = "$got" ] && ok=ok || { ok=MISMATCH; bad=1; }
  printf '%-34s py exp web pages smoke  %-12s want %-12s %s\n' "$label" "$got" "$want" "$ok"
}
total=$(ls reports | wc -l | tr -d ' ')
printf 'reports with a smoke job: %s\n' "$total"
row "one spec"                  false "Y Y Y Y 1"  "experiments/$one.yaml"
row "one report's ledger"       false "Y - Y Y 1"  "reports/$one/ledger.jsonl"
row "pins.env, value moved"     true  "Y Y Y Y $total" pins.env
row "pins.env, comment only"    false "Y Y Y Y 0"  pins.env
row "runner script blis_pin.py" false "Y - - - $total" scripts/blis_pin.py
row "test file only"            false "Y - - - 0"  scripts/test_build.py
row "web/ only"                 false "- - Y Y 0"  web/src/App.tsx
row "docs only"                 false "- - - - 0"  docs/objective-refusals-router-files.md
row "unknown path"              false "Y Y Y Y $total" brand-new-dir/x
[ "$bad" = 0 ] || { echo "harness: a row differs from its expectation" >&2; exit 1; }
