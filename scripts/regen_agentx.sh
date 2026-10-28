#!/usr/bin/env bash
# Regenerate ONE AgentX run directory end to end, on one seed and one pinned stack.
#
#   scripts/regen_agentx.sh <submission.yaml> <corpus.jsonl>
#
# The four steps must run in this order and all four must run: the ad-hoc sequence that produced
# the first set of reports ran the search, then bumped the registry pin, then appended drafting
# arms -- so one ledger held rows priced by two different cost models and two different seeds
# (namasl review #1, #3). Driving it from one script is what keeps a report's rows comparable.
set -euo pipefail

SUB=${1:?usage: regen_agentx.sh <submission.yaml> <corpus.jsonl>}
CORPUS=${2:?usage: regen_agentx.sh <submission.yaml> <corpus.jsonl>}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

RUN_ID=$(basename "$SUB" .yaml)
D=reports/$RUN_ID

# 1. the search itself: every run at the submission's own seed
python3 -I scripts/run_search.py --submission "$SUB" \
  --blis .build/blis --catalog ../blis-catalog --registry ../blis-registry \
  --corpus "$CORPUS" --corpus-cache /tmp/agx-cache \
  --run-timeout 1800 --max-parallel 3

# 2. the drafting arm, at that same seed (add_drafting_arm refuses a seed that disagrees)
python3 -I scripts/add_drafting_arm.py "$D" \
  --blis .build/blis --catalog ../blis-catalog --registry ../blis-registry \
  --corpus "$CORPUS" --corpus-cache /tmp/agx-cache \
  --run-timeout 1800 --max-parallel 3

# 3. the narrative, read out of the data
python3 -I scripts/write_agentx_report.py "$D"

# 4. the rendered page
python3 -I .claude/skills/config-search/scripts/report.py "$D"

# A regenerated directory must be coherent: one seed, one stack, no provenance warning.
python3 -I - "$D" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(".claude", "skills", "config-search", "scripts"))
import report
d = sys.argv[1]
runs = [json.loads(l) for l in open(os.path.join(d, "runs.jsonl")) if l.strip()]
split = report.provenance_split(runs)
if split:
    raise SystemExit(f"{d}: NOT coherent after regeneration: {split}")
print(f"{d}: coherent ({len(runs)} runs, one seed, one stack)")
PY
