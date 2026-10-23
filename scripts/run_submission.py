#!/usr/bin/env python3
"""Automation entrypoint: run config-search for ONE submitted Scenario, write reports/<run_id>/.

    python3 -I scripts/run_submission.py --submission submissions/foo.yaml \
        --blis ./blis --catalog ../blis-catalog --registry ../blis-registry --out reports/

Called by .github/workflows/generate-report.yml AFTER a maintainer approves the gated
environment. It is the seam between a committed Scenario and a committed run directory.

IMPORTANT — the search brain is an LLM. The config-search skill's search (enumerate
topologies, capacity-probe, diagnostic steering, rank) is driven by Claude, not by a plain
script. This entrypoint therefore:

  1. validates the submitted scenario (strict schema; same rules as the webpage/local server),
  2. requires an agent runner to perform the actual search, and
  3. fails LOUDLY if no runner is configured, rather than emitting a placeholder report.

Runner contract (selected by --runner or $CONFIG_SEARCH_RUNNER):
  * invoked as a single executable with no arguments;
  * receives the VALIDATED scenario as a JSON object on STDIN (not YAML, not argv);
  * reads BLIS_BIN / BLIS_CATALOG / BLIS_REGISTRY / OUT_DIR from the environment;
  * must write a complete run directory to $OUT_DIR/<run_id>/ (scenario.json, runs.jsonl,
    pareto.json, report.json) and exit 0 on success, non-zero on failure.
This keeps the non-hermetic LLM step pluggable and explicit — CI wires in the maintainers'
approved agent runner there. BLIS itself (invoked by the skill's run_blis.py/sweep_pool.py)
is deterministic and LLM-free.

This script never fabricates results and never publishes; the workflow commits reports/ to the
PR branch for the normal review + merge -> publish path.
"""
import argparse
import json
import os
import re
import subprocess
import sys

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
KNOWN_HARDWARE = {"h100", "h200", "gb200-nvl72", "gb300", "a100-80", "a100-sxm", "b200", "b300", "l40s"}
# The shape presets the catalog has always carried, used as the floor when no catalog path is
# supplied. With one, the catalog is authoritative — see _shape_workloads.
BASE_WORKLOADS = {"chatbot", "summarization", "contentgen", "multidoc"}


def _shape_workloads(catalog):
    """Workload presets a search can run, read from `catalog` when one is given.

    Only SHAPES qualify. `blis run --workload <name>` resolves the name to
    <catalog>/workloads/<name>.yaml through blis-schemas' LoadWorkload, which REFUSES a
    replay workload ("is a replay workload, not a shape"), so accepting a trace-driven
    replay here would queue a run that cannot start. A replay also needs a corpus fetched
    and digest-checked plus a lane count, none of which this Scenario can express.

    Reading the catalog means a preset added upstream is submittable with no edit here."""
    wdir = os.path.join(catalog, "workloads") if catalog else None
    if not wdir or not os.path.isdir(wdir):
        return set(BASE_WORKLOADS), set()
    shapes, replays = set(), set()
    for fn in sorted(os.listdir(wdir)):
        if not fn.endswith(".yaml"):
            continue
        name = fn[:-5]
        try:
            with open(os.path.join(wdir, fn), encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        (replays if isinstance(doc.get("replay"), dict) else shapes).add(name)
    return (shapes or set(BASE_WORKLOADS)), replays


def validate(scenario, catalog=None):
    errs = []
    model = str(scenario.get("model", "")).strip()
    hardware = str(scenario.get("hardware", "")).strip().lower()
    wl = scenario.get("workload") or {}
    workload = str((wl.get("preset") if isinstance(wl, dict) else wl) or "").lower()
    try:
        budget = int(scenario.get("gpu_budget"))
    except (TypeError, ValueError):
        budget = None
    if not model:
        errs.append("model required")
    if hardware not in KNOWN_HARDWARE:
        errs.append(f"unknown hardware {hardware!r}")
    shapes, replays = _shape_workloads(catalog)
    if workload in replays:
        errs.append(f"workload {workload!r} is a trace-driven replay, which a search scenario "
                    f"cannot run; choose a shape preset: {', '.join(sorted(shapes))}")
    elif workload not in shapes:
        errs.append(f"unknown workload {workload!r} (catalog defines: {', '.join(sorted(shapes))})")
    if not budget or not (1 <= budget <= 100000):
        errs.append("gpu_budget out of range")
    # interactivity_floor must be a mapping (run_search/report.py call .get on it). A bare number —
    # what the old Submit form emitted — would crash the pipeline with AttributeError downstream,
    # so reject it here with a clear message instead.
    floor = scenario.get("interactivity_floor")
    if floor is not None and not isinstance(floor, dict):
        errs.append("interactivity_floor must be a mapping, e.g. {tok_per_s_per_user: 50}")
    return errs, (model, hardware, workload, budget)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run config-search for one submitted Scenario.")
    ap.add_argument("--submission", required=True, help="submissions/<name>.yaml")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports"))
    ap.add_argument("--blis"); ap.add_argument("--catalog"); ap.add_argument("--registry")
    ap.add_argument("--runner", default=os.environ.get("CONFIG_SEARCH_RUNNER"),
                    help="executable that performs the agentic search and writes the run dir")
    args = ap.parse_args(argv)

    sub = os.path.abspath(args.submission)
    if not sub.startswith(ROOT + os.sep) or ".." in args.submission:
        raise SystemExit("unsafe submission path")
    if not os.path.isfile(sub):
        raise SystemExit(f"no such submission: {args.submission}")

    with open(sub, encoding="utf-8") as f:
        scenario = yaml.safe_load(f) or {}
    errs, (model, hardware, workload, budget) = validate(scenario, args.catalog)
    if errs:
        raise SystemExit("invalid scenario:\n  - " + "\n  - ".join(errs))
    print(f"validated: {model} on {budget}x {hardware}, {workload}")

    if not args.runner:
        raise SystemExit(
            "no --runner / $CONFIG_SEARCH_RUNNER configured.\n"
            "The config-search search is LLM-driven and cannot run headless without an agent "
            "runner. Configure the approved runner in CI (it receives the validated scenario "
            "and must write reports/<run_id>/). Refusing to emit a placeholder report."
        )

    # Hand the validated scenario to the approved agent runner. The runner is responsible for
    # the full search and for writing reports/<run_id>/ using the skill's scripts + the pinned
    # BLIS binary passed through here.
    env = dict(os.environ)
    for k, v in (("BLIS_BIN", args.blis), ("BLIS_CATALOG", args.catalog),
                 ("BLIS_REGISTRY", args.registry), ("OUT_DIR", args.out)):
        if v:
            env[k] = os.path.abspath(v) if os.path.exists(v) else v
    print(f"invoking runner: {args.runner}")
    proc = subprocess.run([args.runner], input=json.dumps(scenario), text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"runner failed (exit {proc.returncode}) — no report written")
    print("runner completed; verify reports/<run_id>/ was written")


if __name__ == "__main__":
    main()
