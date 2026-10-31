# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comment style

Keep comments in agent-generated files (code, YAML specs, workflows, `pins.env`) short and
human-readable: one or two lines that say *why*, not a narrative. No multi-paragraph comment
blocks, no history of how the code got here. Longer rationale belongs in a README or the PR.

## Commands

Local `python3` has no PyYAML; run Python through the repo's `.venv` (`pyyaml`, `jsonschema`).
Scripts are always invoked with `-I`.

```bash
# Python tests (hermetic: no BLIS, no network) — same set CI runs in checks.yml
python3 -I scripts/test_build.py
python3 -I scripts/test_backend.py
python3 -I scripts/test_experiment.py
python3 -I .claude/skills/config-search/scripts/test_report.py
python3 -I .claude/skills/config-search/scripts/test_run_blis.py
python3 -I scripts/test_search.py
python3 -I scripts/test_topologies.py

# Single test: test_experiment.py / test_run_blis.py are unittest (run from the script's dir)
cd scripts && python3 -m unittest test_experiment.GridExpansion   # or .GridExpansion.test_x
# the others are plain-function runners; call the function from the __main__ block

# Experiment specs
python3 -I scripts/validate_experiment.py experiments/<name>.yaml   # or experiments/
python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run

# Site data + web app
python3 -I scripts/build_gallery.py [--mode local] [--no-mock]      # reports/ -> site/
cd web && npm ci && npm run dev | build | test | typecheck
```

`npm run dev/build` first runs `stage:data`, which calls `build_gallery.py` and copies `site/`
into `web/public/data/`. Env: `SITE_MODE=local`, `NO_MOCK=1`, `REUSE_SITE=1`.

## Architecture

Two pipelines feed one static site:

- **Experiments (preferred):** `experiments/<name>.yaml` (schema: `experiments/schema.json`,
  doc: `experiments/README.md`) → `scripts/validate_experiment.py` → `scripts/run_experiment.py`
  (deterministic sweeper, no LLM) → `reports/<name>/`. The `blis-experiment` skill only *writes*
  specs; it is the sole LLM step.
- **Submissions (legacy):** `submissions/<name>.yaml` → `config-search` skill (LLM-driven search
  via `.claude/skills/config-search/scripts/run_blis.py`, report via `report.py`) → `reports/`.
- **Site:** `reports/` (committed run dirs, the source of truth) → `scripts/build_gallery.py`
  (+ `build_catalog.py` for model/GPU/workload spec-sheets) → `site/` → `web/` (Vite + React SPA,
  static on GitHub Pages, `base: './'`).

Spec rules worth knowing: every knob value is a list; omitted knobs take the vLLM default; only
combos with `tp × dp × num-instances == scenario.gpus` are run; knob names must be BLIS's exact
names (validated against sibling checkouts). Routing policies live in `routing/*.yaml`.

### Pinned stack and siblings

`pins.env` is the single source of truth for which BLIS produced the reports. `BLIS_REF`/`BLIS_SHA`,
`CATALOG_REF`, `REGISTRY_REF` are hand-pinned; the block at the end is generated from
inference-sim's `go.mod` by `scripts/sync_pins.py` — don't edit it by hand. `test_build.py` checks
the pins against `SETUP.md`, the workflows, and `go.mod`.

Several scripts read sibling checkouts in `../` (`inference-sim`, `blis-catalog`, `blis-registry`,
`blis-schemas`, `blis-vllm`, `blis-latency-kernel`) and degrade gracefully when absent.

### CI (`.github/workflows/`)

- `checks.yml` — the test gate above, plus validate + `--dry-run` every committed spec (fork-safe).
- `run-experiment.yml` / `generate-report.yml` — run sweeps/searches and open a bot PR with `reports/`.
- `pages.yml` — PR previews at `/pr-preview/pr-<N>/`; publishes on main.

`deploy/` runs sweeps as OpenShift Jobs: one BLIS run is single-threaded and `run_experiment.py`
runs serially, so scale with more pods, not bigger ones.
