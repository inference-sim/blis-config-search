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
python3 -I scripts/test_experiment.py
python3 -I scripts/test_run_blis.py
python3 -I scripts/test_blis_pin.py

# Single test: test_experiment.py / test_run_blis.py are unittest (run from the script's dir)
cd scripts && python3 -m unittest test_experiment.GridExpansion   # or .GridExpansion.test_x
# the others are plain-function runners; call the function from the __main__ block

# Experiment specs
python3 -I scripts/fetch_fields.py fields.json      # the pinned engine's field list, until `blis fields` exists
python3 -I scripts/validate_experiment.py experiments/<name>.yaml --blis ./blis --fields fields.json   # or experiments/
python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run --blis ./blis --fields fields.json

# Site data + web app
python3 -I scripts/build_gallery.py                                 # reports/ -> site/
cd web && npm ci && npm run dev | build | test | typecheck
```

`npm run dev/build` first runs `stage:data`, which calls `build_gallery.py` and copies `site/`
into `web/public/data/`. `REUSE_SITE=1` skips the rebuild.

## Architecture

One pipeline feeds one static site:

- **Experiments:** `experiments/<name>.yaml` (schema: `experiments/schema.json`,
  doc: `experiments/README.md`) → `scripts/validate_experiment.py` → `scripts/run_experiment.py`
  (deterministic sweeper, no LLM; runs BLIS via `scripts/run_blis.py`) → `reports/<name>/`.
  The `blis-experiment` skill only *writes* specs; it is the sole LLM step.
- **Site:** `reports/` (committed run dirs, the source of truth) → `scripts/build_gallery.py`
  (+ `build_catalog.py` for model/GPU/workload spec-sheets; `report.py` provides the run-dir and
  feasibility helpers) → `site/` → `web/` (Vite + React SPA, static on GitHub Pages, `base: './'`).
  The SPA is the only report surface; the former standalone HTML report was retired (#69).

Spec rules worth knowing: every knob value is a list; omitted knobs take the vLLM default; only
combos with `tp × dp × num-instances == scenario.gpus` are run; knob names are read from the
simulator (`blis fields`, `blis schema deployment`; `scripts/settings.py`), never kept here.
Routing policies live in `routing/*.yaml`.
`report/lexicon.json` is mirrored byte-for-byte at `web/src/lexicon.json`; edit the first, copy it over.

### Pinned stack and siblings

`pins.env` is the single source of truth for which BLIS produced the reports. `BLIS_REF`/`BLIS_SHA`,
`CATALOG_REF`, `REGISTRY_REF` are hand-pinned; the block at the end is generated from
inference-sim's `go.mod` by `scripts/sync_pins.py` — don't edit it by hand. `test_build.py` checks
the pins against `SETUP.md`, the workflows, and `go.mod`.

The validator and runner reach the simulator only through its commands (`scripts/blis_cli.py`):
`blis schema`, `blis schema validate` and `blis catalog resolve`, and the field list in the file `--fields` names (`blis fields` replaces the file when it exists). Tests use the
TEMPORARY stand-in `scripts/testdata/blis-m1/blis` until a pinned build has the commands. The
site build (`build_catalog.py`), `knob_profile.py` and `fetch_corpus.py` still read a catalog
checkout; see the M1 pull request for why. `test_build.py` reads `../inference-sim`'s go.mod.

### CI (`.github/workflows/`)

- `checks.yml` — the test gate above, plus validate + `--dry-run` every committed spec (fork-safe).
- `run-experiment.yml`: runs sweeps and opens a bot PR with `reports/`.
- `bump-blis-pin.yml` → `pin-compat.yml` → `regen-reports.yml`: the pin-bump loop, which re-sweeps every report.
- `pages.yml` — PR previews at `/pr-preview/pr-<N>/`; publishes on main.

`deploy/` runs sweeps as OpenShift Jobs: one BLIS run is single-threaded and `run_experiment.py`
runs serially, so scale with more pods, not bigger ones.
