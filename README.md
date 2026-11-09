# blis-config-search

Deterministic BLIS experiment sweeps, and the static site that publishes their reports.

```
experiments/<name>.yaml  ──▶  scripts/run_experiment.py  ──▶  reports/<name>/  ──▶  web/ (GitHub Pages)
   the spec                     expand · filter · run · rank     (bot PR)
```

An **experiment** is one scenario plus a few sweeps that tell one story: colocated vs.
disaggregated, drafting off vs. on, one routing policy vs. another. An LLM (or you) writes the
spec; everything after it is deterministic and runs in CI.

| Path | What it is |
|---|---|
| `experiments/` | the specs, their [JSON Schema](experiments/schema.json), and the [contributor doc](experiments/README.md) |
| `scripts/validate_experiment.py` | rejects a bad spec before any run (unknown knobs, impossible GPU arithmetic, SLOs without units) |
| `scripts/run_experiment.py` | the sweeper, **no LLM**: same spec + same [`pins.env`](pins.env) gives the same report |
| `scripts/run_blis.py` | runs one BLIS config and records it |
| `reports/` | committed run dirs, the source of truth the site is built from |
| `scripts/build_gallery.py` | `reports/` → `site/` (JSON + one HTML report per run dir, rendered by `scripts/report.py`) |
| `web/` | the Vite + React site, static on GitHub Pages |
| `.claude/skills/blis-experiment/` | the skill that *designs* a spec; the only LLM in the path |

## Submit an experiment

Write `experiments/<name>.yaml` (the site's Submit tab drafts one), then:

```bash
python3 -I scripts/fetch_fields.py fields.json   # the pinned engine's field list, until `blis fields` exists
python3 -I scripts/validate_experiment.py experiments/<name>.yaml --blis ./blis --fields fields.json
python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run --blis ./blis --fields fields.json   # check the run budget
```

and open a PR. CI validates it (no secrets, safe for forks); after merge the sweeper runs and a
bot PR adds `reports/<name>/`. To run a sweep yourself, see [`SETUP.md`](SETUP.md).

## Site

```bash
cd web && npm ci && npm run dev   # or: build | test | typecheck
```

`npm run dev/build` first runs `build_gallery.py` and copies `site/` into `web/public/data/`
(`REUSE_SITE=1` skips the rebuild). Vite's `base: './'` keeps every request relative, so
`web/dist/` serves from any static host with no server.
