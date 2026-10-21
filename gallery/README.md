# config-search report gallery

A static GitHub Pages site that surfaces the HTML reports produced by the `config-search`
skill: a filterable landing page (sortable table + Pareto plot over the same cross-filter)
that links into each self-contained report. No server runs at view time.

## Layout

```
reports/<run_id>/          committed run dirs — the source of truth
  scenario.json            resolved Scenario (model, hw, budget, workload, floor, headroom)
  pareto.json              finalists + winner (report.py / build input)
  runs.jsonl               one row per BLIS run (optional for the gallery row)
  report.json              brain-authored narrative (optional)
  gallery.json             display sidecar: {synthetic, isl_osl, generated, n_runs, n_servable, reason}
  MOCK                     presence marks the dir illustrative (second signal beside gallery.synthetic)
gallery/index.template.html the gallery shell (spec-sheet design; table + plot views)
scripts/build_gallery.py   reports/ -> site/  (validates, renders reports, emits data.js + index.html)
scripts/gen_demo_reports.py synthesizes the clearly-mock demo run dirs
site/                      build output = Pages publish root (gitignored; CI rebuilds)
```

## Build and view locally

```bash
python3 -I scripts/gen_demo_reports.py      # (demo only) create the mock run dirs
python3 -I scripts/build_gallery.py         # reports/ -> site/
open site/index.html                        # browse the gallery; cards link to reports
```

`--no-mock` excludes synthetic run dirs (the production view); `--reports` / `--out`
override the input and output directories.

## Adding a scenario

1. Run `config-search` locally against your scenario (it runs BLIS on *your* machine, not
   in CI), producing a run directory.
2. Add a `gallery.json` sidecar (see fields above) and commit the run dir under `reports/`.
3. Open a PR. CI builds the site to catch breakage but publishes nothing.
4. A maintainer reviews the committed data and merges. The push to `main` publishes to Pages.

## Why this is safe

- The published site is **pure static HTML/CSS/JS** — no LLM, no server, no form that feeds
  text to a model at view time, so there is no prompt surface to inject into.
- CI runs **only** `build_gallery.py` — a JSON-read + HTML-render transform that **never**
  runs BLIS or the LLM search. A malicious PR cannot make CI execute arbitrary work.
- Every run dir is **validated against a strict whitelist** (known keys, enum hardware,
  `run_id` pattern, numeric ranges) and all strings are HTML-escaped on render. Each report
  is **regenerated in the build from the committed JSON**, so published HTML is always
  `report.py` output, never free-form contributor markup.
- Publishing is gated on a **maintainer merge to `main`**; the PR job is read-only and never
  holds Pages or write scope.

## Determinism

`build_gallery.py` is a pure function of the committed data (INV-6): run dirs are processed
in sorted `run_id` order and the dataset is serialized stably, so identical inputs produce
byte-identical output. It never runs BLIS.
