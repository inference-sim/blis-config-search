# config-search webpage

A static, multi-tab webpage that surfaces BLIS `config-search` reports and the BLIS catalog.
No server at view time; on GitHub Pages it is read-only and reports are added by PR.

## Tabs

- **Reports** — gallery of scenarios (sortable table + Pareto plot) over a cross-filter
  (model / hardware / workload / GPU budget, live counts). A row opens the in-app **scenario
  detail**: an executive briefing (recommended config, headline metrics, why it won, finalists)
  plus a technical Pareto explorer that cross-filters that scenario's own runs, and a link to
  the full standalone report.
- **Models / Workloads / GPUs** — the BLIS catalog (35 models, 4 workload presets, 9 GPUs),
  ingested from the sibling repos. Sortable spec-sheet tables; each row cross-links to the
  Reports tab filtered to it.
- **Submit** — a Scenario builder (model + workload + hardware + GPU budget + floor). Behavior
  depends on the build mode (below).

## Layout

```
reports/<run_id>/            committed run dirs — source of truth (JSON); index.html is built
  scenario.json pareto.json runs.jsonl report.json gallery.json [MOCK]
gallery/
  index.template.html        the SPA shell (tab nav + mount)
  app/                       vanilla ES modules: app.js reports.js detail.js models.js
                             workloads.js gpus.js submit.js + app.css
scripts/
  build_gallery.py           reports/ -> site/ (validate, render reports, emit data.js,
                             catalog.js, mode.js, per-scenario records.json; copy app/)
  build_catalog.py           sibling BLIS repos -> catalog.js (read-only, pinned, graceful)
  gen_demo_reports.py        synthesize the clearly-MOCK demo run dirs (incl. runs.jsonl)
  serve_local.py             local companion server (local mode only)
  run_submission.py          CI automation entrypoint (validate + delegate to approved runner)
  test_build.py              hermetic tests (validation, determinism, --no-mock, catalog)
site/                        build output = Pages publish root (gitignored; CI rebuilds)
.github/workflows/
  pages.yml                  PR: build+validate (no publish); main: build+publish to Pages
  generate-report.yml        PR scenario -> maintainer-approved search -> report -> PR branch
```

## Build and view locally

```bash
python3 -I scripts/gen_demo_reports.py        # (demo only) create the mock run dirs
python3 -I scripts/build_gallery.py           # reports/ -> site/ (mode=pages, default)
python3 -m http.server -d site 8137           # open http://localhost:8137/
```

Flags: `--mode local` enables the live submission form; `--no-mock` excludes synthetic run
dirs (the production view); `--reports` / `--out` / `--siblings` override paths.

### Local mode (run the search yourself)

```bash
python3 -I scripts/build_gallery.py --mode local
python3 -I scripts/serve_local.py             # serves site/ + POST /api/search
```

In local mode the Submit tab is live: it validates your Scenario, writes
`submissions/<name>.yaml`, and hands back the exact `/config-search` command to run in your
own Claude session against your own LLM. Nothing is sent to a hosted service.

## Modes

- **pages** (default; GitHub Pages): fully static. The Submit form is an editable *builder* —
  it produces a copy-ready `scenario.yaml` + a prefilled GitHub PR link + PR steps — but
  **cannot execute** a search (no run button, no network). This is the only mode CI publishes.
- **local** (`--mode local`, your machine): same site, but the Submit form can POST to the
  local companion server to run the search. Never published.

## Adding a scenario (the PR → CI → publish path)

1. Build the Scenario on the Submit tab (or write `submissions/<name>.yaml`) and open a PR.
2. `pages.yml` builds the site on the PR to catch breakage — it publishes nothing.
3. A maintainer reviews the submission and approves the protected `report-generation`
   environment; `generate-report.yml` builds the pinned BLIS stack, runs the search, and
   commits the generated `reports/<run_id>/` back to the PR branch.
4. The maintainer merges. The push to `main` triggers `pages.yml` to publish to Pages.

## Why this is safe

- The published site is **pure static HTML/CSS/JS** — no LLM, no server, no form that feeds
  text to a model at view time, so there is no prompt surface to inject into. The Submit form
  on Pages cannot run anything (mode baked at build; no backend endpoint).
- CI's publish path runs **only** `build_gallery.py` — a JSON-read + HTML-render transform
  that never runs BLIS or the LLM search. The search backend is a **separate** workflow gated
  behind a maintainer-approved environment; an untrusted PR cannot trigger it automatically.
- Every run dir is **validated against a strict whitelist** (known keys, enum hardware,
  `run_id` pattern, numeric ranges) and all strings are HTML-escaped. Each report is
  **regenerated from the committed JSON**, so published HTML is always `report.py` output.
- Publishing is gated on a **maintainer merge to `main`**; the PR job is read-only.

## Determinism

`build_gallery.py` and `build_catalog.py` are pure functions of their inputs (INV-6): run
dirs processed in sorted order, datasets serialized stably. Identical inputs → identical
output. Neither runs BLIS.

## Tests

```bash
python3 -I scripts/test_build.py    # validation, determinism, --no-mock, catalog accuracy
```
