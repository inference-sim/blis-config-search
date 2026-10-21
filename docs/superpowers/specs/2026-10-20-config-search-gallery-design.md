# config-search report gallery — design

**Status:** draft for review · **Date:** 2026-10-20

A static GitHub Pages site that surfaces the HTML reports produced by the `config-search`
skill: a filterable gallery landing page that links out to each self-contained report. No
server at view time; reports are added by PR and published by CI.

## 1. Problem & goal

`config-search` already turns a run directory into a self-contained magazine report
(`report.py` → `index.html`: exec summary, operating-point table, crossfilter explorer,
inline-SVG plots, glossary). What is missing is the **front door**: a single page that lists
every scenario, lets a reader cross-filter to the ones they care about, and links into the
full report for each — hosted statically on GitHub Pages so anyone can browse without
running anything.

**Goal.** Stand up `gallery/index.html` + a tiny build that scans committed run directories,
emits one gallery dataset, and deploys to GitHub Pages — plus a PR/CI flow for adding
reports that is safe against untrusted contributors.

**Non-goals.** (a) Re-implementing the per-report page — `report.py` owns that. (b) Running
BLIS or the LLM search in CI — see the threat model (§5). (c) A database or any runtime
backend — the site is pure static files.

## 2. Design language

The gallery matches the sibling `../leaderboard` **spec-sheet** aesthetic (the chosen
north-star): IBM Plex Sans for prose/labels, IBM Plex Mono + `tabular-nums` for every
measured number, a single teal accent reserved for "the number to look at", hairline rules,
a faint graph-paper ground, soft-chip tags — a lab instrument panel, not a SaaS dashboard.
Light and dark, responsive to phone width. The reports it links to keep their existing serif
magazine look; the gallery is the spec-sheet shell around them.

## 3. The gallery page (frontend)

One page, `gallery/index.html`, authored as a self-contained HTML file (inline CSS+JS, the
only external resource the Google Fonts stylesheet). It reads a single `data.js`
(`window.SCENARIOS = [...]`) emitted by the build (§4) and renders:

### 3.1 Surfaces
- **Masthead** — title, one-line dek, and an always-visible "Simulator predictions · not
  measured" trust badge. Theme toggle (system/light/dark, persisted in `localStorage`).
- **Summary strip** — filter-scoped counts only (*N scenarios · N servable runs · N models ·
  N hardware types in view*). Deliberately **no** single "best config" crowned across the
  filter: throughput is not comparable across different models/workloads, so a global crown
  would mislead. Each scenario surfaces **its own** recommended config instead.
- **Facet bar** — cross-filter over `model`, `hardware`, `workload`, `gpu_budget`, each value
  a chip with a **live count** that reflects the other active facets. Clicking toggles;
  zero-count values dim and disable. "Clear filters" appears when any facet is active.
- **Two views over the same filtered set**, chosen by a Table | Plot toggle (persisted):
  - **Table** (default) — one sortable row per scenario: model, hardware, GPUs, workload,
    recommended config, tok/s, tok/s/GPU, tok/s/user, ITL p99, servable fraction, load (c),
    → report. Numbers mono/right-aligned; any column sorts (stable, aria-sort). A row expands
    to its finalists. Best-in-column teal highlight applies **only among real rows** and only
    when ≥2 real rows are in view. Whole row links to the report. Reflows to stacked cards
    below ~700px.
  - **Plot** — a hand-built inline-SVG Pareto scatter: X = cluster tok/s, Y = per-user tok/s
    (the throughput↔interactivity tradeoff), marker **size** = tok/s/GPU, **color** = model
    (non-teal categorical ramp), **shape** = hardware, dashed/hollow = mock. Hover → tooltip;
    click → report. Same facet selection, same rows.
- **How-it-works panel** — three cells: how a report is made (`run-dir → index.html` via
  `report.py`, no server), how to add your own (run locally, commit, PR, CI publishes), how
  to read the numbers (usable operating point, finalists).
- **Trust note** — figures are simulator predictions; validate on real hardware before
  committing capacity.

### 3.2 Data contract (`window.SCENARIOS[]`)
One object per published scenario, derived by the build from each run directory's
`scenario.json` + `pareto.json`:

```
run_id, model, hardware, gpu_budget, workload, isl_osl, floor, headroom,
n_runs, n_servable, generated (ISO date),
winner:   { label, tp, ni, ep, quant, gpus, c_use, c_star, tps, tps_per_gpu, per_user, itl_p99 },
finalists:[ { label, tp, ni, gpus, c_use, tps, tps_per_gpu, per_user, itl_p99 } ],
reason
```

The report link is always `<run_id>/index.html`, relative — the gallery sits at the site
root and each report is a sibling directory. `run_id` is escaped on interpolation and
URL-encoded in hrefs.

### 3.3 Provenance & the mock question
Every value shown is a BLIS **simulator prediction**, surfaced as such. During design the
dataset carries one real scenario (the committed GLM-5.3-Flash fixture) and several
clearly-flagged `synthetic:true` placeholders so the filter is demonstrable; mock rows are
badged everywhere and never receive the teal "look here" treatment. **In production the
gallery shows only real committed run directories** — the synthetic rows exist only in the
design dataset and are dropped by the build (§4) when `--include-mock` is not passed.

## 4. The build (run-dirs → site)

A single deterministic Python entry, extending the existing `report.py` multi-run path
(which already renders each report and a card index). It:

1. Takes a parent directory of run dirs (e.g. `reports/`), each containing
   `scenario.json`, `pareto.json`, `runs.jsonl`, and the pre-generated `index.html`.
2. For each run dir: validates it against a strict schema (§5), copies the report into
   `site/<run_id>/`, and projects the gallery row (§3.2).
3. Emits `site/data.js` (`window.SCENARIOS=[…]`, sorted by `run_id` for byte-stable output)
   and `site/index.html` (the gallery shell, from a committed template with the dataset
   referenced, not inlined — mirrors how `report.py` already templates).
4. Is a **pure function of the committed data** (INV-6, matching `report.py`): same inputs →
   byte-identical `site/`. It **never** runs BLIS and **never** runs the LLM search; it only
   reads JSON and copies HTML.

`site/` is the GitHub Pages publish root. Nothing in it executes server-side.

## 5. Threat model & safety (the "stable, clear, sound backend")

The public site must not become an injection surface, and PR contributors are untrusted.

- **No runtime compute, no prompts.** The published site is static HTML/CSS/JS. There is no
  LLM, no server, and no form that feeds text to a model at view time — so there is no prompt
  to inject into. The `config-search` LLM search runs only on a contributor's own machine,
  locally, before they open a PR; CI never invokes it on PR-supplied prose.
- **CI runs only the pure transform.** The Pages workflow runs the §4 build (JSON-read +
  HTML-copy + template) over committed data. It does not run BLIS, the search, or any
  contributor-supplied code. A malicious PR cannot make CI execute arbitrary work beyond the
  fixed build.
- **Strict input validation + sanitization.** The build rejects any run dir whose
  `scenario.json`/`pareto.json` fail a strict JSON schema (known keys, numeric ranges, enum
  `model`/`hardware`/`workload`, `run_id` matching `^[a-z0-9][a-z0-9-]{0,63}$`). All string
  fields are HTML-escaped on render. Committed report `index.html` files are
  **report.py-generated and content-checked**, not free-form contributor HTML (see open
  question Q3).
- **Human merge gate.** GitHub Pages deploys only from the default branch after a
  **maintainer approves and merges** the PR. `pull_request` CI builds the site to catch
  breakage but publishes nothing; publish happens on `push: main` (merge) only. A random
  contributor can open a PR but cannot publish without review — the same model the sibling
  repos already use.
- **Least-privilege workflow.** The publish job uses `permissions: contents: read, pages:
  write, id-token: write`; the PR-validation job uses `contents: read` only and never has
  Pages or write scope.

## 6. Repository layout

```
reports/<run_id>/            committed run dirs (scenario.json, pareto.json, runs.jsonl, index.html)
gallery/index.template.html  the spec-sheet gallery shell (teal, facets, table+plot)
scripts/build_gallery.py     run-dirs → site/ (extends report.py multi-run)
scripts/gallery_schema.*     strict validation schema for committed run dirs
site/                        build output = Pages publish root (gitignored or built in CI)
.github/workflows/pages.yml  PR: build+validate (no publish); main: build+publish to Pages
docs/superpowers/specs/…     this spec
```

The `design/mockups/` directory is the design record: `data.js` (shared demo dataset), the
four explored directions (`mock-A`…`mock-D`), `EVAL.md` (the bake-off), and the chosen
**`mock-FINAL-gallery.html`** — the unified Table | Plot gallery the shipping
`gallery/index.template.html` derives from.

## 7. Build order

1. Lock the gallery template from the approved unified mockup (frozen HTML/CSS/JS).
2. `gallery_schema` + validation, with unit tests (valid/invalid/malicious run dirs).
3. `build_gallery.py` (run-dirs → `site/`), deterministic, with a test over
   `examples/sample-run` + a second fixture; assert byte-stability.
4. `pages.yml`: PR build-only job (no publish) + main build+publish job, least privilege.
5. Enable Pages on the repo; verify the deployed site links resolve to reports.

## 8. Open questions

- **Q1.** Publish root — a built `site/` in CI (reports copied), or serve `reports/` + a
  root `gallery/` in place (no copy)? In-place is simpler but mixes data and presentation.
- **Q2.** Should the gallery also expose each report's crossfilter *inline* (cross-scenario),
  or keep cross-scenario filtering to the winner only (current design) and leave per-run
  exploration to the report? Current design: winners only in the gallery.
- **Q3.** How strictly to treat committed `index.html` — regenerate it in CI from the
  committed JSON (guarantees it is report.py output, ignores contributor HTML entirely), or
  trust+content-scan the committed file? Regenerating in CI is the safer default and keeps
  INV-6; it costs a `report.py` run per report (pure transform, no BLIS).
```
