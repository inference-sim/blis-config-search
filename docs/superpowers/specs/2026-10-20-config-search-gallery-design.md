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

---

# v2 — the full webpage (multi-tab SPA)

**Status:** building · **Date:** 2026-10-21

v1 shipped the gallery + static build + PR/CI. v2 grows it into the full site the project
needs: a multi-tab shell, richer per-scenario exploration, the BLIS catalog surfaces
(models / workloads / GPUs), a Scenario-submission flow that is interactive locally and
read-only on GitHub Pages, and the CI that actually runs the search on merged PRs.

## 9. Information architecture — tabs

One static SPA shell (`gallery/app.*`, hash-routed, no framework/build step — vanilla JS
modules, same spec-sheet design tokens). Tabs:

1. **Reports** (default) — the v1 gallery (table + Pareto plot over the cross-filter). A row
   now opens an in-app **scenario detail** (§10), with a link to the full standalone report.
2. **Models** — every model BLIS can run (from blis-catalog): params, architecture
   (dense/MoE, experts), heads, context, quantization. Filterable; each links to the reports
   that used it.
3. **Workloads** — the workload presets (from inference-sim): ISL/OSL, arrival shape, trace
   source. Each links to reports using it.
4. **GPUs** — the hardware BLIS knows (from inference-sim hardware config): memory, bandwidth,
   FLOPs. Each links to reports on it.
5. **Submit** — build a Scenario (model + workload + hardware + GPU budget + floor). Behavior
   is **mode-dependent** (§12): interactive locally, read-only (PR instructions) on Pages.

The catalog tabs are generated at build time from the sibling repos (§11) into `catalog.js`,
so they are static data at view time — no network, no runtime fetch.

## 10. Scenario detail (per-scenario Pareto cross-filtering)

Selecting a report row opens a detail view that surfaces the scenario's own run data richly,
not just the winner:

- **Briefing** (executive): the recommended config, headline metrics (tok/s, tok/s/GPU,
  tok/s/user, ITL p99), the floor it holds, the "why it won" reason, and the finalists it
  beat — scannable in one screen.
- **Pareto charts with cross-filtering** (technical): the full set of the scenario's runs
  (`runs.jsonl` projected to RECORDS) in an interactive explorer — the same d3+crossfilter
  machinery `report.py` already bakes into each report, now reachable from the gallery.
  Cross-filter dimensions: config/topology, concurrency, servable/not, bottleneck. Charts:
  operating curve (per-user vs cluster tok/s), scale bars, tail scatter. Filtering any
  dimension updates every chart + the run table together.
- **Open the full report →** the standalone `report.py` `index.html` (deep narrative +
  glossary) remains the canonical artifact.

Implementation: the detail view reuses the report's RECORDS + explorer so there is one
cross-filter implementation, not two. `build_gallery.py` emits each scenario's RECORDS into
the dataset (or the detail view lazy-loads the report's data files, already copied to
`site/<run_id>/`).

## 11. Catalog ingestion (sibling repos → static data)

`scripts/build_catalog.py` reads the three read-only sibling repos and emits
`site/catalog.js` (`window.CATALOG = {models, hardware, workloads, registry, versions}`):

- **Models** ← blis-catalog (pinned tag). Per model: name, params, architecture, experts,
  heads, context, quantization — whatever the catalog entry carries, projected to a stable
  shape; unknown fields dropped, never invented.
- **Hardware** ← inference-sim hardware config. Per GPU: name, memory, bandwidth, FLOPs.
- **Workloads** ← inference-sim presets. Per preset: name, ISL/OSL, arrival, trace source.
- **Registry** ← blis-registry (pinned tag): the coefficient/cost-model sets by name.
- **Versions** ← the pinned tags/commits (SETUP.md / .env), shown in a footer so the catalog
  is traceable to exact upstream versions.

The exact pins live in a single `.env`-style source (mirroring the leaderboard) so the
catalog and the docs can't drift. The ingestion is read-only and pure; it never mutates the
sibling repos. If a sibling repo is absent at build time, the tab degrades to "catalog
unavailable — built without <repo>" rather than failing the whole build.

## 12. Submission flow — two modes

A single Scenario form, two behaviors chosen by a build-time flag baked into the site
(`window.MODE = "local" | "pages"`):

- **local** (`build_gallery.py --mode local`, for `python3 -m http.server` on a dev box):
  the form is live. The user picks model/workload/hardware/budget/floor; on submit the page
  POSTs to a **local companion server** (`scripts/serve_local.py`) that invokes the
  config-search skill against a user-provided LLM endpoint, streams progress, and drops a new
  run dir into `reports/`. This is the only part that needs an LLM, and it runs only on the
  user's own machine with their own key. Clearly labeled "runs on your machine".
- **pages** (default, for GitHub Pages): the form is shown but **disabled** — it renders the
  exact Scenario the user built as a ready-to-copy `scenario.yaml` plus step-by-step "open a
  PR to run this" instructions and a prefilled GitHub PR link. No compute, no LLM, no network
  calls. This showcases the capability without exposing any execution surface.

The mode is a static constant in the built site; the Pages build can never emit `local` mode.

## 13. Backend — PR → CI → reports

Two workflows:

- **`pages.yml`** (v1, extended) — on PR: build the site (gallery + catalog) and validate; no
  publish. On push to main: build + publish to Pages. Least privilege as in v1.
- **`generate-report.yml`** (new) — the report-generation backend. Triggered on a PR that adds
  a `scenario.yaml` under `submissions/` (or by a maintainer via `workflow_dispatch` /a label).
  It:
  1. Builds the pinned BLIS stack (inference-sim v0.10.0 + catalog 0.2.1 + registry v0.1.1),
     exactly as SETUP.md pins them.
  2. Runs config-search for the submitted Scenario — **the search/LLM step is gated to run
     only after a maintainer approves** (GitHub `environment` protection / approval), never
     automatically on an untrusted PR. BLIS itself is deterministic and LLM-free; the search
     brain (LLM) is the only non-hermetic part and is what the approval gates.
  3. Writes the resulting run dir into `reports/`, regenerates the report, and commits it back
     to the PR branch (or opens a follow-up PR) so the normal review + merge → publish path
     applies. Nothing reaches the live site without the human merge gate.

**Threat model holds:** the public site is still static (no runtime LLM/prompt surface); the
only place prose from a submitter meets an LLM is inside CI, behind an explicit maintainer
approval, on infrastructure the maintainers control — never the public Pages site.

## 14. Repository layout (v2 additions)

```
gallery/app/                 the SPA shell + tab modules (vanilla JS, spec-sheet CSS)
  index.template.html        shell with tab nav + mount points
  app.js  reports.js  models.js  workloads.js  gpus.js  submit.js  detail.js  explorer.js
scripts/build_catalog.py     sibling repos → site/catalog.js  (read-only, pure, pinned)
scripts/build_gallery.py     reports/ → site/ (extended: emits mode, RECORDS, includes catalog)
scripts/serve_local.py       local companion server (local mode only; invokes the skill)
submissions/<name>.yaml      PR-submitted Scenarios the generate-report workflow consumes
pins.env                     single source of truth for upstream repo tags/commits
.github/workflows/generate-report.yml   PR scenario → (approved) search → report → PR
```

## 15. Build order (v2)

1. Catalog ingestion (`build_catalog.py`) from the three repos + a test; emit `catalog.js`.
2. SPA shell + tab nav; port the v1 gallery into the Reports tab unchanged.
3. Models / Workloads / GPUs tabs from `catalog.js`; cross-link to reports.
4. Scenario detail view with the report's crossfilter explorer reused in-app.
5. Submit tab, both modes; `serve_local.py` for local mode.
6. `generate-report.yml` with the maintainer-approval gate; dry-run tested.
7. Agent review loop until the panel agrees the site serves the purpose.
