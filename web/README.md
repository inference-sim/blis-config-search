# config-search webpage (Vite + React)

The spec-sheet webpage for BLIS config-search reports and catalog. Design language and stack
match the sibling [`../../accuracy-dashboard`](../../accuracy-dashboard): IBM Plex + teal, a
left sidebar rail + tabbed main area, dense tables, **light theme only**, custom dropdowns
(never native `<select>`). Builds to static files that serve on GitHub Pages with **no server**.

## Develop

```bash
cd web
npm ci                       # install (use --cache <dir> if ~/.npm is unwritable)
npm run dev                  # predev stages data, then Vite dev server
npm run build                # prebuild stages data, tsc --noEmit, vite build -> dist/
npm run test                 # vitest
npm run typecheck            # tsc --noEmit
```

`npm run stage:data` runs `../scripts/build_gallery.py` (the data producer) and lifts its
output into `public/data/` as `scenarios.json`, `catalog.json`, `mode.json`, and
`reports/<run_id>/{index.html,records.json}`. Env: `SITE_MODE=local` enables the live Submit
form; `NO_MOCK=1` drops synthetic scenarios (production); `REUSE_SITE=1` skips the Python
rebuild if `../site/` already exists.

## Why it needs no server on GitHub Pages

Vite `base: './'` makes every asset and data request **relative**, so `web/dist/` is plain
static files that work at any path — including a GitHub Pages project subpath
(`user.github.io/<repo>/`). The SPA `fetch`es the staged JSON as same-origin relative
requests, which a static host serves directly. No Python/Node runs at view time. (Bare
`file://` is the one context where `fetch` is blocked — use `npm run dev` or any static
server locally; Pages is never `file://`.)

## Layout

```
src/
  main.tsx App.tsx route.ts lib.ts types.ts load.ts vite-env.d.ts styles.css
  components/ Dropdown.tsx Rail.tsx Reports.tsx ScenarioDetail.tsx
              Models.tsx Workloads.tsx Gpus.tsx CatalogTable.tsx Submit.tsx
scripts/stage-data.mjs       site/ -> public/data/
public/data/                 staged data (gitignored; regenerated)
dist/                        build output = Pages artifact (gitignored)
```

Tabs: **Reports** (exec "configs to try" summary → cross-filter → gallery cards → scenario
detail with briefing + Pareto explorer), **Models / Workloads / GPUs** (catalog spec-sheets),
**Submit** (Scenario builder; read-only on Pages, live in local mode).
