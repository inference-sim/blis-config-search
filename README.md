# blis-config-search

Config search wrappers over the BLIS backend.

- **`.claude/skills/config-search/`** — the skill: drives BLIS to find Pareto-optimal serving
  configs for a model + GPU budget + workload, and renders each run directory into a
  self-contained HTML report (`report.py`).
- **`web/`** — the webpage: a Vite + React + TypeScript SPA (spec-sheet design matching
  `../accuracy-dashboard`) surfacing the reports and the BLIS catalog (Models / Workloads /
  GPUs tabs). Static, light-theme, serves on GitHub Pages with no server. See
  [`web/README.md`](web/README.md): `cd web && npm ci && npm run build`.
- **`gallery/` + `scripts/build_gallery.py`** — the data producer: committed run dirs →
  dataset + per-scenario reports the SPA stages in. See [`gallery/README.md`](gallery/README.md).
- **`reports/`** — committed run directories, the source of truth the site is built from.
