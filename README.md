# blis-config-search

Config search wrappers over the BLIS backend.

- **`.claude/skills/config-search/`** — the skill: drives BLIS to find Pareto-optimal serving
  configs for a model + GPU budget + workload, and renders each run directory into a
  self-contained HTML report (`report.py`).
- **`gallery/`** — a static GitHub Pages gallery that surfaces those reports: a filterable
  landing page (sortable table + Pareto plot) linking into each report. See
  [`gallery/README.md`](gallery/README.md). Build with `python3 -I scripts/build_gallery.py`
  and open `site/index.html`.
- **`reports/`** — committed run directories, the source of truth the gallery is built from.
- **`docs/superpowers/specs/`** — design specs; **`design/mockups/`** — the gallery design
  exploration (mockups + evaluation).
