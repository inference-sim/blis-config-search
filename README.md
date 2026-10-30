# blis-config-search

Config search wrappers over the BLIS backend.

- **`.claude/skills/config-search/`** — the skill: drives BLIS to find Pareto-optimal serving
  configs for a model + GPU budget + workload, and renders each run directory into a
  self-contained HTML report (`report.py`).
- **`web/`** — the webpage: a Vite + React + TypeScript SPA (spec-sheet design matching
  `../accuracy-dashboard`) surfacing the reports, a Coverage view of every searched
  model × GPU × workload × budget combination, and the BLIS catalog spec-sheets behind it.
  Static, light-theme, serves on GitHub Pages with no server. See
  [`web/README.md`](web/README.md): `cd web && npm ci && npm run build`.
- **`gallery/` + `scripts/build_gallery.py`** — the data producer: committed run dirs →
  dataset + per-scenario reports the SPA stages in. See [`gallery/README.md`](gallery/README.md).
- **`reports/`** — committed run directories, the source of truth the site is built from.

## Submitting an experiment

An **experiment** is one scenario plus a few sweeps that tell one story — colocated vs.
disaggregated, drafting off vs. on, one routing policy vs. another. The pipeline splits at one
seam: an LLM (or you) writes the spec, and everything after it is deterministic and runs in CI.

```
experiments/<name>.yaml  ──▶  scripts/run_experiment.py  ──▶  reports/<name>/  ──▶  site
   the Experiment Spec          expand · filter · run · rank        (bot PR)
```

- **`experiments/`** — the specs, the [JSON Schema](experiments/schema.json), and the
  [contributor doc](experiments/README.md). Five worked examples are committed.
- **`scripts/validate_experiment.py`** — rejects a bad spec before any run: unknown knob names
  (checked against the `blis-vllm` / `blis-schemas` checkouts), GPU arithmetic that can never be
  satisfied, SLOs without units, missing router files.
- **`scripts/run_experiment.py`** — the deterministic sweeper. **No LLM.** Same spec + same
  [`pins.env`](pins.env) ⇒ same report.
- **`.claude/skills/blis-experiment/`** — the skill that *designs* a spec from a model ×
  hardware × workload, reading the catalog to pick sweeps that apply. The only LLM in the path,
  and it stops at the spec.

To submit one: write `experiments/<name>.yaml`, run

```
python3 -I scripts/validate_experiment.py experiments/<name>.yaml
python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run   # check the run budget
```

and open a PR. CI validates it (no secrets, safe for forks); after merge the sweeper runs and a
bot PR adds `reports/<name>/`. See [`experiments/README.md`](experiments/README.md) for the full
format.

> The older **`submissions/`** path (one Scenario → an LLM-driven config *search*) still works and
> its reports are unchanged. New work should prefer `experiments/`, which answers a stated
> question and needs no model at measurement time.
