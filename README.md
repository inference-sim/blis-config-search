# blis-config-search

Deterministic BLIS experiment sweeps, and the site that publishes their reports.

- **`scripts/`**: the sweeper (`run_experiment.py`), its BLIS runner (`run_blis.py`), and the
  report renderer (`report.py`, with assets in `report/`). See [`SETUP.md`](SETUP.md) to run a
  sweep locally.
- **`web/`** — the webpage: a Vite + React + TypeScript SPA (spec-sheet design matching
  `../accuracy-dashboard`) surfacing the reports, a Coverage view of every swept
  model × GPU × workload × budget combination, linking out to blis-catalog for spec sheets.
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
