# Site data producer

`scripts/build_gallery.py` turns the committed run dirs in `reports/` into the dataset and
per-report pages that `web/scripts/stage-data.mjs` stages into the SPA. The webpage itself is
in [`../web/`](../web/).

```
reports/<run_id>/     committed run dirs (source of truth):
                      scenario.json pareto.json runs.jsonl report.json experiment.json spec.yaml
site/                 build output (gitignored):
  data.js             window.SCENARIOS, one row per report
  catalog.js          window.CATALOG, the model / GPU / workload spec-sheets (build_catalog.py)
  mode.js             window.MODE, "pages" or "local"
  <run_id>/           index.html (rendered by scripts/report.py), records.json, the data files
```

## Build

```bash
python3 -I scripts/build_gallery.py               # reports/ -> site/ (mode=pages)
python3 -I scripts/build_gallery.py --mode local  # Submit tab also shows the local run command
```

`--no-mock` excludes synthetic run dirs (the production view); `--reports`, `--out` and
`--siblings` override paths. `cd web && npm run dev` runs this for you.

## Why it is safe

- The published site is static HTML/CSS/JS: no LLM, no server, and the Submit tab on Pages only
  drafts a spec for a PR. It cannot run anything.
- The publish path runs only `build_gallery.py`, a JSON-read and HTML-render transform that never
  runs BLIS. Sweeps run in `run-experiment.yml` and land as a bot PR a human merges.
- Every run dir is validated against a strict whitelist (known keys, hardware and `run_id` form,
  numeric ranges) before it contributes a row, and all strings are HTML-escaped.

## Determinism

`build_gallery.py` and `build_catalog.py` are pure functions of their inputs: run dirs are read
in sorted order and the datasets serialized stably, so identical inputs give identical output.
