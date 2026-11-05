# `report.json`: the narrative seam

A report draws on four inputs: three measurements (`scenario.json`, `pareto.json`, `runs.jsonl`)
and **`report.json`**, the narrative, which `scripts/run_experiment.py` writes from the spec and
the measured results. It carries the prose, the hero panel, the numbered findings, the operating-
point table, and the glossary, and is rendered by the React SPA under `web/`.

It is **optional**. Without it the report still shows the header, the explorer over every run, and
an auto operating-point table — just without the analysis. With it, you get the full article. The
reader is **defensive**: every field below degrades gracefully if absent.

Keep the prose grounded in the run's actual numbers (cite `pareto.json` / `runs.jsonl`). Prose
strings may contain inline HTML, including glossary links
`<a class="gl" href="#g-ID" data-def="short definition">term</a>`.

```jsonc
{
  "title": "Serving GLM-5.3-Flash on sixteen H100s",   // <h1>; falls back to model/hardware
  "standfirst": "A configuration search over … .",      // the lede sentence under the title
  "meta": ["BLIS discrete-event simulation", "…"],       // small sans-serif tags under the standfirst

  // Hero panel: up to 3 cells. The last renders in the accent "what to run" style.
  "hero": [
    {"h": "What was tested", "html": "…inline HTML allowed…"},
    {"h": "What was found",  "html": "…"},
    {"h": "What to run",     "html": "…", "run": true}
  ],

  // Section 1 is always "The measurements" + the explorer; this is its intro prose.
  "explore_prose": ["…", "…"],

  // Numbered findings (sections 2…N). Each may carry a figure + a prose block; both optional.
  "sections": [
    {
      "num": 2,
      "title": "The operating curve, and where it stops",
      "figure": "operating_curve",      // see "Figures" below; omit for a prose-only section
      // The caption's first sentence is SHARED, from report/lexicon.json — do not write it here.
      // `caption_note` appends what is specific to THIS run (inline HTML allowed).
      "caption_note": "Here that is a four-fold spread in capacity.",
      "prose": ["paragraph one…", "paragraph two…"]
    }
  ],

  // "What to run" section (number defaults to 9). Prose + an operating-point table.
  // The table is DERIVED from runs.jsonl whenever there are runs — see "The shared lexicon".
  "recommendation_num": 9,
  "recommendation_prose": ["…", "…"],
  "operating_points": [
    // FALLBACK ONLY: used when the run dir has no usable ledger. With runs.jsonl present the
    // table is derived from the records instead, so both report surfaces show the same rows.
    // `flags` keys are read in the lexicon's fixed order (tp, dp, epw, ni, bs, mnbt, mns, gmu,
    // mtp, route, pd — any subset).
    {"conc": 20, "flags": {"tp": 4, "ni": 4, "mnbt": 8192, "mns": 256, "mtp": 0},
     "pu": 70.0, "cl": 1471.5, "itl": 14.29, "hit": null}
  ],

  // Glossary (section number defaults to 10). `id` must match the `href="#id"` used in prose.
  "glossary": [
    {"id": "g-per-user-token-rate", "term": "per-user token rate",
     "def": "The bold one-line definition.", "long": "Optional extra sentence(s)."}
  ]
}
```

## The shared lexicon

The site renders the same runs on **two** surfaces: this standalone report and the React
dashboard in `web/`. They had grown two vocabularies for one set of numbers and, worse, two
ranking objectives for "What to run" — the report ranked operating points by per-user rate while
the dashboard ranked them by tok/s per GPU, so at high load the two tables recommended *different*
configurations for the same scenario (issue #13).

Both surfaces now read **`report/lexicon.json`**:

| block | what it fixes |
|---|---|
| `objective` | the one ranking rule for "What to run" (tok/s per GPU, with explicit tiebreaks), plus the blurb that *states* it to the reader |
| `metrics` / `facets` | one label, unit, rounding and one-line definition per RECORDS field |
| `axes` | the axis titles the figures draw |
| `figures` | per figure: title, axis refs, the shared caption, and the "how to read it" line |
| `flags` / `flag_skip` | the deploy flags the operating table prints, and the values worth no ink |
| `operating_columns` | which columns that table shows, in order |

`scripts/lexicon.py` is the Python reader. `web/src/lexicon.json` is a **byte-identical mirror**
(Vite cannot import from outside `web/`), and `scripts/test_build.py` fails the build if the two
drift. Change `report/lexicon.json`, then copy it over the mirror.

**So:** do not write a metric name, an axis title, or a ranking rule into `report.json`. Add it to
the lexicon (both copies) and let both surfaces read it.

## `RECORDS` field reference (what the explorer consumes)

Assembled in `scripts/explore_data.py`. Knobs the engine doesn't vary come through constant (e.g.
`route: "default"`, `mtp: 0`) and the explorer automatically hides single-value dimensions.

`id, src, conc, tp, dp, ni, gpus, series, route, pd, mtp, mnbt, mns, a2a, epw,`
`pu (per-user tok/s: authoritative tok_per_s_per_user if present; else 1000/ITL-p99 when one step`
`   == one token (mtp==0) -- the p99 basis matches the floor + pareto; else null, never a step rate),`
`pu_is_step_estimate (True = under MTP with no authoritative field, so pu is null not a wrong number),`
`cl (=tokens_per_sec), itl (=mean ITL ms), ttft (=ttft_p99_ms),`
`hit (prefix hit or null), to (timed out), drop (unservable), p95 (=e2e_p95_ms),`
`ok (met completion+floor+SLOs; absent SLO metric = violation), status, viol[] (reasons when not ok)`

## `runs.jsonl` record fields (the raw ledger)

One JSON object per BLIS run, written by `scripts/run_blis.py` `run_one`. The explorer's `RECORDS`
above are derived from these. Fields that feasibility, ranking and the site read:

| field | what it is |
|---|---|
| `config` | the knob dict this run was given, `concurrency` included |
| `config_hash` | sha1 (12 hex) of `config`: one run at one load |
| `config_key` | the same hash over `config` without `concurrency`: one configuration, shared by its records at every load. Group on it to get one config's curve |
| `metrics` | the **whole-run** aggregate under the legacy MetricsOutput names. Every verdict, `per_user_rate` and the floor read this; `itl_*` here is the decode-step interval |
| `metrics_all_instances` | the per-instance MetricsOutput rows, same basis |
| `measured` | `{metrics, metric_defs}`: the BLIS `Result` document's top-level **measured-window** block, copied verbatim with BLIS's own names and definitions. Absent (not empty) when BLIS printed no such block, including every run on a pre-v0.15.0 binary. Read by nothing in this repo's verdicts |
| `result` | `{path, sha256}` of the full `Result` document, gzipped. `path` is always relative to the run dir (the directory holding `runs.jsonl`; for `run_blis.py` without `--out`, the parent of the `--keep-results` dir). Present only when the run used `--keep-results` |

`measured.metrics` at v0.15.0 holds `output_tok_s`, `output_tok_s_per_gpu`, `request_throughput`,
TTFT and E2E at mean/p50/p90/p95/p99, per-request ITL (TPOT excluding the first token) at
mean/p50/p90/p99, and per-request output tok/s per user as `interactivity_mean`,
`interactivity_p50` and `interactivity_p90`. These are a different population from `metrics`:
per request over the measured window, not per decode step over the whole run, so the two
`itl_p99_ms` values can differ by several times on the same run.

BLIS's `interactivity_p90` is `1000 / itl_p90_ms` (`sim/metrics_measured.go`), so it is the
**slow tail** of the per-user rate distribution (aiperf's per-user p10), not its p90.
`interactivity_mean` is the mean of each request's own output tok/s, which is the figure to
compare with aiperf's `output_token_throughput_per_user`.

Records written before this change (every committed report at the time, issue #97) have
`config_key`, which was backfilled, but no `measured` or `result`; those need a rerun.
