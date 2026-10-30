# `report.json` — the brain-authored narrative seam

`report.py` renders a magazine-style analytical report (modeled on the internal "glm53flash" reference
format) from a run directory. Three of its four inputs already exist: `scenario.json`, `pareto.json`,
`runs.jsonl`. **`report.json` is the fourth** — the *narrative* Claude (the search brain) writes after
reading the runs. It carries the prose, the hero panel, the numbered findings, the figures each
section shows, the operating-point table, and the glossary.

It is **optional**. Without it, `report.py` still renders the header, the crossfilter explorer over
every run, and an auto operating-point table — just without the analysis. With it, you get the full
article. The reader is **defensive**: every field below degrades gracefully if absent.

The report is a pure function of the run directory, so keep the prose grounded in the run's actual
numbers (cite `pareto.json` / `runs.jsonl`). Prose strings may contain inline HTML, including glossary
links `<a class="gl" href="#g-ID" data-def="short definition">term</a>` — the `data-def` gives a
CSS-only hover tooltip, and `href="#g-ID"` jumps to the matching glossary entry.

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

config-search renders the same runs on **two** surfaces: this standalone report and the React
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

`scripts/lexicon.py` is the Python reader (`report.py` and `figures.py` import it, and `report.py`
bakes the whole file into the page so the explorer's JS labels match too). `web/src/lexicon.json`
is a **byte-identical mirror** — Vite cannot import out of the read-only skill dir — and
`scripts/test_build.py` fails the build if the two drift. Change the skill's copy, then copy it
over the mirror.

**So:** do not write a metric name, an axis title, a figure caption's leading sentence, or a
ranking rule into `report.json`. Add it to the lexicon (both copies) and let both surfaces read it.

## Figures

`figure` names a builder registered in `scripts/figures.py`'s `FIGURES` map. Each entry is
`{"builder": fn, "keys": (RECORDS field(s) the figure's story depends on)}`. `report.render_figure`
first checks that **every** `keys` field actually varies across the runs; if one does not (the search
never swept it), it returns a captioned placeholder instead of a misleading flat chart. An unknown
figure name likewise placeholders.

Each figure's caption, axis titles and "how to read it" line come from the lexicon's `figures`
block, so the dashboard's charts of the same story are worded identically. A section supplies only
its `caption_note`.

| name | what it draws | keys (must vary) |
|---|---|---|
| `operating_curve` | tok/s per user vs cluster tok/s, one line per config, concurrency-labelled; filled = servable | `conc` |
| `scale_bars` | max servable concurrency per config, with the first-break marker | `series` |
| `tail_scatter` | tok/s per user vs p95 latency; crosses = unservable | `conc` |

Builders for knobs the v1 engine does not yet vary (routing, PD split, speculative decoding, dispatch
collective) are **not shipped**; `figures.py` lists them only as commented-out registry examples.
When such a builder lands, add its `FIGURES` entry with the appropriate `keys` (e.g. `("route",)`);
until the search sweeps that knob the variation check renders the placeholder automatically.

## `RECORDS` field reference (what the explorer + figures consume)

Assembled in `report.build_records`. Short keys match the reference report so the explorer JS is
unchanged. Knobs the engine doesn't vary come through constant (e.g. `route: "default"`, `mtp: 0`) and
the explorer automatically hides single-value dimensions.

`id, src, conc, tp, dp, ni, gpus, series, route, pd, mtp, mnbt, mns, a2a, epw,`
`pu (per-user tok/s: authoritative tok_per_s_per_user if present; else 1000/ITL-p99 when one step`
`   == one token (mtp==0) -- the p99 basis matches the floor + pareto; else null, never a step rate),`
`pu_is_step_estimate (True = under MTP with no authoritative field, so pu is null not a wrong number),`
`cl (=tokens_per_sec), itl (=mean ITL ms), ttft (=ttft_p99_ms),`
`hit (prefix hit or null), to (timed out), drop (unservable), p95 (=e2e_p95_ms),`
`ok (met completion+floor+SLOs; absent SLO metric = violation), status, viol[] (reasons when not ok)`
