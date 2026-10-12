# `report.json` — the brain-authored narrative seam

`report.py` renders a magazine-style analytical report (modeled on the CurveBender "glm53flash"
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
      "caption": "<b>Figure 1.</b> …",  // figcaption HTML
      "prose": ["paragraph one…", "paragraph two…"]
    }
  ],

  // "What to run" section (number defaults to 9). Prose + an operating-point table.
  "recommendation_num": 9,
  "recommendation_prose": ["…", "…"],
  "operating_points": [
    // One row per chosen operating point. `flags` keys are read in a fixed order into the
    // monospace flag list: tp, ni, mnbt, mns, mtp, route, pd (any subset).
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

## Figures

`figure` names a builder registered in `scripts/figures.py`'s `FIGURES` map. Each entry is
`{"builder": fn, "keys": (RECORDS field(s) the figure's story depends on)}`. `report.render_figure`
first checks that **every** `keys` field actually varies across the runs; if one does not (the search
never swept it), it returns a captioned placeholder instead of a misleading flat chart. An unknown
figure name likewise placeholders.

| name | what it draws | keys (must vary) |
|---|---|---|
| `operating_curve` | per-user tok/s vs cluster tok/s, one line per config, concurrency-labelled; filled = servable | `conc` |
| `scale_bars` | max servable concurrency per config, with the first-break marker | `series` |
| `tail_scatter` | per-user tok/s vs p95 latency; crosses = unservable | `conc` |

Builders for knobs the v1 engine does not yet vary (routing, PD split, speculative decoding, dispatch
collective) are **not shipped**; `figures.py` lists them only as commented-out registry examples.
When such a builder lands, add its `FIGURES` entry with the appropriate `keys` (e.g. `("route",)`);
until the search sweeps that knob the variation check renders the placeholder automatically.

## `RECORDS` field reference (what the explorer + figures consume)

Assembled in `report.build_records`. Short keys match the reference report so the explorer JS is
unchanged. Knobs the engine doesn't vary come through constant (e.g. `route: "default"`, `mtp: 0`) and
the explorer automatically hides single-value dimensions.

`id, src, conc, tp, dp, ni, gpus, series, route, pd, mtp, mnbt, mns, a2a, epw,`
`pu (per-user tok/s: authoritative tok_per_s_per_user if present; else 1000/mean-ITL when one step`
`   == one token (mtp==0); else null -- never a known-wrong step rate),`
`pu_is_step_estimate (True = under MTP with no authoritative field, so pu is null not a wrong number),`
`cl (=tokens_per_sec), itl (=mean ITL ms), ttft (=ttft_p99_ms),`
`hit (prefix hit or null), to (timed out), drop (unservable), p95 (=e2e_p95_ms),`
`ok (met completion+floor+SLOs; absent SLO metric = violation), status, viol[] (reasons when not ok)`
