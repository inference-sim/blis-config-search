# config-search

Find the best BLIS serving configuration for a model, a GPU budget, and a workload -- then turn the
results into a magazine-style interactive HTML report. It drives the simulator many times, steering
toward configs that **maximize throughput while staying interactive** (>= 50 tok/s per user by
default), writes a run directory you can inspect or re-plot without re-running anything, and renders
that directory into a shareable analytical report (numbered findings, inline-SVG figures, a crossfilter
explorer over every run, and a glossary).

Both halves of the skill are present: the **sweeper / engine** (runs BLIS, writes the run directory)
and the **report** (`scripts/report.py`, reads the same run directory, writes `index.html` -- it
never runs BLIS). They meet only at the run-directory format; see `examples/sample-run/` for its shape.

## What "best" means (all overridable)

By default the search reports a **dual Pareto view**:

- **Budget-max** -- the most output tok/s you can get on the GPUs you have, and
- **Efficiency** -- output tok/s per GPU,

both subject to an **interactivity floor** of **>= 50 tok/s per user** (ITL p99 <= 20 ms) with a
**headroom buffer** for load spikes. You can retarget the floor, add TTFT/E2E SLOs, change the
headroom, or ask for a different headline -- just say so when you invoke it.

## Prerequisites

See `SETUP.md`. In short: build BLIS release `v0.10.0` (@ `0d092d4`) as `./blis` (runs on the accurate
`blis-latency-kernel` backend), and point `BLIS_CATALOG` at blis-catalog `0.2.1` and `BLIS_REGISTRY`
at blis-registry `v0.1.1`. All example models (`glm-5.3-flash`, `qwen3-14b`) already ship in `0.2.1`.

## Use it

```
/config-search find the best config for GLM-5.3-Flash on 16 H100s for a ShareGPT chatbot,
                keep each user above 50 tok/s
```

Claude will restate the scenario it understood (model, hardware, budget, workload, objective, floor,
headroom), wait for your OK, then search and write a run directory under `runs/<run-id>/`.

For a repeatable run, hand it a scenario file instead:

```
/config-search --scenario examples/scenario-glm53-flash.yaml
```

## Examples

| Scenario | What it shows |
|---|---|
| **GLM-5.3-Flash, 16x H100, chatbot, floor 50 tok/s/user** | the headline: best config within budget; reproduces the real 2x TP8+EP shape as one candidate. `examples/scenario-glm53-flash.yaml` |
| **GLM-5.3-Flash, 16x H100, synthetic ISL 8000 / OSL 512** | long-context behavior; how the winner shifts when prefill dominates |
| **qwen3-14b, H100, chatbot** | a fast smoke run (small model already in the catalog) to sanity-check the whole loop |

BLIS workload presets are `chatbot`, `summarization`, `contentgen`, `multidoc`, `distribution`
(a ShareGPT-style chatbot maps to `chatbot`). Example cluster sizes and ISL/OSL (synthetic ISL 8000 /
OSL 512, concurrency ladders, H100/H200) are drawn from the real llm-d campaigns in
the internal calibration reports.

## Output

```
runs/<run-id>/
  scenario.json   # exactly what was searched (+ pinned_stack, pinned workload)
  runs.jsonl      # every kernel run: config, resolved deployment, metrics (cluster) + metrics_all_instances[]
  pareto.json     # finalists (c_star/c_use/enriched curve[]), dropped topologies, chosen winner + move trail
  report.json     # OPTIONAL brain-authored narrative for the report (see scripts/report_schema.md)
```

How it's produced today: `scripts/run_blis.py` + `scripts/sweep_pool.py` price configs on the kernel
(generating a scenario per per-instance topology) and write `runs.jsonl`; the **search brain (Claude,
per `SKILL.md`)** decides which configs to run, assembles `scenario.json`, computes/writes
`pareto.json`, and writes the `report.json` narrative. A single-command driver that emits them is the
TODO in Status.

Then render it:

```
python3 -I scripts/report.py runs/<run-id>     # -> runs/<run-id>/index.html
python3 -I scripts/report.py runs/             # several run dirs -> each report + a card index
```

`scripts/report.py` is a pure `run-dir -> index.html` transform (it never runs BLIS): a magazine-style
analytical report with a numbered-finding narrative, pre-rendered inline-SVG figures
(`scripts/figures.py`), a crossfilter explorer over every run, and a glossary. The prose comes from
`report.json` (absent -> header + explorer + an auto operating table still render). The report is
fully self-contained and renders offline: d3 + crossfilter (the explorer's only JS deps) are inlined
from `report/vendor/`, so there is no network dependency at view time. See `SKILL.md` and
`scripts/report_schema.md`.

Reproducibility: each `blis run` is deterministic (INV-6) and the pool writes `runs.jsonl` in a
stable order, so the *data* is reproducible. The end-to-end *recommendation* is only as reproducible
as the search policy, which in this v1 is prompt-driven -- so treat it as a strong default, not a
byte-identical guarantee. Re-plot or re-analyze from the run directory without touching BLIS again.

## How it runs in parallel

The search spends spare cores: `scripts/sweep_pool.py` runs a batch of configs concurrently, capped
at `min(core_cap, memory_cap)` computed from the host's **available** CPUs and free memory at runtime
(`--max-parallel` overrides). On a 24-core / 250 GiB host that is 19 parallel BLIS runs; on a small or
containerized host the memory cap binds instead. `scripts/run_blis.py` runs a single config and is the
one place that invokes BLIS.

## Status

First engine version, **verified end-to-end on the kernel backend**: `examples/sample-run/` is a real
kernel mini-search (GLM-5.3-Flash, 16x H100) -- topologies priced via generated scenarios, bisected to
C* at the interactivity floor, winner `4x TP4 + EP`. The runner parses the JSON metrics and
per-instance objects, the parallel pool hits ~4-14x, and the unit suite passes. The **report step is
implemented** (`scripts/report.py` + `scripts/figures.py` + `report/template.html`): it renders
`examples/sample-run/` into a magazine-style `index.html` -- numbered findings, inline-SVG figures
(operating curve, scale bars, tail scatter), a crossfilter explorer, and a glossary -- from the
committed `report.json` narrative; `scripts/test_report.py` covers the records/FACTS mapping, each
figure, report.json tolerance, and the seam-consistency checks.
Caveat: outputs are simulator **predictions** (the kernel prices MLA/KDA from fitted coefficients) --
spot-check against the real llm-d measurements in the internal calibration reports before acting. Still to
do: tune the steering rules against that real data, and wrap the capacity-probe/finalist driver as one
command (which would also assemble `curve[]` in code rather than by the brain).
