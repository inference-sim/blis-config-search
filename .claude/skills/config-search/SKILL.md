---
name: config-search
description: >
  Use when choosing or tuning a BLIS serving deployment for a given model + GPU budget + workload --
  which tp / num-instances / PD / EP and engine knobs maximize throughput while holding an
  interactivity floor. Triggers: "what config should I run", "best config for X on N GPUs",
  right-sizing or capacity-planning a cluster in the simulator, comparing serving topologies,
  finding Pareto-optimal serving configs. Simulation only (BLIS), not a live cluster.
---

# config-search

## Overview

Drive BLIS many times to find the deployment config(s) that get the most out of a GPU budget while
staying interactive. **Claude is the search brain**: it enumerates candidate shapes, probes each to
its interactivity limit, reads the binding bottleneck, and steers knobs with *explained* moves. It
writes a run directory that the report step (added later) turns into an HTML report.

**Everything below is a default the user can override on any run** -- the objective, the floor, the
knobs searched, the budget. That flexibility is the point of this skill.

## When to use / not

- **Use when:** picking a serving config; right-sizing a cluster to a budget; "best config for this
  model on N GPUs"; comparing topologies (TP vs replicas vs PD vs EP); capacity planning -- all in
  simulation.
- **Not for:** a real cluster; a single known config (just run `blis run` directly).

## Pinned stack (required)

The search is priced by the **`blis-latency-kernel`** backend -- the accurate per-operator model, the
reason to run BLIS at all here. It reads the deployment from a **scenario** + a **registry**, so:

- **BLIS** release **`v0.10.0`** (commit `0d092d4`; kernel `v0.1.0`, pinned in its `go.mod`), built
  as `./blis`. Always run with `--latency-model blis-latency-kernel`.
- **blis-catalog @ `0.2.1`** (`--catalog`/`BLIS_CATALOG`); a model runs **iff** it is in the catalog
  (NS-6).
- **blis-registry @ `v0.1.1`** (`--registry`) -- holds the 6 coefficient sets a scenario names,
  including `cost-model-memory` (older clones lack it; use the `v0.1.1` tag).
- A **scenario file** states model, hardware, tp/pp/dp/ep and engine knobs. The kernel **refuses**
  `--model/--hardware/--tp/--dp/--enable-expert-parallel` on the CLI; the runner writes them into a
  generated scenario instead (see "Running BLIS").

See `SETUP.md` for the exact build + pins. The example models (`glm-5.3-flash`, `qwen3-14b`) ship in
catalog `0.2.1`.

## Invocation

```
/config-search <natural-language scenario>
/config-search --scenario path/to/scenario.yaml
```

Claude resolves the request into a **Scenario**, echoes it back, and runs **only after the user
confirms**. Natural language is enough; a `scenario.yaml` is for reproducibility / power users.

## Scenario (the input Claude resolves)

| Field | Meaning | Default |
|---|---|---|
| `model` | catalog short-name | (required) |
| `hardware` + `gpu_budget` | GPU type + how many are available | (required) |
| `workload` | named preset, or ISL/OSL + arrival shape | (required) |
| `objective` | what "best" means | **dual Pareto**: maximize output tok/s **and** output tok/s per GPU |
| `interactivity_floor` | per-user rate the winner must hold | **>= 50 tok/s/user == ITL p99 <= 20 ms** |
| `headroom` h | spike buffer; size to usable capacity `C_use = h x C*` | **0.8** |
| `slos` | optional p99 TTFT / E2E caps | none |
| `space` | knob overrides (knobs + candidate values + constraints) | the default knob tiers below |
| `budget` | max BLIS runs and/or wall-clock | sensible small default; ask if ambiguous |
| `max_parallel` | worker cap | **host-adaptive** (see Parallelism) |

Load axis is **concurrency** (closed-loop virtual users, `--concurrency`), not open-loop `--rate`.

## The search (what Claude does) -- adaptive diagnostic, v1

1. **Enumerate feasible topologies within budget.** Valid `tp` (divides attention/KV heads and, for
   MoE, expert count), instance counts with `tp x instances <= gpu_budget`, optional PD and EP shapes.
   Each per-instance topology (tp/pp/dp/ep + engine knobs) becomes a **generated scenario**; instance
   count stays a CLI knob. Cheap arithmetic feasibility only; **BLIS stays the arbiter** -- a shape
   BLIS refuses is dropped with its error recorded. (A vendored capacity pre-filter is a later
   upgrade; see disc #3.)
2. **Capacity probe to C\*.** For each candidate, probe `--concurrency` upward then **bisect** to
   `C*` = the largest concurrency whose p99s still meet the interactivity floor (and any SLOs).
   ~5-8 runs per candidate. With no floor/SLO, `C*` = the throughput knee.
3. **Diagnostic steering.** Read the **binding signal** at `C*` and make **one explained knob move**
   (keep it only if `C_use/G` improves), then re-probe. Rules:

   | Binding signal at C* | Likely cause | Move |
   |---|---|---|
   | ITL p99 binds, batch full, KV healthy | decode compute/bandwidth ceiling | accept as ceiling; add decode instances / raise `tp` (spec-decode is NOT supported in v1 -- the runner rejects it) |
   | TTFT / scheduling-delay binds, queue growing | prefill/admission-bound | raise `max_num_batched_tokens` (chunked-prefill `long_prefill_token_threshold` is NOT a kernel field in v1 -- rejected) |
   | `preemption_count` > 0 / KV thrash | KV-starved | `--kv-cache-dtype fp8`; raise `--gpu-memory-utilization`; **check `--block-size-in-tokens`** (effective KV tokens = total_kv_blocks x block_size); lower `--max-num-seqs` |
   | `--max-num-seqs` saturated, tput below ceiling, no KV failures | batch-width limited | raise `--max-num-seqs` / `--max-num-batched-tokens` |
   | cluster shape with an idle pool / imbalance | routing/placement limited | fix prefill:decode ratio; open routing (`--routing-policy`/`--routing-scorers`) |

   Under the kernel, the per-instance knobs above (`block_size`, `max_num_seqs`,
   `max_num_batched_tokens`, `gpu_memory_utilization`, `quantization`/`cache_dtype`, `tp`/`dp`/
   `enable_expert_parallel`) are **engine/parallel fields in the generated scenario**; only
   `scheduler`/`routing_policy`/`routing_scorers`/`admission_policy`/`preemption_policy`,
   `num_instances`, and `concurrency` stay CLI. Idle-pool/imbalance is read from
   `metrics_all_instances` (per-instance), which the aggregate hides.

   The runner **rejects any other candidate key loudly** (recorded as an error row, never silently
   dropped). Not supported yet in v1: PD prefill/decode split (`prefill_tp`/`decode_tp`/
   `prefill_instances`/`decode_instances`), pipeline parallel (`pp`>1), speculative decoding, chunked-
   prefill `long_prefill_token_threshold`, and a multi-node *single* instance (`tp*dp` > 8 on H100 --
   scale out with `num_instances` instead). **Precision is never defaulted**: set `quantization`/
   `cache_dtype` explicitly (GLM-5.3-Flash is fp8); unset lets the checkpoint govern (e.g. qwen3-14b
   stays bf16).

4. **Rank + finalists.** Compute the dual Pareto over **usable capacity** `C_use = h x C*`
   (budget-max output tok/s, and output tok/s per GPU `= tps/G`). Dense-sweep the top finalists'
   concurrency for clean curves, pick the winner, and report its margin + the **move trail** ("why
   this won").

**Determinism (scope it honestly):** every `blis run` is deterministic (INV-6) and the pool writes
`runs.jsonl` in stable input order, so the *data* is reproducible. The end-to-end *recommendation* is
only as reproducible as your search decisions -- in this v1 those are prompt-driven, so make them
reproducible by always deciding the next step from a **stable ordering of the batch** (not
worker-finish order) and recording the move trail. Do not claim a byte-identical recommendation the
way the formal sweeper (with an executable policy) could. **Caching:** hash the config; identical
configs coalesce (runs are exact and free to re-read). **Parallelism:** see below.

## Running BLIS (the runner + the parallel pool)

Two scripts, verified end-to-end against the pinned kernel binary:

- **`python3 -I scripts/run_blis.py`** -- the single source of truth for one run. From a candidate's
  per-instance topology (tp/pp/dp/ep + engine knobs) it **generates a scenario YAML** into
  `--scenarios`, then runs `./blis run --latency-model blis-latency-kernel --scenario <generated>
  --scenarios <dir> --registry <blis-registry>` with the still-CLI knobs (`--num-instances`,
  `--concurrency`, `--workload`, `--num-requests`, `--seed`). It parses the `=== Simulation Metrics
  ===` JSON -- the `cluster` aggregate into `metrics` and the per-instance objects into
  `metrics_all_instances` -- flags incomplete/overloaded runs, and records the resolved `deployment`
  (`parallel`/`engine`/`num_instances`/`gpus`/`scenario`). `--defaults-filepath` resolves next to the
  binary so it is cwd-independent. See `--help`.
- **`python3 -I scripts/sweep_pool.py`** -- runs a **batch** of configs in parallel under the
  host-adaptive cap, appends records to `runs.jsonl` in stable input order, prints cap + speedup.

> A first-class `RunResult` document upstream (declarative API, inference-sim#1853 / #1855) would
> remove the JSON-block extraction.

## Parallelism

**Yes -- the search runs BLIS in parallel**, bounded by a host-adaptive cap that `sweep_pool.py`
computes at runtime:

```
cap = min(core_cap, memory_cap)        # whichever is tighter; --max-parallel overrides
core_cap   = floor(available_cpus * util_fraction)        # sched_getaffinity, respects cgroup/affinity
memory_cap = floor(MemAvailable * mem_budget_fraction / peak_rss_per_run)   # MemAvailable at runtime
```

from the host's **available** (not installed) cpu/ram -- never assuming we own the box. BLIS is ~1
core and ~0.1-0.6 GiB RSS per run (conservative default 1 GiB), so a big dedicated host binds on
cores and a small / shared / containerized one binds on memory. Verified: 24-core/250 GiB host ->
cap 19 (cores), ~7x speedup on an 8-config batch; `--max-parallel 2` -> cap 2. **Non-Linux note:**
without `/proc/meminfo` (e.g. macOS) free memory is unknown, so the cap falls back to CPU-only and
`sweep_pool.py` warns -- pass `--max-parallel` there to bound RAM.

The adaptive steps (bisection, coordinate-ascent) are sequential, so the width comes from independent
work: many candidate topologies, the finalists' dense sweeps, k-ary boundary probes. Baseline is
**synchronous wide-batch rounds** -- emit a wide batch, run it through the pool, decide the next. Keep
the search decision order **stable** (sort returned results by input order, not finish order) so the
recommendation stays reproducible despite parallelism.

## The data dump (seam to the report step)

A run directory, **convention-based, not a frozen schema** (see `examples/sample-run/`). Today the
scripts write `runs.jsonl`; **you (the search brain) write `scenario.json` and `pareto.json`** from
the runs. (Folding that assembly into a one-command driver is the open follow-up.)

```
runs/<run-id>/
  scenario.json   # resolved Scenario (model, hw, budget, workload, objective, floor, headroom, slos, pinned_stack)  -- brain
  runs.jsonl      # one row per blis run: {config, concurrency, deployment{parallel,engine,num_instances,gpus,scenario},
                  #   metrics (cluster), metrics_all_instances[], error, phase}  -- run_blis/sweep_pool
  pareto.json     # finalists{c_star,c_use,at_c_use,curve[]}, dropped[], winner{reason,move_trail}  -- brain
```

`curve[]` carries per point: `concurrency`, `tokens_per_sec`, `itl_p99_ms`, `ttft_p99_ms`,
`scheduling_delay_p99_ms`, `preemption_count`, and **`tok_per_s_per_user`**. The last is written by
the brain, not derived as `1000/itl_p99_ms`: BLIS appends ITL per decode *step*, so under
speculative decoding one entry covers several tokens and the naive axis understates the per-user rate.

## Report step

The report half of this skill (disc #4, report track) reads the run directory and emits a
**magazine-style analytical report** modeled on the CurveBender "glm53flash" format: a numbered-finding
narrative, pre-rendered inline-SVG figures, a client-side **crossfilter explorer** over every run, and
a glossary. It **never runs BLIS** at generation time -- it is a pure `run-dir -> index.html`
transform, so a report is regenerable from the committed data alone.

```
python3 -I scripts/report.py runs/<run-id>              # -> runs/<run-id>/index.html (next to the data)
python3 -I scripts/report.py runs/<run-id> --out p.html # custom output path
python3 -I scripts/report.py runs/                      # a parent of several run dirs -> each report + a card index
```

Inputs (all optional-tolerant -- three of four are brain-authored):

- **`scenario.json` / `pareto.json` / `runs.jsonl`** -- the search outputs. `report.py` maps each
  run into a flat **`RECORDS`** row (short keys: `conc,tp,ni,series,pu,cl,itl,ttft,p95,ok,viol…`) and
  classifies each run servable/not against the interactivity floor + SLOs (a declared SLO whose
  metric is absent from both the cluster aggregate and every instance counts as a *violation*, not a
  silent pass). `pu` (per-user tok/s) prefers the engine's authoritative `tok_per_s_per_user` field;
  it falls back to `1000 / ITL p99` only when that field is absent **and** one decode step equals
  one token (`mtp == 0`). The p99 basis is deliberate: it matches `pareto.json`'s `tok_per_s_per_user`
  and the interactivity floor (p99 <= 20 ms), so the page never shows two incompatible per-user
  numbers for a point. Under speculative decoding with no authoritative field, ITL p99 is a
  per-*step* interval that would undercount the per-token rate, so `pu` is left **null** (flagged
  `pu_is_step_estimate`) rather than publishing a known-wrong number. It also computes a **`FACTS`**
  stats object the prose cites.
- **`report.json`** -- the **brain-authored narrative** (standfirst, hero panel, numbered sections +
  which figure each shows, the operating-point table, the glossary). Documented in
  `scripts/report_schema.md`. **Absent -> the report still renders** the header, the explorer, and an
  auto operating table, with no analysis prose.

Pieces:

- **`scripts/figures.py`** -- deterministic inline-SVG figure builders (`operating_curve`,
  `scale_bars`, `tail_scatter`, …), themed via the stylesheet's `--fig-*` variables. Each is
  registered with the `keys` (RECORDS fields) its story depends on; `report.render_figure` checks
  that every keyed field actually **varies** across the runs and otherwise returns a captioned
  placeholder. So a figure keyed on a knob the search never swept (e.g. a routing figure, `keys=
  ("route",)`) placeholders honestly. The three builders that ship are all keyed on fields the v1
  engine does vary (concurrency / topology), so they draw. Routing/PD/spec-decode figures are **not
  shipped yet** -- `figures.py` carries them only as commented-out registry examples; add the entry
  (with the right `keys`) when the matching builder lands, and the variation check placeholders it
  until the search sweeps that knob.
- **`report/template.html`** -- the magazine shell (reference CSS, `__BODY__` mount point, the
  explorer JS). The report is **fully self-contained / offline**: `report.py` inlines
  `report/vendor/d3.min.js` + `crossfilter.min.js` (the explorer's only JS deps) into the HTML, so
  there is no CDN or any other network dependency at view time. The article and baked inline-SVG
  figures need no JS at all; the explorer additionally guards on `typeof d3 === "undefined"` so even
  a stripped vendor block degrades to the static article. (A sibling `report/index-template.html` is
  the card-index landing page for the multi-run case.)

Determinism (INV-6): a pure function of the run-directory **contents** -- identical data renders
byte-identical HTML regardless of where the directory lives or what it is named
(`json.dumps(..., sort_keys=True)`; the inlined vendor libs are fixed bytes; the footer identifies the run by the
`run_id` carried inside `scenario.json`/`pareto.json`, not by the directory path or name). A run
directory with no `run_id` in its data shows "unidentified" in the footer -- it never falls back to
the directory basename, so the output stays independent of location even then. `examples/sample-run/` is the
committed fixture (now including a worked `report.json`); `scripts/test_report.py` covers the records
mapping, FACTS, each figure, report.json tolerance, and the seam-consistency checks.

## Examples

See `README.md`. Headline example: **GLM-5.3-Flash on 16x H100, ShareGPT chatbot, floor 50 tok/s/user**
-> best config within budget (reproduces the real 2x TP8+EP shape as one candidate).

## Common mistakes

- **Carrying over the Nous campaign's numbers** (block_size thresholds, Pareto densities, pop sizes):
  they are from an older latency model and were never validated against real clusters. Re-measure on
  the pinned stack. Keep `block_size` as a *knob*; drop the stale *values*.
- **Treating `--total-kv-blocks` as the lever** instead of `--gpu-memory-utilization` -- in a real
  deployment KV capacity is derived from GPU memory + model/hardware, not dialed directly.
- **Sizing to the edge** -- recommend `C_use = h x C*`, not `C*`.
- **Letting parallelism make the search non-deterministic** -- always decide from a stable batch order.
- **Trusting a prediction as ground truth** -- label outputs as simulator predictions; where real data
  exists (e.g. the curvebender campaign reports), spot-check against it.
