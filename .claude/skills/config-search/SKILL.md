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
  as `./blis`. The kernel backend is selected by the scenario's `coefficients:` list, **not** a
  `--latency-model` flag (v0.10.0 removed that flag; passing it errors "unknown flag").
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

1. **Enumerate feasible topologies within budget.** Use **`scripts/topologies.py`**
   (`candidate_topologies(gpu_budget, model_entry)`) -- the same enumeration the automated
   `run_search.py` path uses, so the two cannot drift. Valid `tp` (divides attention/KV heads and,
   for MoE, expert count), instance counts with `tp x dp x instances <= gpu_budget`, optional PD and
   EP shapes. Each per-instance topology (tp/pp/dp/ep + engine knobs) becomes a **generated
   scenario**; instance count stays a CLI knob. Cheap arithmetic feasibility only; **BLIS stays the
   arbiter** -- a shape BLIS refuses is dropped with its error recorded in `pareto.json`'s
   `dropped[]` (never silently absent -- a lone survivor must not look like the whole space). (A
   vendored capacity pre-filter is a later upgrade; see disc #3.)

   **MoE knob tier (required for any `arch: MoE` model).** Dense models take `tp` alone; an MoE
   model must also cross **`dp`** and **`enable_expert_parallel`**, because:

   - The **expert-parallel width is derived, not declared**: `tp x pcp x dp` (blis-schemas
     `Parallelism.ExpertParallelWidth`). With `pcp` unset that is `tp x dp`, so EP over a width of
     **1 shards nothing** -- skip those candidates instead of burning a run on a no-op.
   - A width of 1 is also **why big MoE models refuse to load**: deepseek-v4-pro at `tp=4` fails with
     *"191.2 GiB of per-rank fixed occupancy on a 141.0 GiB device (tp=4, expert-parallel width 1)"*,
     while `tp=4,dp=2` runs. If narrow-`tp` shapes are dying on memory, the missing axis is the
     **footprint** (`tp x dp`), not `tp`.
   - **Always run the non-EP control** beside each EP arm. On deepseek-v4-pro / 256x H200 every EP
     arm beat its control (`32x TP4xDP2`: 7994.8 vs 6349.2 tok/s at `C_use`), but the comparison is
     the finding -- do not assume the direction, measure it.
   - **Dense models reject both**: `dp=2` on qwen3-14b errors *"states dp 2 for a dense model; DP
     placement is modelled for MoE models"*, so read `arch` from the catalog (never guess from the
     name) and keep the dense grid narrow.
   - `all2all_backend` (`allgather_reducescatter` / `deepep_high_throughput` / `deepep_low_latency`)
     is a real engine field the kernel prices, but it moved throughput ~2% here against EP's ~24% --
     it is **not** in the default grid. Sweep it only when asked; the report carries it as a facet.

   **Engine/routing knob tier (coordinate refinement, `topologies.knob_variants`).** After the
   topology sweep, refine the **top `REFINE_TOP` shapes one knob at a time** -- a full cross
   product (4 routing x 3 batch x 3 seqs x 12 shapes x 12 loads = 5184 runs) buys nothing a
   coordinate sweep does not. Measured on kimi-k2.5 / 512x H200 / multidoc:

   | knob | finding |
   |---|---|
   | `routing_policy` | **the biggest lever.** `least-loaded` lifted `64x TP4xDP2 + EP` from 36.4 to **67.0 tok/s/GPU (+84%)** by raising `C*` from 512 to 1024. `always-busiest` is pathological (-58% tput, ITL 103 ms, TTFT 4.6 s) -- never recommend it, and do not spend runs on it. Valid: `round-robin` (default), `least-loaded`, `weighted`, `always-busiest`. |
   | `max_num_seqs` | binds **downward**: 256 -> 64 cost 29%, -> 16 cost 64%. Raising it above the default changed nothing. |
   | `max_num_batched_tokens` | 8192 -> 2048 raised ITL p99 47%; 32768 changed nothing. |
   | `gpu_memory_utilization` | 0.9 -> 0.7/0.5 starves KV (1270 preemptions at 0.5, -27% tput). |

   The lesson: these knobs **constrain** rather than improve. Probe the constrained side (where a
   real deployment is often forced to sit), not headroom that does not exist.

   **Ranking is tie-tolerant (`rank_finalists`, `TIE_TOLERANCE` 0.5%).** Two configs within half a
   percent on tok/s/GPU are a tie, and the one with **fewer knobs set** wins. Without this,
   qwen3-14b reported `32x TP8 · seq64` beating plain `32x TP8` by 0.3 tok/s out of 39,801
   (0.0008% -- noise), implying `max_num_seqs=64` was the reason. A recommendation should only
   carry a knob that earned its place.

   **The label is topology; knob deltas are a separate field.** `topologies.label()` returns
   instances x TP x DP x EP only -- what dominates an LLM serving decision and what a reader scans
   down a column. Knobs go in **`variant`** (`topologies.knob_phrase` /
   `report._disambiguate_series`), spelled out in words: *"batch tokens 2,048"*, never an invented
   `b2k`. Squeezing them into the label produced
   `64x TP4xDP2 + EP · ll · b8k · seq256 · mem0.9 · a2a=ep`, which is why they are separated.
   `variant` is scoped to the topology group and measured against that group's baseline, so an
   unrefined shape and the all-defaults row both get `""`.

   The report then surfaces knobs where they belong: **tiered facets** (`tier:'primary'` --
   sessions / configuration / knob delta / GPUs / TP / DP / EP width / instances -- always
   visible; `tier:'detail'` -- routing, scheduler, admission, batch tokens, max seqs, block size,
   gpu-mem-util -- behind a "show N more filters" disclosure, with any filter set there staying
   applied while collapsed), the hover card's knob block, and the click panel's **Routing &
   policy** section.

   **Everywhere a config is NAMED, it is named `label (variant)`.** A bare label does not identify
   a config once knobs are swept -- five finalists read `64x TP2xDP4 + EP` with nothing to tell
   them apart. So: `pareto.json`'s `winner.chosen` (plus explicit `label`/`variant` fields),
   `winner.reason` (which states the knobs or says "at default engine and routing settings"), the
   gallery's `winner.name`/`finalists[].name`, and the web page's recommendation and
   alternative cards. Two consumers matched on the bare label and broke when `chosen` gained its
   variant -- `build_gallery` silently showed the LAST finalist as winner, and the detail page
   dropped every same-topology variant from its runners-up -- so both now match on the
   **(label, variant) pair**.

   **The operating table ("What to run") must state the whole deployment.** It is the table a
   reader copies flags from, so an omission is a wrong deployment: it printed only
   `--tensor-parallel-size` and `--num-instances`, leaving out `--enable-expert-parallel`,
   `--data-parallel-size` and routing. The flag list and the skip list come from the shared lexicon
   (`report/lexicon.json`), which covers every knob the search can vary and drops the ones sitting
   at their default so the two or three that matter stay visible -- and means the dashboard prints
   the same deployment for the same run.

   **It ranks on tok/s per GPU, and says so.** One row per load, the most efficient *servable*
   run there, ties broken on fleet tok/s then per-user rate then run id. Per-GPU is the efficiency
   half of the dual-Pareto objective the search picks its own winner on, so the table, the
   recommendation card and the winner all agree. With `runs.jsonl` present the rows are **derived**
   from the records; a brain-authored `operating_points` list is a fallback for a run dir with no
   usable ledger (an authored list carried its own rows and its own metrics, which is how the two
   surfaces drifted apart in the first place).
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
   `prefill_instances`/`decode_instances`), pipeline parallel (`pp`>1), speculative decoding, and
   chunked-prefill `long_prefill_token_threshold`. **Precision is never defaulted**: set
   `quantization`/`cache_dtype` explicitly (GLM-5.3-Flash is fp8); unset lets the checkpoint govern
   (e.g. qwen3-14b stays bf16).

   **Multi-node instances ARE supported**, and necessary: the biggest MoE models do not fit in one
   node at any `tp`. `tp*dp` may exceed 8 as long as it fills **whole nodes** (a straddling
   footprint like 12 leaves half a node idle and misprices the cross-node term), and the generated
   scenario names a `fabric` automatically once the pool spans nodes. The automated search starts
   at one node and widens a node-doubling at a time **only when nothing single-node is servable**,
   so a model that fits never pays the fabric term. Measured footprint floors on H200:
   glm-5.2 and inkling need 2 nodes, kimi-k3 needs 4 (332.3 GiB/rank at tp=8 on a 141 GiB device),
   nemotron-3-ultra-550b needs 8.

   **`block_size` is model-dependent -- never hardcode it.** The kernel adapter requires a positive
   value ("deployment states no block_size"), but a DSA sparse-MLA model (glm-5, glm-5.2, glm-5.3,
   glm-5.3-flash, deepseek-v4-pro) runs a 64-token indexer and refuses a stated 16. `run_blis.py`
   fills it per model (`default_block_size`, reading `sparse_mla` out of the catalog graph): 64 for
   DSA, 16 otherwise. A stated value always wins, so it stays a searchable knob.

   A model whose catalog entry has **no `graph.yaml`** cannot run at all (granite-5.0-230b-base,
   sarvam-105b) -- that is an upstream blis-catalog gap, fixed by its `derive_graph.py`, not here.

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
  `--scenarios`, then runs `./blis run --scenario <generated>
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

The explorer is **two-part cross-filtering**, both halves driving one shared selection:

- **"Filter by outcome: brush a range"** -- a row of brushable histograms over the continuous
  metrics (`pu` speed per user, `tpg` output per GPU, `ttft`, `itl`). Bins and scales are fixed to
  the full record set so a bar never moves as filters change; only its highlight does.
- **"Filter by setting: click a level"** -- the categorical knob facets (`conc`, `tp`, `dp`, `epw`,
  `a2a`, `mnbt`, `mns`, `route`, `pd`, `ok`, ...) with live counts that respect every *other*
  filter. A knob the search never varied **auto-hides** (the `varies()` guard), so a dense run set
  shows no EP facets rather than a dead single-value column.

A brushed range excludes runs whose metric was never measured (a null never satisfies a range),
which is correct -- you cannot select on an outcome that was not measured -- and the empty-state
banner says so.

```
python3 -I scripts/report.py runs/<run-id>              # -> runs/<run-id>/index.html (next to the data)
python3 -I scripts/report.py runs/<run-id> --out p.html # custom output path
python3 -I scripts/report.py runs/                      # a parent of several run dirs -> each report + a card index
```

Inputs (all optional-tolerant -- three of four are brain-authored):

- **`scenario.json` / `pareto.json` / `runs.jsonl`** -- the search outputs. `report.py` maps each
  run into a flat **`RECORDS`** row (short keys: `conc,tp,dp,epw,ni,gpus,series,pu,cl,tpg,itl,ttft,
  p95,a2a,ok,viol…`; `tpg` = output per GPU from the *unrounded* throughput, `epw` = the derived
  expert-parallel width or null when EP is off) and
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
  which figure each shows, run-specific caption notes, the glossary). Documented in
  `scripts/report_schema.md`. **Absent -> the report still renders** the header, the explorer, and an
  auto operating table, with no analysis prose. Note what it does **not** carry: metric names, axis
  titles, a figure caption's leading sentence and the "What to run" ranking rule all come from the
  shared lexicon below, so the narrative cannot re-word them for one surface only.

Pieces:

- **`report/lexicon.json` + `scripts/lexicon.py`** -- the **shared vocabulary of both report
  surfaces**. config-search renders the same runs twice: this standalone HTML report and the React
  dashboard in `web/`. They had drifted into two vocabularies for one set of numbers and two
  ranking objectives for "What to run" -- the report ranked operating points by per-user rate while
  the dashboard ranked them by tok/s per GPU, so at high load the two tables recommended
  *different* configurations for one scenario (issue #13). The lexicon fixes the objective (tok/s
  per GPU, with explicit tiebreaks, plus the blurb that states it), every metric/facet label, unit
  and definition, the axis titles, each figure's caption, and the deploy-flag list the operating
  table prints. `report.py` and `figures.py` read it through `lexicon.py`, and `report.py` bakes
  the whole file into the page so the explorer's JS labels come from it too. **`web/src/lexicon.json`
  is a byte-identical mirror** (Vite cannot import out of this read-only skill dir); change the copy
  here, then copy it over the mirror -- `scripts/test_build.py` fails the build on drift.
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
