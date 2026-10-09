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

- **BLIS** at the `kernel-exclusive` branch (PR #1851), built as `./blis` -- the kernel-exclusive
  latency model is the most accurate we have, and a search is only as good as its latency model.
- **blis-catalog @ `0.2.1`**, **blis-registry @ `v0.1.1`**, located via `--catalog` / `BLIS_CATALOG`.
- A model runs **iff** it is in the catalog (NS-6). No HuggingFace fetch at run time.

See `SETUP.md` for building the pinned binary. The example models (`glm-5.3-flash`, `qwen3-14b`) ship
in catalog `0.2.1` -- no overlay needed.

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
   Cheap arithmetic feasibility only (weights + minimum KV must fit); **BLIS stays the arbiter** -- a
   shape BLIS refuses at startup is dropped with its error recorded. (A vendored capacity pre-filter
   is a later upgrade; see disc #3.)
2. **Capacity probe to C\*.** For each candidate, probe `--concurrency` upward then **bisect** to
   `C*` = the largest concurrency whose p99s still meet the interactivity floor (and any SLOs).
   ~5-8 runs per candidate. With no floor/SLO, `C*` = the throughput knee.
3. **Diagnostic steering.** Read the **binding signal** at `C*` and make **one explained knob move**
   (keep it only if `C_use/G` improves), then re-probe. Rules:

   | Binding signal at C* | Likely cause | Move |
   |---|---|---|
   | ITL p99 binds, batch full, KV healthy | decode compute/bandwidth ceiling | accept as ceiling; more decode TP/instances, or spec-decode |
   | TTFT / scheduling-delay binds, queue growing | prefill/admission-bound | chunked prefill (`--long-prefill-token-threshold`), raise `--max-num-batched-tokens` |
   | `preemption_count` > 0 / KV thrash | KV-starved | `--kv-cache-dtype fp8`; raise `--gpu-memory-utilization`; **check `--block-size-in-tokens`** (effective KV tokens = total_kv_blocks x block_size); lower `--max-num-seqs` |
   | `--max-num-seqs` saturated, tput below ceiling, no KV failures | batch-width limited | raise `--max-num-seqs` / `--max-num-batched-tokens` |
   | cluster shape with an idle pool / imbalance | routing/placement limited | fix prefill:decode ratio; open routing (`--routing-policy`/`--routing-scorers`) |

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

Two scripts, verified against the pinned pr-1851 binary:

- **`python3 -I scripts/run_blis.py`** -- the single source of truth for invoking BLIS and recording
  **one** run. Maps Scenario knobs to real pr-1851 flags, runs `./blis run`, and parses the metrics.
  `blis run` prints a JSON object under a `=== Simulation Metrics ===` header; the runner extracts
  the `cluster` object (fields: `tokens_per_sec`, `responses_per_sec`, `ttft_p99_ms`, `itl_p99_ms`,
  `e2e_p99_ms`, `scheduling_delay_p99_ms`, `preemption_count`, ...). It passes `--defaults-filepath`
  (resolved next to the binary) so it does not depend on cwd. See `--help`.
- **`python3 -I scripts/sweep_pool.py`** -- runs a **batch** of configs in parallel under the
  host-adaptive cap, appends all records to `runs.jsonl` in stable input order, and prints the cap +
  speedup. This is how the search spends surplus cores (see Parallelism).

> A first-class `RunResult` document upstream (declarative API, inference-sim#1853 / #1855) would
> remove even the JSON-block extraction.

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
  scenario.json   # the resolved Scenario (model, hw, budget, workload, objective, floor, headroom, slos)  -- written by the brain
  runs.jsonl      # one row per blis run: {config, concurrency, metrics, diagnostics, binding_signal, config_hash, phase}  -- written by run_blis/sweep_pool
  pareto.json     # finalists, dual-Pareto points, chosen winner, margins, explained move trail  -- written by the brain
```

## Report step (added later)

The report half of this skill (disc #4, report track) reads the run directory and emits a
self-contained interactive HTML report. **Not implemented here yet** -- `examples/sample-run/` is the
committed fixture that defines the seam so the report work can proceed independently and be added on.

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
