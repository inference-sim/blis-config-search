# config-search

Find the best BLIS serving configuration for a model, a GPU budget, and a workload -- then (later)
turn the results into an interactive HTML report. It drives the simulator many times, steering toward
configs that **maximize throughput while staying interactive** (>= 50 tok/s per user by default), and
writes a run directory you can inspect or re-plot without re-running anything.

This is the **sweeper / engine** half of the skill. The report half is added on later and reads the
same run directory (see `examples/sample-run/` for the shape of that data).

## What "best" means (all overridable)

By default the search reports a **dual Pareto view**:

- **Budget-max** -- the most output tok/s you can get on the GPUs you have, and
- **Efficiency** -- output tok/s per GPU,

both subject to an **interactivity floor** of **>= 50 tok/s per user** (ITL p99 <= 20 ms) with a
**headroom buffer** for load spikes. You can retarget the floor, add TTFT/E2E SLOs, change the
headroom, or ask for a different headline -- just say so when you invoke it.

## Prerequisites

See `SETUP.md`. In short: build BLIS at the `kernel-exclusive` branch (PR #1851) as `./blis`, and
point `BLIS_CATALOG` at a catalog (blis-catalog `0.2.1`). All example models (`glm-5.3-flash`,
`qwen3-14b`) already ship in `0.2.1` -- no model setup needed.

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
`project-cb-lab/curvebender-tools/reports`.

## Output

```
runs/<run-id>/
  scenario.json   # exactly what was searched
  runs.jsonl      # every BLIS run: config, concurrency, metrics, diagnostics
  pareto.json     # the finalists, the chosen winner, and why it won
```

How it's produced today: `scripts/run_blis.py` + `scripts/sweep_pool.py` run BLIS and write
`runs.jsonl`; the **search brain (Claude, per `SKILL.md`)** decides which configs to run, assembles
`scenario.json`, and computes/writes `pareto.json`. A single-command driver that emits all three is
the TODO in Status.

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

First engine version, **smoke-tested on the pinned stack**: the runner parses BLIS's JSON metrics
correctly, the parallel pool hit ~7x on an 8-config batch (qwen3-14b), and the GLM-5.3-Flash headline
example (2x TP8 + EP) runs clean. Still to do: tune the diagnostic-steering rules against real
curvebender measurements, and build out the capacity-probe/finalist driver end to end.
