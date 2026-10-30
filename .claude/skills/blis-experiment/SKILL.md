---
name: blis-experiment
description: >
  Use when designing a BLIS EXPERIMENT -- a set of sweeps that answer a question about serving a
  model on a given GPU budget, rather than a single best-config search. Triggers: "design an
  experiment for X on N GPUs", "does drafting pay for model X", "compare colocated vs P/D",
  "which routing policy for this workload", "write me an experiment spec", "sweep engine knobs
  for X". Produces experiments/<name>.yaml (an Experiment Spec, RFC #21) plus a short narrative;
  the deterministic sweeper runs it afterwards. Simulation only (BLIS), not a live cluster.
---

# blis-experiment

## Overview

Turn a scenario -- **model x hardware x workload**, a GPU budget and an SLO -- into a valid
**Experiment Spec**: one shared scenario plus a few sweeps that together tell one story.

This skill is the **only LLM in the pipeline**, and it stops at the spec. Everything after it is
deterministic: `scripts/run_experiment.py` expands the grids, runs BLIS and picks the winners with
no model in the loop, which is why CI can run it unattended. So the spec is the whole deliverable
— and a spec that encodes a bad question wastes the entire run.

Contributors may also write specs by hand, or with any other tool. Nothing downstream knows or
cares that this skill wrote one.

## When to use / not

- **Use when:** the user wants to *learn something* about a deployment — does drafting pay here,
  is disaggregation worth it, which routing policy suits this traffic, where do the batching knobs
  bind, does fp8 KV buy throughput.
- **Not for:** "just give me the best config" on a budget (that is `config-search`, which probes
  adaptively for capacity); a single known config (run `blis run` directly); a real cluster.
- **Out of scope by design** (RFC #21): adaptive SLO probes, open-loop `--rate`, adding or
  removing router plugins, defaults shared across sweeps, and any experiment spanning more than
  one hardware type or one workload. Two hardware types means two experiments.

## Inputs you need

| Input | Notes |
|---|---|
| model | a blis-catalog short-name; a model runs **iff** the catalog carries it |
| hardware | one GPU type per experiment |
| workload | one per experiment — a shape preset or a replay scenario |
| GPU budget | `scenario.gpus`; **every** run uses exactly this many |
| SLO | at minimum `itl_p99` (20ms == 50 tok/s/user) |
| run budget | how many BLIS runs the campaign may spend — **the single input that decides how deep the spec can go** |

Ask only for what is genuinely missing, and infer what the catalog can tell you. The model,
hardware, workload and GPU budget are enough to begin; default the SLO to `{itl_p99: 20ms}` and
say so.

**The run budget is the exception worth asking about.** It is not a detail — it decides whether
the spec can afford interaction-aware sweeps or only coordinate ones, and that changes what the
experiment can conclude. A run is a few seconds for a generated shape and ~10-15s for an agentic
replay, so a few hundred runs is routine and a few thousand is a real campaign.

If the user gives no run budget, **do not quietly produce the cheapest possible spec.** Either
ask, or design to a stated default and say what you assumed and what it bought:

> Designed to ~400 runs. That afforded the batch-cap sweep across two topologies (which is where
> the 46× interaction lives); a third topology or an `all2all_backend` arm would need ~700.

**A one-line scenario deserves a real experiment, not a thin one.** Given only "deepseek-v4.1-flash
on 32x H100, agentx", the catalog already tells you the architecture, expert count, attention
kind, context window, declared speculator and quantization — which is enough to pick 3-4 sweeps
that interact, state a hypothesis for each, and name what you left out. Returning five
single-knob sweeps pinned to one topology is under-delivering on the information you had.

## Step 1: profile the combination — do not guess, and do not read only the catalog row

**Run this first, every time:**

```
python3 -I scripts/knob_profile.py --model <m> --hardware <hw> --workload <wl>
```

It prints, per knob: whether it **applies** to this exact combination, the values that are
**legal here**, and **why** — read from the sources, not from memory:

| Source | What only it knows |
|---|---|
| catalog model row | `arch`, `experts`, `top_k`, `attn_kind`, `context`, `quant` |
| the model's **`graph.yaml`** | the **declared speculator and its max depth** (so whether drafting is legal at all, and the K ceiling), the sliding window, and the `bytes_by_cache_dtype` tables — *which KV dtypes this model is priced for* |
| **blis-vllm's rules pack** | the **architecture-keyed** rules: what `auto`/`fp8` resolve to on this chip's capability family, which draft methods the architecture refuses, whether `engram_config` applies |
| the hardware entry | `ComputeCapability` → the capability family that selects among those resolutions, plus memory for the footprint floor |

**The catalog row alone is not enough**, and this is the trap: it carries no speculator, no
sliding window, no KV-dtype table and no capability gating. Design from the row only and you will
either omit a real axis or sweep an impossible one. Worked example, verified against the pinned
binary:

> `deepseek-v4.1-flash` prices `nvfp4_ds_mla` (288 B/entry vs 528) **only** under capability
> family 10. On **b200** the profile reports `cache_dtype: [fp8_ds_mla, nvfp4_ds_mla]` — a real
> KV-precision axis. On **h100/h200** it reports `fp8_ds_mla` alone and says the other dtype "is
> priced only on another capability family, so it is not an axis here". BLIS agrees: the same run
> is accepted on b200 and refused on h100 with *"nvfp4_ds_mla needs the V4 … on SM90"*.

Nothing in the catalog row says that. Take the profile's verdict as the input to the grid: sweep
what it marks **APPLIES**, and do not sweep what it marks **DOES NOT APPLY** — every one of those
would return a column of identical numbers or a refused run.

## Step 2: group the applicable knobs by what they CONTEND for

**The default to avoid: one sweep per knob.** It feels tidy and it answers the wrong question —
"what is the best value of knob X at the one topology I pinned" — while leaving the thing the
user actually wants to know, *how these settings interact*, untested.

**Measured** (deepseek-v2-lite, 8 GPUs, 64 sessions), `max_num_seqs` 8 → 256: **+320%** at
`1x TP8`, **+6.9%** at `8x TP1`. Same knob, same budget, same load, 46× apart. A single-topology
sweep reports one of those as "the" answer.

So build each sweep around a **contended resource**, and put the knobs that fight over it in the
*same* sweep so the grid crosses them:

| Sweep built around | Knobs that contend | The question it answers |
|---|---|---|
| **the batch** | `max_num_seqs` × `max_num_batched_tokens`, across ≥2 instance counts | which cap binds first, and does that change with fleet shape |
| **KV capacity** | `cache_dtype` × `gpu_memory_utilization`, across ≥2 `tp` | do cheaper entries and more memory add, or substitute |
| **the MoE collective** | `all2all_backend` × `dp`/EP (and `eplb` if the budget allows) | the backend's cost *is* a function of `tp × dp`, so it cannot be answered at one width |
| **prefix reuse** | `enable_prefix_caching` × `router.endpoint_picker_config` | prefix-affinity routing is worthless with the cache off — only the cross shows the pairing |
| **the draft** | `num_spec_tokens` × topology (K=0 control in the grid) | drafting changes throughput *and* C\*, so it can move the best topology |
| **per-step overhead** | `cudagraph_mode` alone is fine | does not interact meaningfully (+3.6% at `1x TP8`, +2.4% at `4x TP2`) |

Two rules follow:

1. **A knob that interacts with topology is swept across ≥2 topologies** — widest and narrowest
   the budget allows. The GPU filter prunes mismatches, so `num-instances: [1, 8]` with
   `parallel: {tp: [8, 1]}` costs 2 shapes, not 4. `references/knobs.md` → *Interactions* lists
   which knobs these are.
2. **Two knobs that contend for the same resource go in one sweep, crossed** — not in two sweeps
   that each pin the other at its default.

A lone-knob coordinate sweep is still right for a knob the profile and `references/knobs.md`
agree does not interact (`cudagraph_mode`, `quantization`), and as a first pass over a space too
big to cross. Otherwise prefer the cross: a 4×5 cross at one topology (20 configs) usually beats
two coordinate sweeps (9 configs) *plus* a conclusion that does not hold.

**Spend the budget on interactions before breadth.** If it forces a choice, drop a knob rather
than collapse an interacting knob to one topology.

## The knob space, and what each knob is worth

**Read `references/knobs.md` before choosing a grid.** It is the full sweepable surface — 31
engine knobs plus three nested blocks, with the legal values for each, the measured leverage, and
the cross-knob constraints. The summary:

1. **Tier 1, measured leverage** — sweep freely. `cudagraph_mode` is the biggest engine lever
   (`NONE` cost 44%, `FULL` beat the default by 5%); `quantization` moved +30% (`modelopt_fp4`);
   `all2all_backend` moved −33% *but only with expert parallelism on*; the batching and KV knobs
   bind downward.
2. **Tier 2, inert without a precondition** — `enable_prefix_caching` is byte-identical on every
   *synthetic shape* workload (generated prompts share no prefix; it needs a **replay**),
   `scheduling_policy: priority` needs requests that carry priorities, `swa_bounded_replay` is
   read by DeepSeek-V4.1 alone. A column of identical numbers reads as "this knob does nothing" —
   a wasted campaign, not a finding.
3. **Tier 3, not carried by the pinned stack** — vLLM v0.32.0 has ~51 engine fields; pinned BLIS
   (blis-schemas v0.5.0) has 31. Naming one of the other 20 fails **every** run. If one of them is
   what the user actually wants (`watermark`, `max_num_active_seqs`, `kv_cache_memory_bytes`), say
   it needs a pins bump — do not silently substitute a different knob.

**Every knob you sweep needs a reason it could move on *this* model and workload.** Gate it on
its precondition; when a precondition fails, leave the knob out and record that in a `#` comment.

### Let the model pick the knobs

The catalog row you already read decides most of this. Map it directly:

| Catalog field | Knobs it unlocks — or rules out |
|---|---|
| `arch: MoE` + `experts` | `dp`, `enable_expert_parallel`, `all2all_backend`, `moe_backend`, `eplb`. **Dense ⇒ none of these**: no experts to dispatch or balance, and the validator rejects them. |
| `experts` count | `eplb.num_redundant_experts` must make `experts + redundant` divisible by the EP width (`tp × dp`), or the layout **does not start**. |
| `attn_kind: mla` / sparse | KV is already small, so `cache_dtype` is a **weak** lever — spend the runs elsewhere. `gqa` with a large KV ⇒ `cache_dtype` and `gpu_memory_utilization` matter much more. |
| `attn_kind` mamba/recurrent | `mamba_cache_dtype`, `mamba_ssm_cache_dtype`, `mamba_cache_mode`. On anything else these size a state that does not exist. |
| `context` | the ceiling for `max_model_len`, and therefore for `long_prefill_token_threshold`. |
| graph's declared speculator | whether a drafting sweep is possible at all, and which `method` is legal (`dspark`, `mtp`, `deepseek_mtp` — DeepSeek-V4.1 refuses every MTP name). |
| `quant` / checkpoint format | whether `quantization` is a real arm or already fixed by the checkpoint. |
| hardware `memory_gib` vs `params` | the footprint floor; a model that needs multiple nodes makes narrow-`tp` grids die on memory. |
| workload: prefix reuse | `enable_prefix_caching`, prefix-affinity routing — a **replay** only. |
| workload: input/output mix | long prompts ⇒ `long_prefill_token_threshold`, P/D, prefill routing. Long outputs ⇒ batching, drafting, `cudagraph_mode`. |

### The validator is not optional

It checks the name, the **value** (`cudagraph_mode: [full]` is refused — the enum is upper-case;
`cache_dtype: [fp16]` is `float16`; `dspar` suggests `dspark`), and the **cross-knob** rules that
BLIS itself does not: `long_prefill_token_threshold` above `max_model_len`, a capture maximum that
is not the largest captured size, `max_model_len` past the model's context window, EPLB
redundancy that breaks expert divisibility, MoE knobs on a dense model.

That last group matters more than it looks: blis-vllm's 32 rules are **not evaluated at run
time** (the pinned simulator does not depend on blis-vllm at all), so BLIS will happily simulate a
deployment real vLLM would refuse to launch — a `long_prefill_token_threshold` of 999,999 against
a 32k context ran fine. A number from such a run is meaningless.

## Every sweep gets a hypothesis — ONE sentence

Each sweep's `narrative` is **one sentence, at most two**, naming what you expect and the reason.
The report prints it directly above that sweep's measured result, so it has to be scannable:

> `Eight narrow instances give a scorer more to exploit than two wide ones.`

That is a hypothesis. `Sweep routing` is not. Neither is a paragraph.

**Hard limit: ~200 characters.** Past that it stops being a caption and starts crowding the
numbers it is supposed to frame. Measured prose from an unbounded run came in at 600–670
characters per sweep — five sentences of hypothesis, rationale, prior measurements and design
justification stacked into one field, which pushed the actual results off the screen.

**What does NOT belong in the narrative:**

| Keep out | Where it goes instead |
|---|---|
| why you chose this grid, or skipped another sweep | a `#` comment at the top of the spec |
| prior measurements ("NONE cost 44% on a reference part") | a `#` comment, if it is worth recording at all |
| that a control arm exists, or shares a topology grid | the grid says so; the report pairs the rows |
| which knobs are inert and why | a `#` comment — it explains the *absence* of a sweep |
| the measured outcome | the report writes that itself, under "Measured:" |

The narrative is the one line a reader sees beside the result, so spend it on the prediction.

### Comments are cheap, but not free — keep the header under ~10 lines

Comments never reach the report, so they are the right home for rationale. They are still read by
the next contributor, and an unbounded header buries the spec: a real generated spec opened with
**36 comment lines** (including a duplicated paragraph) before `name:`, so the grid — the part
anyone actually needs — started below the first screen.

**A spec header is at most ~10 lines**, and says only what a reader cannot get from the grid:

- the question the experiment answers, in a sentence;
- the two or three catalog/profile facts that *picked* the axes (`arch MoE, 128 experts` →
  EP is real; `family 10` → nvfp4 is an axis);
- anything deliberately left out that a reader would otherwise expect, one line each.

Do not transcribe the profile output, re-explain a knob that `references/knobs.md` already
covers, restate a measurement the report will print, or repeat the same note in six sibling
specs. Per-sweep comments follow the same rule: one or two lines, only where the grid is not
self-explanatory.

Together the narratives should still read as one argument, with a control arm wherever a
comparison needs one (the no-drafting baseline beside the drafting arm, the aggregated arm beside
the disaggregated one, on the *same* topology grid so each row has a twin).

## Keep the grid inside the run budget

Total runs = (configs surviving the GPU filter) x `len(scenario.concurrency)`, summed over sweeps.
Check it before you hand the spec over:

```
python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run
```

It prints per-sweep config and run counts and how many combos the GPU filter dropped. If it is
over budget, **narrow the grid deliberately** — fewer knob values, a coarser concurrency ladder —
and note in a `#` comment that the grid was bounded. Do not silently drop a control arm.

## The format (RFC #21 — see experiments/README.md for the full contract)

- **Every knob value is a list.** `tp: [8]`, never `tp: 8`.
- **An omitted knob takes the vLLM default.** Do not restate defaults.
- **Knob names are BLIS's exact names** — `engine:` keys are blis-vllm engine-block keys (== vLLM's
  own config field names), `parallel:` keys are blis-schemas pool fields. Nothing invented.
- **Every run uses exactly `scenario.gpus`.** For a colocated sweep
  `tp x dp x num-instances == gpus`; for a disaggregated one both pools count together.
- **Concurrency is set once, in `scenario`,** so every sweep is measured at the same loads.
- **Routing is a file**, listed under `router.endpoint_picker_config` (see `routing/`). A new
  policy is a new file. Overrides edit fields the base file already has, against one base.
- **`speculative` needs both `method` and `num_spec_tokens`.** Acceptance comes from the registry.
- **A replay workload is already fully described by the catalog.** Naming it names all of it.

### Replay workloads: state the corpus path, nothing else

`blis-catalog/workloads/<name>.yaml` **is** the workload. For `agentx-mvp` it states the harness
scenario id (`inferencex-agentx-mvp`), the corpus loader with its pinned HF revision, the
`sha256`, `random_seed`, `benchmark_duration_s`, `grace_period_s`, `max_context_length`, the
cache-bust and trajectory settings, and the `category`/`thinking` pair that selects the
acceptance profile. The sweeper reads all of it, and BLIS enforces the catalog's digest — a
corpus whose bytes differ is refused even when the spec states no sha at all.

So a replay spec states only what the catalog cannot know:

```yaml
scenario:
  workload: agentx-mvp
  replay:
    corpus: .corpus/traces.jsonl   # where the file sits on THIS machine (not vendored)
    corpus_cache: .corpus/cache    # optional local parse cache; -46% per run
    grace_period: inf              # a deliberate override of the descriptor's 30s
```

Do **not** restate `scenario:`, `corpus_sha256`, `benchmark_duration` or `max_context_length` —
they are catalog data, and a copy can only drift from it. Use a relative corpus path, never an
absolute one: an absolute path pins the spec to one machine. A field that *does* differ from the
descriptor is recorded as an override, so the report can say the run departed from the catalog.

## You MUST validate before returning

Run the validator and fix every error — a spec that fails is not a deliverable:

```
python3 -I scripts/validate_experiment.py experiments/<name>.yaml
```

It checks knob names against the sibling `blis-vllm` / `blis-schemas` checkouts when present,
the GPU arithmetic per sweep, SLO units, and that every router file exists. Then run `--dry-run`
to confirm the grids expand to the run count you intended. Return the spec path, the run count,
and a short plain-language summary of the experiment's story.

## Common mistakes

- **Inferring architecture from a name.** Read `arch` from the catalog. A dense model with a `dp`
  grid fails every run.
- **A drafting sweep on a model with no speculator.** Every run is refused; the ledger then shows
  "drafter: off" beside "draft tokens: 1". Check the graph first.
- **A grid that expands to nothing** because no combination hits `gpus`. The validator catches it;
  run it.
- **Restating vLLM defaults** as swept values — it doubles the grid and measures nothing.
- **A sweep with no control.** "Drafting gives 900 tok/s" means nothing without the same topology
  at K=0.
- **Comparing across hardware or workloads in one spec.** Out of scope; write two experiments.
- **Sizing the concurrency ladder past the knee.** For a replay the load axis is session-tree
  *lanes*, worth many requests each, so the ladder is far shorter than for a generated shape.
