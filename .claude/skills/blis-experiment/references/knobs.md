# The knob space: what a spec may sweep, with what values, and why

Five separate questions. Confusing them is how a campaign gets spent on nothing.

1. **Is the name real?** `engine:` keys are blis-vllm's engine-block keys (== vLLM's own config
   field names). ~51 exist at v0.32.0.
2. **Can the pinned stack carry it?** **All 51 now decode.** At the stack `pins.env` pins, a
   pool's `engine:` is an *opaque block* opened by the engine model that `engine_version` names —
   blis-vllm's `rules/engineconfig.Engine` — so the decode surface IS blis-vllm's field list. The
   pinned engine model is a blis fork tag (note the suffix), whose `rules/engineconfig` is
   byte-identical to the plain upstream v0.32.0 tag's, so these 51 names did not move when the
   pin last moved. (Until v0.12.0 it was blis-schemas' own 31-field struct, and the other 20 keys
   failed every run with `field <name> not found in type deployment.Engine`.)
3. **Will the cost model *price* it?** A different and now more important question, because the
   bump turned loud failures into silent ones. **Eighteen** knobs decode and are still refused:
   eleven run to completion and measure byte-identical (`ENGINE_INERT`), three have no cost term
   (`ENGINE_NOT_MODELLED`), four are priced only in one shape (`ENGINE_CONDITIONAL`). All three
   lists were measured against the pinned binary, not read off the struct. Of the **21** keys the
   bump opened up (3 priced + these 18; not to be confused with the 20 above, which is `51 − 31`),
   only `max_num_active_seqs`, `max_num_scheduled_tokens` and `frontend` move a
   number — and the first only once concurrency gives it a queue to cap (v4-pro/h200 tp2 dp4:
   inert at c=8; 2405.3 → 808.4 tok/s at c=256 with 8 active seqs).
4. **Is the value legal?** Most of these knobs are closed enums. The engine refuses an unknown
   member at startup, and so does BLIS (measured: `cudagraph_mode: BOGUS`, `cache_dtype: fp16`
   and `max_num_seqs: 0` are all refused). Note the gap the bump opened here: the keys that moved
   to the opaque engine block are NOT all enum-checked — `mla_prefill_backend: flashattn` and
   `linear_backend: cutlass` are both invented values and both ran rc=0. That is another reason
   those two are refused outright.
5. **Will it move anything on *this* model and workload?** The hard question, and the one that
   decides whether a sweep is worth its runs.

The validator answers 1–4 for you, and the model-aware half of 5. **Run it.** This file is for
the judgement it cannot make.

> ### Where the constraints come from
> blis-vllm's `rules/v0_32/pack.go` holds **32 named rules**, each transcribing one of the
> engine's own startup assertions with file and line. Those rules are *not evaluated at run
> time*: the pinned simulator depends on `blis-latency-kernel` and `blis-schemas` only, never on
> blis-vllm. Measured consequence — BLIS accepted a `long_prefill_token_threshold` of 999,999
> against a 32k context, which **real vLLM refuses at startup**.
>
> So a spec can describe a deployment that simulates fine and could never be launched, and the
> number would be meaningless. `scripts/validate_experiment.py` now applies the cross-knob rules
> pre-flight for exactly this reason. Don't rely on BLIS to catch them.

> Measurements below were taken on **BLIS v0.12.0** (catalog 0.4.0, registry v0.2.0),
> `deepseek-v2-lite` on `h100`, `chatbot`, seed 42, 2025-10-29 — the pin at the time. They have
> NOT been re-measured since the pin moved to v0.14.0, so treat them as *direction and rough
> magnitude* only, which is all they were ever meant to be. Re-measure on your own scenario.

---

## Tier 1 — measured leverage. Sweep these.

| Knob | Values | What it does, and what it measured |
|---|---|---|
| `cudagraph_mode` | `NONE`, `PIECEWISE` (default), `FULL`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE` | **The biggest engine lever.** Sets capture strategy ⇒ the host launch term. tp8 c=8: `PIECEWISE` 1983 → `FULL` **2086 (+5%)**; `NONE` 606 vs 1083 at c=4 (**−44%**). Worth a sweep on almost any scenario. |
| `quantization` | 30 values; `fp8`, `modelopt_fp4`, `mxfp4`, `awq`, `gptq`, `compressed-tensors`, … | **The served WEIGHT format** — sets weight-read bytes, compute peak and GEMM efficiency *at once*. Measured 1083 (checkpoint bf16) → `fp8` **1271 (+17%)** → `modelopt_fp4` **1411 (+30%)**. Omit it to let the checkpoint govern; that is the honest baseline arm. |
| `max_num_seqs` | `[16, 32, 64, 128, 256, 512, 1024]`; default 256 | Caps concurrent sequences, and also sizes the runner and graph capture. **Binds downward**: 256→64 cost 29%, →16 cost 64%; raising it past the default measured as a no-op. |
| `max_num_batched_tokens` | `[2048, 4096, 8192, 16384, 32768]`; default 8192 | Caps one scheduler step. Also binds downward — 8192→2048 raised ITL p99 47%. |
| `gpu_memory_utilization` | `[0.5, 0.7, 0.8, 0.85, 0.9, 0.95]`; default 0.9 | Sets KV capacity. Starving it causes preemptions (0.5 → 1270 preemptions, −27%). |
| `cache_dtype` | 19 values; `auto`, `fp8`, `fp8_ds_mla`, `nvfp4_ds_mla`, `float16`, … | KV precision ⇒ tokens per byte. **Model-specific**: DeepSeek-V4.1 refuses an empty cache dtype and serves only `fp8_ds_mla` / `nvfp4_ds_mla`. Weak on a sparse-MLA model, which already keeps KV small. |
| `block_size` | `[16, 32, 64, 128, 256]` | KV paging granularity. **Never hardcode** — DSA sparse-MLA models need a multiple of 64, DeepSeek-V4 wants 256, V4.1 wants 64 on Hopper / 128 on Blackwell. `run_blis` fills it per model. |
| `all2all_backend` | 14 values; `allgather_reducescatter`, `deepep_high_throughput`, `deepep_low_latency`, `deepep_v2`, `pplx`, … | MoE dispatch/combine; sets the volume basis (differs by a factor of `top_k`). **Real −33% spread, but ONLY with expert parallelism on**: tp4/dp2/EP baseline 928 → `deepep_low_latency` **620**. With EP off, byte-identical. |
| `long_prefill_token_threshold` | `[256, 512, 1024, 2048, 4096]`; 0 = uncapped | Caps one prefill's tokens per step so a long prompt cannot stall the decodes beside it. Measured 1983.2 → 1973.9 at 256. **Must not exceed `max_model_len`** (see constraints). |
| `max_cudagraph_capture_size` | `[64, 128, 256, 512, 1024]` | Largest step captured; bigger steps run eager. Matters under drafting, where a step verifies K+1 tokens per request — at K=5, a size of 64 sends every step of >10 requests eager. |

## Tier 2 — real knobs, INERT unless a precondition holds

Sweeping one of these without its precondition gives a column of identical numbers, which reads
as "this knob does nothing". That is a wasted campaign, not a finding. **The validator now
rejects the model-dependent ones** (MoE knobs on a dense model, mamba knobs on a non-recurrent
model); the rest are your judgement.

| Knob | Values | Precondition — without it, measured byte-identical |
|---|---|---|
| `enable_prefix_caching` | `[true, false]` | **A replay workload.** Byte-identical on `chatbot` AND `multidoc`: synthetic shapes generate unique prompts, so there is no prefix to reuse. Only an agentic replay, which re-sends a growing shared context, makes this a lever. Note the default is **true**, so omitting it describes a deployment *with* caching. |
| `scheduling_policy` | `fcfs` (default), `priority` | Requests must carry priorities. The workloads here set none, so `priority` measured identical. |
| `moe_backend` | 23 values; `auto`, `triton`, `deep_gemm`, `cutlass`, `flashinfer_trtllm`, … | An **MoE** model. Also: a mega-MoE backend is one of the conditions under which DeepSeek-V4.1 turns on its *own* sequence parallelism, replacing each attention all-reduce with all-gather + reduce-scatter. Measured identical on this part; treat as exploratory, always keep a control. |
| `disable_cascade_attn` | `[true, false]` | A shared prefix across a batch for cascade attention to read once instead of per request. |
| `async_scheduling` | `[true, false]` | Tri-state with disqualifying conditions. **Interacts with drafting**: only the EAGLE family, MTP names, `ngram_gpu`, `draft_model` and `dspark` leave async scheduling enabled; any other method disables it (or is refused when async is explicitly requested). |
| `allreduce_backend` | `custom`, `nccl` | `custom` (the SM-consuming kernel) supports only certain rank counts, and only within one node absent multi-node NVLink — outside those it falls back to the NIC whatever you state. |
| `disable_custom_all_reduce` | `[true, false]` | Same: it is a *request*, and the resolver reports what runs. |
| `swa_bounded_replay` | `[true, false]` | **DeepSeek-V4.1 only** — nothing else reads it. |
| `dcp_comm_backend` | `ag_rs` (default), `a2a` | Decode-context parallelism (`dcp` > 1). `ag_rs` = query all-gather + LSE all-gather + output reduce-scatter; `a2a` replaces the last two with one all-to-all. |
| `dcp_q_replicate` | `[true, false]` | Same. Replicates the MLA query projection to skip the query all-gather, at the cost of redundant compute. |
| `cp_kv_cache_interleave_size` | `[1, 16, 64]` | Context parallelism on. `1` stripes token-by-token, `block_size` block-by-block. **`block_size` must be ≥ it and divisible by it.** |
| `mamba_cache_dtype`, `mamba_ssm_cache_dtype` | `auto`, `float32`, `float16`, `bfloat16` | A **Mamba-family / recurrent** model. Note `float32` is valid here but *not* in `cache_dtype`. |
| `mamba_cache_mode` | `none`, `align` | Same. Decides whether recurrent state is fixed per sequence or proportional to the context bound — three orders of magnitude apart. |
| `cudagraph_capture_sizes` | list **of lists**: `[[1,2,4],[1,2,4,8]]` | Each value is a whole capture plan. Sizes above `max_num_batched_tokens` are dropped, and `cudagraph_mode: NONE` ignores the plan entirely. |

## Nested engine blocks

Three knobs are **mappings** whose inner fields are each a list; the sweeper expands one
candidate per combination. Inner names are the engine's own and are decoded strictly — `eplb`
carries `enable_eplb`, not `enabled`.

```yaml
engine:
  eplb: {enable_eplb: [true], num_redundant_experts: [0, 32, 64]}
```

| Block | Inner fields | Notes |
|---|---|---|
| `eplb` | `enable_eplb`, `num_redundant_experts`, `window_size`, `step_interval` | **Expert-parallel load balancing — the most interesting of the three.** `num_redundant_experts` is *not cosmetic*: with EPLB on, the engine asserts that the EP width divides `experts + redundant`, so this field decides whether the layout **starts at all**. Measured tp4/dp2/EP: 928.2 → 917.2 with 32 redundant (−1.2%). MoE only. |
| `dbo` | `enable_dbo`, `dbo_decode_token_threshold`, `dbo_prefill_token_threshold` | Dual-batch overlap: split a batch so one microbatch's communication hides under the other's compute. **State a threshold when enabling it** — which one applies turns on batch uniformity and the two defaults differ by 16×. Refused with Engram layers on DeepSeek-V4.1. |
| `engram_config` | `cpu_offload`, `dp_shared_memory`, `use_thp` | Where DeepSeek-V4.1's Engram tables live. Default `cpu_offload: true` (host memory). The engine refuses `dp_shared_memory` without `cpu_offload`, refuses `use_thp` with `dp_shared_memory`, and refuses the whole block on a model without Engram layers. |

`speculative` is nested too, but it **flattens** into drafting candidates — see **Drafting**.

## Tier 3 — NOT sweepable on the pinned stack

Real vLLM v0.32.0 fields that blis-schemas v0.5.0 has no field for. The validator rejects these
by name with that reason; they become available only when `pins.env` moves.

`max_num_active_seqs`, `max_num_scheduled_tokens`, `watermark`, `prefill_schedule_interval`,
`stream_interval`, `prefix_match_unit`, `prefix_cache_retention_interval`,
`disable_hybrid_kv_cache_manager`, `kv_cache_memory_bytes`, `kv_events`, `max_loras`,
`max_cpu_loras`, `custom_ops`, `decoder_replay_cudagraph_capture_sizes`, `use_trtllm_attention`,
`mla_prefill_backend`, `indexer_kv_dtype`, `linear_backend`,
`long_prefill_token_threshold_adaptive`, `env`, `frontend`.

Several are genuinely interesting (`watermark` for admission, `max_num_active_seqs` for
decoupling admission from runner sizing, `kv_cache_memory_bytes` for fixing KV size directly). If
one is what the user actually wants, say it needs a pins bump rather than substituting a
different knob silently.

---

## Cross-knob constraints the validator enforces

These are real vLLM startup assertions that **BLIS does not check**, so a violating spec would
simulate happily and describe an unlaunchable deployment.

| Rule | Constraint |
|---|---|
| `long-prefill-threshold-within-max-model-len` | `long_prefill_token_threshold` ≤ `max_model_len` (default 32,768 if unstated) |
| `cudagraph-capture-sizes` | `max_cudagraph_capture_size` must equal the largest of `cudagraph_capture_sizes`; sizes above `max_num_batched_tokens` are dropped; `cudagraph_mode: NONE` ignores the plan |
| `max-model-len-within-context-window` | `max_model_len` ≤ the model's catalog `context` |
| `expert-divisibility-under-eplb` | with EPLB on, EP width (`tp × dp`) must divide `experts + num_redundant_experts`. The validator computes the nearest fix for you |
| `dbo-thresholds-stated-when-enabled` | enabling `dbo` without a threshold measures an unchosen default (16× apart) |
| `mamba-cache-mode-matches-model` | mamba knobs only on a recurrent model |
| (dense models) | no `all2all_backend` / `moe_backend` / `eplb` without experts |

Others worth knowing, not yet machine-checked: `dcp` must divide `tp`; `block_size` ≥ and
divisible by `cp_kv_cache_interleave_size`; `moonep` is refused for a quantized or non-BF16 model
and with EPLB; `all2all_backend: passthrough` is bound to the FlashInfer MoE-EP megakernels;
DeepSeek-V4.1 refuses every MTP-family draft method and serves `dspark`.

## Parallel knobs

| Knob | Notes |
|---|---|
| `tp` | Powers of two. Past 8 the instance spans nodes and must fill whole nodes. |
| `dp`, `enable_expert_parallel` | **MoE only** — BLIS refuses `dp: 2` on a dense model. EP width is derived as `tp × dp`, so EP over width 1 shards nothing. |
| `pp` | **Sweepable from the v0.15.0 pin** (kernel v0.5.0 prices it; v0.4.0 and earlier refused it outright). Each stage is its own step over its share of the layers, plus a boundary send and the receiving stage's TP all-gather. The boundary *send* is priced from `pipeline_boundary_bytes_per_token`, which only **5 of the catalog's 42 models** declare (`muse-spark-open-816b-a42b` and its `-nvfp4` twin, `deepseek-v4-pro`, `deepseek-v4.1-flash`, `inkling`). A model without it still runs `pp: 2` at rc=0 — measured on `qwen3-14b` — so this is **not** a refusal but a silent omission of the boundary term: the stages are priced, the wire between them is not. Prefer a model that declares it when `pp` is the thing being measured, and read the run's own `pipeline_stages` / `pipeline_binding_stage` assumption warnings. Consumes GPUs like `tp` and `dp`: the footprint is `tp × pp × dp`, so a `pp` ladder changes the footprint unless `tp` moves with it. Measured on Muse armB (h100 roce, 16 GPUs, EP on): tp8 × pp2 = 96.80 tok/s vs tp16 × pp1 = 99.47 at the same footprint. |
| `pcp`, `dcp` | Context parallelism. `dcp` must divide `tp` when `pcp` is off. |

## Router knobs

Routing is a **file** (`routing/*.yaml`), not a policy name. Override fields *inside* one base
file with `schedulingProfiles` (profile → pluginRef → `weight`) and `plugins` (plugin →
`parameters` → field). Sweepable parameters: any scorer's `weight`;
`approx-prefix-cache-producer`'s `lruCapacityPerServer` / `blockSizeTokens` /
`maxPrefixTokensToMatch`; `prefix-cache-scorer`'s `matchLengthWeight` /
`matchLengthScaleTokens`; `load-aware-scorer`'s `threshold`; `prefix-based-pd-decider`'s
`nonCachedTokens` / `promptTokens`; `probabilistic-admitter`'s `queueDepthThreshold` /
`kvCacheUtilThreshold` / `power` / `k`.

Routing only matters with **several instances**, and separates most when traffic is uneven or
re-sends prefixes. On four equal instances with uniform synthetic requests, five arms measured
byte-identical.

## Drafting

`speculative: {method: [...], num_spec_tokens: [1, 3, 5]}` — **both** fields required; acceptance
comes from the registry, not a knob. The pinned schema carries *only* these two (v0.32.0 adds
`draft_model`, `kv_cache_dtype`, `draft_sample_method`, `rejection_sample_method`,
`enable_adaptive_verification`, none sweepable here).

Valid methods: `ngram`, `medusa`, `mlp_speculator`, `draft_model`, `suffix`, `eagle`, `eagle3`,
`dflash`, `ngram_gpu`, `dspark`, plus the 27 MTP names. **But the method is the model's**, read
from its catalog graph — `dspark` on deepseek-v4.1-flash, `mtp` on deepseek-v4-pro,
`deepseek_mtp` on glm-5.2. On a model with no declared speculator every run is refused, and
DeepSeek-V4.1 specifically refuses every MTP-family name.

The direction is **not monotonic at small K**: measured K=0 924.5, K=1 692.8 (−25%), K=3 954.7,
K=5 1041.5 (+13%). Sweep the depth; always keep the K=0 control.

## Interactions: when one knob's answer depends on another

A sweep that pins `4x TP8` and varies one engine knob measures **that knob at that topology**.
Whether the answer generalises depends on the knob, and the difference is not small.

**Measured** (deepseek-v2-lite, h100, chatbot, 8 GPUs, 64 sessions, 240 requests):

| | `max_num_seqs: 8` | `max_num_seqs: 256` | gain |
|---|---|---|---|
| `1x TP8` | 2123 | 8908 | **+320%** |
| `8x TP1` | 740 | 791 | **+6.9%** |

Same knob, same budget, same load — and a 46× difference in how much it is worth. The cause is
structural: 64 sessions over one instance is 64 per instance, where the cap at 8 bites hard; over
eight instances it is ~8 each, so the cap barely binds. A coordinate sweep at `1x TP8` would
report "max_num_seqs is worth 320%" and a reader would carry that to the many-instance shape,
where it is worth 7%.

Compare `cudagraph_mode`, which does **not** interact meaningfully: `FULL` beat `PIECEWISE` by
+3.6% at `1x TP8` and +2.4% at `4x TP2` — same direction, similar magnitude. One topology is
enough to establish it.

### Which knobs interact with topology

| Interacts — sweep over ≥2 topologies | Independent — one topology is enough |
|---|---|
| `max_num_seqs`, `max_num_batched_tokens`, `max_num_active_seqs` — per-instance batch share scales with instance count | `cudagraph_mode`, `max_cudagraph_capture_size` — per-step host cost, largely topology-independent |
| `gpu_memory_utilization`, `cache_dtype`, `block_size` — KV per GPU scales with `tp` | `quantization` — weight bytes scale with `tp` the same way everywhere |
| `all2all_backend`, `moe_backend`, `eplb` — the collective's cost *is* a function of `tp × dp` | `scheduling_policy`, `async_scheduling` |
| `router.*` — a scorer needs endpoints; routing is meaningless at one instance and sharpens as instances grow | `enable_prefix_caching` (gated on the workload, not the topology) |
| `speculative.num_spec_tokens` — drafting changes both throughput and C\*, so it can move the best topology | |
| `long_prefill_token_threshold` — competes with decode for a *per-instance* token budget | |

### The ≥2-topology rule

For any knob in the left column, sweep it across **at least two topologies from opposite ends of
the viable range** — the widest and the narrowest shape the budget allows. That costs 2× the runs
of a coordinate sweep and turns "best value here" into "best value, and whether it travels".

```yaml
# Interaction-aware: the SAME knob grid at two topologies, so the comparison is readable
# both down each column (the knob) and across them (does the answer travel?).
- name: batch-width-vs-shape
  narrative: The batch cap should bind hard on few wide instances and barely at all on many
    narrow ones, because per-instance session count scales with the fleet.
  num-instances: [1, 8]
  parallel: {tp: [8, 1]}          # the GPU filter keeps 1xTP8 and 8xTP1, drops the mismatches
  engine:
    max_num_seqs: [8, 32, 128, 256]
```

The GPU filter does the pruning for you: `num-instances × tp` must equal `gpus`, so listing both
instance counts and both widths yields the two *matched* shapes and silently drops the other two
combinations. 4 knob values × 2 topologies = 8 configs, not 16.

### When a full cross is the right answer

Cross two knobs when **the interaction is the question**, not as a default:

- `cache_dtype` × `gpu_memory_utilization` — both buy KV headroom, so the question is whether
  they add or substitute.
- `max_num_seqs` × `max_num_batched_tokens` — two caps on the same batch; whichever binds first
  makes the other irrelevant, and a cross shows where the crossover sits.
- `enable_prefix_caching` × `router.endpoint_picker_config` — prefix-affinity routing is only
  worth anything if the cache it steers toward is on.

A 2-knob cross at one topology (4×5 = 20 configs) is usually a better spend than two coordinate
sweeps at one topology (9 configs) *plus* a wrong conclusion.

## Budget arithmetic

Total runs = (configs surviving the GPU filter) × `len(scenario.concurrency)`, summed over
sweeps. A full cross explodes fast: 6 `max_num_seqs` × 5 `max_num_batched_tokens` × 9 loads = 270
runs for one sweep.

A coordinate sweep — one knob against a fixed baseline — is the cheap default, but it answers a
*narrower* question than it looks: **"what is the best value of this knob at that one
topology"**, not "what is the best value". That is fine for a knob whose direction holds across
topologies, and wrong for one that interacts. See **Interactions** below for which is which, and
for the ≥2-topology rule that keeps a coordinate sweep honest.

Always check:

```
python3 -I scripts/validate_experiment.py experiments/<name>.yaml
python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run
```
