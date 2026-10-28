# Knob map: spec ↔ blis-vllm

Every `engine:` knob in an experiment spec is a key of
[blis-vllm](https://github.com/inference-sim/blis-vllm)'s engine block (`rules/engineconfig.Engine`,
documented in its [`docs/engine-config.md`](https://github.com/inference-sim/blis-vllm/blob/main/docs/engine-config.md)).
Each `parallel:` knob is a key of the blis-schemas pool `parallel:` block that blis-vllm reads. The
spec uses the same key, the same nesting and the same values. blis-vllm models **vLLM v0.32.0**
(`5df63a72`), and its keys are vLLM's own config field names, so a spec knob is also a vLLM knob.

**Source:** blis-vllm @ `99d3fd7` (main), blis-schemas v0.8.0. blis-vllm decodes the block strictly, so
a key it doesn't define fails at decode time. That's the guarantee that no spec knob is invented.

**Status** is blis-vllm's own [feature matrix](https://github.com/inference-sim/blis-vllm/blob/main/docs/feature-matrix.md):
- **supported:** diff-tested against vLLM v0.32.0 at zero mismatches.
- **kernel follow-up:** the scheduler side is in blis-vllm, but blis-latency-kernel still owes the pricing.

## `parallel:` (blis-schemas pool layout)

| Spec knob | blis-vllm reads | vLLM v0.32.0 | Feature-matrix status |
|---|---|---|---|
| `parallel.tp` | `Parallel.TP` | `parallel_config.tensor_parallel_size` | supported (decides nothing about scheduling; priced by the kernel) |
| `parallel.dp` | `Parallel.DP` | `parallel_config.data_parallel_size` | supported (DP lockstep and wave cadence, T3) |
| `parallel.enable_expert_parallel` | `Parallel.EnableExpertParallel` | `parallel_config.enable_expert_parallel` | kernel follow-up (EP is priced by the kernel) |

## `engine:` (blis-vllm engine block)

| Spec knob | blis-vllm key | vLLM v0.32.0 | Feature-matrix status |
|---|---|---|---|
| `engine.max_num_seqs` | `max_num_seqs` | `scheduler_config.max_num_seqs` | supported (T2) |
| `engine.max_num_batched_tokens` | `max_num_batched_tokens` | `scheduler_config.max_num_batched_tokens` | supported (T2) |
| `engine.long_prefill_token_threshold` | `long_prefill_token_threshold` | `scheduler_config.long_prefill_token_threshold` | supported (T2) |
| `engine.max_num_active_seqs` | `max_num_active_seqs` | `scheduler_config.max_num_active_seqs` | supported (T2) |
| `engine.watermark` | `watermark` | `scheduler_config.watermark` | supported (T2) |
| `engine.scheduling_policy` | `scheduling_policy` | `scheduler_config.policy` (`--scheduling-policy`) | supported (T2) |
| `engine.block_size` | `block_size` | `cache_config.block_size` | supported (T2) |
| `engine.enable_prefix_caching` | `enable_prefix_caching` | `cache_config.enable_prefix_caching` | supported (T2) |
| `engine.max_model_len` | `max_model_len` | `model_config.max_model_len` | supported (T2) |
| `engine.gpu_memory_utilization` | `gpu_memory_utilization` | `cache_config.gpu_memory_utilization` | kernel follow-up (KV sizing) |
| `engine.cache_dtype` | `cache_dtype` | `cache_config.cache_dtype` (`--kv-cache-dtype`) | kernel follow-up |
| `engine.quantization` | `quantization` | `model_config.quantization` | kernel follow-up |
| `engine.all2all_backend` | `all2all_backend` | `parallel_config.all2all_backend` | kernel follow-up |
| `engine.speculative.method` | `speculative.method` | `speculative_config.method` | supported for DSpark, MTP, EAGLE3 (T2) |
| `engine.speculative.num_spec_tokens` | `speculative.num_spec_tokens` | `speculative_config.num_speculative_tokens` | supported (T2) |

## Rules that come from blis-vllm

- **Leaving a knob out means vLLM's default.** blis-vllm treats every engine key as optional and
  an omitted one as unset, so the engine picks vLLM v0.32.0's default. For
  `gpu_memory_utilization`, 0 is the unset value.
- **`all2all_backend` is an engine key, not a parallel key.** It sits in blis-vllm's engine block,
  and blis-schemas has no `parallel.all2all_backend`.
- **`speculative.method` and `speculative.num_spec_tokens` are both required** when `speculative`
  is set. The method must be one the v0.32 rules pack accepts (`dspark`, `deepseek_mtp`, `eagle3`,
  …). The acceptance rate is not a blis-vllm key: BLIS reads it from the registry.
- **Valid string values are release data.** blis-vllm's `rules/v0_32` pack accepts the
  `cache_dtype`, `quantization`, `all2all_backend` and `speculative.method` values. A value it
  refuses fails validation before any run.

## Cluster-level knobs (not blis-vllm keys)

These describe the stack around the engine, so blis-vllm doesn't own them:

| Spec knob | Where it maps |
|---|---|
| `num-instances` | `blis run --num-instances`: engine replicas |
| `prefill-instances` / `decode-instances` | `blis run --prefill-instances` / `--decode-instances` |
| `router.*` | Deployment `router:`, the llm-d EndpointPickerConfig (llm-d-r fork) |

## Pin caveat

blis-vllm models v0.32.0 behind blis-schemas v0.8.0. The repo's pinned BLIS (v0.12.0, and v0.13.0
too) still decodes the engine block itself at blis-schemas v0.5.0/v0.7.0, and its scenarios declare
`engine_version: "0.29.0"`. All the keys above already exist in that older block with the same names,
except `max_num_active_seqs` and `watermark`, which are new in blis-vllm. Running a spec through
blis-vllm needs a BLIS release that embeds it (inference-sim#2021).
