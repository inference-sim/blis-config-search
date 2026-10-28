# Experiment spec for config-search

**Status:** proposal. Builds on [#1](https://github.com/inference-sim/blis-config-search/discussions/1)
(sweep spec) and [#3](https://github.com/inference-sim/blis-config-search/discussions/3) (search
methodology). Nothing implements it yet.

## The idea

An **experiment** is one **scenario** (model × hardware × workload × GPU budget) and a list of
**sweeps**. Each sweep is one part of the story: for example "colocated serving" next to
"disaggregated serving", or "no drafting" next to "drafting". All sweeps share the scenario, so
their results are comparable by construction.

```
user ─► config-search skill ─► experiment.yaml ─► sweeper ─► runs ─► report
        (LLM: picks the sweeps     (human-readable,   (no LLM: grid × C* probe)
         and suggests values)       reviewed)
```

The **skill** does the thinking. It reads the catalog (MoE or dense? does the model ship a drafter?
how many nodes does it need?), proposes sweeps that tell a story, and fills each knob with a
generous set of reasonable values. The **spec** is what a human reviews, so it is kept as plain as
possible. The **sweeper** runs it deterministically.

## Format

```yaml
name: <experiment id>

scenario:
  model: <blis-catalog model>
  hardware: <blis-catalog hardware>
  workload: <blis-catalog workload>       # chatbot, multidoc, agentx-mvp, ...
  gpus: <int>                             # EVERY run uses exactly this many
  slo: {itl_p99: 20ms, ttft_p99: 5s}      # defines C*
  concurrency: [1, 2, 4, 8, 16, 32, 64]   # blis run --concurrency: the load levels every config runs at

sweeps:
  - name: <sweep id>
    num-instances: [...]                  # blis run flag
    parallel: {<field>: [...]}            # Deployment pool `parallel:` fields
    engine:   {<field>: [...]}            # Deployment pool `engine:` fields
    router:   {endpoint_picker_config: [...]}   # Deployment `router:` block (llm-d)
```

A disaggregated sweep swaps the single pool for one pool per role, as a v0.12.0 Deployment does:

```yaml
  - name: disaggregated
    prefill-instances: [...]              # blis run flags
    decode-instances: [...]
    prefill: {parallel: {...}, engine: {...}}
    decode:  {parallel: {...}, engine: {...}}
    router:  {endpoint_picker_config: [routing/pd-always.yaml, routing/pd-prefix.yaml]}
```

In BLIS the P/D decision belongs to the router (a `disagg-profile-handler` with a P/D decider), so
a disaggregated sweep always names a router config.

That is the whole format. Every value is a list. A one-element list pins a knob, and a knob that is
left out takes BLIS's default.

## Knobs map one-to-one onto blis-vllm

Each `engine:` knob is a key of [blis-vllm](https://github.com/inference-sim/blis-vllm)'s engine
block (`rules/engineconfig.Engine`, which models vLLM **v0.32.0**). Each `parallel:` knob is a key
of the blis-schemas pool layout that blis-vllm reads. The spec uses the same names, the same
nesting and the same values, and invents nothing. Because blis-vllm's keys are vLLM's own config
field names, a spec knob is also a vLLM knob. Full table with feature-matrix status:
[`knobs.md`](knobs.md).

| Spec knob | blis-vllm / vLLM v0.32.0 |
|---|---|
| `parallel.tp`, `parallel.dp`, `parallel.enable_expert_parallel` | `parallel_config.{tensor_parallel_size, data_parallel_size, enable_expert_parallel}` |
| `engine.max_num_seqs`, `max_num_batched_tokens`, `long_prefill_token_threshold`, `max_num_active_seqs`, `watermark`, `scheduling_policy` | `scheduler_config.*` |
| `engine.block_size`, `enable_prefix_caching`, `gpu_memory_utilization`, `cache_dtype` | `cache_config.*` |
| `engine.max_model_len`, `quantization` | `model_config.*` |
| `engine.all2all_backend` | `parallel_config.all2all_backend` |
| `engine.speculative.{method, num_spec_tokens}` | `speculative_config.{method, num_speculative_tokens}` |

- **Checked by blis-vllm itself.** Every engine block the examples expand to (97 of them) decodes
  with blis-vllm's strict `engineconfig.Decode`, and its values pass the v0.32 rules pack. A
  kebab-case key, a vLLM flag name such as `kv_cache_dtype`, a parallel key under `engine:`, or an
  unknown drafter is refused.
- **Leaving a knob out means vLLM's default.** blis-vllm treats every engine key as optional.
- **`all2all_backend` is an engine key.** blis-vllm puts it there, and blis-schemas has no
  `parallel.all2all_backend`.
- **`speculative` needs both `method` and `num_spec_tokens`.** The acceptance rate isn't a
  blis-vllm key; BLIS reads it from the registry.
- **Instance counts and `router:` aren't engine keys.** They describe the stack around the engine:
  `blis run` replica counts and the llm-d EndpointPickerConfig.
- **Pin caveat.** The pinned BLIS (v0.12.0, and v0.13.0 too) doesn't embed blis-vllm yet. It reads
  the same keys under the same names, except `max_num_active_seqs` and `watermark`, which only
  blis-vllm has.

**Routing is in scope.** Any run with more than one instance is already a routing experiment,
because the llm-d router decides where every request goes. A routing value is an llm-d
**EndpointPickerConfig** file: its scorers, weights and picker, its P/D decider, and its admission
and flow-control plugins. BLIS runs it unchanged, the same file a real deployment uses. So there are
no invented routing knobs. To try a new policy you add a file, as the arms in
[`routing/`](../../routing/README.md) do. If you leave `router` out, BLIS uses the llm-d router's
built-in default (queue scoring).

### Sweeping knobs inside a router config

Name a base file, then list values for fields **inside** it. The names are the ones the file itself
uses, so nothing is invented:

```yaml
router:
  endpoint_picker_config: [routing/queue-kv.yaml]   # the base
  schedulingProfiles:                                # profile name -> pluginRef -> weight
    default:
      queue-scorer: {weight: [1, 2, 4]}
      kv-cache-utilization-scorer: {weight: [0, 1, 2, 4]}
  plugins:                                           # plugin name -> parameters
    approx-prefix-cache-producer:
      parameters: {lruCapacityPerServer: [15625, 31250, 62500]}
```

- **`schedulingProfiles.<profile>.<pluginRef>.weight`** is a scorer's weight in that profile.
- **`plugins.<name>.parameters.<field>`** is any plugin parameter. A plugin's name defaults to its
  `type`.
- **The sweeper makes one config per combination** (the base file with those fields replaced) and
  passes it to BLIS.
- **The reference is the `llm-d-r` branch of
  [inference-sim/llm-d-router](https://github.com/inference-sim/llm-d-router/tree/llm-d-r).** That's
  the fork BLIS v0.12.0 embeds; it pins commit `c8dc911a93a6`, which is on that branch.
- **Typos fail at startup.** The fork decodes plugin parameters with `DisallowUnknownFields`, so a
  misspelled field fails before any run instead of being silently ignored. The sweeper also checks
  that each named plugin and profile entry exists in the base file.

Parameters worth sweeping (field names from the fork):

| Plugin | Parameters |
|---|---|
| any scorer in a profile | `weight` |
| `approx-prefix-cache-producer` | `lruCapacityPerServer` (default 31250), `blockSizeTokens` (64), `maxPrefixTokensToMatch` (131072) |
| `prefix-cache-scorer` | `matchLengthWeight`, `matchLengthScaleTokens` |
| `load-aware-scorer` | `threshold` (default 128) |
| `token-load-scorer` | `queueThresholdTokens` (default 4194304) |
| `active-request-scorer` | `idleThreshold`, `maxBusyScore` |
| `prefix-based-pd-decider` | `nonCachedTokens` (0 turns prefix-based P/D off), `promptTokens` |
| `probabilistic-admitter` | `queueDepthThreshold`, `kvCacheUtilThreshold`, `power`, `k` |
| `utilization-detector` | `queueDepthThreshold`, `kvCacheUtilThreshold` |
| `max-score-picker`, `random-picker` | `maxNumOfEndpoints` |

## What the sweeper does (implicit, not in the spec)

1. **Expands each sweep to its full grid** and keeps only the combos that use **exactly `gpus`
   GPUs**: `tp × dp × num-instances`, or the sum over the prefill and decode pools. Every run in
   every sweep is on the same hardware, so every comparison is like-for-like. The sweeper reports
   how many combos it dropped. Instances must also fit in one node or fill whole nodes.
2. **Runs every config at every `concurrency` level** and reads `C*` off the curve: the highest
   level that still meets `slo`. The load lives in `scenario`, not in a sweep, so every sweep is
   measured on the same load axis and their curves are comparable. Its unit
   comes from the workload: users for chatbot, session lanes for a replay such as agentx-mvp.
3. **Ranks every config on usable capacity per GPU**, within each sweep and across the whole
   experiment. The report gives each sweep its own section and overlays their fronts.

## Examples

| File | Story | Configs per sweep (all on the same GPUs) |
|---|---|---|
| [`agg-vs-disagg.yaml`](examples/agg-vs-disagg.yaml) | kimi-k2.5, 64 H200, multidoc: does splitting prefill from decode pay? | `aggregated` 12, `disaggregated` 6, `pd-threshold` 4 |
| [`drafting-on-off.yaml`](examples/drafting-on-off.yaml) | deepseek-v4.1-flash, 32 H100, AgentX: is speculative decoding worth it? | `no-drafting` 6, `drafting` 30 |
| [`routing.yaml`](examples/routing.yaml) | deepseek-v4.1-flash, 64 H200, AgentX: which llm-d policy wins, and how sensitive is it to its own knobs? | `policies` 10, `queue-kv-weights` 12, `prefix-affinity-cache` 9 |
| [`engine-tuning.yaml`](examples/engine-tuning.yaml) | glm-5.3-flash, 16 H100, chatbot: which engine knobs move capacity? | `batching` 30, `kv-memory` 24, `admission` 27 |
| [`kv-precision.yaml`](examples/kv-precision.yaml) | qwen3-14b, 16 H100, chatbot: does an fp8 KV cache pay on a dense model? | `bf16-kv` 4, `fp8-kv` 4 |

Each config runs at every `concurrency` level.

Schema: [`experiment.schema.json`](experiment.schema.json). It only accepts the knob names above.

## Limitations / future work

- **Load is a fixed concurrency list.** There is no adaptive search for the highest load that
  meets the SLO (#3's probe), and no open-loop `--rate`.
- **Router overrides only change existing fields.** A sweep can't add a plugin to a profile or
  remove one. That takes a separate base file.
- **No shared defaults between sweeps.** Knobs that two sweeps share are repeated in each one.
- **One hardware type and one workload per experiment.** Comparing H100 vs. H200, or chatbot vs.
  multidoc, takes separate experiments.
