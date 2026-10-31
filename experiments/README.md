# Experiment Specs

An **experiment** is one scenario plus a few **sweeps** that tell one story — colocated vs.
disaggregated serving, drafting off vs. on, one routing policy vs. another. You write the spec;
a deterministic sweeper runs it and publishes the report.

The pipeline splits at exactly one seam:

```
          you (or the blis-experiment skill)          deterministic, no LLM, runs in CI
   ──────────────────────────────────────────── │ ────────────────────────────────────────────
   experiments/<name>.yaml  ────────────────────┼──▶  run_experiment.py  ──▶  reports/<name>/
            the Experiment Spec                 │     expand · filter · run · rank
```

Everything to the right of the seam is a pure function of the spec plus the pinned stack
([`pins.env`](../pins.env)), so the same spec gives the same report. That is what lets CI run it
unattended on merge.

## Submitting one

1. Write `experiments/<name>.yaml` (copy a committed example — of the seven below, five are
   [discussion #21](https://github.com/inference-sim/blis-config-search/discussions/21)'s own).
2. Validate it:
   ```
   python3 -I scripts/validate_experiment.py experiments/<name>.yaml
   ```
3. Check the size of the campaign you are proposing:
   ```
   python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run
   ```
4. Open a PR. CI validates every spec on the PR (no secrets, safe for forks). A maintainer runs
   the sweeper after merge, and a bot PR adds `reports/<name>/` and the gallery entry.

You do not run BLIS yourself, and you never commit results by hand.

## The format

```yaml
name: deepseek-v4-1-flash-h100-agentx-drafting   # also the reports/<name>/ directory

scenario:                       # shared by EVERY sweep
  model: deepseek-v4.1-flash    # a blis-catalog short-name
  hardware: h100                # one GPU type per experiment
  workload: agentx-mvp          # one workload per experiment
  gpus: 32                      # EVERY run uses exactly 32 GPUs
  slo: {itl_p99: 20ms}          # units are required
  concurrency: [4, 8, 16, 32, 64, 128]

sweeps:
  # Both sweeps share this topology grid, so every drafting row has an exact K=0 twin at the
  # same topology and load. That pairing is what makes the comparison readable -- and it is a
  # COMMENT, because it explains the design rather than predicting the outcome.
  - name: no-drafting
    narrative: The baseline; if drafting does not beat it at the same topology and load, the
      drafter is not paying for itself.
    num-instances: [1, 2, 4]
    parallel: {tp: [4, 8], dp: [1, 2, 4, 8], enable_expert_parallel: [true]}

  - name: drafting
    # Measured elsewhere: K=1 -25%, K=3 +3%, K=5 +13% -- which is why depth is swept, not assumed.
    narrative: Drafting should win on decode-bound traffic, and the win should grow with depth.
    num-instances: [1, 2, 4]
    parallel: {tp: [4, 8], dp: [1, 2, 4, 8], enable_expert_parallel: [true]}
    engine:
      speculative: {method: [dspark], num_spec_tokens: [1, 3, 5]}
```

### The rules

- **Every knob value is a list**, even a single one: `tp: [8]`, never `tp: 8`. One rule, so a grid
  always reads as a product.
- **An omitted knob takes the vLLM default.** Don't restate defaults — it doubles the grid and
  measures nothing.
- **Every run uses exactly `scenario.gpus`.** The sweeper keeps only combos where
  `tp x dp x num-instances == gpus` (for a disaggregated sweep, both pools together), so every
  comparison is like-for-like. A grid where nothing totals `gpus` expands to **no runs**, and the
  validator rejects it.
- **Concurrency is set once, in `scenario`,** so every sweep is measured at the same loads.
- **Knob names are BLIS's exact names** (see below). Nothing is invented here.
- **One hardware type and one workload per experiment.** Comparing H100 vs. H200 takes two.
- **Each sweep needs a `narrative`** — **one sentence**, at most two, under ~200 characters.
  It is the single line the report prints above that sweep's measured result, so it states the
  prediction and nothing else. Grid rationale, prior measurements, and why you skipped some other
  sweep all belong in a `#` comment, which stays in the spec and never reaches the report. The
  validator prints a style note (not a failure) for anything longer.

## Knobs map one-to-one onto blis-vllm

| Spec knob | blis-vllm / vLLM v0.32.0 |
|---|---|
| `parallel.tp`, `dp`, `pp`, `pcp`, `dcp`, `enable_expert_parallel` | `parallel_config.*` (blis-schemas pool layout) |
| `engine.max_num_seqs`, `max_num_batched_tokens`, `max_num_active_seqs`, `watermark`, `scheduling_policy` | `scheduler_config.*` |
| `engine.block_size`, `enable_prefix_caching`, `gpu_memory_utilization`, `cache_dtype` | `cache_config.*` |
| `engine.max_model_len`, `quantization` | `model_config.*` |
| `engine.all2all_backend` | `parallel_config.all2all_backend` |
| `engine.speculative.{method, num_spec_tokens}` | `speculative_config.{method, num_speculative_tokens}` |
| `num-instances`, `prefill-instances`, `decode-instances` | `blis run` replica-count flags, not engine keys |
| `router.*` | llm-d's EndpointPickerConfig, not an engine key |

The validator checks every name against the sibling `../blis-vllm` and `../blis-schemas`
checkouts when they are present, and against the names pinned at blis-vllm v0.32.0 / blis-schemas
v0.8.0 otherwise. A typo fails before any run — and the message suggests the right name.

**Values are checked too.** `cudagraph_mode: [full]` is refused (the enum is upper-case),
`cache_dtype: [fp16]` is refused (it is `float16`), and `quantization: [fp_8]` suggests `fp8` —
the valid set or the near miss is always named. The value sets are read from blis-vllm's
`rules/v0_32/pack.go` when that checkout is present.

### Cross-knob constraints BLIS does not check

blis-vllm's rules pack holds **32 named rules**, each transcribing one of vLLM's own startup
assertions. They are **not evaluated at run time** — the pinned simulator depends on
`blis-latency-kernel` and `blis-schemas` only, never on blis-vllm. Measured consequence: BLIS
happily simulated a `long_prefill_token_threshold` of 999,999 against a 32k context, which real
vLLM refuses at startup. A spec like that produces numbers for a deployment nobody could launch.

So the validator applies them pre-flight:

| Constraint | Rule |
|---|---|
| `long_prefill_token_threshold` ≤ `max_model_len` | `long-prefill-threshold-within-max-model-len` |
| `max_cudagraph_capture_size` = the largest `cudagraph_capture_sizes` entry; sizes above `max_num_batched_tokens` are dropped; `cudagraph_mode: NONE` ignores the plan | `cudagraph-capture-sizes` |
| `max_model_len` ≤ the model's catalog `context` | `max-model-len-within-context-window` |
| with EPLB on, `tp × dp` must divide `experts + num_redundant_experts` (the error names the nearest fix) | `expert-divisibility-under-eplb` |
| enabling `dbo` without a threshold measures an unchosen default (the two differ by 16×) | `dbo-thresholds-stated-when-enabled` |
| mamba knobs only on a recurrent model; MoE knobs and `eplb` only on an MoE model | `mamba-cache-mode-matches-model` |

The model-aware rules read `../blis-catalog` when present and are skipped without it, so the
check stays safe on a fork PR with no catalog checkout.

### `model` / `hardware` / `workload` must name something real

With a catalog available the validator also checks that those three **exist**, and suggests the
nearest name when one doesn't:

```
scenario.hardware: 'h200x' is not in the catalog (did you mean 'h200'?)
  -- known: a100-80, a100-sxm, b200, b300, gb200-nvl72, gb300, h100, h200, l40s
```

This is existence, not feasibility. A name the catalog has no row for provably cannot run — BLIS
exits `rc=1` with `opening .../hardware/h200x.yaml: no such file or directory` on *every* run —
so it belongs with the GPU arithmetic, caught before a clone and build rather than after. Whether
a KV cache actually **fits** stays BLIS's call; the validator adds no physics.

Point it at a catalog with `--catalog`, or let it find `../blis-catalog`. Without one the check is
skipped like the model-aware rules, and the summary line says so rather than implying it passed:

```
7/7 valid (knob names checked against ...; model/hardware/workload NOT checked -- no blis-catalog checkout)
```

### What the pinned stack can actually carry

vLLM v0.32.0 defines ~51 engine fields, but the **pinned** BLIS (v0.12.0, which bundles
blis-schemas v0.5.0) has only 31, and the scenario decodes strictly. Naming one of the other 20 —
`max_num_active_seqs`, `watermark`, `prefix_match_unit`, `stream_interval`, `linear_backend`, … —
would fail **every** run with `field … not found in type deployment.Engine`, so the validator
rejects them up front with that reason. They become sweepable when `pins.env` moves.

The 31 that do work, grouped: batching/admission (`max_num_seqs`, `max_num_batched_tokens`,
`long_prefill_token_threshold`, `scheduling_policy`, `async_scheduling`), graph capture
(`cudagraph_mode`, `max_cudagraph_capture_size`, `cudagraph_capture_sizes`), memory/precision
(`cache_dtype`, `quantization`, `block_size`, `gpu_memory_utilization`, `max_model_len`),
attention (`enable_prefix_caching`, `disable_cascade_attn`, `swa_bounded_replay`), MoE and
collectives (`all2all_backend`, `moe_backend`, `allreduce_backend`,
`disable_custom_all_reduce`), context parallelism (`dcp_comm_backend`, `dcp_q_replicate`,
`cp_kv_cache_interleave_size`), recurrent caches (`mamba_cache_dtype`, `mamba_ssm_cache_dtype`,
`mamba_cache_mode`), and the nested blocks below.

### Nested blocks

Four knobs are mappings whose inner fields are each a list. Inner names are the engine's own and
are decoded strictly (`eplb` carries `enable_eplb`, not `enabled`):

```yaml
engine:
  eplb: {enable_eplb: [true], num_redundant_experts: [0, 32, 64]}   # MoE expert-load balancing
  dbo:  {enable_dbo: [true, false]}                                 # dual-batch overlap
  engram_config: {cpu_offload: [true, false]}
```

`speculative` is also nested and needs **both** `method` and `num_spec_tokens`; acceptance comes
from the registry, not a knob.

### Not every accepted knob is a lever

A knob can be accepted and still measure byte-identical — `enable_prefix_caching` does on every
*synthetic shape* workload, because generated prompts share no prefix (it only matters on a
replay), and `all2all_backend` does with expert parallelism off. Sweeping one of those produces a
column of identical numbers that reads as "this knob does nothing", which costs runs and teaches
nothing. The skill's
[`references/knobs.md`](../.claude/skills/blis-experiment/references/knobs.md) records the
measured leverage and each knob's precondition — worth reading before choosing a grid.

## Replay workloads: the catalog already describes them

`blis-catalog/workloads/<name>.yaml` **is** the workload. For `agentx-mvp` it states the harness
scenario id, the corpus loader with its pinned HF revision, the `sha256`, `random_seed`,
`benchmark_duration_s`, `grace_period_s`, `max_context_length`, the cache-bust and trajectory
settings, and the `category`/`thinking` pair that selects the acceptance profile. The sweeper
reads all of it, and **BLIS enforces the catalog's digest even when the spec states no sha** — a
corpus whose bytes differ is refused regardless.

So a replay spec states only what the catalog cannot know:

```yaml
scenario:
  workload: agentx-mvp
  replay:
    corpus: .corpus/traces.jsonl   # where the file sits on THIS machine (it is not vendored)
    corpus_cache: .corpus/cache    # optional local parse cache; measured -46% per run
    grace_period: inf              # a deliberate override of the descriptor's 30s
```

Don't restate `scenario`, `corpus_sha256`, `benchmark_duration`, `max_context_length`,
`category` or `thinking` — a copy can only drift from the catalog, and the validator prints a
style note for each. Keep the corpus path **relative**: an absolute one pins the spec to one
machine. A field that genuinely differs from the descriptor is recorded as an override, so the
report can say the run departed from the catalog.

**To fetch the corpus**, let the descriptor drive it — it already names the dataset, the pinned
revision and the digest:

```
python3 -I scripts/fetch_corpus.py experiments/<name>.yaml --catalog ../blis-catalog
```

It downloads to the path the spec states, verifies the sha256 before moving the file into place,
and is a no-op once the corpus is there — so `run-experiment.yml` runs it on every sweep. A spec
whose workload is a generated shape needs nothing and the script says so. `.corpus/` is
gitignored; the corpus is 568 MB and deliberately not vendored.

## Which knobs apply to your model × hardware × workload

```
python3 -I scripts/knob_profile.py --model <m> --hardware <hw> --workload <wl>
```

Prints, per knob: whether it **applies** here, the values **legal here**, and **why** — read
from the catalog row, the model's `graph.yaml`, blis-vllm's rules pack, and the hardware entry.

**The catalog row alone is not enough.** It carries no speculator, no sliding window, no
KV-dtype table and no capability gating. For example, `deepseek-v4.1-flash` prices
`nvfp4_ds_mla` only under capability family 10, so the profile calls it a real KV axis on
**b200** and not an axis on **h100/h200** — and BLIS agrees, refusing it on SM90. Design from
the row alone and you will either miss a real axis or sweep an impossible one.

Sweep what it marks APPLIES; leave out what it marks DOES NOT APPLY.

## Knob interactions: what a coordinate sweep does not tell you

A sweep that pins one topology and varies one knob measures **that knob at that topology**. For
some knobs that generalises; for others it does not, and the gap is large.

Measured (deepseek-v2-lite, 8 GPUs, 64 sessions), `max_num_seqs` 8 → 256:

| topology | gain |
|---|---|
| `1x TP8` | **+320%** |
| `8x TP1` | **+6.9%** |

Same knob, same budget, same load. 64 sessions over one instance is 64 per instance, where a cap
of 8 bites hard; over eight instances it is ~8 each, where it barely binds. A single-topology
sweep would report one of those numbers and a reader would carry it to the other shape.

**Knobs that interact with topology** — sweep across ≥2 shapes: batch caps (`max_num_seqs`,
`max_num_batched_tokens`), KV capacity (`gpu_memory_utilization`, `cache_dtype`, `block_size`),
MoE collectives (`all2all_backend`, `moe_backend`, `eplb`), `router.*`, drafting depth, and
`long_prefill_token_threshold`.

**Knobs that largely do not** — one topology is enough: `cudagraph_mode`,
`max_cudagraph_capture_size`, `quantization`, `scheduling_policy`, `async_scheduling`.

The GPU filter makes the two-topology form cheap, because it drops the mismatched combinations:

```yaml
- name: batch-width-vs-shape
  narrative: The batch cap should bind hard on few wide instances and barely on many narrow ones.
  num-instances: [1, 8]
  parallel: {tp: [8, 1]}       # keeps 1xTP8 and 8xTP1; drops 1xTP1 and 8xTP8 as off-budget
  engine:
    max_num_seqs: [8, 32, 128, 256]
```

That is 8 configs, not 16. Cross two knobs when the interaction *is* the question —
`cache_dtype` × `gpu_memory_utilization` (add or substitute?), `max_num_seqs` ×
`max_num_batched_tokens` (which cap binds first?), prefix caching × routing (prefix-affinity is
worthless with the cache off).

## Routing

Any run with more than one instance is already a routing experiment, because the llm-d router
decides where every request goes. So routing is a knob like any other:
`router.endpoint_picker_config` lists **EndpointPickerConfig** files from [`routing/`](../routing),
which BLIS runs unchanged. A new policy is a new file, never an invented knob. Omit `router` to
use the llm-d default (itself a legitimate baseline arm).

```yaml
router:
  endpoint_picker_config: [random.yaml, queue.yaml, load-aware.yaml, queue-kv.yaml, prefix-affinity.yaml]
```

### Sweeping fields inside a router config

Name one base file, then list values for fields **the base file already has**, using its own
names. The sweeper writes one config per combination into `reports/<name>/routing/`, each named by
a hash of its contents:

```yaml
router:
  endpoint_picker_config: [queue-kv.yaml]        # exactly one base
  schedulingProfiles:                            # profile -> pluginRef -> field
    default:
      queue-scorer: {weight: [1, 2, 4]}
      kv-cache-utilization-scorer: {weight: [0, 1, 2, 4]}
  plugins:                                       # plugin -> parameters -> field
    approx-prefix-cache-producer:
      parameters: {lruCapacityPerServer: [15625, 31250, 62500]}
```

Overrides only **change** existing fields; adding or removing a plugin takes a separate base file.
An override naming a field the base lacks is refused, so a typo can't silently produce the base
config and read as "this knob does nothing".

Parameters worth sweeping, per #21:

| Plugin | Parameters |
|---|---|
| any scorer | `weight` |
| `approx-prefix-cache-producer` | `lruCapacityPerServer`, `blockSizeTokens`, `maxPrefixTokensToMatch` |
| `prefix-cache-scorer` | `matchLengthWeight`, `matchLengthScaleTokens` |
| `load-aware-scorer` | `threshold` |
| `prefix-based-pd-decider` | `nonCachedTokens`, `promptTokens` |
| `probabilistic-admitter` | `queueDepthThreshold`, `kvCacheUtilThreshold`, `power`, `k` |

## Disaggregated (P/D) sweeps

In BLIS the P/D split is a **router** decision, so a disaggregated sweep states two pools and
varies the P/D decider. The budget counts both pools together.

```yaml
- name: disaggregated
  narrative: Dedicated prefill and decode pools, swept over the prefill:decode ratio.
  prefill-instances: [2, 3, 4]
  decode-instances: [4, 5, 6]
  prefill: {parallel: {tp: [8], enable_expert_parallel: [true]}}
  decode:  {parallel: {tp: [8], enable_expert_parallel: [true]}}
  router: {endpoint_picker_config: [pd-always.yaml, pd-prefix.yaml]}
```

BLIS imposes real constraints on P/D, all checked before any run:

- `prefill-instances` + `decode-instances` must equal the total instance count — every instance
  has a role.
- The two pools must **agree** on `dp` (one replica factor applies to both), `block_size` (a
  handoff moves whole KV blocks), and KV cache format.
- Drafting is **one configuration per run**, not per pool.
- A prefill pool needs a decode pool, and vice versa; a colocated pool can't sit beside either.
- Pools never share a node, so a P/D run always spans ≥ 2 nodes and uses the inter-node fabric.
- The handoff is NIXL (`NixlConnector` by default).

A sweep is colocated **or** disaggregated, never both.

## The committed examples

| Spec | Question |
|---|---|
| [`deepseek-v4-1-flash-h100-agentx-drafting.yaml`](deepseek-v4-1-flash-h100-agentx-drafting.yaml) | Does drafting pay on an agentic replay, and at what depth? |
| [`deepseek-v4-1-flash-h200-agentx-routing.yaml`](deepseek-v4-1-flash-h200-agentx-routing.yaml) | Does routing matter more on few big instances or many small ones? |
| [`kimi-k2-5-h200-multidoc-pd.yaml`](kimi-k2-5-h200-multidoc-pd.yaml) | Colocated vs. disaggregated, and where should the P/D split trigger? |
| [`glm-5-3-flash-h100-chatbot-engine.yaml`](glm-5-3-flash-h100-chatbot-engine.yaml) | Where do the batching and KV-memory knobs bind? |
| [`glm-5-3-flash-h100-chatbot-execution.yaml`](glm-5-3-flash-h100-chatbot-execution.yaml) | Which part of the execution layer binds first — graph capture, prefill admission, batch width or KV capacity? |
| [`qwen3-14b-h100-chatbot-execution.yaml`](qwen3-14b-h100-chatbot-execution.yaml) | The same question on a **dense** model, where the catalog rules out every expert knob and GQA makes KV precision a strong lever |
| [`qwen3-14b-h100-chatbot-kv-precision.yaml`](qwen3-14b-h100-chatbot-kv-precision.yaml) | Does an fp8 KV cache buy throughput on a dense model? |

[`schema.json`](schema.json) (JSON Schema 2020-12) describes the spec's **structure**, so an
editor can flag a malformed spec as you type. It is an aid, not the authority:
**`scripts/validate_experiment.py` is what CI runs**, it does not load the schema, and it checks
everything a schema can't reach — knob names, enum values, GPU arithmetic, SLO units, router
files, and whether the model/hardware/workload exist. A spec your editor likes can still be
rejected; run the validator before opening a PR.

The two are kept honest by `SchemaAgreesWithValidator` in `scripts/test_experiment.py`: every
shipped spec must satisfy both, and a corpus of structurally-broken specs must be rejected by
both. So the schema can be looser than the validator only in the three classes it documents as
out of reach — it can never be *stricter*, which is the case that would have your editor
red-underline a spec CI accepts.

## What the sweeper writes

`reports/<name>/` — the same run-directory format the rest of the repo reads:

| File | Contents |
|---|---|
| `runs.jsonl` | one row per BLIS run: config, deployment, metrics, per-instance metrics, errors |
| `experiment.json` | per-sweep results, narratives, the best config per sweep and overall |
| `scenario.json` | the resolved scenario + pinned stack, as `report.py` / `build_gallery.py` read it |
| `pareto.json` | finalists (one per sweep), the winner and the reason, and dropped combos |
| `spec.yaml` | the spec verbatim, so the directory can be re-rendered on its own |
| `routing/` | any generated router configs, named by content hash |
| `scenarios/` | the generated BLIS scenario YAMLs |

**"Best" is an executable rule, not a judgement:** most output tok/s per GPU among runs that hold
the scenario's SLOs, breaking ties on fleet tok/s then the lighter load. A run that misses an SLO
can never win, however fast it is, and a declared SLO that wasn't measured counts as a miss rather
than a silent pass.

## When the pinned simulator moves

Every report is a function of its spec **and** the pins, so a pin bump makes every committed
report stale at once. And a new simulator can need code changes here first: v0.14.0 needed 14
files (#25), and turned 11 knobs silently inert. So the loop has a gate between the bump and
the regeneration, and the code changes land on the pin PR itself:

```
inference-sim release --bump-blis-pin.yml-->  DRAFT pin PR  <--fix commits--  pin-compat.yml
   --green, ready, merge-->  regen-reports.yml  --one job per report-->  one PR per model batch
```

* **`bump-blis-pin.yml`** polls upstream daily (or on `repository_dispatch: inference-sim-release`,
  or by hand). For a release newer than `BLIS_REF` it rewrites `pins.env` + SETUP.md, checks the
  tag's commit and that the release builds, and opens a **draft** PR with a checklist. It does not
  require this repo's tests to pass: a release that needs code changes still needs its PR. It
  never guesses `CATALOG_REF` / `REGISTRY_REF`.
* **`pin-compat.yml`** runs on every PR that changes `pins.env`; make it a required check. Against
  the new binary it runs:
  * `blis_pin.py drift`: the validator's pinned knob tables against the `blis-vllm` /
    `blis-schemas` versions the release's go.mod names;
  * the suite twice, once hermetic and once with those modules as `../` siblings;
  * `blis_pin.py smoke`: a few rows from each report's own ledger, re-run. It fails on a row that
    ran before and fails now (**LOUD**), and on a knob that moved the numbers before and now
    measures identically to its baseline (**SILENT**).

  It comments what is left to fix. Push the fixes (knob tables, `run_blis.py`, a submission
  field the new stack refuses) onto the same branch until it is green.
* **`regen-reports.yml`** runs when a push to main changes a pin's *value*, which normally means
  the merged pin PR, with its fixes already on main. It re-checks `drift` first, so a pin that
  skipped the gate cannot start a campaign. `scripts/blis_pin.py plan` classifies every
  `reports/<id>/` by its source:
  * an `experiments/` spec, re-swept by `run_experiment.py`;
  * a `submissions/` file, re-searched by `run_search.py`.

  It batches them by model, at most four per PR. Each report is its own job, and a failed one is
  listed in its batch's PR instead of being left silently stale. Each PR carries an old → new
  table of stack, winner, tok/s/GPU and servable runs.

Every PR this loop opens carries the **`auto-update`** label, plus **`blis-pin`** (the draft pin
PR) or **`report-regen`** (a batch of regenerated reports), so `label:auto-update` lists them all.

So **a report with no spec and no submission cannot be regenerated**, and the test suite refuses
one (`test_plan_covers_the_real_gallery`). Re-run a subset by hand with the `only` input.

The 64-GPU AgentX replays need more memory than a hosted runner has; point the `REGEN_RUNS_ON`
repository variable at a larger runner (see the workflow header for all four sizing variables).
The smoke jobs use the same runner.

## Not supported (per #21)

- Adaptive SLO probes or open-loop `--rate` — load is the fixed `concurrency` list.
- Adding or removing router plugins via overrides — that takes another base file.
- Defaults shared across sweeps — knobs two sweeps share are repeated in each.
- More than one hardware type or workload per experiment.
