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
   python3 -I scripts/validate_experiment.py experiments/<name>.yaml --blis ./blis
   ```
3. Check the size of the campaign you are proposing:
   ```
   python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run --blis ./blis
   ```
4. Open a PR. CI validates every spec on the PR (no secrets, safe for forks). A maintainer runs
   the sweeper after merge, and a bot PR adds `reports/<name>/` and the gallery entry.

You do not run BLIS yourself, and you never commit results by hand.

## The format

```yaml
name: deepseek-v4-1-flash-h100-agentx-4node   # also the reports/<name>/ directory

scenario:                       # shared by EVERY sweep
  model: deepseek-v4.1-flash    # a blis-catalog short-name
  hardware: h100                # one GPU type per experiment
  workload: agentx-mvp          # one workload per experiment
  gpus: 32                      # EVERY run uses exactly 32 GPUs
  slo: {itl_p99: 20ms}          # latency caps need units; throughput_per_gpu is a tok/s floor
  concurrency: [4, 8, 16, 32, 64, 128]
  cluster: {fabric: ib-400g}    # inter-node interconnect; omit for the ib-400g default (H100 on RoCE: roce-200g)

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
- **The cluster fabric is a knob, not the hardware.** A fabric is a property of the cluster, not
  the GPU die, so it is stated separately: `scenario.cluster.fabric` names a catalog `networks/`
  class (`ib-400g`, `roce-200g`, `ethernet-100gbe`) and prices the cross-node traffic a
  multi-node run generates. Omit it for the `ib-400g` default; an H100 fleet on RoCE must set
  `roce-200g`, or it is silently priced on InfiniBand at ~2× the real cross-node bandwidth. The
  validator checks the name against the catalog. Single-node runs have no cross-node traffic, so
  the fabric is ignored.
- **Each sweep needs a `narrative`** — **one sentence**, at most two, under ~200 characters.
  It is the single line the report prints above that sweep's measured result, so it states the
  prediction and nothing else. Grid rationale, prior measurements, and why you skipped some other
  sweep all belong in a `#` comment, which stays in the spec and never reaches the report. The
  validator prints a style note (not a failure) for anything longer.

## Knobs are vLLM's own settings

| Spec knob | vLLM setting |
|---|---|
| `parallel.tp`, `dp`, `pp`, `pcp`, `dcp`, `enable_expert_parallel` | `parallel_config.*` (the deployment document's pool layout) |
| `engine.max_num_seqs`, `max_num_batched_tokens`, `max_num_active_seqs`, `watermark`, `scheduling_policy` | `scheduler_config.*` |
| `engine.block_size`, `enable_prefix_caching`, `gpu_memory_utilization`, `cache_dtype` | `cache_config.*` |
| `engine.max_model_len`, `quantization` | `model_config.*` |
| `engine.all2all_backend` | `parallel_config.all2all_backend` |
| `engine.speculative.{method, num_spec_tokens}` | `speculative_config.{method, num_speculative_tokens}` |
| `num-instances`, `prefill-instances`, `decode-instances` | `blis run` replica-count flags, not engine keys |
| `router.*` | llm-d's EndpointPickerConfig, not an engine key |

The validator reads every name from the simulator it is given with `--blis` (or `$BLIS`):
engine settings from `blis fields --engine vllm`, the engine's field list at the release that
binary runs, and pool-layout keys from `blis schema deployment`. This repository keeps no list of
settings (CFG-3). A typo fails before any run, and the message suggests the right name. Four
engine-block names differ from vLLM's (`cache_dtype` for `kv_cache_dtype`, `cudagraph_mode` for
`compilation_config.cudagraph_mode`, and the `speculative` block's `method` and
`num_spec_tokens`); `scripts/settings.py` maps them to their field-list entries until the engine
block takes vLLM's names.

A swept setting that the field list marks `proven_inert` or `inert_unproven` is refused: every
arm of the sweep would report the same number. The message quotes the field list's evidence.

**Values are the engine's.** The validator keeps no value sets. The engine refuses a bad value
at start-up with its own message (X15), and the run records the refusal.

### Cross-knob constraints BLIS does not check

The validator still refuses two combinations the engine accepts and real vLLM refuses at start-up:
`long_prefill_token_threshold` above `max_model_len`, and a CUDA-graph capture plan that no
`max_num_batched_tokens` or `cudagraph_mode` in the sweep can use. Rules that depend on the
model (its context window, expert count or attention kind) are not checked here, because they
would read catalog files directly (design §5); the engine refuses those runs itself.

### `model` and `hardware` must name something real

The validator asks `blis catalog resolve` for each and refuses a name the catalog has no entry
for, with the command's reason:

```
scenario.hardware: 'h200x': the catalog has no hardware entry of this name
```

This is existence, not feasibility. Whether a KV cache actually **fits** stays BLIS's call; the
validator adds no physics. `--catalog` names a catalog directory; without it the binary's
embedded catalog answers.

### What the pinned stack can actually carry

vLLM v0.32.0 defines ~51 engine fields, and at the stack [`pins.env`](../pins.env) pins
**all of them decode**: a pool's `engine:` is an opaque block opened by the engine model that
`engine_version` names, which is blis-vllm's own `engineconfig.Engine`. The pinned engine model is
a blis fork tag (note the suffix), whose `rules/engineconfig` is byte-identical to the plain
upstream v0.32.0 tag's — so these 51 names did not move when the pin last moved (checked
tag-to-tag, both directions). Before v0.14.0 the decode surface was blis-schemas' narrower
31-field struct, and naming one of the other 20 failed **every** run with
`field … not found in type deployment.Engine`.

**Decoding is no longer the interesting question, though — leverage is.** The bump opened up
**21** keys, and only **three** change a number: `max_num_active_seqs`,
`max_num_scheduled_tokens` and `frontend`. The other **18** are refused, in three measured
groups: **11** decode, run to completion and measure *byte-identical* (the kernel cost model has
no term for them, so they are carried as launch provenance), **3** have no cost term at all, and
**4** are priced only in one shape. That was established by running each against the pinned
binary with two contrasting values on `deepseek-v4-pro`/h200, `qwen3-14b`/h100 and
`deepseek-v4.1-flash`/h200, beside a `max_num_seqs` control that moved every time.

(Two different figures are in play, so to be explicit: **21** is what the bump opened up — the
names that used to fail the decode and now run. The **20** above is `51 − 31`, the names the old
31-field struct did not carry at all. They are not the same set.)

So the validator refuses **eighteen** knobs, in three groups, each with its measured reason:

| Why refused | Knobs |
|---|---|
| Decodes and runs, but measures identical | `watermark`, `stream_interval`, `kv_cache_memory_bytes`, `max_loras`, `custom_ops`, `use_trtllm_attention`, `mla_prefill_backend`, `indexer_kv_dtype`, `linear_backend`, `long_prefill_token_threshold_adaptive`, `decoder_replay_cudagraph_capture_sizes` |
| No cost term at all | `prefix_match_unit`, `max_cpu_loras`, `prefix_cache_retention_interval` |
| Priced only in one shape | `prefill_schedule_interval` (needs dp>1), `disable_hybrid_kv_cache_manager` (refused on sliding-window MLA), `env` (one variable is read), `kv_events` (a struct, not a bool) |

A sweep over any of them runs to completion and reports a column of equal numbers, which looks
like a finding. That is the expensive failure, so it is refused up front rather than left to burn
the runs — the same reasoning as "Not every accepted knob is a lever" below.

The 33 that do work, grouped: batching/admission (`max_num_seqs`, `max_num_batched_tokens`,
`long_prefill_token_threshold`, `scheduling_policy`, `async_scheduling`), graph capture
(`cudagraph_mode`, `max_cudagraph_capture_size`, `cudagraph_capture_sizes`), memory/precision
(`cache_dtype`, `quantization`, `block_size`, `gpu_memory_utilization`, `max_model_len`),
attention (`enable_prefix_caching`, `disable_cascade_attn`, `swa_bounded_replay`), MoE and
collectives (`all2all_backend`, `moe_backend`, `allreduce_backend`,
`disable_custom_all_reduce`), context parallelism (`dcp_comm_backend`, `dcp_q_replicate`,
`cp_kv_cache_interleave_size`), recurrent caches (`mamba_cache_dtype`, `mamba_ssm_cache_dtype`,
`mamba_cache_mode`), the three the v0.14.0 pin opened up (`max_num_active_seqs`,
`max_num_scheduled_tokens`, `frontend`), and the nested blocks below.

`max_num_active_seqs` is the one worth knowing about: it is a large lever where it binds and inert
where it does not. On `deepseek-v4-pro`/h200 tp2 dp4 chatbot it did nothing at c=8, and took
2405.3 tok/s down to 808.4 (8 active seqs) and 1684.0 (32) at c=256 — an admission cap only bites
once there is a queue to cap.

### Nested blocks

Four knobs are mappings whose inner fields are each a list. Inner names are the engine's own and
are decoded strictly (`eplb` carries `enable_eplb`, not `enabled`):

```yaml
engine:
  eplb: {enable_eplb: [true], num_redundant_experts: [0, 32, 64]}   # MoE expert-load balancing
  dbo:  {enable_dbo: [true, false]}                                 # dual-batch overlap
  engram_config: {cpu_offload: [true, false]}
```

`speculative` is also nested and needs **both** `method` and `num_spec_tokens`. Other fields
stay nested, including an external catalog `draft_model` and
`enable_adaptive_verification`. Their availability comes from the pinned engine's field list,
not from a second list here. Fields marked inert remain fixed in `scenario.engine`, as the
validator requires. Fixed nested fields survive a sweep over other fields in the same block.

Acceptance normally comes from the registry. An experiment can instead fix
`scenario.acceptance` to an inline value in the native Scenario schema:

```yaml
acceptance:
  value:
    num_spec_tokens: 7
    acceptance_length: 2.5
    source: "Explicit example assumption, not a measurement"
```

The runner validates the value against `blis schema scenario`, requires its depth to match,
records its source, and derives per-token rates from that same value. Acceptance is a fixed
workload assumption, not a deployment knob to optimize. Named-profile pins are not yet wired
through this interface and are refused, as are competing CLI acceptance overrides. A borrowed
or aggregate value does not establish measured accuracy at every load or hardware configuration.

A sweep can cross complete native `Deployment.offload` blocks through `offload`. `null` is the
no-tier control. The pinned deployment schema validates their fields; the runner does not keep
another offload schema or translate weight/Engram offload into KV offload:

```yaml
offload:
  - null
  - connector: OffloadingConnector
    spec: TieringOffloadingSpec
    tiers:
      - tier: cpu_dram
        bytes: 1073741824
```

The native block is deployment-level: every engine receives its stated capacity, following the
owner's connector semantics. It is not a per-pool or per-rank override. The block participates in
the configuration identity, is retained in the ledger, and must produce read-back evidence in
the raw Result before a report claims the CPU tier served requests.

### Not every accepted knob is a lever

A knob can be accepted and still measure byte-identical — `enable_prefix_caching` does on every
*synthetic shape* workload, because generated prompts share no prefix (it only matters on a
replay), and `all2all_backend` does with expert parallelism off. Sweeping one of those produces a
column of identical numbers that reads as "this knob does nothing", which costs runs and teaches
nothing. The skill's
[`references/knobs.md`](../.claude/skills/blis-experiment/references/knobs.md) records the
measured leverage and each knob's precondition — worth reading before choosing a grid.

## Replay workloads: the catalog already describes them

The catalog's `workloads/<name>.yaml` **is** the workload. For `agentx-mvp` it states the harness
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

For diagnostic collection, `scenario.replay.server_series_interval_s` passes the
explicit interval to BLIS's native `--server-series-interval-s` flag. The raw
Result contains per-engine counters, including CPU-tier prompt-token transfers,
under `reports.engine.cluster.server_series`. Omission leaves collection unchanged;
this setting does not change the ranking basis.

For long sweeps, `run_experiment.py --checkpoint-runs` writes and syncs each
finished attempt to `run-checkpoints.jsonl`, including refused and incomplete
attempts. Use a fresh output directory: an existing journal is never truncated.
The journal follows completion order and may be partial; it is not a report or
a resume command. A completed sweep still writes the usual ordered `runs.jsonl`
and `scenario.json`, with unchanged ranking rules. Use bounded experiment shards
and retain the inputs and raw outputs alongside both files.

`scripts/shard_experiment.py` splits a validated experiment into bounded native
specs. It retains every listed choice and every load, and checks each generated
batch with the reference `expand_sweep`. It does not price, rank or prune points.
GPU-budget refusals remain the runner's decision, not a sharder filter.

```bash
SPEC=experiments/muse-spark-open-816b-a42b-gb200-agentx-rack.yaml
python3 -I scripts/shard_experiment.py "$SPEC" --max-runs 64
python3 -I scripts/shard_experiment.py "$SPEC" --max-runs 64 --index 0 --out shard-0.yaml
```

The first command gives the shard count and unfiltered point count. Generate
every index from zero through shard count minus one; validate and execute each
with the usual runner and keep its standard artifacts. Each generated filename
must be new. The parent spec, its hash, runner pins, index mapping and completed
batch records belong in the execution package. A completed batch is not evidence
that the whole parent grid has finished.

**To fetch the corpus**, let the descriptor drive it — it already names the dataset, the pinned
revision and the digest:

```
python3 -I scripts/fetch_corpus.py experiments/<name>.yaml --catalog "$SUBSTRATE_CATALOG"
```

It downloads to the path the spec states, verifies the sha256 before moving the file into place,
and is a no-op once the corpus is there — so `run-experiment.yml` runs it on every sweep. A spec
whose workload is a generated shape needs nothing and the script says so. `.corpus/` is
gitignored; the corpus is 568 MB and deliberately not vendored.

### A workload the catalog does not have: `workload_spec`

A spec names its workload by catalog name (`workload:`) or by a file in this repository
(`workload_spec:`), **never both**. The file is the `replay:` block of a catalog workload entry, and
the sweeper passes it to `blis replay --workload-spec`:

A complete example is `examples/custom-workload-example.yaml` (kept outside `experiments/`, which CI runs).

```yaml
scenario:
  workload_spec: workloads/agentx-uncapped-30s-grace.yaml
  replay:
    corpus: .corpus/traces.jsonl
```

The validator requires `corpus.sha256` (64 lowercase hex digits) and `corpus.source`, either a
Hugging Face `huggingface` dataset with a pinned `revision`, or a `url`, so every report names the
bytes it replayed; the simulator refuses a corpus whose digest differs. The path is relative to the
repository root. The report labels the workload "custom (not in the catalog)" and records the
file's own sha256 and the document as read, in `scenario.json` and `experiment.json`. A custom
workload runs wherever its corpus can be fetched by its declared source and checked against the
sha256.

## Which knobs apply to your model × hardware × workload

```
python3 -I scripts/knob_profile.py --model <m> --hardware <hw> --workload <wl>
```

Prints, per knob: whether it **applies** here, the values **legal here**, and **why** — read
from the catalog row, the model's `graph.yaml`, and the hardware entry. blis-vllm's architecture
rules are not read here: no simulator command publishes them yet.

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
- Pools never share a node, so a P/D run always spans ≥ 2 nodes and prices its handoff over
  `scenario.cluster.fabric` (default `ib-400g`).
- The handoff is NIXL (`NixlConnector` by default).

A sweep is colocated **or** disaggregated, never both.

## A worked P/D example

| Spec | Question |
|---|---|
| [`kimi-k2-5-h200-multidoc-pd.yaml`](kimi-k2-5-h200-multidoc-pd.yaml) | Colocated vs. disaggregated, and where should the P/D split trigger? |

[`schema.json`](schema.json) (JSON Schema 2020-12) describes the spec's **structure**, so an
editor can flag a malformed spec as you type. It is an aid, not the authority:
**`scripts/validate_experiment.py` is what CI runs**, it does not load the schema, and it checks
everything a schema can't reach — knob names, enum values, GPU arithmetic, SLO units, router
files, and whether the model/hardware/workload/fabric exist. A spec your editor likes can still
be rejected; run the validator before opening a PR.

The two are kept honest by `SchemaAgreesWithValidator` in `scripts/test_experiment.py`: every
shipped spec must satisfy both, and a corpus of structurally-broken specs must be rejected by
both. So the schema can be looser than the validator only in the three classes it documents as
out of reach — it can never be *stricter*, which is the case that would have your editor
red-underline a spec CI accepts.

## What the sweeper writes

`reports/<name>/` — the same run-directory format the rest of the repo reads:

| File | Contents |
|---|---|
| `runs.jsonl` | one row per BLIS run: config, deployment, metrics (the measured window), the Result's measured-window metrics verbatim, a load-independent `config_key`, errors; see `report/schema.md` |
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
  or by hand). For a release newer than `BLIS_REF` it rewrites `pins.env` + `SETUP.md`, checks the
  tag's commit and that the release builds, and opens a **draft** PR with a checklist. It does not
  require this repo's tests to pass: a release that needs code changes still needs its PR. It
  needs no catalog or registry pin: the release embeds both.
* **`pin-compat.yml`** runs on every PR, and does its work only when a pin's *value* moves against
  the base branch. Mark its **`gate`** job as the required check: it always reports, and passes when
  no pin moved. Against
  the new binary it runs:
  * `validate_experiment.py experiments/ --blis upstream/blis`: every spec against the new
    binary's field list, schemas and catalog;
  * the hermetic suite;
  * `blis_pin.py smoke`: a few rows from each report's own ledger, re-run. It fails on a row that
    ran before and fails now (**LOUD**), and on a knob that moved the numbers before and now
    measures identically to its baseline (**SILENT**).

  It comments what is left to fix. Push the fixes (`run_blis.py`, a spec field
  the new stack refuses) onto the same branch until it is green.
* **`regen-reports.yml`** runs when a push to main changes a pin's *value*, which normally means
  the merged pin PR, with its fixes already on main. It validates every spec first, so a pin that
  skipped the gate cannot start a campaign. `scripts/blis_pin.py plan` finds each
  `reports/<id>/`'s `experiments/` spec, which `run_experiment.py` re-sweeps.

  It batches them by model, at most four per PR. Each report is its own job, and a failed one is
  listed in its batch's PR instead of being left silently stale. Each PR carries an old → new
  table of stack, winner, tok/s/GPU and servable runs.

Every PR this loop opens carries the **`auto-update`** label, plus **`blis-pin`** (the draft pin
PR) or **`report-regen`** (a batch of regenerated reports), so `label:auto-update` lists them all.

So **a report with no spec cannot be regenerated**, and the test suite refuses
one (`test_plan_covers_the_real_gallery`). Re-run a subset by hand with the `only` input.

The 64-GPU AgentX replays need more memory than a hosted runner has; point the `REGEN_RUNS_ON`
repository variable at a larger runner (see the workflow header for all four sizing variables).
The smoke jobs use the same runner. `deploy/openshift/arc/` sets one up on OpenShift.

## Not supported (per #21)

- Adaptive SLO probes or open-loop `--rate` — load is the fixed `concurrency` list.
- Adding or removing router plugins via overrides — that takes another base file.
- Defaults shared across sweeps — knobs two sweeps share are repeated in each.
- More than one hardware type or workload per experiment.
