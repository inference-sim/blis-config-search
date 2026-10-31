#!/usr/bin/env python3
# run_blis.py -- the ONE place config-search invokes BLIS and records a single run.
#
#   python3 -I scripts/run_blis.py ...        (importable: sweep_pool.py reuses run_one())
#
# BLIS is driven on the **blis-latency-kernel** backend. The kernel reads the deployment (model,
# hardware, tp/pp/dp/ep, engine knobs) from a committed SCENARIO and REFUSES
# --model/--hardware/--tp/--dp/--enable-expert-parallel on the CLI. So a candidate's per-instance
# topology is written into a generated Scenario+Deployment YAML; only workload, concurrency, instance
# counts, scheduler/routing/admission policy, and seed stay on the CLI.
#
# Every candidate key is classified to its scenario/CLI slot; UNKNOWN or NOT-YET-SUPPORTED keys are
# REJECTED (recorded as an error row), never silently dropped -- else two candidates could differ in
# the ledger yet run the identical command.
#
# Pinned stack: see pins.env, the single source of truth for the BLIS/catalog/registry refs that
# price every run (scripts/test_build.py checks the docs and workflows against it). The registry
# must carry the 7 coefficient sets named below, incl cost-model-memory and cost-model-fabric.
import argparse, hashlib, json, os, subprocess, sys, tempfile, time

# cost-model-fabric is the cross-node/cross-rack interconnect term, and it is the set REGISTRY_REF
# v0.3.0 adds (v0.2.0 has no such file, which is why the registry pin had to move with the BLIS
# one). Every one of upstream's 40 Muse scenarios names it, so a scenario this script generates
# names it too.
#
# It is NOT a hard requirement -- a scenario omitting it runs rc=0 -- which is exactly why it is
# listed here rather than left to chance. Measured on the Muse armB layout (h100 roce, 2 nodes,
# tp8 x pp2), holding the workload fixed: 48.61 tok/s with the set against 48.38 without, so the
# term is small on this layout but real, and omitting it is a silent mispricing rather than an
# error. The size of the gap is a property of this layout and should not be read as a bound: it
# prices the links a collective actually crosses, so a layout that crosses more of them pays more.
COEFFICIENTS = ["cost-model-primitives", "cost-model-collectives", "cost-model-host-overheads",
                "cost-model-attention", "cost-model-recurrent", "cost-model-memory",
                "cost-model-fabric"]

# --- candidate-key classification (allowlist; anything else is rejected, never silently ignored) ---
SCENARIO_PARALLEL = ("tp", "pp", "dp", "enable_expert_parallel")       # -> scenario Deployment.parallel
# Engine fields the generated scenario may carry. The authority IS blis-vllm v0.32.0's
# rules/engineconfig.Engine (51 top-level yaml tags), because the pinned BLIS (pins.env) makes a
# pool's `engine:` an opaque block (deployment.EngineBlock) opened by the engine model that
# the scenario's `engine_version` names. Before v0.14.0 the authority was blis-schemas' own
# 31-field struct, and the 20 names below this comment's old cut-off were hard decode failures
# ("field max_num_active_seqs not found in type deployment.Engine"); they decode now.
#
# The scenario still decodes STRICTLY, so a key outside those 51 fails every run of a sweep. The
# seven that decode but are not priced (or are priced only in one shape) are refused by name in
# UNSUPPORTED below, with the measured reason, rather than being omitted from this list.
#
# Valid VALUES for the enumerated ones come from blis-vllm's rules pack
# (rules/v0_32/pack.go Pack(): CUDAGraphModes, CacheDTypes, All2AllBackends, MoEBackends,
# SchedulerPolicies, MambaDTypes, DCPCommBackends) -- see scripts/validate_experiment.py, which
# checks a swept value against them so a typo fails before the sweep rather than per run.
SCENARIO_ENGINE = (
    # --- the original eight (every committed report was produced with these) ---
    "quantization", "cache_dtype", "block_size", "max_num_batched_tokens",
    "max_num_seqs", "max_model_len", "cudagraph_mode", "gpu_memory_utilization",
    # --- scheduling / admission ---
    # `long_prefill_token_threshold` was previously listed as UNSUPPORTED ("not a kernel
    # scenario field"). That is stale: it IS a field of the v0.5.0 Engine struct, and a run
    # naming it completes and moves the number (measured: 1983.2 -> 1973.9 tok/s at 256 on
    # chatbot tp8 c=8). It is the chunked-prefill cap a prefill-bound deployment is forced onto.
    "long_prefill_token_threshold",
    "scheduling_policy",                  # fcfs (default) | priority
    "async_scheduling",                   # tri-state bool
    # --- CUDA graph capture: the biggest engine lever measured on the pinned stack ---
    # chatbot tp8 c=8: PIECEWISE 1983.2, FULL 2086.1 (+5%), FULL_AND_PIECEWISE 2085.1,
    # FULL_DECODE_ONLY 2074.2; NONE costs 44% (605.9 vs 1083.2 at c=4).
    "max_cudagraph_capture_size", "cudagraph_capture_sizes",
    # --- attention / prefix reuse ---
    "enable_prefix_caching",              # tri-state bool; see the INERT note in the skill
    "disable_cascade_attn", "swa_bounded_replay",
    # --- MoE dispatch. all2all_backend is a real lever, but ONLY with expert parallelism on
    # (measured tp4/dp2/EP: 928.2 -> 619.5 with deepep_low_latency, a 33% drop; with EP off it
    # is byte-identical). The skill gates it on EP for that reason.
    "all2all_backend", "moe_backend",
    # --- collectives ---
    "allreduce_backend", "disable_custom_all_reduce",
    # --- context parallelism ---
    "dcp_comm_backend", "dcp_q_replicate", "cp_kv_cache_interleave_size",
    # --- recurrent (Mamba-family) state caches ---
    "mamba_cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode",
    # --- NESTED blocks (rendered as sub-blocks, like `speculative`) ---
    # `eplb` is the interesting one: expert-load balancing adds redundant experts to even out a
    # skewed routing distribution, which is an MoE-only lever with a real memory cost.
    # `dbo` is dual-batch overlap; `engram_config` places Engram's host-side tables.
    "eplb", "dbo", "engram_config",
    # --- opened up by the v0.12.0 -> v0.14.0 pin bump ---
    # These were hard decode failures under blis-schemas v0.5.0's fixed Engine struct and were
    # listed in UNSUPPORTED for that reason. v0.14.0 reads `engine:` as an opaque block against
    # blis-vllm v0.32.0's engineconfig.Engine, which carries all 21 of them.
    #
    # Only the three below are PRICED, and that was established by running each against the
    # pinned binary with two contrasting values, not by reading the struct. `max_num_active_seqs`
    # is a large lever where it binds (deepseek-v4-pro/h200 tp2 dp4 chatbot: 2405.3 tok/s at
    # c=256 falls to 808.4 at 8 active seqs, 1684.0 at 32) and inert at c=8, which is the usual
    # admission-cap shape. The other eleven decode and run but are byte-identical on every
    # scenario measured -- see ENGINE_DECODED_BUT_INERT below for what that means for a sweep.
    "max_num_active_seqs", "max_num_scheduled_tokens", "frontend",
)
# Carried by the pinned engine model, accepted at run time, and measured BYTE-IDENTICAL: sweeping
# one burns runs to produce a column of equal numbers that reads as "this knob does nothing".
#
# Measured on deepseek-v4-pro/h200 (tp2 dp4, chatbot, c=8 and c=256), qwen3-14b/h100 (tp4) and
# deepseek-v4.1-flash/h200 (tp2 dp4), two contrasting values each, against a `max_num_seqs`
# control that moved as expected on every one. The kernel cost model has no term for them; the
# engine carries them as launch provenance.
#
# `frontend` is deliberately NOT here: it is inert on v4-pro and qwen3-14b but DOES move
# deepseek-v4.1-flash (4656.6 -> 4563.5 python->rust), so it is a real if narrow host-overhead
# lever and belongs in SCENARIO_ENGINE above.
#
# These are refused rather than silently carried, for the reason experiments/README.md gives under
# "Not every accepted knob is a lever": the sweep that teaches nothing is the expensive failure,
# because it looks like a result. A spec that genuinely wants one recorded can state it as a
# fixed engine default rather than a swept knob.
ENGINE_DECODED_BUT_INERT = {
    "watermark": "the KV-watermark fraction",
    "stream_interval": "how often decoded tokens are flushed to the client",
    "kv_cache_memory_bytes": "an explicit KV pool size (gpu_memory_utilization is the live knob)",
    "max_loras": "the LoRA slot count",
    "custom_ops": "the custom-op enable/disable list",
    "use_trtllm_attention": "the TRT-LLM attention path",
    "mla_prefill_backend": "the MLA prefill kernel choice",
    "indexer_kv_dtype": "the DSA indexer's KV dtype",
    "linear_backend": "the linear-layer kernel choice",
    "long_prefill_token_threshold_adaptive": "adaptive chunked-prefill sizing",
    "decoder_replay_cudagraph_capture_sizes": "the decode-replay capture plan",
}
# Nested engine blocks and the fields each accepts, from the pinned engine model's structs
# (blis-vllm v0.32.0 rules/engineconfig). A spec states these as a mapping, e.g.
# `eplb: {enable_eplb: [true], num_redundant_experts: [0, 32]}`; the runner renders them as
# sub-blocks and the validator checks their inner field names.
#
# `speculative` gained five fields at the v0.14.0 pin (it was method + num_spec_tokens under
# blis-schemas v0.5.0). Only method and num_spec_tokens select an acceptance profile and so move
# a number; the rest are recorded provenance -- validate_experiment.SPECULATIVE_RECORDED_ONLY
# refuses a sweep over them, with that reason.
NESTED_ENGINE_FIELDS = {
    "eplb": ("enable_eplb", "num_redundant_experts", "window_size", "step_interval"),
    "dbo": ("enable_dbo", "dbo_decode_token_threshold", "dbo_prefill_token_threshold"),
    "engram_config": ("cpu_offload", "dp_shared_memory", "use_thp"),
    "speculative": ("method", "num_spec_tokens", "draft_model", "kv_cache_dtype",
                    "draft_sample_method", "rejection_sample_method",
                    "enable_adaptive_verification"),
}
# Speculative decoding IS priced by the kernel (since BLIS v0.12.0 + registry v0.2.0; v0.14.0 added
# only the five extra fields above). It is a nested `speculative: {method, num_spec_tokens}` block
# on the pool's engine (the pinned engine model's engineconfig.Engine.Speculative), so the two
# candidate keys below are folded into that block rather than emitted as flat engine fields.
#
# The ACCEPTANCE (how many drafted tokens land) is not a deployment property: it depends on the
# model, drafter and traffic. BLIS resolves it from the registry's acceptance/ profiles, selected
# by the workload's category and thinking mode, and REFUSES a drafting scenario whose acceptance
# does not resolve ("its acceptance does not resolve: ... state one"). So a spec-decode run must
# also carry the selectors -- see SPECULATIVE_CLI.
SCENARIO_SPECULATIVE = ("speculative_method", "num_speculative_tokens")
# Acceptance selectors / overrides. These are CLI, not scenario: they describe the WORKLOAD's
# acceptance behaviour, not the deployment.
SPECULATIVE_CLI = {"speculative_category": "--speculative-category",
                   "speculative_thinking": "--speculative-thinking",
                   "speculative_acceptance_length": "--speculative-acceptance-length",
                   "speculative_acceptance_rate": "--speculative-acceptance-rate",
                   "speculative_fallback_category": "--speculative-fallback-category"}
CLI_FLAGS = {"num_instances": "--num-instances", "concurrency": "--concurrency",
             "scheduler": "--scheduler", "preemption_policy": "--preemption-policy"}
# The llm-d EndpointPickerConfig this deployment routes with, as a path. Named by the scenario's
# DEPLOYMENT-level `router:` block (blis-schemas spec/deployment/deployment.go, pinned in
# pins.env) -- see ROUTING_MOVED below for why this replaced the former --routing-policy flag.
SCENARIO_ROUTER = ("router_config",)
PARALLEL_DEFAULTS = {"tp": 1, "pp": 1, "dp": 1, "enable_expert_parallel": False}
GPUS_PER_NODE = 8   # single-node H100; a per-instance footprint beyond this spans nodes (needs a fabric)
# Cross-node cost resolves from the fabric (a catalog networks/ entry), required once nodes > 1.
DEFAULT_FABRIC = "ib-400g"   # InfiniBand NDR 400 Gb/s; the standard H100 multi-node interconnect
# NON-PRECISION engine defaults only. quantization/cache_dtype are intentionally NOT defaulted: an
# empty precision lets the checkpoint format govern (e.g. qwen3-14b stays BF16), so forcing FP8 here
# would silently mis-price every non-FP8 model. `block_size` is also absent here because its right
# value depends on the MODEL -- see default_block_size(), which fills it in per run.
#
# `max_model_len` is not merely a convenience here: blis-vllm v0.32.0's rules make it MANDATORY
# ("must be stated; an unstated or auto window is the model's or the memory's"), so the pinned
# stack refuses a scenario without it. Because this default always emits one, the five shipped
# specs that never mention max_model_len still run at the pin -- the requirement is met
# by the runner, not by the spec. Do not "fix" those specs by adding it; stating it there would
# pin a context window the spec does not actually care about.
ENGINE_DEFAULTS = {"max_num_batched_tokens": 8192, "max_num_seqs": 256,
                   "max_model_len": 32768, "cudagraph_mode": "PIECEWISE", "gpu_memory_utilization": 0.9}
# A DSA sparse-MLA model's indexer runs on 64-token kernel blocks, so a stated block_size must be a
# multiple of 64 or the engine refuses the layout at startup (blis-latency-kernel sparse_mla.go,
# `dsaBlockSize`). vLLM v0.31.0 picks 64 for these models for the same reason.
DSA_BLOCK_SIZE = 64
DEFAULT_BLOCK_SIZE = 16     # vLLM's own default for everything else
# The DeepSeek-V4 family runs its OWN FlashMLA backends, not the DSA ones, and their kernel block
# depends on the PART as well as the architecture (blis-latency-kernel@v0.3.0 sparse_mla.go:95,
# `v4BlockSize`): DeepSeek-V4 runs 256 everywhere, while DeepSeek-V4.1 runs 64 on Hopper (SM90) and
# 128 on everything newer. A flat DSA 64 therefore fails every V4.1 Blackwell run ("block_size 64
# on a DeepSeek-V4-family model: its sparse attention and indexer run 128-token kernel blocks on
# this part"). The kernel would pick the right value itself, but only when block_size is UNSET --
# and the adapter refuses that ("deployment states no block_size"), so it is computed here.
V4_ARCH = "DeepseekV4ForCausalLM"
V41_ARCH = "DeepseekV41ForCausalLM"
V4_BLOCK_SIZE = 256             # DeepSeek-V4, every part
V41_BLOCK_SIZE_HOPPER = 64      # DeepSeek-V4.1 on SM90
V41_BLOCK_SIZE_NEWER = 128      # DeepSeek-V4.1 on Blackwell and later
HOPPER_CC_MAJOR = 9             # compute capability 9.x == Hopper (SM90)
# inference-sim v0.12.0 ("Merge llm-d-r into main: all routing through the embedded llm-d router",
# #2000) REMOVED every routing/admission CLI flag this runner used to emit. Routing is now a real
# llm-d EndpointPickerConfig named by the scenario's deployment-level `router:` block, so the knob
# is `router_config` (a path to an EPP YAML) rather than a policy name. The authoritative mapping
# from each former flag is inference-sim docs/guide/routing.md, "From BLIS's former routing
# features". Note that round-robin / least-loaded / always-busiest are DROPPED outright there --
# "the llm-d router has no such policy" -- so `least-loaded`, the biggest lever the earlier
# campaigns measured, has no exact equivalent; the nearest are a `queue-scorer` or
# `load-aware-scorer` profile with `max-score-picker`, or `random-picker`.
ROUTING_MOVED = (
    "removed in inference-sim v0.12.0: routing is now an llm-d EndpointPickerConfig named by the "
    "scenario's `router:` block -- pass `router_config` (a path to an EPP YAML) instead. See "
    "docs/guide/routing.md 'From BLIS's former routing features' for the per-flag mapping; "
    "round-robin/least-loaded/always-busiest are dropped upstream (nearest: a queue-scorer or "
    "load-aware-scorer profile with max-score-picker, or random-picker)")
# --- P/D disaggregation ------------------------------------------------------------------------
#
# A disaggregated run states TWO pools -- one `role: prefill`, one `role: decode` -- and BLIS
# routes each request's prefill and decode to the matching pool. The constraints below are
# BLIS's own, read from blis-vllm compat/build/pools.go (openPools) and blis-schemas
# spec/deployment/validate.go, not inferred:
#
#   * `--prefill-instances` + `--decode-instances` must equal `--num-instances`: "in a
#     disaggregated run every instance is a prefill or a decode instance, so the counts must
#     add up" (pools.go:155). An instance with no role would run the first pool's engine.
#   * A prefill pool REQUIRES `pd_transfer.connector`, and BLIS models only NIXL's handoff:
#     NixlConnector / NixlPullConnector / NixlPushConnector, or a MultiConnector over one
#     (RequireNIXLHandoff, pools.go:218).
#   * The two pools must AGREE on `block_size` (a handoff moves whole blocks), on KV cache
#     format, on `dp` (one replica factor applies to both pools) and on the speculative config
#     (one draft width per run). Each disagreement is a hard refusal upstream, so they are
#     checked here -- a refusal costs a whole sweep of identical errors otherwise.
#   * blis-schemas refuses a prefill pool with no decode pool, a decode pool with no prefill
#     pool, and a colocated pool alongside either (validate.go:499-506).
#   * Pools never share a node, so a P/D run always spans >= 2 nodes and needs `cluster.fabric`.
PD_ROLES = ("prefill", "decode")
# Per-role layout/engine knobs, as `<role>_<knob>` candidate keys (prefill_tp, decode_block_size).
# Flat rather than nested so a candidate dict stays one level deep, like every other config.
PD_PARALLEL = tuple(f"{r}_{k}" for r in PD_ROLES for k in SCENARIO_PARALLEL)
PD_ENGINE = tuple(f"{r}_{k}" for r in PD_ROLES for k in SCENARIO_ENGINE)
# Per-role DRAFTING keys exist only to be refused with the real reason. BLIS applies one
# speculative configuration to the whole run ("one speculative configuration applies to the run,
# so the pools must agree"), so `prefill_num_speculative_tokens` is not a narrower knob than
# `num_speculative_tokens` -- it is a misunderstanding of where drafting is configured. Accepting
# the key here and rejecting it in resolve_pd gives that message instead of "unknown key".
PD_SPECULATIVE = tuple(f"{r}_{k}" for r in PD_ROLES for k in SCENARIO_SPECULATIVE)
PD_CLI = {"prefill_instances": "--prefill-instances", "decode_instances": "--decode-instances"}
# The handoff connector is a DEPLOYMENT field (`pd_transfer.connector`), not a flag, so it is
# accepted as a candidate key and consumed by resolve_pd rather than emitted on the CLI.
PD_DEPLOYMENT = ("pd_connector",)
# BLIS prices only NIXL handoffs; the default is the plain connector llm-d deploys.
NIXL_CONNECTORS = ("NixlConnector", "NixlPullConnector", "NixlPushConnector", "MultiConnector")
DEFAULT_PD_CONNECTOR = "NixlConnector"
# Knobs a P/D run must hold IDENTICAL across its two pools (see the module note above). Drafting
# is NOT here: it is one configuration per run rather than a per-pool knob, so a per-role
# drafting key is refused outright (PD_SPECULATIVE) and the folded `speculative` block is
# compared separately.
# Transcribed from blis-vllm v0.32.0 compat/build/pools.go, not guessed: block_size (:172) and
# dp (:188) are each checked on their own, and requireSameKVCacheFormat (:329) compares exactly
# four cache-format fields. `quantization` was listed here and is NOT compared by any of them --
# it is the weights' format, which a handoff never moves -- so a P/D sweep over it is legitimate.
PD_MUST_MATCH = ("dp", "block_size", "cache_dtype", "mamba_cache_dtype",
                 "mamba_ssm_cache_dtype", "mamba_cache_mode")

# Deliberately unsupported in this runner -> fail loudly with the reason (do not silently drop).
UNSUPPORTED = {
    "routing_policy": ROUTING_MOVED,
    "routing_scorers": ROUTING_MOVED,
    "admission_policy": ROUTING_MOVED,
    "prefill_routing_scorers": ROUTING_MOVED,
    "decode_routing_scorers": ROUTING_MOVED,
    "prefill_decode_instances":
        "--prefill-decode-instances is not supported: a blis-schemas deployment's roles are "
        "colocated, prefill and decode, so no scenario describes a shared instance's engine",
    "encode_instances":
        "--encode-instances is not supported: there is no encode role in a blis-schemas "
        "deployment",
    # Engine fields the pinned stack DECODES but its engine model does not price, and ones it
    # prices only in a particular shape. Under BLIS v0.12.0 this covered 21 names and was about
    # decoding (blis-schemas v0.5.0's Engine was a fixed 31-field struct); v0.14.0 makes `engine:`
    # an opaque block opened by blis-vllm v0.32.0's engineconfig.Engine, so all 21 decode and 3
    # are swept normally. These seven are refused because a sweep over them reports noise; the
    # other eleven are refused as ENGINE_DECODED_BUT_INERT, for measuring identical.
    #
    # Kept in sync with validate_experiment.ENGINE_NOT_WIRED by test_experiment.py.
    "prefix_match_unit":
        "prefix_match_unit decodes under the pinned engine model but is not priced by it: prefix "
        "matching is modelled at block granularity, so sweeping the unit reports noise "
        "(upstream blis-vllm#7)",
    "max_cpu_loras":
        "max_cpu_loras decodes under the pinned engine model but is not priced by it: there is no "
        "CPU-side LoRA swap term in the cost model (upstream blis-vllm#5)",
    "prefix_cache_retention_interval":
        "prefix_cache_retention_interval decodes under the pinned engine model but is not priced "
        "by it: the prefix cache is modelled without eviction over time, so retention has no "
        "effect on any run",
    # These three are refused OUTRIGHT, not gated on their condition. The condition is stated so
    # the reason is auditable, but do not read it as "satisfy it and the sweep runs" -- the runner
    # refuses every shape. Gating on dp / model family / env key would be a real improvement and
    # is deliberately not in this change.
    "prefill_schedule_interval":
        "prefill_schedule_interval is refused: it is priced only when the pool has dp > 1 (with a "
        "single data-parallel rank there is no cross-rank prefill schedule to interval), and the "
        "runner does not gate on dp -- it refuses the knob in every shape",
    "disable_hybrid_kv_cache_manager":
        "disable_hybrid_kv_cache_manager is refused: the binary rejects it on a sliding-window "
        "MLA model (the DeepSeek-V4 family), where the hybrid manager is what models the sliding "
        "window, and the runner refuses it on every model rather than gating on the family",
    "env":
        "env is refused: the pinned engine model reads exactly one variable, "
        "VLLM_USE_V2_MODEL_RUNNER, and any other key is carried but never read, so a sweep over "
        "it reports noise -- the runner refuses the block rather than gating on the key",
    "kv_events":
        "kv_events is a struct under the pinned engine model (publisher/topic/buffer), not a "
        "bool, and none of its inner fields is sweepable at this pin either (no spec can express "
        "them) -- so it is not sweepable at all here, in any form",
    # Decoded and accepted, but measured byte-identical on every scenario tried.
    **{k: (f"{k} ({what}) is carried by the pinned engine model but the kernel cost model has no "
           f"term for it: measured byte-identical on deepseek-v4-pro/h200, qwen3-14b/h100 and "
           f"deepseek-v4.1-flash/h200 with two contrasting values each, so a sweep over it "
           f"returns a column of equal numbers. State it as a fixed engine default if the run "
           f"should record it.")
       for k, what in ENGINE_DECODED_BUT_INERT.items()},
}


class ConfigError(ValueError):
    pass


def classify_config(config):
    """Split a candidate into (parallel, engine, cli, router); raise ConfigError on unknown keys.

    The speculative keys are folded into engine["speculative"], the nested block the schema wants,
    and the acceptance selectors go to the CLI.

    A DISAGGREGATED candidate (one carrying `prefill_instances`/`decode_instances` or any
    `prefill_*`/`decode_*` layout key) additionally gets a `pd` block on the returned parallel
    dict, holding each role's resolved layout and engine. See the P/D note at the top of this
    module for the constraints BLIS enforces on it.
    """
    parallel, engine, cli, router = dict(PARALLEL_DEFAULTS), dict(ENGINE_DEFAULTS), {}, {}
    spec = {}
    pd_parallel = {r: {} for r in PD_ROLES}
    pd_engine = {r: {} for r in PD_ROLES}
    for k, v in config.items():
        if k in SCENARIO_PARALLEL:
            parallel[k] = v
        elif k in SCENARIO_ENGINE:
            engine[k] = v
        elif k in SCENARIO_SPECULATIVE:
            spec[k] = v
        elif k in SPECULATIVE_CLI:
            cli[k] = v
        elif k in SCENARIO_ROUTER:
            router[k] = v
        elif k in PD_CLI or k in PD_DEPLOYMENT:
            cli[k] = v
        elif k in PD_PARALLEL:
            role, knob = k.split("_", 1)
            pd_parallel[role][knob] = v
        elif k in PD_ENGINE or k in PD_SPECULATIVE:
            role, knob = k.split("_", 1)
            pd_engine[role][knob] = v
        elif k in CLI_FLAGS:
            cli[k] = v
        elif k in UNSUPPORTED:
            raise ConfigError(f"{k!r}: {UNSUPPORTED[k]}")
        else:
            supported = (sorted(SCENARIO_PARALLEL) + sorted(SCENARIO_ENGINE)
                         + sorted(SCENARIO_SPECULATIVE) + sorted(SPECULATIVE_CLI)
                         + sorted(SCENARIO_ROUTER) + sorted(CLI_FLAGS)
                         + sorted(PD_CLI) + sorted(PD_DEPLOYMENT) + sorted(PD_PARALLEL)
                         + sorted(PD_ENGINE))
            raise ConfigError(f"{k!r}: unknown key; supported = {supported}")
    if spec:
        k = int(spec.get("num_speculative_tokens") or 0)
        if k > 0:
            # `method` defaults to the speculator the model's own graph declares; naming it here
            # would hardcode one model's drafter (deepseek-v4.1-flash is dspark, glm-5.2 is mtp).
            method = spec.get("speculative_method")
            engine["speculative"] = ({"method": method, "num_spec_tokens": k} if method
                                     else {"num_spec_tokens": k})
        elif spec.get("speculative_method"):
            raise ConfigError(
                "speculative_method with num_speculative_tokens 0 (or unset): drafting 0 tokens is "
                "speculative decoding OFF, so naming a method would record a drafter that never ran")
    rc = router.get("router_config")
    if rc is not None and not os.path.exists(rc):
        # Fail here rather than let BLIS fail per-run: a typo'd EPP path would otherwise cost a
        # whole sweep of identical "no such file" rows.
        raise ConfigError(f"router_config {rc!r}: no such file")
    # pp>1 was refused through KERNEL_REF v0.4.0 ("the pinned kernel does not price pipeline
    # parallelism"). Kernel v0.5.0 prices it: each stage is a step over its share of the layers
    # (get_pp_indices), the embedding on the first and the head on the last, plus the boundary
    # send and the receiving stage's TP all-gather; the binding rank is the stage whose pool holds
    # the fewest tokens.
    #
    # It is a real lever, not merely accepted. Measured on the Muse armB layout (h100 roce, 2
    # nodes, 16 GPUs, EP on, 8 requests of 4096/64): tp8 x pp2 = 96.80 tok/s against
    # tp16 x pp1 = 99.47 at the SAME footprint -- so the two layouts price apart. The gap's size
    # is a property of that workload; it is not a bound, and pp should be swept against the
    # topology rather than assumed to cost a fixed amount.
    #
    # The boundary SEND is priced from the graph's pipeline_boundary_bytes_per_token, which only 5
    # of the catalog's 42 models declare. A model without it is NOT refused: measured on
    # qwen3-14b, tp2 x pp2 runs rc=0 and prices the stages while omitting the wire between them.
    # That is a mispricing this layer cannot detect without reading every graph, and it is not an
    # error, so it is left to the spec author (knobs.md says which models carry the field) and to
    # the run's own pipeline_stages / pipeline_binding_stage assumption warnings.
    fp = footprint_gpus(parallel)
    # A multi-node single instance IS supported: the generated scenario names a fabric once the
    # pool spans nodes (see scenario_yaml), and BLIS prices the cross-node collectives from it.
    # This matters because the largest MoE models cannot fit in one node at all -- kimi-k3 needs
    # 332.3 GiB per rank at tp=8 on a 141 GiB device and only fits from tp=32 (4 nodes) up.
    # What is still refused is a footprint that STRADDLES a node boundary: pools are allocated in
    # whole nodes, so the remainder would sit idle and the cross-node term would be mispriced.
    if fp > GPUS_PER_NODE and fp % GPUS_PER_NODE:
        raise ConfigError(
            f"per-instance footprint tp*pp*dp={fp} spans {fp / GPUS_PER_NODE:.2g} nodes: a "
            f"multi-node instance must fill whole {GPUS_PER_NODE}-GPU nodes (use a footprint that "
            f"is a multiple of {GPUS_PER_NODE}, or scale out with num_instances instead)")
    pd = resolve_pd(cli, pd_parallel, pd_engine, parallel, engine)
    if pd:
        parallel = {**parallel, "pd": pd}
    return parallel, engine, cli, router


def _pd_requested(cli, pd_parallel, pd_engine):
    """True when a candidate describes a disaggregated run at all.

    `pd_connector` counts: naming a handoff connector on a colocated run means the candidate
    was written as a P/D one and lost its pools, which should fail loudly rather than run as a
    single pool with the connector quietly dropped.
    """
    return bool(cli.get("prefill_instances") is not None
                or cli.get("decode_instances") is not None
                or cli.get("pd_connector") is not None
                or any(pd_parallel.values()) or any(pd_engine.values()))


def resolve_pd(cli, pd_parallel, pd_engine, base_parallel, base_engine):
    """Resolve a disaggregated candidate into {role: {parallel, engine, instances}}, or {}.

    Each role's layout is the shared `parallel`/`engine` overridden by that role's own
    `prefill_*`/`decode_*` keys, so a sweep states only what differs between the pools. Every
    constraint BLIS would refuse later is checked here instead -- a per-run refusal costs the
    whole sweep the same error (see the P/D note at the top of this module).
    """
    if not _pd_requested(cli, pd_parallel, pd_engine):
        return {}
    pi, di = cli.get("prefill_instances"), cli.get("decode_instances")
    if pi is None or di is None:
        missing = "decode_instances" if pi is not None else "prefill_instances"
        raise ConfigError(
            f"a disaggregated candidate states no {missing}: BLIS refuses a pool with no "
            f"counterpart pool (a prefill pool has nowhere to send KV, a decode pool no source)")
    # Checked BEFORE the pools are built: a per-role drafting key must not reach a pool's engine
    # block, where it would be written into the scenario as a field no engine accepts.
    for role in PD_ROLES:
        bad = sorted(k for k in pd_engine[role] if k in SCENARIO_SPECULATIVE)
        if bad:
            raise ConfigError(
                f"{role}_{bad[0]}: drafting is one configuration per run, not per pool -- BLIS "
                f"applies one speculative configuration to the whole run, so state "
                f"{bad[0]} once for the candidate instead")
    pi, di = int(pi), int(di)
    if pi < 1 or di < 1:
        raise ConfigError(f"prefill_instances={pi}, decode_instances={di}: each role needs at "
                          f"least one instance in a disaggregated run")
    # BLIS checks this itself and refuses, but the message there names only the three numbers;
    # raising it here lets the sweeper drop the combo with the reason recorded.
    ni = cli.get("num_instances")
    if ni is None:
        cli["num_instances"] = ni = pi + di
    elif int(ni) != pi + di:
        raise ConfigError(
            f"num_instances={ni}, but prefill_instances={pi} + decode_instances={di}={pi + di}: "
            f"in a disaggregated run every instance is a prefill or a decode instance, so the "
            f"counts must add up")
    out = {}
    for role, instances in (("prefill", pi), ("decode", di)):
        par = {**{k: v for k, v in base_parallel.items() if k != "pd"}, **pd_parallel[role]}
        eng = dict(base_engine)
        # The shared `speculative` block is already folded into base_engine; a per-role override
        # of it is refused below (one draft width per run), so only flat keys merge here.
        eng.update(pd_engine[role])
        # pp>1 is priced from kernel v0.5.0 on each side of a disaggregated run too; see the
        # colocated site in classify_config for what moved and what still refuses.
        fp = footprint_gpus(par)
        if fp > GPUS_PER_NODE and fp % GPUS_PER_NODE:
            raise ConfigError(
                f"{role} pool footprint tp*pp*dp={fp} straddles a node boundary: a multi-node "
                f"pool must fill whole {GPUS_PER_NODE}-GPU nodes")
        out[role] = {"parallel": par, "engine": eng, "instances": instances,
                     "footprint_gpus_per_instance": fp}
    # The pools must agree on everything a handoff depends on. Each of these is a hard refusal
    # upstream; checking them together means one clear message instead of a sweep of them.
    for knob in PD_MUST_MATCH:
        a = out["prefill"]["parallel"].get(knob, out["prefill"]["engine"].get(knob))
        b = out["decode"]["parallel"].get(knob, out["decode"]["engine"].get(knob))
        if a != b:
            raise ConfigError(_pd_mismatch_reason(knob, a, b))
    if out["prefill"]["engine"].get("speculative") != out["decode"]["engine"].get("speculative"):
        raise ConfigError(
            "the prefill and decode pools state different speculative configurations: one "
            "speculative configuration applies to the whole run, so the pools must agree")
    connector = cli.pop("pd_connector", DEFAULT_PD_CONNECTOR)
    if connector not in NIXL_CONNECTORS:
        raise ConfigError(
            f"pd_connector {connector!r}: BLIS models NIXL's handoff (one of "
            f"{', '.join(NIXL_CONNECTORS)})")
    out["connector"] = connector
    return out


def _pd_mismatch_reason(knob, a, b):
    """Why BLIS refuses two pools that disagree on `knob` -- its own reason, not a generic one."""
    why = {
        "dp": "one replica factor applies to both pools, so their dp must match",
        "block_size": "a P/D handoff moves whole KV blocks, so the pools must page identically",
        "cache_dtype": "the handoff moves the prefill pool's KV representation into the decode "
                       "pool's blocks with no conversion modelled, so the cache formats must agree",
        # No `quantization` entry: requireSameKVCacheFormat compares cache_dtype and the three
        # mamba_* fields and does NOT compare quantization, so PD_MUST_MATCH dropped it. A reason
        # for a knob that is never checked would be dead text.
        "mamba_cache_dtype": "requireSameKVCacheFormat compares the recurrent cache format too, "
                             "and the handoff moves those pages without conversion",
        "mamba_ssm_cache_dtype": "requireSameKVCacheFormat compares the SSM cache format too, and "
                                 "the handoff moves those pages without conversion",
        "mamba_cache_mode": "requireSameKVCacheFormat compares the recurrent cache mode, which "
                            "decides the page layout the handoff has to move",
    }.get(knob, "BLIS requires the two pools to agree on this")
    return (f"the prefill pool states {knob}={a!r} and the decode pool {knob}={b!r}: {why}")


def pd_total_gpus(pd):
    """GPUs a disaggregated deployment occupies: both pools' instances x their footprints."""
    return sum(pd[r]["footprint_gpus_per_instance"] * pd[r]["instances"] for r in PD_ROLES)


def footprint_gpus(parallel):
    """GPUs one instance occupies = tp x pp x dp (EP is within the TP ranks, not an extra factor)."""
    return parallel["tp"] * parallel["pp"] * parallel["dp"]


_DSA_CACHE = {}


def is_dsa_model(catalog, model):
    """True when the model's catalog graph has a DSA sparse-MLA attention node.

    DSA is a property of the GRAPH, not of the model's name -- glm-5.3 and deepseek-v4-pro are DSA
    while qwen3-14b is not, and nothing in the names says so. Read it from the authoritative
    graph.yaml (NS-6: the catalog is the source of truth) with a plain substring scan so the runner
    keeps its no-external-yaml-dep property. An unreadable or absent graph answers False: the
    non-DSA default is the safe one, and BLIS reports the real error for a missing model anyway.
    """
    if not catalog or not model:
        return False
    key = (catalog, model)
    if key not in _DSA_CACHE:
        path = os.path.join(catalog, "models", model, "graph.yaml")
        try:
            with open(path, encoding="utf-8") as f:
                _DSA_CACHE[key] = "sparse_mla" in f.read()
        except OSError:
            _DSA_CACHE[key] = False
    return _DSA_CACHE[key]


# `blis run --workload <preset>` is SILENTLY IGNORED in closed-loop mode, which is the only mode
# config-search uses. cmd/root.go resolves the workload as an if/else chain:
#
#     --workload-spec  ->  concurrency > 0  ->  workloadType == "distribution"  ->  preset loader
#
# Because every search run passes --concurrency, it takes the SECOND branch, which synthesizes from
# SynthesizeFromDistribution using the CLI distribution flags and never reads the named preset.
# Verified on v0.10.0 and v0.12.0: `chatbot`, `multidoc`, `summarization` and even a nonexistent
# `bogus-name` all produce byte-identical metrics (the 512/512-token CLI defaults). That is how
# reports came to be labelled with a workload their data does not reflect -- a committed "multidoc"
# run averages 545 prompt tokens where multidoc's shape states 10,240.
#
# The fix does not need --workload-spec (which refuses --concurrency outright: "define concurrency
# in the spec file using clients[].concurrency"). A preset IS just distribution parameters --
# SynthesizeFromPreset forwards them straight to SynthesizeFromDistribution -- and the
# concurrency branch reads exactly those CLI flags. So read the preset out of the catalog and pass
# its shape explicitly, which makes the closed-loop run price the traffic it claims.
WORKLOAD_IGNORED = (
    "`--workload <preset>` is ignored when --concurrency is set (cmd/root.go resolves the "
    "concurrency branch before the preset loader), so the preset's shape is passed as explicit "
    "--prompt-tokens/--output-tokens flags instead")
# preset YAML key -> the CLI flag carrying it on a closed-loop run.
PRESET_SHAPE_FLAGS = (
    ("prompt", "tokens", "--prompt-tokens"),
    ("prompt", "tokens_stdev", "--prompt-tokens-stdev"),
    ("prompt", "tokens_min", "--prompt-tokens-min"),
    ("prompt", "tokens_max", "--prompt-tokens-max"),
    ("output", "tokens", "--output-tokens"),
    ("output", "tokens_stdev", "--output-tokens-stdev"),
    ("output", "tokens_min", "--output-tokens-min"),
    ("output", "tokens_max", "--output-tokens-max"),
)

_PRESET_CACHE = {}


def load_workload_shape(catalog, workload):
    """The prompt/output token shape of a catalog SHAPE preset, as {flag: value}.

    Returns {} for a replay workload (no ISL/OSL by construction -- its lengths live in the
    corpus) and raises ConfigError for a name the catalog does not carry, so a typo'd preset
    cannot quietly fall back to the 512/512 defaults the way `--workload` does.

    Parsed with a deliberately small two-level scanner rather than a YAML dependency, matching
    is_dsa_model's approach: these files are flat `prompt:`/`output:` blocks of scalars.
    """
    if not catalog or not workload:
        return {}
    key = (catalog, workload)
    if key in _PRESET_CACHE:
        return _PRESET_CACHE[key]
    path = os.path.join(catalog, "workloads", workload + ".yaml")
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise ConfigError(f"workload {workload!r}: cannot read {path} ({e})")
    section, vals = None, {}
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw[0].isspace():                      # a top-level key
            section = raw.split(":", 1)[0].strip()
            # A replay workload states no shape at all; its traffic is the corpus. Refuse it here
            # rather than return {}: `blis run` would then quietly serve the 512/512 CLI defaults
            # and the run would be labelled with a replay workload it never replayed -- the same
            # mislabelling this function exists to prevent. A replay belongs on the replay path
            # (--agentic-scenario), which is a different subcommand.
            if section == "replay":
                raise ConfigError(
                    f"workload {workload!r} is a REPLAY workload, not a shape: it has no ISL/OSL "
                    f"to generate (its traffic is the recorded corpus) and cannot run through "
                    f"`blis run`. Replay it with --agentic-scenario instead.")
            continue
        if ":" not in raw or section not in ("prompt", "output"):
            continue
        k, v = raw.split(":", 1)
        vals[(section, k.strip())] = v.strip()
    out = {}
    for sec, k, flag in PRESET_SHAPE_FLAGS:
        v = vals.get((sec, k))
        if v is not None:
            try:
                out[flag] = int(v)
            except ValueError:
                continue
    if not out:
        raise ConfigError(
            f"workload {workload!r}: {path} states no prompt/output token shape; a shape preset "
            f"must carry one (a replay workload is not runnable through `blis run`)")
    _PRESET_CACHE[key] = out
    return out


# The acceptance length a drafting run resolved. BLIS applies it internally but does NOT echo it
# in the metrics, and it is the number the per-user axis needs: under speculative decoding BLIS
# logs ITL per VERIFICATION STEP, and one step emits AL tokens, so the per-token rate is
# AL x 1000 / ITL_p99 and the naive 1000/ITL understates it by the full acceptance factor. On
# deepseek-v4.1-flash at K=5 that is the difference between 27.7 tok/s/user (looks like a floor
# violation) and 97.4 (the real rate) -- i.e. the naive axis would reject every drafting config.
#
# The profile key is exactly what the run already states: (model, method, K, thinking, category),
# so the same lookup BLIS does can be reproduced here and RECORDED on the row.
_AL_CACHE = {}


def acceptance_length(registry, model, method, k, thinking, category):
    """The measured acceptance length for this (model, method, K, thinking, category), or None.

    Read from the registry's acceptance/ profiles (registry v0.2.0+; v0.1.1 has no acceptance/).
    None means no profile matched -- the caller must then NOT publish a derived per-user rate,
    because the acceptance factor is unknown.

    Returns (al, source) where source is the profile set's name plus the hardware it was measured
    on, so a report can state that an H100 run reuses a B200 measurement rather than implying it
    was measured there. The profile set itself asks for that: "Acceptance depends mostly on the
    model, drafter and workload rather than the chip; a resolution on another part records the
    reuse."
    """
    if not (registry and model and method and k and thinking and category):
        return None, None
    key = (registry, model, method, int(k), str(thinking), str(category))
    if key in _AL_CACHE:
        return _AL_CACHE[key]
    found = (None, None)
    adir = os.path.join(registry, "acceptance")
    try:
        names = sorted(n for n in os.listdir(adir) if n.endswith((".yaml", ".yml")))
    except OSError:
        _AL_CACHE[key] = found
        return found
    for name in names:
        try:
            import yaml                      # only needed on the drafting path
            doc = yaml.safe_load(open(os.path.join(adir, name), encoding="utf-8")) or {}
        except Exception:
            continue
        for prof in doc.get("profiles") or []:
            if (str(prof.get("model")) == model
                    and str(prof.get("speculative_method")) == method
                    and int(prof.get("num_spec_tokens") or 0) == int(k)
                    and str(prof.get("thinking")) == str(thinking)
                    and str(prof.get("category")) == str(category)):
                al = prof.get("acceptance_length")
                hw = ((prof.get("provenance") or {}).get("hardware") or "unrecorded")
                if al:
                    found = (float(al), f"{doc.get('name') or name} ({hw})")
                break
        if found[0] is not None:
            break
    _AL_CACHE[key] = found
    return found


def draft_depth(run):
    """The draft depth a recorded run actually ran at, 0 when drafting was off.

    The authoritative place is the RESOLVED deployment's engine.speculative block (what BLIS was
    told); the flat candidate key is the fallback for a row written before that block existed."""
    cfg = run.get("config") or {}
    eng = ((run.get("deployment") or {}).get("engine") or {})
    return int((eng.get("speculative") or {}).get("num_spec_tokens")
               or cfg.get("num_speculative_tokens") or 0)


def run_acceptance(run):
    """The acceptance length this run was priced with, or None when none was resolved."""
    acc = ((run.get("deployment") or {}).get("acceptance") or {})
    try:
        al = float(acc.get("acceptance_length"))
    except (TypeError, ValueError):
        return None
    return al if al > 0 else None


def worst_metric(run, key):
    """The WORST (max) value of `key` across the cluster aggregate AND every per-instance metric.

    An SLO can be met in the aggregate yet violated on one instance, so feasibility is judged on
    the worst instance -- for EVERY latency SLO, not just ITL. Judging ITL on the worst instance
    while judging TTFT/E2E on the aggregate is the same displayed-servable-while-an-instance-is-
    over split the ITL basis was reworked to remove (#19): in the committed multidoc-prefill
    ledger three rows sit under the 10s TTFT cap in the aggregate while an instance is over it
    (susiejojo review).

    The aggregate is included rather than trusted, because it is NOT always the max over the
    instances -- measured across the committed ledgers, an instance exceeds it on most rows for
    every one of the three p99 metrics. Neither side alone is the worst case.

    An idle instance reports 0, not null. For a latency metric `max` discards that naturally, and
    taking the max (never the mean) is also what stops an idle 0 from dragging the worst case
    down. Returns (worst, missing_everywhere).
    """
    vals = []
    try:
        vals.append(float((run.get("metrics") or {}).get(key)))
    except (TypeError, ValueError):
        pass
    for inst in (run.get("metrics_all_instances") or []):
        if isinstance(inst, dict):
            try:
                vals.append(float(inst.get(key)))
            except (TypeError, ValueError):
                pass
    return (max(vals), False) if vals else (None, True)


def worst_itl_p99(run):
    """The worst-instance p99 ITL -- `worst_metric` at the key the per-user rate is built on."""
    return worst_metric(run, "itl_p99_ms")


def per_user_rate(run):
    """THE per-user token rate for a recorded run -- the one definition the search, the report's
    feasibility test and the displayed number all use (namasl review #2).

    Two things make this more than 1000/ITL:

    * Under speculative decoding BLIS reports the ITL of a VERIFICATION STEP, and one step emits
      the ACCEPTANCE LENGTH of tokens. The per-TOKEN rate is therefore AL x 1000/ITL; the naive
      form understates a drafting arm by the whole acceptance factor.
    * The basis is the WORST INSTANCE, not the aggregate, because that is what the floor promises
      every user. Scoring the displayed rate on the aggregate while testing feasibility on the
      worst instance is how a row came to be shown at 101.9 tok/s/user and simultaneously marked
      infeasible at 31.8 ms ITL.

    Returns (rate_or_None, basis, itl_used) where basis is "" for a direct rate,
    "derived-from-acceptance" for an AL-scaled one, and "step-estimate-withheld" when drafting is
    on and the acceptance is unknown -- an unknown multiple is never published as a number.
    """
    itl, missing = worst_itl_p99(run)
    if missing or not itl:
        return None, "", None
    mtp = draft_depth(run)
    if mtp:
        al = run_acceptance(run)
        if al is None:
            return None, "step-estimate-withheld", itl
        return al * 1000.0 / itl, "derived-from-acceptance", itl
    return 1000.0 / itl, "", itl


_SPEC_CACHE = {}


def graph_speculator(catalog, model):
    """The speculator method the model's own catalog graph declares, or "" when it has none.

    The engine REQUIRES a method on a drafting deployment ('engine.speculative.method "" is not
    accepted ... it accepts deepseek_mtp, dflash, ... dspark, eagle, ...'), and the right value is
    a property of the MODEL, not of this search: deepseek-v4.1-flash drafts with `dspark`,
    glm-5.2 with `mtp`. So read it out of graph.yaml's `speculator:` block rather than hardcoding
    one model's drafter or making every caller state it.

    An empty answer means the model ships no speculator, which is also the honest signal that
    speculative decoding is not available for it.
    """
    if not catalog or not model:
        return ""
    key = (catalog, model)
    if key not in _SPEC_CACHE:
        path = os.path.join(catalog, "models", model, "graph.yaml")
        method, in_spec = "", False
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line[:1].isspace():
                        in_spec = line.startswith("speculator:")
                        continue
                    if in_spec:
                        t = line.strip()
                        if t.startswith("method:"):
                            method = t.split(":", 1)[1].strip().strip('"\'')
                            break
        except OSError:
            method = ""
        _SPEC_CACHE[key] = method
    return _SPEC_CACHE[key]


_ARCH_CACHE = {}


def model_architecture(catalog, model):
    """The architecture the model's catalog graph NAMES, or "" when unreadable.

    The kernel decides the DeepSeek-V4 family by this string and nothing else -- a 512-wide
    sparse-MLA head is not the family's mark, since GLM-5.3-Flash's sparse layers have one too
    (blis-latency-kernel sparse_mla.go, `deepseekV4Family`). So match the architecture, never the
    model's name.
    """
    if not catalog or not model:
        return ""
    key = (catalog, model)
    if key not in _ARCH_CACHE:
        path = os.path.join(catalog, "models", model, "graph.yaml")
        arch = ""
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("architecture:"):
                        arch = s.split(":", 1)[1].strip().strip('"\'')
                        break
        except OSError:
            arch = ""
        _ARCH_CACHE[key] = arch
    return _ARCH_CACHE[key]


_CC_CACHE = {}


def compute_capability_major(catalog, hardware):
    """The major CUDA compute capability of a catalog hardware entry, or None when unstated.

    This is the only thing that distinguishes Hopper from Blackwell for block sizing: the catalog
    states `ComputeCapability: "9.0"` for h100/h200 and "10.0"/"10.3" for the Blackwell parts.
    None means "unstated", and callers treat that as Hopper -- the kernel's own fallback, which
    infers the family and records the assumption rather than failing.
    """
    if not catalog or not hardware:
        return None
    key = (catalog, hardware)
    if key not in _CC_CACHE:
        path = os.path.join(catalog, "hardware", hardware + ".yaml")
        major = None
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    # Skip the _comment_cc prose, which also contains "ComputeCapability".
                    if s.startswith("ComputeCapability:"):
                        val = s.split(":", 1)[1].strip().strip('"\'')
                        try:
                            major = int(val.split(".")[0])
                        except ValueError:
                            major = None
                        break
        except OSError:
            major = None
        _CC_CACHE[key] = major
    return _CC_CACHE[key]


def default_block_size(catalog, model, hardware=None):
    """The block_size to state when a candidate does not pin one.

    It cannot be a single constant in any direction: the kernel adapter REQUIRES a positive
    block_size on the deployment ("kernelmodel: deployment states no block_size"), so omitting it
    fails every model; a flat 16 is refused by every DSA sparse-MLA model, which is what made
    glm-5 / glm-5.2 / glm-5.2-fp8 / glm-5.3 unrunnable in this runner; and a flat 64 is refused by
    a DeepSeek-V4-family model on a Blackwell part, which is what made every deepseek-v4.1-flash
    B200 run fail. The V4 family's sizes come from `v4BlockSize` (see the constants above).
    """
    arch = model_architecture(catalog, model)
    if arch == V4_ARCH:
        return V4_BLOCK_SIZE
    if arch == V41_ARCH:
        major = compute_capability_major(catalog, hardware)
        # Unstated compute capability is treated as Hopper, matching the kernel's own inference.
        if major is None or major == HOPPER_CC_MAJOR:
            return V41_BLOCK_SIZE_HOPPER
        return V41_BLOCK_SIZE_NEWER
    return DSA_BLOCK_SIZE if is_dsa_model(catalog, model) else DEFAULT_BLOCK_SIZE


def _iter_json_objects(text):
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if isinstance(obj, dict):
                yield obj
            i = end
        else:
            i += 1


def _find_metrics_dicts(obj, out):
    if isinstance(obj, dict):
        if "tokens_per_sec" in obj:
            out.append(obj)
        for v in obj.values():
            _find_metrics_dicts(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _find_metrics_dicts(v, out)
    return out


def _all_metrics(stdout):
    out = []
    for o in _iter_json_objects(stdout):
        _find_metrics_dicts(o, out)
    return out


def _result_doc(stdout):
    """The v0.15.0 Result document on stdout, or None for an older flat MetricsOutput.

    v0.15.0's `feat(output)` made stdout ONE document (kind: Result) instead of the flat
    MetricsOutput dict every earlier pin printed. Recognised by `kind` plus the sections this
    module reads, never by `schema_version` alone -- a replay document is the same shape.
    """
    if stdout.lstrip().startswith("kind: Result"):    # see INF_NEEDS_YAML
        import yaml
        o = yaml.load(stdout, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        return o if isinstance(o.get("engine"), dict) else None
    for o in _iter_json_objects(stdout):
        if o.get("kind") == "Result" and isinstance(o.get("engine"), dict):
            return o
    return None


# The legacy MetricsOutput key -> where the same number lives in a v0.15.0 Result document.
#
# `engine.cluster.whole_run` is the WHOLE RUN, which is what every earlier pin's stdout reported
# and what the committed ledgers therefore hold; top-level `metrics` is the MEASURED WINDOW, a
# different population (result_build.go standardMetrics). Mixing the two would make this bump a
# silent re-baselining of every latency in the repo, so the whole-run block is the source and the
# measured window is left to the Result document's own readers.
_WHOLE_RUN_KEYS = {k: k for k in ("tokens_per_sec", "responses_per_sec",
                                   "ttft_mean_ms", "ttft_p90_ms", "ttft_p95_ms", "ttft_p99_ms",
                                   "e2e_mean_ms", "e2e_p90_ms", "e2e_p95_ms", "e2e_p99_ms")}

# itl_* was RENAMED to step_interval_*, not dropped: result_inputs.go fills whole_run's
# step_interval_* from whole.ITL*Ms, the very fields MetricsOutput's itl_* keys held ("the
# step-interval series the legacy itl_* keys held"). So this is a rename to follow, NOT the
# top-level metrics.itl_* -- those are FormulaTPOTExclFirst, per-request TPOT over the measured
# window, a genuinely different measurement. Measured on one Muse run: whole-run
# step_interval_p99 294.03 ms against measured-window itl_p99 33.87 ms, ~9x apart because a step
# serves the whole batch. Reading the per-request figure into itl_p99_ms would silently rebase
# every SLO verdict and every per-user rate in the repo against 12 committed reports. The same
# holds for the step_interval_* on `cluster` itself: result_inputs.go fills that copy from `sum`
# (the measured window), so it is not a fallback for whole_run's either.
_ITL_FROM_STEP_INTERVAL = {"itl_mean_ms": "step_interval_mean_ms", "itl_p90_ms": "step_interval_p90_ms",
                           "itl_p95_ms": "step_interval_p95_ms", "itl_p99_ms": "step_interval_p99_ms"}

# Counters that moved out of the metrics dict into the document's own `requests` block, under
# shorter names. _completion_issue reads all five, so without this remapping an incomplete or
# overloaded run is ranked as a clean capacity point.
_REQUEST_COUNTS = {"injected_requests": "injected", "completed_requests": "completed",
                   "still_queued": "still_queued", "still_running": "still_running",
                   "dropped_unservable": "dropped_unservable", "timed_out_requests": "timed_out",
                   "length_capped_requests": "length_capped"}

# Counters that stayed on the cluster aggregate but outside whole_run.
_CLUSTER_KEYS = {**{k: k for k in ("total_input_tokens", "total_output_tokens",
                                    "scheduling_delay_p99_ms", "kv_allocation_failures")},
                 "preemption_count": "preemptions",
                 "vllm_estimated_duration_s": "simulated_duration_s"}


def _flatten_result(doc):
    """A v0.15.0 Result document as the flat MetricsOutput dict the rest of this module reads.

    The aggregate was restructured by the bump while `engine.instances` kept emitting a verbatim
    MetricsOutput per instance, so WITHOUT this a single-instance run (no instances block) has no
    itl_* anywhere and every ITL SLO fails as "declared but not measured", while a multi-instance
    run resolves normally off the per-instance rows. That asymmetry is why it reads as a finding
    rather than a crash: a sweep reports 0 servable, or judges only its multi-instance arms.
    """
    cluster = ((doc.get("engine") or {}).get("cluster")) or {}
    whole = cluster.get("whole_run") or {}
    if not whole:
        return {}   # no whole-run aggregate: fail as "no cluster metrics parsed", not later
    out = {"instance_id": "cluster"}
    # itl_* ONLY from whole_run, never the cluster's own step_interval_* (see
    # _ITL_FROM_STEP_INTERVAL); hence the guard above rather than a fallback.
    for src, mapping in ((whole, _WHOLE_RUN_KEYS), (whole, _ITL_FROM_STEP_INTERVAL),
                         (cluster, _CLUSTER_KEYS), (doc.get("requests") or {}, _REQUEST_COUNTS)):
        for legacy, new in mapping.items():
            if new in src:
                out[legacy] = src[new]
    return out


def parse_metrics(stdout):
    doc = _result_doc(stdout)
    if doc is not None:
        return _flatten_result(doc)
    cands = _all_metrics(stdout)
    for o in cands:
        if o.get("instance_id") == "cluster":
            return o
    return cands[0] if cands else {}


def parse_all_instances(stdout):
    doc = _result_doc(stdout)
    if doc is not None:
        inst = (doc.get("engine") or {}).get("instances") or {}
        # Each value is a verbatim MetricsOutput, already carrying its own instance_id.
        return [v for _, v in sorted(inst.items()) if isinstance(v, dict)]
    return [o for o in _all_metrics(stdout) if o.get("instance_id") not in (None, "cluster")]


def _completion_issue(m):
    """Why a run is NOT a usable capacity/ranking point (incomplete/overloaded), else None."""
    inj, comp = m.get("injected_requests"), m.get("completed_requests")
    if inj is not None and comp is not None and inj != comp:
        return f"incomplete: injected={inj} != completed={comp}"
    for f in ("still_queued", "still_running", "dropped_unservable", "timed_out_requests"):
        v = m.get(f)
        if v:
            return f"overloaded: {f}={v}"
    return None


def pool_nodes(parallel, num_instances=1):
    """Nodes the colocated pool must hold for `num_instances` copies of this instance shape.

    BLIS v0.10.0 validates `--num-instances` against the pool's rank capacity: a pool of
    `nodes` x 8 GPUs holds floor(nodes*8 / footprint) instances, and asking for more is refused
    ("N instance(s) requested, but ...'s colocated pool holds M rank(s) at its layout"). Sizing
    the pool from ONE instance's footprint therefore capped every search at a single node's
    worth of instances; size it for the whole fleet instead."""
    total = footprint_gpus(parallel) * max(1, num_instances)
    return max(1, -(-total // GPUS_PER_NODE))           # ceil(total / 8)


def _pd_scenario_yaml(model, hardware, engine_version, pd, name, block, fabric, router):
    """Render a DISAGGREGATED Scenario+Deployment: one pool per role plus `pd_transfer`.

    Pools never share a node (blis-vllm places each role on its own nodes and asserts they do
    not overlap), so each pool is sized in WHOLE nodes and the cluster is their sum -- which is
    also why a P/D cluster always names a fabric: every handoff crosses nodes.
    """
    pools, total_nodes = [], 0
    for role in PD_ROLES:
        spec = pd[role]
        nodes = pool_nodes(spec["parallel"], spec["instances"])
        total_nodes += nodes
        pools.append(
            f"  - role: {role}\n"
            f"    nodes: {nodes}\n"
            f"    parallel:\n{block(spec['parallel'], 6)}\n"
            f"    engine:\n{block(spec['engine'], 6)}")
    rc = (router or {}).get("router_config")
    router_block = ""
    if rc:
        router_block = f"\nrouter:\n  endpoint_picker_config: {json.dumps(os.path.abspath(rc))}"
    return f"""kind: Scenario
name: {name}
engine_version: "{engine_version}"
model: {model}
coefficients: [{", ".join(COEFFICIENTS)}]
cluster:
  hardware: {hardware}
  fabric: {fabric}
  nodes: {total_nodes}
  gpus_per_node: {GPUS_PER_NODE}
---
kind: Deployment
name: {name}
pools:
{chr(10).join(pools)}
pd_transfer:
  connector: {pd.get("connector", DEFAULT_PD_CONNECTOR)}{router_block}
"""


def scenario_yaml(model, hardware, engine_version, parallel, engine, name, num_instances=1,
                  fabric=DEFAULT_FABRIC, router=None):
    """Render a two-doc Scenario+Deployment YAML (no external yaml dep). Omitted engine keys (e.g.
    quantization when unset) let the checkpoint/engine govern.

    A `pd` block on `parallel` (written by resolve_pd) renders a DISAGGREGATED deployment: two
    role pools plus the `pd_transfer` connector, instead of one colocated pool.
    """
    def block(d, indent):
        """Render a flat mapping, plus the one NESTED value the engine takes (`speculative`).

        json.dumps on a dict would emit JSON-flow style, which the strict loader rejects for this
        field, so a dict value is rendered as an indented sub-block instead.
        """
        pad = " " * indent
        out = []
        for k, v in d.items():
            if k == "pd":
                continue                  # not a schema field; it SELECTS the pool layout below
            if isinstance(v, dict):
                out.append(f"{pad}{k}:")
                out.extend(f"{pad}  {kk}: {json.dumps(vv)}" for kk, vv in v.items())
            else:
                out.append(f"{pad}{k}: {json.dumps(v)}")
        return "\n".join(out)
    pd = (parallel or {}).get("pd")
    if pd:
        return _pd_scenario_yaml(model, hardware, engine_version, pd, name, block, fabric, router)
    nodes = pool_nodes(parallel, num_instances)
    # A multi-node cluster MUST name a fabric: BLIS resolves cross-node cost from the fabric,
    # not the chip, and refuses the scenario otherwise ("scenario.cluster.fabric: required for
    # a N-node cluster"). Single-node clusters omit it (no cross-node traffic to price).
    fabric_line = f"\n  fabric: {fabric}" if nodes > 1 else ""
    # `router:` is a DEPLOYMENT-level key (a sibling of `pools:`), not a pool key: blis-schemas
    # puts Router on Deployment. Absent means the router's built-in standalone configuration over
    # every pod, which is what every pre-v0.12.0 run effectively used.
    rc = (router or {}).get("router_config")
    router_block = ""
    if rc:
        router_block = f"\nrouter:\n  endpoint_picker_config: {json.dumps(os.path.abspath(rc))}"
    return f"""kind: Scenario
name: {name}
engine_version: "{engine_version}"
model: {model}
coefficients: [{", ".join(COEFFICIENTS)}]
cluster:
  hardware: {hardware}{fabric_line}
  nodes: {nodes}
  gpus_per_node: 8
---
kind: Deployment
name: {name}
pools:
  - role: colocated
    nodes: {nodes}
    parallel:
{block(parallel, 6)}
    engine:
{block(engine, 6)}{router_block}
"""


def write_scenario(scenarios_dir, model, hardware, engine_version, parallel, engine, num_instances=1,
                   fabric=DEFAULT_FABRIC, router=None):
    """Write a scenario ATOMICALLY (temp + os.replace) so concurrent pool workers never read a
    half-written file for the same hashed deployment; return its file name.

    `num_instances` participates in the hash because it sets the pool's node count — two runs
    that differ only in instance count need DIFFERENT scenario files, or the second reuses a
    pool too small for it and BLIS refuses the run."""
    # The router config is part of the key: two routing arms differ ONLY in their `router:` block,
    # so sharing one scenario file would silently run the second arm with the first arm's router.
    key = json.dumps([model, hardware, engine_version, parallel, engine, num_instances, fabric,
                      router or {}], sort_keys=True)
    name = f"{model}-{hardware}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    os.makedirs(scenarios_dir, exist_ok=True)
    path = os.path.join(scenarios_dir, name + ".yaml")
    if not os.path.exists(path):
        body = scenario_yaml(model, hardware, engine_version, parallel, engine, name, num_instances,
                             fabric, router)
        fd, tmp = tempfile.mkstemp(dir=scenarios_dir, prefix=".tmp-", suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(body)
            os.replace(tmp, path)           # atomic publish; a concurrent writer's replace is idempotent
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return name + ".yaml"


def resolve_defaults(blis, defaults):
    if defaults:
        return defaults
    beside = os.path.join(os.path.dirname(os.path.abspath(blis)), "defaults.yaml")
    return beside if os.path.exists(beside) else None


def _cli_flag(key):
    """The CLI flag for a cli-slot candidate key: a deployment knob, a P/D instance count, or an
    acceptance selector."""
    for table in (CLI_FLAGS, PD_CLI, SPECULATIVE_CLI):
        if key in table:
            return table[key]
    raise KeyError(key)


# The Result document echoes its input, and JSON has no infinity: an infinite grace makes BLIS exit
# fatally AFTER simulating ("result: .inf has no JSON form"). A large finite grace is no stand-in --
# the measured window then runs to the grace end, so its rates shrink by orders of magnitude.
# YAML encodes `.inf`, so those runs ask for YAML; the document is otherwise the same.
INF_NEEDS_YAML = ("inf", ".inf")


def build_replay_cmd(blis, catalog, registry, scenarios_dir, scenario_name, seed, defaults, cli,
                     replay):
    """The `blis replay --agentic-scenario` command for a trace-REPLAY workload.

    A replay is a different subcommand, not a --workload value: `blis run --workload agentx-mvp`
    cannot replay one (the preset reader refuses a replay, and in concurrency mode --workload is
    not even read -- see WORKLOAD_IGNORED). The corpus is real recorded traffic, so there is no
    ISL/OSL to synthesize and no --num-requests: the sending window (--benchmark-duration) and the
    lane count bound the run instead.

    `concurrency` here means SESSION-TREE LANES (live trees), not in-flight requests -- one lane
    drives a whole agentic trajectory whose sub-requests fan out from it.
    """
    cmd = [blis, "replay",
           "--scenario", scenario_name, "--scenarios", scenarios_dir, "--registry", registry,
           "--agentic-scenario", replay["scenario"],
           "--corpus", replay["corpus"],
           # Required: the harness reads the dataset's default branch rather than a pinned
           # revision, so only the file's digest ties a run to its bytes. BLIS refuses a replay
           # that states no sha256, and refuses a corpus whose digest does not match.
           "--corpus-sha256", replay["corpus_sha256"]]
    if catalog:
        cmd += ["--catalog", catalog]
    if resolve_defaults(blis, defaults):
        cmd += ["--defaults-filepath", resolve_defaults(blis, defaults)]
    # max_model_len: the deployment's own resolved window. Every trace whose peak prompt + output
    # exceeds it is DROPPED, and the AgentX corpus caps requests at 256k -- so a 32768-token
    # deployment drops the entire corpus ("no trace fits max_context_length 32768").
    if replay.get("max_context_length"):
        cmd += ["--max-context-length", str(replay["max_context_length"])]
    if replay.get("benchmark_duration"):
        cmd += ["--benchmark-duration", str(replay["benchmark_duration"])]
    if replay.get("grace_period"):
        cmd += ["--grace-period", str(replay["grace_period"])]
        if str(replay["grace_period"]) in INF_NEEDS_YAML:
            cmd += ["--format", "yaml"]
    if replay.get("corpus_cache"):
        cmd += ["--corpus-cache", replay["corpus_cache"]]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    for k, v in cli.items():
        cmd += [_cli_flag(k), str(v)]
    return cmd


def build_cmd(blis, catalog, registry, scenarios_dir, scenario_name, workload, num_requests, seed,
              horizon, defaults, cli, shape=None):
    # BLIS selects the kernel cost-model backend from the scenario's `coefficients:` list (written
    # into the generated scenario), not a CLI flag: the old `--latency-model blis-latency-kernel`
    # flag was removed in v0.10.0 and is rejected ("unknown flag"), so it must not be passed.
    cmd = [blis, "run",
           "--scenario", scenario_name, "--scenarios", scenarios_dir, "--registry", registry]
    if catalog:
        cmd += ["--catalog", catalog]
    if resolve_defaults(blis, defaults):
        cmd += ["--defaults-filepath", resolve_defaults(blis, defaults)]
    if workload:
        # Still passed for provenance (it names the preset in the recorded cmd), but it does NOT
        # decide the traffic under --concurrency -- the shape flags below do. See WORKLOAD_IGNORED.
        cmd += ["--workload", workload]
    for flag, value in sorted((shape or {}).items()):
        cmd += [flag, str(value)]
    if num_requests is not None:
        cmd += ["--num-requests", str(num_requests)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if horizon is not None:
        cmd += ["--horizon", str(horizon)]
    for k, v in cli.items():
        cmd += [_cli_flag(k), str(v)]
    return cmd


_STACK_CACHE = {}


def row_stack(blis, catalog, registry):
    """A compact per-ROW identity of the cost model that priced this run.

    report.provenance_split() detects a ledger whose rows were not all priced the same way, but its
    stack branch could never fire: the stack was recorded only at SCENARIO level, so no run row
    carried one and the branch's unit test passed by injecting a key the pipeline never wrote
    (susiejojo review #2). Recording it per row is what makes the check real -- it is what would
    have caught the v0.1.1/v0.2.0 split in the committed reports from the artifacts alone, rather
    than from the commit dates.

    Cheap: the digests/commits are resolved once per process and cached."""
    key = (blis, catalog, registry)
    if key in _STACK_CACHE:
        return _STACK_CACHE[key]
    out = {}
    for name, path in (("catalog", catalog), ("registry", registry)):
        head = None
        if path and os.path.isdir(path):
            try:
                pr = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                                    capture_output=True, text=True, timeout=10)
                head = (pr.stdout or "").strip()[:12] or None if pr.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                head = None
        if head:
            out[name] = head
    if blis and os.path.isfile(blis):
        h = hashlib.sha256()
        try:
            with open(blis, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out["blis"] = h.hexdigest()[:12]
        except OSError:
            pass
    _STACK_CACHE[key] = out
    return out


def _record_shell(config, phase):
    return {"config": config, "concurrency": config.get("concurrency"),
            "config_hash": hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
            "phase": phase}


def run_one(blis, catalog, registry, scenarios_dir, model, hardware, engine_version,
            workload, num_requests, seed, horizon, config, phase="probe", timeout=120, defaults=None,
            replay=None):
    """Run one kernel-priced BLIS config and return a record dict. No runs.jsonl I/O (caller writes).

    `replay` selects the trace-REPLAY path (`blis replay --agentic-scenario`) instead of
    `blis run`: pass the resolved replay descriptor (scenario/corpus/corpus_sha256/...). When it is
    None the run is a generated-workload `blis run` as before.
    """
    rec = _record_shell(config, phase)
    # Which cost model priced THIS row (see row_stack): makes provenance_split's stack branch able
    # to fire on real data instead of guarding an unreachable path.
    rec["pinned_stack"] = row_stack(blis, catalog, registry)
    try:
        parallel, engine, cli, router = classify_config(config)
        # Resolve the preset's token shape so a closed-loop run actually prices the traffic it
        # names (see WORKLOAD_IGNORED). A replay carries its lengths in the corpus, so no shape.
        shape = {} if replay else load_workload_shape(catalog, workload)
    except ConfigError as e:
        # Same row SHAPE as a real run so neither sweep_pool's summary nor the report trips on a
        # missing key (every record has wall_s, deployment, metrics, ...). returncode=None marks it
        # "not run"; error carries the reason.
        rec.update({"deployment": None, "cmd": None, "returncode": None, "wall_s": 0.0,
                    "metrics": {}, "metrics_all_instances": [], "error": f"unsupported config: {e}",
                    "stdout_tail": "", "stderr_tail": ""})
        return rec
    num_instances = cli.get("num_instances", 1)
    pd = (parallel or {}).get("pd")
    fp = footprint_gpus(parallel)
    # Fill block_size per MODEL when the candidate did not pin one (see default_block_size): the
    # adapter requires a value, and DSA models require it to be a multiple of 64.
    engine.setdefault("block_size", default_block_size(catalog, model, hardware))
    if pd:
        # Both pools page identically or BLIS refuses the handoff, so ONE resolved value is
        # filled into both rather than each pool defaulting independently.
        for role in PD_ROLES:
            pd[role]["engine"].setdefault("block_size", engine["block_size"])
    # Fill the drafter from the MODEL's graph when a candidate asked for drafting without naming
    # one (the engine requires a method; see graph_speculator). A model with no speculator is
    # refused here rather than by BLIS, so the error names the model instead of an empty string.
    spec = engine.get("speculative")
    if spec and not spec.get("method"):
        method = graph_speculator(catalog, model)
        if not method:
            rec.update({"deployment": None, "cmd": None, "returncode": None, "wall_s": 0.0,
                        "metrics": {}, "metrics_all_instances": [],
                        "error": (f"unsupported config: speculative decoding requested but "
                                  f"{model!r} declares no speculator in its catalog graph"),
                        "stdout_tail": "", "stderr_tail": ""})
            return rec
        spec["method"] = method
    # Resolve and RECORD the acceptance length a drafting run will be priced with. BLIS applies it
    # but never echoes it, and the per-user axis is wrong without it (see acceptance_length).
    acceptance = None
    if spec:
        al, al_src = acceptance_length(registry, model, spec.get("method"),
                                       spec.get("num_spec_tokens"),
                                       cli.get("speculative_thinking"),
                                       cli.get("speculative_category"))
        if al:
            acceptance = {"acceptance_length": al, "source": al_src,
                          "num_spec_tokens": spec.get("num_spec_tokens"),
                          "method": spec.get("method"),
                          # True when the profile was measured on another part. The profile set
                          # says acceptance is mostly model/drafter/workload rather than chip and
                          # asks that reuse be recorded, so a report can state it.
                          "reused_from_other_hardware": bool(
                              al_src and hardware and f"({hardware})" not in al_src)}
    scenario_name = write_scenario(scenarios_dir, model, hardware, engine_version, parallel, engine,
                                   num_instances, DEFAULT_FABRIC, router)
    if replay:
        cmd = build_replay_cmd(blis, catalog, registry, scenarios_dir, scenario_name, seed,
                               defaults, cli, replay)
    else:
        cmd = build_cmd(blis, catalog, registry, scenarios_dir, scenario_name, workload,
                        num_requests, seed, horizon, defaults, cli, shape)
    t0 = time.time()
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, rc = pr.stdout, pr.stderr, pr.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, rc = "", "timeout", -1
    metrics = parse_metrics(stdout) if rc == 0 else {}
    if rc != 0:
        error = stderr.strip().splitlines()[-1] if stderr.strip() else f"rc={rc}"
    elif not metrics:
        error = "rc=0 but no cluster metrics parsed from stdout (output format changed?)"
    else:
        error = _completion_issue(metrics)
    rec.update({
        # `router` and `workload` are recorded so a ledger row states the WHOLE deployment and the
        # traffic it was priced on. Without the workload field a replay row and a generated-shape
        # row were indistinguishable after the fact.
        # A disaggregated row's GPU count is BOTH pools together; `fp * num_instances` would be
        # the shared layout's footprint, which no pool necessarily runs.
        "deployment": {"parallel": parallel, "engine": engine, "num_instances": num_instances,
                       "footprint_gpus_per_instance": fp,
                       "gpus": pd_total_gpus(pd) if pd else fp * num_instances,
                       "disaggregated": bool(pd),
                       "scenario": scenario_name, "router": router or None,
                       # The acceptance the drafting run was priced with, so the report can
                       # derive a correct per-token rate instead of a per-STEP one.
                       "acceptance": acceptance,
                       "workload": ({"kind": "replay", **replay} if replay
                                    else {"kind": "shape", "preset": workload})},
        "cmd": cmd, "returncode": rc, "wall_s": round(time.time() - t0, 3),
        "metrics": metrics, "metrics_all_instances": parse_all_instances(stdout) if rc == 0 else [],
        "error": error, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:],
    })
    return rec


def add_common_args(p):
    p.add_argument("--blis", default="./blis")
    p.add_argument("--catalog", default=os.environ.get("BLIS_CATALOG"))
    p.add_argument("--registry", default=os.environ.get("BLIS_REGISTRY"), help="blis-registry v0.1.1 clone root")
    p.add_argument("--scenarios", required=True, help="dir to write/read generated scenario YAMLs")
    p.add_argument("--model", required=True, help="catalog short-name, e.g. glm-5.3-flash")
    p.add_argument("--hardware", required=True, help="scenario hardware key, e.g. h100")
    p.add_argument("--engine-version", dest="engine_version", default="0.29.0")
    p.add_argument("--workload")
    p.add_argument("--num-requests", type=int, dest="num_requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int)
    p.add_argument("--defaults", help="path to defaults.yaml (default: next to the blis binary)")
    p.add_argument("--phase", default="probe")
    p.add_argument("--timeout", type=int, default=120)
    # --- trace-REPLAY workload (blis replay --agentic-scenario) ---------------------------------
    # A replay is real recorded traffic, so it has no ISL/OSL to synthesize: naming an agentic
    # scenario switches the whole run onto `blis replay`. See build_replay_cmd.
    p.add_argument("--agentic-scenario", dest="agentic_scenario",
                   help='replay an agentic harness scenario instead of a generated workload '
                        '(e.g. "inferencex-agentx-mvp"); requires --corpus')
    p.add_argument("--corpus", help="the replay corpus file (JSONL, one trace per line)")
    p.add_argument("--corpus-sha256", dest="corpus_sha256",
                   help="the corpus file's sha256; the replay refuses a corpus that does not match")
    p.add_argument("--corpus-cache", dest="corpus_cache",
                   help="dir for parsed-corpus snapshots (output is identical with or without it)")
    p.add_argument("--benchmark-duration", dest="benchmark_duration",
                   help="the profiling sending window, e.g. 900s")
    p.add_argument("--grace-period", dest="grace_period",
                   help='how long in-flight requests may finish after the window (duration or "inf")')
    p.add_argument("--max-context-length", dest="max_context_length",
                   help='drop traces whose peak prompt+output exceeds this ("max_model_len" or a '
                        'token count). NOTE: the AgentX corpus caps requests at 256k, so a '
                        '32768-token deployment drops EVERY trace')


def replay_from_args(args):
    """The replay descriptor from parsed args, or None for a generated-workload run.

    Validated here rather than per-run: a missing corpus or digest would otherwise fail every run
    of a sweep identically, and BLIS refuses a replay that states no sha256 anyway.
    """
    if not getattr(args, "agentic_scenario", None):
        return None
    if not getattr(args, "corpus", None):
        raise ValueError("--agentic-scenario requires --corpus")
    if not os.path.exists(args.corpus):
        raise ValueError(f"--corpus {args.corpus!r}: no such file")
    if not getattr(args, "corpus_sha256", None):
        raise ValueError(
            "--agentic-scenario requires --corpus-sha256: the harness reads the dataset's default "
            "branch rather than a pinned revision, so only the file's digest ties a run to its bytes")
    r = {"scenario": args.agentic_scenario, "corpus": os.path.abspath(args.corpus),
         "corpus_sha256": args.corpus_sha256}
    for k in ("corpus_cache", "benchmark_duration", "grace_period", "max_context_length"):
        v = getattr(args, k, None)
        if v:
            r[k] = v
    return r


def require_env(args):
    """Raise ValueError (do NOT sys.exit) -- this module is imported as a library; callers in a
    main() catch it and exit."""
    missing = []
    if not args.catalog:
        missing.append("--catalog or BLIS_CATALOG (NS-6)")
    if not args.registry:
        missing.append("--registry or BLIS_REGISTRY (blis-latency-kernel needs it)")
    if missing:
        raise ValueError("missing required: " + "; ".join(missing))


def main():
    p = argparse.ArgumentParser(description="Run one kernel-priced BLIS config; print its record")
    add_common_args(p)
    p.add_argument("--config", required=True, help='JSON knob dict, e.g. {"tp":8,"enable_expert_parallel":true,"num_instances":2,"concurrency":32}')
    p.add_argument("--out", help="optional runs.jsonl to append to (single-run use only)")
    args = p.parse_args()
    try:
        require_env(args)
        replay = replay_from_args(args)
    except ValueError as e:
        sys.exit(str(e))
    rec = run_one(args.blis, args.catalog, args.registry, args.scenarios, args.model, args.hardware,
                  args.engine_version, args.workload, args.num_requests, args.seed, args.horizon,
                  json.loads(args.config), args.phase, args.timeout, args.defaults, replay)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
