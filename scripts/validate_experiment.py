#!/usr/bin/env python3
"""Validate an Experiment Spec (RFC #21) BEFORE any BLIS run -- the contract's gatekeeper.

    python3 -I scripts/validate_experiment.py experiments/foo.yaml
    python3 -I scripts/validate_experiment.py experiments/            # every spec
    python3 -I scripts/validate_experiment.py experiments/foo.yaml --json

Exit 0 when every spec is valid, 1 otherwise, with one `file: path: message` line per error.
Safe for an untrusted PR: it reads YAML with safe_load, touches no network, and runs no BLIS.

WHY THIS EXISTS SEPARATELY FROM THE SCHEMA. experiments/schema.json describes STRUCTURE (every
value is a list, a sweep is colocated or disaggregated but not both) for a contributor's EDITOR.
This script never loads it -- the structural rules below are the ENFORCED copy, and
test_experiment.py's SchemaAgreesWithValidator is what stops the two definitions drifting apart.
Three classes of error no schema can reach are exactly the ones that waste a whole sweep:

  1. a knob name that no engine accepts. The names are not ours to invent -- an `engine:` key
     is a key of blis-vllm's engine block (which models vLLM v0.32.0), a `parallel:` key is a
     blis-schemas pool-layout field. A typo like `max_num_seq` would decode-fail per run, so
     every name is checked against the SIBLING CHECKOUTS when they are present
     (../blis-vllm, ../blis-schemas), falling back to a pinned list when they are not.
  2. GPU arithmetic that cannot be satisfied. `gpus` is the budget every run must use exactly,
     so a grid where no combination totals it expands to NOTHING. Reporting "0 runs" after a
     clone and build is a wasted CI job; reporting "no combination of tp x dp x num-instances
     reaches 32" is a fixable message.
  3. a router file that does not exist. A typo'd EndpointPickerConfig path costs a whole sweep
     of identical "no such file" rows (the same reason run_blis.classify_config checks it).

The validator is deliberately STRICTER than BLIS on names and LOOSER on physics: it never
claims a config will fit in memory or that a shape is fast. BLIS stays the arbiter of what
runs; this only rejects what provably cannot.
"""
import argparse
import json
import os
import re
import sys

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SIBLINGS = os.path.abspath(os.path.join(ROOT, os.pardir))
ROUTING_DIR = os.path.join(ROOT, "routing")

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# A sweep's `narrative` is the ONE line the report prints above that sweep's measured result, so
# it has to stay scannable. Past this it stops framing the numbers and starts crowding them:
# unbounded prose measured 600-670 characters per sweep, five sentences of hypothesis, rationale,
# prior measurements and design justification in the field meant for the prediction alone.
# Advisory, not an error -- it is a style bound, and failing CI over prose length would block a
# spec whose experiment is sound. Rationale belongs in a `#` comment, which never reaches the
# report.
NARRATIVE_CHARS = 200
# Comment lines a spec may open with before its first real key. Rationale belongs in the header
# (it never reaches the report), but an unbounded one buries the grid -- measured 36 lines,
# duplicated paragraph included, on a generated spec. Advisory, like NARRATIVE_CHARS.
HEADER_LINES = 12
# Replay fields the CATALOG owns: blis-catalog/workloads/<name>.yaml states the harness scenario
# id, the corpus digest and revision, the sending window, the grace period, the context filter and
# the acceptance selectors, and the sweeper reads them from there. Verified against the pinned
# binary: a replay with none of these stated ran to completion and still refused a corpus whose
# bytes did not match the descriptor's sha256. A spec restating one holds a copy that can drift
# from the catalog, which is the whole reason the descriptor exists.
#
# NOT here, because the catalog cannot know them: `corpus` (the local file path), `corpus_cache`
# (local parse cache) and `grace_period` (the one override a saturation search legitimately
# needs -- the descriptor's 30s makes most saturated loads report injected != completed).
CATALOG_OWNED_REPLAY_FIELDS = frozenset({
    "scenario", "corpus_sha256", "benchmark_duration", "benchmark_duration_s",
    "max_context_length", "random_seed", "category", "thinking", "harness_version",
    "cache_bust", "system_idle_gap_cap_s", "trajectory_start", "warmup",
    "split_flattened_agents",
})
DURATION_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(us|ms|s)$")
GPUS_PER_NODE = 8          # a per-instance footprint past this spans nodes (run_blis.GPUS_PER_NODE)

# ---------------------------------------------------------------------------------------------
# Knob names. Each set below has an authoritative upstream source, read when the checkout is
# present so a spec is validated against the stack that will actually run it. The fallbacks are
# the names verified at blis-vllm v0.32.0 / blis-schemas v0.8.0 and are what CI uses, where only
# this repo is checked out.

# blis-schemas spec/deployment/deployment.go, Parallelism's yaml tags.
PARALLEL_KNOBS = frozenset({
    "tp", "pp", "dp", "dp_local", "pcp", "dcp", "enable_expert_parallel",
    "data_parallel_load_balancing",
})

# blis-vllm rules/engineconfig, Engine's yaml names == vLLM's own config field names. The full
# surface is large; this is it as pinned by that package's fieldnames test at v0.32.0.
ENGINE_KNOBS = frozenset({
    "all2all_backend", "disable_custom_all_reduce", "allreduce_backend", "cudagraph_mode",
    "max_cudagraph_capture_size", "cudagraph_capture_sizes", "async_scheduling", "frontend",
    "disable_cascade_attn", "enable_prefix_caching", "swa_bounded_replay", "block_size",
    "max_num_batched_tokens", "max_num_seqs", "long_prefill_token_threshold", "max_model_len",
    "quantization", "cache_dtype", "mamba_cache_dtype", "mamba_ssm_cache_dtype",
    "mamba_cache_mode", "gpu_memory_utilization", "scheduling_policy", "dbo", "eplb",
    "speculative", "engram_config", "moe_backend", "dcp_comm_backend", "dcp_q_replicate",
    "cp_kv_cache_interleave_size", "long_prefill_token_threshold_adaptive",
    "max_num_scheduled_tokens", "max_num_active_seqs", "watermark", "prefill_schedule_interval",
    "stream_interval", "disable_hybrid_kv_cache_manager", "kv_cache_memory_bytes",
    "prefix_match_unit", "prefix_cache_retention_interval", "kv_events", "max_loras",
    "max_cpu_loras", "decoder_replay_cudagraph_capture_sizes", "custom_ops",
    "use_trtllm_attention", "mla_prefill_backend", "indexer_kv_dtype", "linear_backend", "env",
})

# Engine knobs blis-vllm v0.32.0 defines that the PINNED stack cannot carry. The scenario is
# decoded strictly against the `Engine` struct of the blis-schemas the pinned BLIS was built
# against (v0.5.0 at BLIS v0.12.0), so a key outside it fails EVERY run of the sweep with
# "field <name> not found in type deployment.Engine". Reported with that reason rather than as a
# typo: these are real vLLM knobs, just not sweepable until the pins move.
#
# Kept in sync with run_blis.UNSUPPORTED by test_experiment.py, which fails if the two drift.
ENGINE_NOT_WIRED = {
    k: ("a vLLM v0.32.0 engine field the pinned BLIS (v0.12.0, blis-schemas v0.5.0) has no "
        "field for, so the scenario would fail to decode; sweepable only after the pins move")
    for k in ("max_num_active_seqs", "max_num_scheduled_tokens", "watermark",
              "prefill_schedule_interval", "stream_interval", "prefix_match_unit",
              "prefix_cache_retention_interval", "disable_hybrid_kv_cache_manager",
              "kv_cache_memory_bytes", "kv_events", "max_loras", "max_cpu_loras", "custom_ops",
              "decoder_replay_cudagraph_capture_sizes", "use_trtllm_attention",
              "mla_prefill_backend", "indexer_kv_dtype", "linear_backend",
              "long_prefill_token_threshold_adaptive", "env", "frontend")
}

# Valid VALUES for the enumerated engine knobs, transcribed from blis-vllm's rules pack
# (rules/v0_32/pack.go, `Pack()`), which models vLLM v0.32.0. Read from the sibling checkout when
# it is present (see _enums_from_blis_vllm) so the sets track the stack rather than a copy.
#
# Checking VALUES matters as much as checking names: `cudagraph_mode: [full]` is a plausible
# thing to write and the engine refuses it (the enum is upper-case), so without this the whole
# sweep dies one identical error at a time.
ENGINE_ENUMS = {
    "cudagraph_mode": ("NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"),
    "cache_dtype": ("auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc",
                    "fp8_ds_mla", "nvfp4_ds_mla", "turboquant_k8v4", "turboquant_4bit_nc",
                    "turboquant_k3v4_nc", "turboquant_3bit_nc", "ultraquant_4bit",
                    "int4_per_token_head", "int8_per_token_head", "fp8_per_token_head",
                    "nvfp4", "nvfp4_4over6"),
    "all2all_backend": ("naive", "allgather_reducescatter", "pplx", "deepep_high_throughput",
                        "deepep_low_latency", "deepep_v2", "flashinfer_all2allv",
                        "flashinfer_nvlink_one_sided", "flashinfer_nvlink_two_sided",
                        "mori_high_throughput", "mori_low_latency", "moonep", "nixl_ep",
                        "passthrough"),
    "moe_backend": ("auto", "triton", "batched_triton", "deep_gemm", "deep_gemm_mega_moe",
                    "cutlass", "flashinfer_trtllm", "flashinfer_cutlass", "flashinfer_cutedsl",
                    "flashinfer_moe_ep_cutedsl", "flashinfer_b12x", "b12x",
                    "flashinfer_moe_ep_mega_deep_gemm", "marlin", "humming", "triton_unfused",
                    "aiter", "aiter_mega_moe", "aiter_triton_mxfp4_bf16", "flydsl", "hpc",
                    "emulation", "rdna3"),
    "allreduce_backend": ("custom", "nccl"),
    "scheduling_policy": ("fcfs", "priority"),
    # Draft methods, from SpeculativeMethods (the six named ones, the EAGLE family's 27
    # multi-token-prediction names, dflash, ngram_gpu and dspark). Needed in the PINNED set and
    # not only in the live read: CI has no ../blis-vllm checkout, so without this entry a typo'd
    # `speculative.method` passed validation there while failing every run.
    #
    # Membership is necessary, not sufficient -- the method must also be the one the MODEL's
    # graph declares (dspark for deepseek-v4.1-flash, glm5_next_mtp for glm-5.3-flash), and
    # DeepSeek-V4.1 refuses every MTP-family name. scripts/knob_profile.py reports both.
    "speculative_method": (
        "bailing_hybrid_mtp", "bailing_hybrid_v3_mtp", "custom_class", "deepseek_mtp",
        "dflash", "dots3_note_mtp", "draft_model", "dspark", "eagle", "eagle3", "ernie_mtp",
        "exaone4_5_mtp", "exaone_moe_mtp", "extract_hidden_states", "gemma4_mtp",
        "glm4_moe_lite_mtp", "glm4_moe_mtp", "glm5_next_mtp", "glm_ocr_mtp", "hy_v3_mtp",
        "hy_v4_mtp", "inkling_mtp", "kimi_k3_mtp", "longcat_flash_mtp", "medusa", "mimo_mtp",
        "mimo_v2_mtp", "minimax_m3_mtp", "mlp_speculator", "mtp", "nemotron_h_mtp", "ngram",
        "ngram_gpu", "pangu_ultra_moe_mtp", "qwen3_5_mtp", "qwen3_next_mtp", "qwen4_exp_mtp",
        "step3p5_mtp", "suffix"),
    # The served WEIGHT format, which sets weight-read bytes, compute peak and the GEMM
    # efficiency envelope at once -- a strong lever (measured 1083 bf16 -> 1271 fp8 -> 1411
    # modelopt_fp4 tok/s). Empty/unset lets the checkpoint's own format govern.
    "quantization": ("awq", "auto_awq", "fp8", "fbgemm_fp8", "fp_quant", "modelopt",
                     "modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed", "auto_gptq", "gptq",
                     "gptq_marlin", "awq_marlin", "humming", "compressed-tensors",
                     "experts_int8", "quark", "moe_wna16", "torchao", "inc", "mxfp4",
                     "gpt_oss_mxfp4", "deepseek_v4_fp8", "online", "fp8_per_tensor",
                     "fp8_per_block", "fp8_per_channel", "int8_per_channel_weight_only",
                     "nvfp4_per_token", "mxfp8"),
    "dcp_comm_backend": ("ag_rs", "a2a"),
    "mamba_cache_dtype": ("auto", "float32", "float16", "bfloat16"),
    "mamba_ssm_cache_dtype": ("auto", "float32", "float16", "bfloat16"),
    "mamba_cache_mode": ("none", "align"),
}

# Engine knobs whose value is a boolean (tri-state in the schema: unstated means the engine
# decides, which is NOT the same deployment as an explicit false).
ENGINE_BOOLS = ("enable_prefix_caching", "disable_cascade_attn", "swa_bounded_replay",
                "async_scheduling", "disable_custom_all_reduce", "dcp_q_replicate")
# Engine knobs whose value is a list of ints, not a scalar. In a spec these are a list OF lists:
# `cudagraph_capture_sizes: [[1,2,4], [1,2,4,8]]` sweeps two capture plans.
ENGINE_INT_LISTS = ("cudagraph_capture_sizes",)

# NESTED engine blocks: a mapping whose own fields are lists, like `speculative`. Field names are
# blis-schemas v0.5.0's; the engine decodes strictly, so an invented inner field fails every run.
NESTED_ENGINE_FIELDS = {
    "speculative": ("method", "num_spec_tokens"),
    "eplb": ("enable_eplb", "num_redundant_experts", "window_size", "step_interval"),
    "dbo": ("enable_dbo", "dbo_decode_token_threshold", "dbo_prefill_token_threshold"),
    "engram_config": ("cpu_offload", "dp_shared_memory", "use_thp"),
}
# Inner fields of a nested block that take a boolean.
NESTED_BOOL_FIELDS = ("enable_eplb", "enable_dbo", "cpu_offload", "dp_shared_memory", "use_thp")

# llm-d-router plugin parameters worth sweeping, per RFC #21's table. The router itself rejects
# an unknown parameter, so this catches a typo one step earlier with a better message. A plugin
# absent from this map is NOT an error: the table is the documented subset, not the full surface.
ROUTER_PLUGIN_PARAMS = {
    "approx-prefix-cache-producer": {"lruCapacityPerServer", "blockSizeTokens",
                                     "maxPrefixTokensToMatch"},
    "prefix-cache-scorer": {"matchLengthWeight", "matchLengthScaleTokens"},
    "load-aware-scorer": {"threshold"},
    "prefix-based-pd-decider": {"nonCachedTokens", "promptTokens"},
    "probabilistic-admitter": {"queueDepthThreshold", "kvCacheUtilThreshold", "power", "k"},
}


def _names_from_blis_vllm():
    """Engine field names read from a ../blis-vllm checkout, or None.

    Source of truth is that package's fieldnames test, whose `want` map is one quoted field
    name per line -- it exists precisely to pin these names, so parsing it beats guessing at
    struct tags spread over several files.
    """
    path = os.path.join(SIBLINGS, "blis-vllm", "rules", "engineconfig", "fieldnames_test.go")
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return None
    body = src.split("want := map[string]string{", 1)
    if len(body) < 2:
        return None
    names = set(re.findall(r'^\s*"([a-z0-9_]+)":', body[1].split("\n}", 1)[0], re.M))
    return names or None


def _names_from_blis_schemas():
    """Parallelism yaml field names read from a ../blis-schemas checkout, or None."""
    path = os.path.join(SIBLINGS, "blis-schemas", "spec", "deployment", "deployment.go")
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return None
    m = re.search(r"type Parallelism struct \{(.*?)\n\}", src, re.S)
    if not m:
        return None
    names = set(re.findall(r'yaml:"([a-z0-9_]+)', m.group(1)))
    return names or None


def _enums_from_blis_vllm():
    """Engine value sets read from ../blis-vllm's rules pack, or None.

    The pack's `Pack()` builds each set with a `set("a", "b", ...)` call against a named field,
    so the names and their members can be read directly rather than mirrored by hand.
    """
    path = os.path.join(SIBLINGS, "blis-vllm", "rules", "v0_32", "pack.go")
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return None
    # Field name in the pack -> knob name in a spec.
    wanted = {"CUDAGraphModes": "cudagraph_mode", "CacheDTypes": "cache_dtype",
              "All2AllBackends": "all2all_backend", "MoEBackends": "moe_backend",
              "AllReduceBackends": "allreduce_backend", "SchedulerPolicies": "scheduling_policy",
              "DCPCommBackends": "dcp_comm_backend", "MambaDTypes": "mamba_cache_dtype",
              "MambaCacheModes": "mamba_cache_mode", "Quantizations": "quantization",
              "SpeculativeMethods": "speculative_method"}
    # Named `[]string{...}` vars the pack SPREADS into a set, e.g.
    # `set(append([]string{"eagle", ...}, mtpMethods...)...)`. A spread contributes its own
    # members, so they have to be resolved or the set reads short -- SpeculativeMethods lost all
    # 27 MTP names (`mtp`, `deepseek_mtp`, ...) that way, and a spec naming one was rejected as
    # "not a draft method the engine knows" although the engine accepts it.
    spreads = {}
    for m in re.finditer(r"\bvar\s+(\w+)\s*=\s*\[\]string\{(.*?)\}", src, re.S):
        spreads[m.group(1)] = tuple(re.findall(r'"([^"]+)"', m.group(2)))
    out = {}
    for field, knob in wanted.items():
        # Balance parentheses from `set(` so an inner append(...)/[]string{...} call does not
        # end the match early: a non-greedy `(.*?)\)` stopped at the FIRST `)`, which for
        # SpeculativeMethods was the end of the inline literal, before `mtpMethods...`.
        start = re.search(rf"\b{field}:\s*set\(", src)
        if not start:
            continue
        i, depth = start.end(), 1
        while i < len(src) and depth:
            depth += (src[i] == "(") - (src[i] == ")")
            i += 1
        body = src[start.end():i - 1]
        values = tuple(re.findall(r'"([^"]+)"', body))
        for name in re.findall(r"\b(\w+)\.\.\.", body):
            values += spreads.get(name, ())
        if values:
            out[knob] = tuple(dict.fromkeys(values))
    if "mamba_cache_dtype" in out:
        out["mamba_ssm_cache_dtype"] = out["mamba_cache_dtype"]
    return out or None


def engine_enums():
    """(enums, source) -- the valid value sets for the enumerated engine knobs."""
    live = _enums_from_blis_vllm()
    if live:
        merged = dict(ENGINE_ENUMS)
        merged.update(live)
        return merged, "../blis-vllm rules pack"
    return dict(ENGINE_ENUMS), "pinned value sets (blis-vllm v0.32.0)"


def knob_vocabulary():
    """(parallel, engine, source) -- the knob names a spec may use.

    Reads the sibling checkouts when present so the validator tracks the stack rather than a
    copy of it; falls back to the pinned sets (what CI has) otherwise. `source` says which,
    so a failure message can tell a contributor what their names were checked against.
    """
    par, eng = _names_from_blis_schemas(), _names_from_blis_vllm()
    if par and eng:
        return par, eng, "../blis-schemas + ../blis-vllm checkouts"
    if eng:
        return PARALLEL_KNOBS, eng, "../blis-vllm checkout + pinned parallel names"
    if par:
        return par, ENGINE_KNOBS, "../blis-schemas checkout + pinned engine names"
    return PARALLEL_KNOBS, ENGINE_KNOBS, "pinned names (blis-vllm v0.32.0 / blis-schemas v0.8.0)"


# ---------------------------------------------------------------------------------------------


def _suggest(name, vocabulary):
    """The closest known knob name, for a typo message. Nothing clever: one edit apart, or a
    shared prefix long enough to be unambiguous."""
    import difflib
    hit = difflib.get_close_matches(str(name), sorted(vocabulary), n=1, cutoff=0.8)
    return f" (did you mean {hit[0]!r}?)" if hit else ""


def parse_duration_ms(text):
    """A duration string -> milliseconds, or None when it is not one.

    An SLO without a unit is refused rather than assumed: `itl_p99: 20` could be 20ms (the
    interactivity floor) or 20s (no constraint at all), and guessing either way silently
    changes which configs count as servable.
    """
    m = DURATION_RE.match(str(text).strip())
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    return value * {"us": 0.001, "ms": 1.0, "s": 1000.0}[unit]


def footprint(parallel):
    """GPUs one instance occupies = tp x pp x dp. EP lives inside the TP ranks, not beside them.

    Mirrors run_blis.footprint_gpus so the validator's arithmetic and the runner's agree.
    """
    return (int(parallel.get("tp", 1)) * int(parallel.get("pp", 1))
            * int(parallel.get("dp", 1)))


def _is_list(v):
    return isinstance(v, list) and not isinstance(v, (str, bytes))


def _check_value_lists(block, where, errs, allow_nested=()):
    """Every knob value must be a LIST -- the spec's one formatting rule (RFC #21).

    `tp: 8` instead of `tp: [8]` is the most likely hand-authoring mistake, and it is not
    harmlessly equivalent: the sweeper reads a grid as a product of lists, so a scalar would
    either crash or be iterated as characters.
    """
    for key, val in (block or {}).items():
        if key in allow_nested:
            continue
        if not _is_list(val):
            errs.append(f"{where}.{key}: every value must be a list "
                        f"(write `{key}: [{json.dumps(val) if not isinstance(val, str) else val}]`)")
        elif not val:
            errs.append(f"{where}.{key}: empty list -- a knob with no values to probe")


def _check_knob_names(block, where, vocabulary, kind, errs, source, skip=()):
    for key in (block or {}):
        if key in skip:
            continue
        if key in ENGINE_NOT_WIRED and kind == "engine":
            errs.append(f"{where}.{key}: {ENGINE_NOT_WIRED[key]}")
        elif key not in vocabulary:
            errs.append(f"{where}.{key}: not an {kind} knob per {source}"
                        f"{_suggest(key, vocabulary)}" if kind == "engine" else
                        f"{where}.{key}: not a {kind} knob per {source}"
                        f"{_suggest(key, vocabulary)}")


def _grid_size(lists):
    n = 1
    for values in lists:
        n *= max(1, len(values))
    return n


def _parallel_combos(parallel):
    """Every (tp, pp, dp) combination a `parallel:` block expands to, as footprints.

    Only the three factors that consume GPUs matter here; EP and the rest ride along.
    """
    tps = parallel.get("tp", [1]) if _is_list(parallel.get("tp", [1])) else [1]
    pps = parallel.get("pp", [1]) if _is_list(parallel.get("pp", [1])) else [1]
    dps = parallel.get("dp", [1]) if _is_list(parallel.get("dp", [1])) else [1]
    out = []
    for tp in tps:
        for pp in pps:
            for dp in dps:
                try:
                    out.append(int(tp) * int(pp) * int(dp))
                except (TypeError, ValueError):
                    continue
    return out or [1]


def _check_engine_values(engine, where, errs, enums=None):
    """Check each swept engine VALUE against the engine's own accepted set.

    A name check alone is not enough. `cudagraph_mode: [full]` is a natural thing to write and
    the engine refuses it (its enum is upper-case), `cache_dtype: [fp16]` is not a member either
    (it is `float16`), and either mistake kills every run in the sweep with the same error. The
    suggestion matters here too: the sets are large (19 cache dtypes, 14 all2all backends).
    """
    enums = enums if enums is not None else ENGINE_ENUMS
    for key, values in (engine or {}).items():
        if key == "speculative" or not _is_list(values):
            continue
        if key in ENGINE_INT_LISTS:
            # A list-valued knob is swept as a list OF lists; a flat list of ints is almost
            # certainly one plan written without the outer brackets.
            for v in values:
                if not _is_list(v):
                    errs.append(f"{where}.{key}: {key} takes a LIST of sizes, so sweeping it "
                                f"needs a list of lists (write `{key}: [[{v}, ...]]`); got "
                                f"{v!r}")
                elif any(not isinstance(x, int) or isinstance(x, bool) for x in v):
                    errs.append(f"{where}.{key}: every size must be an integer, got {v!r}")
            continue
        if key in ENGINE_BOOLS:
            for v in values:
                if not isinstance(v, bool):
                    errs.append(f"{where}.{key}: {v!r} is not a boolean; {key} is tri-state "
                                f"(omit it to let the engine decide, which is a DIFFERENT "
                                f"deployment from an explicit false)")
            continue
        allowed = enums.get(key)
        if not allowed:
            continue
        for v in values:
            if str(v) not in allowed:
                hint = _suggest(v, allowed) or " (valid: " + ", ".join(allowed) + ")"
                errs.append(f"{where}.{key}: {v!r} is not a value the engine accepts{hint}")


def _check_cross_knob(sweep, scenario, where, errs, model_entry=None):
    """Constraints that span TWO knobs, or a knob and the model -- the ones BLIS does not catch.

    Measured against the pinned binary: BLIS v0.12.0 validates enum membership and simple ranges
    (`cudagraph_mode: BOGUS`, `cache_dtype: fp16` and `max_num_seqs: 0` are all refused at
    startup), but it does NOT check cross-knob agreement -- a
    `long_prefill_token_threshold` of 999,999 against a 32k context ran happily. It cannot: the
    pinned simulator depends on blis-latency-kernel and blis-schemas only, never on blis-vllm,
    so the 32 rules in blis-vllm's v0_32 pack are not evaluated at run time at all.
    They are real vLLM behaviour even so -- the pack transcribes the engine's own startup
    assertions with file and line -- so a spec that violates one describes a deployment that
    could not be launched for real, and the simulated number would be meaningless.
    Each check below names the rule it comes from.

    Deliberately WARNINGS-as-errors only where the engine itself refuses at startup; a merely
    uneven-but-legal layout (expert-imbalance-without-eplb) is left alone.
    """
    engine = dict((sweep.get("engine") or {}))
    shared = scenario.get("engine") or {}
    # A knob can be fixed in scenario.engine and swept in the sweep; the sweep wins per run, so
    # check the sweep's values against the shared value when the sweep does not set it.
    def values(knob):
        if knob in engine and _is_list(engine[knob]):
            return list(engine[knob])
        if knob in shared:
            return [shared[knob]]
        return []

    # long-prefill-threshold-within-max-model-len: "this release refuses at startup a
    # long_prefill_token_threshold greater than max_model_len".
    lens = [v for v in values("max_model_len") if isinstance(v, int)]
    thresholds = [v for v in values("long_prefill_token_threshold") if isinstance(v, int)]
    if lens and thresholds:
        worst = max(thresholds)
        if worst > max(lens):
            errs.append(
                f"{where}.engine.long_prefill_token_threshold: {worst:,} exceeds every "
                f"max_model_len in this sweep (max {max(lens):,}); vLLM refuses a threshold "
                f"above the context window at startup, so those runs describe a deployment "
                f"that could not launch")
    elif thresholds and not lens:
        # No stated context window means the engine default (32,768 in this runner).
        default_len = 32768
        worst = max(thresholds)
        if worst > default_len:
            errs.append(
                f"{where}.engine.long_prefill_token_threshold: {worst:,} exceeds the default "
                f"max_model_len ({default_len:,}); vLLM refuses a threshold above the context "
                f"window, so raise max_model_len in the same sweep or lower the threshold")

    # cudagraph-capture-sizes: "refuses a stated max_cudagraph_capture_size that is not the
    # largest of the stated cudagraph_capture_sizes it keeps".
    sizes = [v for v in values("cudagraph_capture_sizes") if _is_list(v)]
    maxima = [v for v in values("max_cudagraph_capture_size") if isinstance(v, int)]
    for plan in sizes:
        ints = [x for x in plan if isinstance(x, int) and not isinstance(x, bool)]
        for cap in maxima:
            if ints and cap != max(ints):
                errs.append(
                    f"{where}.engine: max_cudagraph_capture_size {cap:,} is not the largest of "
                    f"cudagraph_capture_sizes {ints}; vLLM refuses a stated maximum that is not "
                    f"the largest size it keeps")
                break

    # cudagraph-capture-sizes (second clause): sizes above max_num_batched_tokens are dropped, so
    # a plan made entirely of dropped sizes captures nothing.
    budgets = [v for v in values("max_num_batched_tokens") if isinstance(v, int)]
    if budgets and sizes:
        smallest_budget = min(budgets)
        for plan in sizes:
            ints = [x for x in plan if isinstance(x, int) and not isinstance(x, bool)]
            if ints and min(ints) > smallest_budget:
                errs.append(
                    f"{where}.engine.cudagraph_capture_sizes: every size in {ints} is above "
                    f"max_num_batched_tokens {smallest_budget:,}, and the engine drops sizes "
                    f"above the token budget -- this plan would capture no graph at all")

    # cudagraph-capture-sizes: "cudagraph_mode NONE captures no graph, so the engine ignores the
    # stated capture sizes". Sweeping both at once makes the capture arm meaningless.
    modes = [str(v) for v in values("cudagraph_mode")]
    if modes == ["NONE"] and (sizes or maxima):
        errs.append(
            f"{where}.engine: cudagraph_mode is NONE for every run, which captures no graph, so "
            f"the stated capture sizes are ignored -- the sweep measures nothing")

    # max-model-len-within-context-window, read from the catalog rather than guessed.
    ctx = (model_entry or {}).get("context")
    if ctx and lens and max(lens) > int(ctx):
        errs.append(
            f"{where}.engine.max_model_len: {max(lens):,} exceeds {scenario.get('model')}'s "
            f"catalog context window ({int(ctx):,}); vLLM refuses a stated max_model_len above "
            f"the window it derives from the model configuration")

    # dbo-thresholds-stated-when-enabled: "an unstated threshold silently takes a default that
    # differs by 16x between the two", so an enabled DBO sweep that states neither is measuring
    # an engine default it did not choose.
    dbo = engine.get("dbo") if isinstance(engine.get("dbo"), dict) else None
    if dbo and any(v is True for v in (dbo.get("enable_dbo") or [])):
        if not (dbo.get("dbo_decode_token_threshold") or dbo.get("dbo_prefill_token_threshold")):
            errs.append(
                f"{where}.engine.dbo: enables dual-batch overlap without stating "
                f"dbo_decode_token_threshold or dbo_prefill_token_threshold; which one applies "
                f"turns on batch uniformity and the two defaults differ by 16x, so the sweep "
                f"would measure an engine default it did not choose")

    # expert-divisibility-under-eplb: "with load balancing on, the engine asserts that the
    # expert-parallel width divides the physical expert count". Checked against the catalog's
    # expert count and the sweep's own tp x dp, so a bad redundancy count is caught here.
    eplb = engine.get("eplb") if isinstance(engine.get("eplb"), dict) else None
    experts = (model_entry or {}).get("experts")
    if eplb and experts and any(v is True for v in (eplb.get("enable_eplb") or [])):
        widths = {fp for fp in _parallel_combos(sweep.get("parallel") or {}) if fp > 1}
        reds = [v for v in (eplb.get("num_redundant_experts") or [0]) if isinstance(v, int)]
        for width in sorted(widths):
            bad = [r for r in reds if (int(experts) + r) % width]
            if bad and len(bad) == len(reds):
                need = [(width - (int(experts) + r) % width) % width for r in bad]
                errs.append(
                    f"{where}.engine.eplb.num_redundant_experts: with expert-parallel width "
                    f"{width}, {scenario.get('model')}'s {int(experts)} experts plus "
                    f"{bad} redundant is not divisible by {width}; vLLM asserts divisibility "
                    f"when load balancing is on, so an indivisible layout does not start "
                    f"(nearest fixes: +{sorted(set(need))} redundant experts)")

    # mamba-cache-mode-matches-model: "a cache mode for recurrent layers on a model with none is
    # a setting with no effect". The catalog records the attention kind, so this is checkable.
    kind = str((model_entry or {}).get("attn_kind") or "").lower()
    mamba_knobs = [k for k in ("mamba_cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode")
                   if values(k)]
    if mamba_knobs and kind and "mamba" not in kind and "recurrent" not in kind:
        errs.append(
            f"{where}.engine: sweeps {', '.join(mamba_knobs)} on "
            f"{scenario.get('model')}, whose attention is {kind!r} with no recurrent layers; "
            f"these size a recurrent state that does not exist, so every arm would measure the "
            f"same thing")

    # Dense models take no expert knobs at all. BLIS refuses dp>1 on a dense model itself, but
    # not an MoE-only ENGINE knob, which would simply do nothing.
    if (model_entry or {}).get("arch") == "dense":
        moe_only = [k for k in ("all2all_backend", "moe_backend") if values(k)]
        if moe_only:
            errs.append(
                f"{where}.engine: sweeps {', '.join(moe_only)} on "
                f"{scenario.get('model')}, which the catalog says is a DENSE model; there are "
                f"no experts to dispatch, so every arm would measure the same thing")
        if eplb:
            errs.append(f"{where}.engine.eplb: expert-parallel load balancing on a dense model "
                        f"has no experts to balance")


def _check_nested_block(block, where, fields, errs):
    """A nested engine block (`eplb`, `dbo`, `engram_config`): known inner fields, all lists.

    The engine decodes these strictly too, so an invented inner field is as fatal as an invented
    top-level knob -- and easier to get wrong, since the inner names are not the block's name
    repeated (`eplb` carries `enable_eplb`, not `enabled`).
    """
    if not isinstance(block, dict) or not block:
        errs.append(f"{where}: must be a non-empty mapping of its own fields "
                    f"({', '.join(fields)})")
        return
    for field, values in block.items():
        if field not in fields:
            errs.append(f"{where}.{field}: not a field of this block"
                        f"{_suggest(field, fields) or f_valid(fields)}")
            continue
        if not _is_list(values):
            errs.append(f"{where}.{field}: every value must be a list "
                        f"(write `{field}: [{json.dumps(values)}]`)")
        elif not values:
            errs.append(f"{where}.{field}: empty list")
        elif field in NESTED_BOOL_FIELDS:
            for v in values:
                if not isinstance(v, bool):
                    errs.append(f"{where}.{field}: {v!r} is not a boolean")


def f_valid(fields):
    """" (valid: a, b, c)" -- kept a helper so the message reads the same everywhere."""
    return " (valid: " + ", ".join(fields) + ")"


def _check_gpu_math_colocated(sweep, gpus, where, errs):
    """At least one combo must total EXACTLY `gpus`, and report what the survivors are.

    `gpus` is the experiment's like-for-like guarantee, so a grid that cannot reach it is not
    a narrow sweep -- it is an empty one, and better caught here than after a CI build.
    """
    insts = sweep.get("num-instances") or [1]
    if not _is_list(insts):
        return                                   # already reported by _check_value_lists
    fps = _parallel_combos(sweep.get("parallel") or {})
    ok = [(fp, ni) for fp in fps for ni in insts
          if isinstance(ni, int) and fp * ni == gpus]
    if ok:
        straddling = [fp for fp, _ in ok if fp > GPUS_PER_NODE and fp % GPUS_PER_NODE]
        if len(straddling) == len(ok):
            errs.append(f"{where}: every surviving combo has a per-instance footprint that "
                        f"straddles a node boundary ({sorted(set(straddling))} GPUs on "
                        f"{GPUS_PER_NODE}-GPU nodes); a multi-node instance must fill whole nodes")
        return
    combos = sorted({f"{fp}x{ni}={fp * ni}" for fp in fps for ni in insts
                     if isinstance(ni, int)})
    errs.append(f"{where}: no combination of tp x pp x dp x num-instances totals the "
                f"scenario's {gpus} GPUs, so this sweep expands to no runs "
                f"(tried: {', '.join(combos[:8])}{' ...' if len(combos) > 8 else ''})")


def _check_gpu_math_pd(sweep, gpus, where, errs):
    """A disaggregated sweep's budget counts BOTH pools: prefill GPUs + decode GPUs == gpus.

    BLIS additionally requires the two pools to agree on dp (one replica factor applies to
    both) and refuses pools that share a node, so this also checks what it can of that.
    """
    pf_i = sweep.get("prefill-instances") or []
    dc_i = sweep.get("decode-instances") or []
    if not (_is_list(pf_i) and _is_list(dc_i)):
        return
    pf_par = ((sweep.get("prefill") or {}).get("parallel")) or {}
    dc_par = ((sweep.get("decode") or {}).get("parallel")) or {}
    pf_dp = pf_par.get("dp", [1])
    dc_dp = dc_par.get("dp", [1])
    if _is_list(pf_dp) and _is_list(dc_dp) and sorted(map(str, pf_dp)) != sorted(map(str, dc_dp)):
        errs.append(f"{where}: the prefill pool sweeps dp {list(pf_dp)} and the decode pool "
                    f"dp {list(dc_dp)}; BLIS applies ONE replica factor to both pools, so "
                    f"their dp must match")
    ok = []
    for pfp in _parallel_combos(pf_par):
        for dfp in _parallel_combos(dc_par):
            for pi in pf_i:
                for di in dc_i:
                    if not (isinstance(pi, int) and isinstance(di, int)):
                        continue
                    if pfp * pi + dfp * di == gpus:
                        ok.append((pfp, pi, dfp, di))
    if not ok:
        errs.append(f"{where}: no combination of prefill + decode pools totals the scenario's "
                    f"{gpus} GPUs (prefill footprints {sorted(set(_parallel_combos(pf_par)))} x "
                    f"{list(pf_i)} + decode {sorted(set(_parallel_combos(dc_par)))} x "
                    f"{list(dc_i)}), so this sweep expands to no runs")


def _check_router(router, where, errs, routing_dir=ROUTING_DIR):
    """Router files must exist, and override trees must name list-valued fields.

    A missing EndpointPickerConfig is the cheapest possible catch: BLIS would fail once per
    run in the sweep, all for the same typo.
    """
    for rel in (router.get("endpoint_picker_config") or []):
        if not isinstance(rel, str):
            continue
        if os.path.isabs(rel) or ".." in rel.split("/"):
            errs.append(f"{where}.endpoint_picker_config: {rel!r} must be a plain file name "
                        f"under routing/")
            continue
        if not os.path.isfile(os.path.join(routing_dir, rel)):
            have = sorted(f for f in os.listdir(routing_dir) if f.endswith(".yaml")) \
                if os.path.isdir(routing_dir) else []
            errs.append(f"{where}.endpoint_picker_config: no such file routing/{rel} "
                        f"(available: {', '.join(have) or 'none'})")
    for profile, refs in (router.get("schedulingProfiles") or {}).items():
        for ref, fields in (refs or {}).items():
            _check_value_lists(fields, f"{where}.schedulingProfiles.{profile}.{ref}", errs)
    for plugin, block in (router.get("plugins") or {}).items():
        known = ROUTER_PLUGIN_PARAMS.get(plugin)
        for section, fields in (block or {}).items():
            if section != "parameters":
                errs.append(f"{where}.plugins.{plugin}.{section}: only `parameters` may be "
                            f"swept inside a plugin")
                continue
            _check_value_lists(fields, f"{where}.plugins.{plugin}.parameters", errs)
            for field in (fields or {}):
                if known is not None and field not in known:
                    errs.append(f"{where}.plugins.{plugin}.parameters.{field}: not a documented "
                                f"parameter of {plugin} (RFC #21 lists: "
                                f"{', '.join(sorted(known))})")


_CATALOG_CACHE = {}


def catalog_entry(model, catalog=None):
    """The catalog row for `model`, or None -- `arch`, `experts`, `context`, `attn_kind`.

    Used only to CHECK a spec against the model it names (dense models take no expert knobs, a
    non-recurrent model takes no mamba knobs, EPLB redundancy must divide into the expert count).
    Optional on purpose: the validator must run on a PR with no catalog checkout, where it
    simply skips the model-aware rules rather than failing.

    The built catalog is cached per root: building it walks every model directory (~0.4s), and
    validating a directory of specs would otherwise pay that once per spec.
    """
    return _catalog(catalog)["models"].get(model)


def _catalog(catalog=None):
    """The built catalog's three vocabularies, keyed `models` (name -> row), `hardware`,
    `workloads` (name -> True), plus `present`. All empty with `present` False when there is no
    catalog checkout.

    One build serves both the model-aware rules and the existence check, and `present` is what
    lets the caller tell "not in the catalog" apart from "no catalog to ask".
    """
    root = catalog or os.path.join(SIBLINGS, "blis-catalog")
    if root not in _CATALOG_CACHE:
        got = {"models": {}, "hardware": {}, "workloads": {}, "present": False}
        if os.path.isdir(root):
            try:
                sys.path.insert(0, os.path.join(ROOT, "scripts"))
                import build_catalog
                cat = build_catalog.build(os.path.join(root, os.pardir))
                got["models"] = {m.get("name"): m for m in cat.get("models", [])}
                for key, src in (("hardware", "hardware"), ("workloads", "workloads")):
                    got[key] = {(e.get("name") if isinstance(e, dict) else e): True
                                for e in (cat.get(src) or [])}
                    got[key].pop(None, None)
                # A build that yielded no models is a broken or empty checkout, not an authority
                # to reject names against.
                got["present"] = bool(got["models"])
            except Exception:
                got = {"models": {}, "hardware": {}, "workloads": {}, "present": False}
        _CATALOG_CACHE[root] = got
    return _CATALOG_CACHE[root]


def scenario_identity_errors(sc, catalog=None):
    """`model` / `hardware` / `workload` that the catalog does not define, with a suggestion.

    EXISTENCE, not feasibility. A name the catalog has no row for provably cannot run -- BLIS
    exits rc=1 with `opening .../hardware/h200x.yaml: no such file or directory` on every run of
    the sweep -- so catching it is the same job as the GPU arithmetic check: refuse what cannot
    run before a clone-and-build, rather than after (susiejojo review). Verified: a spec with
    `hardware: h200x` validated clean, dry-ran clean, planned 154 runs, and failed all of them.

    Whether a KV cache FITS on that hardware stays BLIS's call -- this adds no physics.

    Skipped entirely when no catalog is checked out, exactly as the model-aware rules are, so
    fork CI (which validates with no catalog and no secrets) keeps working.
    """
    cat = _catalog(catalog)
    if not cat["present"]:
        return []
    errs = []
    for key, bucket in (("model", "models"), ("hardware", "hardware"), ("workload", "workloads")):
        name = sc.get(key)
        if not name or name in cat[bucket]:
            continue
        known = sorted(cat[bucket])
        errs.append(f"scenario.{key}: {name!r} is not in the catalog{_suggest(name, known)} "
                    f"-- known: {', '.join(known)}")
    return errs


def header_comment_lines(path):
    """How many comment/blank lines a spec file opens with, before its first real key.

    Counted from the file rather than the parsed spec, because YAML comments are not part of the
    document. A header is the right place for rationale -- it never reaches the report -- but an
    unbounded one buries the grid: a generated spec opened with 36 comment lines, duplicated
    paragraph included, so the sweeps started below the first screen.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return 0
    n = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            n += 1
            continue
        break
    return n


def advisories(spec, path=None):
    """Style notes that are NOT errors: things worth fixing that must not fail a build.

    Kept separate from validate() on purpose. An over-long narrative describes a perfectly
    runnable experiment, so blocking the PR over it would be wrong -- but leaving it unsaid is
    how the field drifted to five-sentence paragraphs in the first place.
    """
    notes = []
    if path:
        n = header_comment_lines(path)
        if n > HEADER_LINES:
            notes.append(
                f"header: {n} comment lines before the first key -- aim for under "
                f"{HEADER_LINES}. Keep the question, the two or three facts that picked the "
                f"axes, and anything deliberately left out; drop transcribed profile output "
                f"and anything references/knobs.md already covers")
    # A replay workload is fully described by its catalog descriptor, so a spec that restates
    # part of it has made a copy that can only drift. Verified: BLIS reads the digest, window,
    # grace period and context filter from blis-catalog/workloads/<name>.yaml, and still refuses
    # a corpus whose bytes do not match -- so none of those belong in a spec.
    rep = ((spec.get("scenario") or {}).get("replay") or {})
    for key in sorted(set(rep) & CATALOG_OWNED_REPLAY_FIELDS):
        notes.append(
            f"scenario.replay.{key}: already stated by the catalog's workload descriptor, which "
            f"the sweeper reads -- remove it rather than keeping a copy that can drift")
    corpus = rep.get("corpus")
    if isinstance(corpus, str) and os.path.isabs(corpus):
        notes.append(
            f"scenario.replay.corpus: {corpus!r} is an absolute path, which pins the spec to one "
            f"machine -- use a repo-relative path such as .corpus/traces.jsonl")
    for i, sw in enumerate(spec.get("sweeps") or []):
        if not isinstance(sw, dict):
            continue
        where = f"sweeps[{i}]({sw.get('name', '?')})"
        text = " ".join(str(sw.get("narrative") or "").split())
        if len(text) > NARRATIVE_CHARS:
            # The REASON is printed once per run by main(), not once per sweep: repeating it 51
            # times buries the file names it is meant to point at.
            notes.append(f"{where}.narrative: {len(text)} characters")
    return notes


def validate(spec, routing_dir=ROUTING_DIR, vocabulary=None, catalog=None):
    """Every error in `spec`, as readable `path: message` strings. Empty means valid.

    Collects ALL errors rather than stopping at the first: a contributor fixing a spec by hand
    should see every problem in one run, not discover them one CI round-trip at a time.
    """
    errs = []
    if not isinstance(spec, dict):
        return ["spec must be a mapping"]
    par_vocab, eng_vocab, source = vocabulary or knob_vocabulary()
    enums, _enum_source = engine_enums()
    entry = None

    name = spec.get("name")
    if not name:
        errs.append("name: required")
    elif not NAME_RE.match(str(name)):
        errs.append(f"name: {name!r} must be lowercase letters, digits and dashes "
                    f"(it is also the reports/<name>/ directory)")

    known_top = {"name", "description", "scenario", "sweeps"}
    for key in spec:
        if key not in known_top:
            errs.append(f"{key}: unknown top-level key (allowed: {', '.join(sorted(known_top))})")

    sc = spec.get("scenario")
    if not isinstance(sc, dict):
        errs.append("scenario: required mapping (model, hardware, workload, gpus, concurrency)")
        sc = {}
    else:
        known_sc = {"model", "hardware", "workload", "gpus", "slo", "concurrency", "seed",
                    "engine", "replay"}
        for key in sc:
            if key not in known_sc:
                errs.append(f"scenario.{key}: unknown key "
                            f"(allowed: {', '.join(sorted(known_sc))})")
        for key in ("model", "hardware", "workload"):
            if not sc.get(key):
                errs.append(f"scenario.{key}: required")
        # Present AND real: a name the catalog does not define cannot run anywhere. No-op without
        # a catalog checkout.
        errs.extend(scenario_identity_errors(sc, catalog))
        gpus = sc.get("gpus")
        if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 1:
            errs.append("scenario.gpus: required positive integer -- the budget EVERY run uses")
            gpus = None
        conc = sc.get("concurrency")
        if not _is_list(conc) or not conc:
            errs.append("scenario.concurrency: required non-empty list of load levels")
        else:
            bad = [c for c in conc if not isinstance(c, int) or isinstance(c, bool) or c < 1]
            if bad:
                errs.append(f"scenario.concurrency: {bad} are not positive integers")
            elif list(conc) != sorted(conc):
                errs.append(f"scenario.concurrency: {list(conc)} must be ascending, so a "
                            f"capacity curve reads left to right")
        slo = sc.get("slo")
        if slo is not None:
            if not isinstance(slo, dict) or not slo:
                errs.append("scenario.slo: must be a non-empty mapping, e.g. {itl_p99: 20ms}")
            else:
                known_slo = {"itl_p99", "ttft_p99", "e2e_p99"}
                for key, val in slo.items():
                    if key not in known_slo:
                        errs.append(f"scenario.slo.{key}: unknown SLO "
                                    f"(allowed: {', '.join(sorted(known_slo))})")
                    elif parse_duration_ms(val) is None:
                        errs.append(f"scenario.slo.{key}: {val!r} needs an explicit unit "
                                    f"(e.g. 20ms, 5s) -- a bare number is ambiguous")
        # scenario.engine holds settings every run carries, so its values are SINGLE, not lists.
        # Getting this backwards is easy, and a list here would be written into the scenario
        # verbatim and refused by the engine, so say which way round it goes.
        for key, val in (sc.get("engine") or {}).items():
            if _is_list(val) and key not in ENGINE_INT_LISTS:
                errs.append(f"scenario.engine.{key}: a shared setting takes ONE value, not a "
                            f"list (sweep it under a sweep's `engine:` instead)")
        _check_knob_names(sc.get("engine") or {}, "scenario.engine", eng_vocab, "engine", errs,
                          source)
        # A shared setting is a single value, so wrap each in a list to reuse the same value
        # check the swept ones get -- a bad `cudagraph_mode` is just as fatal here.
        _check_engine_values({k: (v if _is_list(v) and k in ENGINE_INT_LISTS else [v])
                              for k, v in (sc.get("engine") or {}).items()},
                             "scenario.engine", errs, enums)
        seed = sc.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
            errs.append("scenario.seed: must be a non-negative integer")
        # Read once, for the model-aware cross-knob rules below. None (no catalog checkout) just
        # skips them.
        entry = catalog_entry(sc.get("model"), catalog)

    sweeps = spec.get("sweeps")
    if not _is_list(sweeps) or not sweeps:
        errs.append("sweeps: required non-empty list")
        return errs

    seen = set()
    for i, sw in enumerate(sweeps):
        where = f"sweeps[{i}]"
        if not isinstance(sw, dict):
            errs.append(f"{where}: must be a mapping")
            continue
        sname = sw.get("name")
        if not sname:
            errs.append(f"{where}.name: required")
        elif not NAME_RE.match(str(sname)):
            errs.append(f"{where}.name: {sname!r} must be lowercase letters, digits and dashes")
        else:
            where = f"sweeps[{i}]({sname})"
            if sname in seen:
                errs.append(f"{where}: duplicate sweep name -- each sweep is a column in the "
                            f"report and must be distinguishable")
            seen.add(sname)

        known_sw = {"name", "narrative", "num-instances", "parallel", "engine", "router",
                    "prefill-instances", "decode-instances", "prefill", "decode"}
        for key in sw:
            if key not in known_sw:
                errs.append(f"{where}.{key}: unknown sweep key "
                            f"(allowed: {', '.join(sorted(known_sw))})")

        colocated = {"num-instances", "parallel"} & set(sw)
        disagg = {"prefill-instances", "decode-instances", "prefill", "decode"} & set(sw)
        if colocated and disagg:
            errs.append(f"{where}: mixes colocated keys ({', '.join(sorted(colocated))}) with "
                        f"disaggregated ones ({', '.join(sorted(disagg))}); BLIS refuses a "
                        f"colocated pool alongside prefill/decode pools, so a sweep is one "
                        f"or the other")
        elif disagg:
            for a, b in (("prefill-instances", "decode-instances"),
                         ("decode-instances", "prefill-instances"),
                         ("prefill", "decode"), ("decode", "prefill")):
                if a in sw and b not in sw:
                    errs.append(f"{where}: has {a} but no {b}; BLIS refuses a {a.split('-')[0]} "
                                f"pool with no counterpart pool")

        for key in ("num-instances", "prefill-instances", "decode-instances"):
            val = sw.get(key)
            if val is None:
                continue
            if not _is_list(val) or not val:
                errs.append(f"{where}.{key}: must be a non-empty list of instance counts")
            else:
                bad = [v for v in val if not isinstance(v, int) or isinstance(v, bool) or v < 1]
                if bad:
                    errs.append(f"{where}.{key}: {bad} are not positive integers")

        _check_value_lists(sw.get("parallel") or {}, f"{where}.parallel", errs)
        _check_knob_names(sw.get("parallel") or {}, f"{where}.parallel", par_vocab, "parallel",
                          errs, source)
        _check_engine(sw.get("engine") or {}, f"{where}.engine", eng_vocab, source, errs, enums)
        for role in ("prefill", "decode"):
            block = sw.get(role)
            if not isinstance(block, dict):
                if block is not None:
                    errs.append(f"{where}.{role}: must be a mapping with parallel/engine")
                continue
            for key in block:
                if key not in ("parallel", "engine"):
                    errs.append(f"{where}.{role}.{key}: only `parallel` and `engine` may be "
                                f"stated per role")
            _check_value_lists(block.get("parallel") or {}, f"{where}.{role}.parallel", errs)
            _check_knob_names(block.get("parallel") or {}, f"{where}.{role}.parallel", par_vocab,
                              "parallel", errs, source)
            _check_engine(block.get("engine") or {}, f"{where}.{role}.engine", eng_vocab, source,
                          errs, enums)

        router = sw.get("router")
        if router is not None:
            if not isinstance(router, dict) or not router:
                errs.append(f"{where}.router: must be a non-empty mapping")
            else:
                for key in router:
                    if key not in ("endpoint_picker_config", "schedulingProfiles", "plugins"):
                        errs.append(f"{where}.router.{key}: unknown router key (allowed: "
                                    f"endpoint_picker_config, schedulingProfiles, plugins)")
                if not _is_list(router.get("endpoint_picker_config", [])):
                    errs.append(f"{where}.router.endpoint_picker_config: must be a list of files")
                # An override tree edits fields of a BASE file, so there must be exactly one base
                # to edit -- against several, "the same override" would mean different things.
                bases = router.get("endpoint_picker_config") or []
                if (router.get("schedulingProfiles") or router.get("plugins")):
                    if not bases:
                        errs.append(f"{where}.router: overrides name no base "
                                    f"endpoint_picker_config to apply them to")
                    elif _is_list(bases) and len(bases) > 1:
                        errs.append(f"{where}.router: overrides apply to ONE base file, but "
                                    f"{len(bases)} are listed ({', '.join(map(str, bases))}); "
                                    f"split into separate sweeps")
                _check_router(router, f"{where}.router", errs, routing_dir)

        if isinstance(sc.get("gpus"), int) and not isinstance(sc.get("gpus"), bool):
            if disagg:
                _check_gpu_math_pd(sw, sc["gpus"], where, errs)
            elif colocated:
                _check_gpu_math_colocated(sw, sc["gpus"], where, errs)
        # Cross-knob and model-aware constraints LAST: they read the sweep as a whole, so they
        # run once the individual knobs are known to be well-formed.
        if isinstance(sc, dict):
            _check_cross_knob(sw, sc, where, errs, entry)
    return errs


def _check_engine(engine, where, vocabulary, source, errs, enums=None):
    """Engine knobs, with the nested blocks (`speculative`, `eplb`, …) handled as such."""
    _check_value_lists(engine, where, errs,
                       allow_nested=tuple(NESTED_ENGINE_FIELDS) + ENGINE_INT_LISTS)
    _check_knob_names(engine, where, vocabulary, "engine", errs, source)
    _check_engine_values(engine, where, errs, enums)
    for block, fields in NESTED_ENGINE_FIELDS.items():
        if block == "speculative" or block not in (engine or {}):
            continue                        # speculative has its own stricter checks below
        _check_nested_block(engine[block], f"{where}.{block}", fields, errs)
    spec = engine.get("speculative")
    if spec is None:
        return

    if not isinstance(spec, dict):
        errs.append(f"{where}.speculative: must be a mapping with method and num_spec_tokens")
        return
    for key in spec:
        if key not in ("method", "num_spec_tokens"):
            # blis-vllm v0.32.0's Speculative carries seven fields (draft_model, kv_cache_dtype,
            # draft_sample_method, rejection_sample_method, enable_adaptive_verification), but
            # the PINNED blis-schemas v0.5.0 holds only these two, so the rest cannot be swept.
            errs.append(f"{where}.speculative.{key}: not a field the pinned stack carries (only "
                        f"method and num_spec_tokens; blis-schemas v0.5.0's Speculative has no "
                        f"other field, and acceptance comes from the registry, not a knob)")
    methods = (enums or {}).get("speculative_method")
    if methods and _is_list(spec.get("method")):
        for m in spec["method"]:
            if str(m) not in methods:
                errs.append(f"{where}.speculative.method: {m!r} is not a draft method the "
                            f"engine knows{_suggest(m, methods)}")
    # BOTH are required: BLIS refuses a drafting scenario whose acceptance does not resolve, and
    # acceptance is selected by the method AND the depth. Naming one without the other is the
    # mistake RFC #21 calls out explicitly.
    for key in ("method", "num_spec_tokens"):
        if key not in spec:
            errs.append(f"{where}.speculative: needs both method and num_spec_tokens "
                        f"(missing {key})")
    _check_value_lists(spec, f"{where}.speculative", errs)


def load(path):
    """(spec, error). A YAML error is returned, not raised: one bad file in a directory run
    should not hide the rest."""
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f), None
    except OSError as e:
        return None, f"cannot read: {e}"
    except yaml.YAMLError as e:
        return None, f"not valid YAML: {str(e).splitlines()[0]}"


def spec_paths(target):
    """The spec files `target` names: one file, or every *.yaml in a directory (sorted)."""
    if os.path.isdir(target):
        return [os.path.join(target, f) for f in sorted(os.listdir(target))
                if f.endswith((".yaml", ".yml"))]
    return [target]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate Experiment Specs (experiments/<name>.yaml) against RFC #21.")
    ap.add_argument("target", nargs="?", default=os.path.join(ROOT, "experiments"),
                    help="a spec file, or a directory of them (default: experiments/)")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable {file: [errors]} instead of text")
    ap.add_argument("--routing", default=ROUTING_DIR,
                    help="directory holding the EndpointPickerConfig files (default: routing/)")
    ap.add_argument("--catalog",
                    help="blis-catalog checkout, for checking that scenario.model/hardware/"
                         "workload exist (default: ../blis-catalog; skipped when absent)")
    args = ap.parse_args(argv)

    paths = spec_paths(args.target)
    if not paths:
        print(f"no spec files under {args.target}", file=sys.stderr)
        return 1
    vocab = knob_vocabulary()
    report, notes, failed = {}, {}, 0
    for path in paths:
        # Inside the repo a relative path is what a contributor recognises; outside it (a spec
        # under /tmp, say) `relpath` climbs out with ../.. noise, so keep the path as given.
        rel = os.path.relpath(path, ROOT)
        if rel.startswith(os.pardir):
            rel = path
        spec, err = load(path)
        errs = [err] if err else validate(spec, args.routing, vocab, args.catalog)
        report[rel] = errs
        notes[rel] = [] if err else advisories(spec, path)
        if errs:
            failed += 1
    if args.as_json:
        print(json.dumps({"errors": report, "advisories": notes}, indent=2, sort_keys=True))
    else:
        for rel in sorted(report):
            if report[rel]:
                print(f"FAIL {rel}")
                for e in report[rel]:
                    print(f"  - {e}")
            else:
                print(f"ok   {rel}")
            for n in notes.get(rel) or ():
                print(f"  note: {n}")
        # Say whether model/hardware/workload were CHECKED or skipped: "valid" means something
        # weaker without a catalog, and a silent skip is how a typo'd identifier reaches a build.
        cat = _catalog(args.catalog)
        ident = (f"model/hardware/workload checked against the catalog "
                 f"({len(cat['models'])} models, {len(cat['hardware'])} hardware, "
                 f"{len(cat['workloads'])} workloads)") if cat["present"] else (
                 "model/hardware/workload NOT checked -- no blis-catalog checkout")
        print(f"\n{len(paths) - failed}/{len(paths)} valid "
              f"(knob names checked against {vocab[2]}; {ident})")
        all_notes = [n for v in notes.values() for n in v]
        if all_notes:
            print(f"\n{len(all_notes)} style note(s), not failures.")
            if any(".narrative:" in n for n in all_notes):
                print(f"  narrative: the one line the report prints above a sweep's measured "
                      f"result -- keep it to one sentence under {NARRATIVE_CHARS} characters, "
                      f"and put grid rationale, prior measurements and skipped-sweep reasons in "
                      f"a `#` comment, which never reaches the report.")
            if any(n.startswith("header:") for n in all_notes):
                print(f"  header: keep it under {HEADER_LINES} lines -- the question, the two or "
                      f"three facts that picked the axes, and anything deliberately left out. "
                      f"The grid itself is the documentation.")
            if any("scenario.replay." in n for n in all_notes):
                print(f"  replay: the catalog's workload descriptor already states the digest, "
                      f"window, grace period and context filter, and the sweeper reads them from "
                      f"there -- a spec states only the local corpus path.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
