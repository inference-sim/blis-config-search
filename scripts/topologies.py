#!/usr/bin/env python3
"""Readable names for a candidate config: its topology label and its knob settings.

run_experiment.py names every finalist with these, so a report reads "8x TP4xDP2 + EP, batch
tokens 2,048" rather than a field dump.
"""


def shape(tp=1, pp=1, dp=1, enable_expert_parallel=False):
    """One pool's parallel shape, without an instance count: "TP4xPP2xDP2 + EP".

    pp and dp show only when > 1, so a plain `TP8` stays uncluttered. Shared by label() and both
    pools of run_experiment.label_of so they cannot drift; the shape is the per-instance GPU
    footprint (tp x pp x dp) plus EP.
    """
    s = (f"TP{tp}"
         + (f"xPP{pp}" if pp and pp > 1 else "")
         + (f"xDP{dp}" if dp and dp > 1 else ""))
    return s + (" + EP" if enable_expert_parallel else "")


def label(cand):
    """Topology label for a candidate ("32x TP4xPP2xDP2 + EP").

    Deliberately topology-ONLY: instances x TP x PP x DP x EP is what dominates an LLM serving
    decision and what a reader scans down a column; the first four are the per-instance GPU
    footprint (tp x pp x dp) times the instance count. Engine and routing knobs are described
    separately by knob_phrase(); squeezing them in here produced
    `64x TP4xDP2 + EP · ll · b8k · seq256 · mem0.9`, unreadable and full of invented shorthand.
    """
    return (f"{cand['num_instances']}x "
            + shape(cand["tp"], cand.get("pp", 1), cand.get("dp", 1),
                    cand.get("enable_expert_parallel")))


# Readable knob names, spelled out on purpose: "b2k" for max_num_batched_tokens=2048 tells a
# reader nothing. Mirrors report.KNOB_NAMES, keyed by candidate-config key.
KNOB_NAMES = {"num_speculative_tokens": "draft tokens", "speculative_method": "drafter",
              "router_config": "routing", "routing_policy": "routing",
              "routing_scorers": "scorers",
              "max_num_batched_tokens": "batch tokens", "max_num_seqs": "max seqs",
              "block_size": "block size", "gpu_memory_utilization": "gpu mem util",
              "scheduler": "scheduler", "admission_policy": "admission",
              # The wider engine surface an Experiment Spec can sweep. Each is the name a
              # reader would use for the setting, not the engine's field name: a variant
              # reading "cudagraph_mode FULL_AND_PIECEWISE" is a field dump, where "graph
              # capture full+piecewise" is a description.
              "cudagraph_mode": "graph capture",
              "max_cudagraph_capture_size": "max captured step",
              "cudagraph_capture_sizes": "captured steps",
              "long_prefill_token_threshold": "prefill chunk",
              "scheduling_policy": "scheduling", "async_scheduling": "async scheduling",
              "enable_prefix_caching": "prefix caching",
              "disable_cascade_attn": "cascade attention disabled",
              "swa_bounded_replay": "bounded SWA replay",
              "all2all_backend": "all-to-all", "moe_backend": "MoE kernel",
              "allreduce_backend": "all-reduce",
              "disable_custom_all_reduce": "custom all-reduce disabled",
              "cache_dtype": "KV dtype", "quantization": "weights",
              "max_model_len": "context window",
              "dcp_comm_backend": "decode-CP transport",
              "dcp_q_replicate": "decode-CP query replication",
              "cp_kv_cache_interleave_size": "CP interleave",
              "mamba_cache_dtype": "mamba cache dtype",
              "mamba_ssm_cache_dtype": "mamba SSM cache dtype",
              "mamba_cache_mode": "mamba cache mode",
              # nested engine blocks and their inner fields
              "eplb": "expert balancing", "dbo": "dual-batch overlap",
              "engram_config": "engram",
              "enable_eplb": "expert balancing",
              "num_redundant_experts": "redundant experts",
              "window_size": "window", "step_interval": "step interval",
              "enable_dbo": "dual-batch overlap",
              "dbo_decode_token_threshold": "DBO decode threshold",
              "dbo_prefill_token_threshold": "DBO prefill threshold",
              "cpu_offload": "CPU offload", "dp_shared_memory": "DP shared memory",
              "use_thp": "huge pages",
              # Opened up by the v0.14.0 pin (the engine surface moved to blis-vllm's
              # engineconfig.Engine, which carries these; blis-schemas' old 31-field struct
              # did not). Only the three the cost model actually prices are sweepable, so
              # only those need a label. Named for what they do, as above.
              "max_num_active_seqs": "active seqs cap",
              "max_num_scheduled_tokens": "scheduled tokens cap",
              "frontend": "API frontend"}

# Knobs whose value is a BOOLEAN, phrased as a state rather than "name True". "prefix caching
# False" reads as a value dump; "prefix caching off" reads as a deployment.
_BOOL_KNOB_WORDS = {
    "enable_prefix_caching": ("prefix caching on", "prefix caching off"),
    "async_scheduling": ("async scheduling on", "async scheduling off"),
    "disable_cascade_attn": ("cascade attention off", "cascade attention on"),
    "disable_custom_all_reduce": ("custom all-reduce off", "custom all-reduce on"),
    "swa_bounded_replay": ("bounded SWA replay on", "bounded SWA replay off"),
    "dcp_q_replicate": ("decode-CP query replication on", "decode-CP query replication off"),
}


def routing_arm_name(path):
    """The readable name of a routing arm, from its EPP config path.

    "routing/prefix-affinity.yaml" -> "prefix-affinity". The file basename IS the arm's name (see
    routing/README.md), so a reader sees "prefix-affinity routing" rather than a path.
    """
    base = str(path).rsplit("/", 1)[-1]
    return base[:-5] if base.endswith(".yaml") else base


def knob_phrase(key, value):
    """Spelled-out description of one knob setting, e.g. "batch tokens 2,048"."""
    if key == "num_speculative_tokens":
        return "no drafting" if not value else f"{value} draft token{'s' if value != 1 else ''}"
    if key == "router_config":
        return f"{routing_arm_name(value)} routing"
    if key == "routing_policy":
        return f"{value} routing"
    if key in _BOOL_KNOB_WORDS and isinstance(value, bool):
        on, off = _BOOL_KNOB_WORDS[key]
        return on if value else off
    name = KNOB_NAMES.get(key, key)
    if key == "cudagraph_mode":
        # The enum is SHOUTED (FULL_AND_PIECEWISE); spell it as words.
        return f"{name} {str(value).lower().replace('_and_', '+').replace('_', ' ')}"
    if key == "cudagraph_capture_sizes" and isinstance(value, (list, tuple)):
        return f"{name} {', '.join(str(v) for v in value)}"
    if isinstance(value, int) and not isinstance(value, bool) and key in (
            "max_num_batched_tokens", "max_num_seqs", "block_size", "max_model_len",
            "long_prefill_token_threshold", "max_cudagraph_capture_size",
            "cp_kv_cache_interleave_size"):
        return f"{name} {value:,}"
    return f"{name} {value}"
