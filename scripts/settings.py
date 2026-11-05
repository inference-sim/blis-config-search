#!/usr/bin/env python3
"""The engine settings and pool-layout keys a spec may name, read from the simulator.

Config search keeps no list of settings (CFG-3). The engine's settings are the entries of its
field list, which `blis fields --engine <kind>` prints at the release the binary runs; the
pool-layout keys are the properties of the deployment document's Parallelism, which `blis schema
deployment` prints (SUB-19). This module turns those two outputs into one Vocabulary that the
validator and the runner share.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_cli  # noqa: E402

# The engine kind every scenario this repository writes names. SGLang searches come in M3.
ENGINE_KIND = "vllm"

# Engine-block names that differ from vLLM's own name for the setting, mapped to the field-list
# entry ("config:<name>" or "env:<NAME>"). blis-vllm's engine block (rules/engineconfig/engine.go
# at e68846f) spells these differently from vLLM's EngineArgs and envs, which X1 does not allow.
# A spec must use the engine block's name, or the block does not decode; the field list gives the
# setting's status. Each entry goes when the engine block takes vLLM's name. Engine-block names
# with no field-list entry (allreduce_backend, env) are not settable here.
ENGINE_BLOCK_NAMES = {
    "cache_dtype": "config:kv_cache_dtype",
    "cudagraph_mode": "config:compilation_config.cudagraph_mode",
    "custom_ops": "config:compilation_config.custom_ops",
    "decoder_replay_cudagraph_capture_sizes":
        "config:compilation_config.decoder_replay_cudagraph_capture_sizes",
    "use_trtllm_attention": "config:attention_config.use_trtllm_attention",
    "mla_prefill_backend": "config:attention_config.mla_prefill_backend",
    "indexer_kv_dtype": "config:attention_config.indexer_kv_dtype",
    "frontend": "env:VLLM_USE_RUST_FRONTEND",
    "dbo.enable_dbo": "config:enable_dbo",
    "dbo.dbo_decode_token_threshold": "config:dbo_decode_token_threshold",
    "dbo.dbo_prefill_token_threshold": "config:dbo_prefill_token_threshold",
    "eplb.enable_eplb": "config:enable_eplb",
    "eplb.num_redundant_experts": "config:eplb_config.num_redundant_experts",
    "eplb.window_size": "config:eplb_config.window_size",
    "eplb.step_interval": "config:eplb_config.step_interval",
    "kv_events.enable_kv_cache_events": "config:kv_events_config.enable_kv_cache_events",
    "kv_events.publisher": "config:kv_events_config.publisher",
    "kv_events.topic": "config:kv_events_config.topic",
    "speculative.method": "config:speculative_config.method",
    "speculative.num_spec_tokens": "config:speculative_config.num_speculative_tokens",
    "speculative.draft_model": "config:speculative_config.model",
    "speculative.kv_cache_dtype": "config:speculative_config.kv_cache_dtype",
    "speculative.draft_sample_method": "config:speculative_config.draft_sample_method",
    "speculative.rejection_sample_method": "config:speculative_config.rejection_sample_method",
    "speculative.enable_adaptive_verification":
        "config:speculative_config.enable_adaptive_verification",
}


class Vocabulary:
    """The names a spec may use, as one simulator build publishes them.

    `parallel`: the pool-layout keys. `flat`: the top-level engine keys. `blocks`: each nested
    engine block and its inner keys. `release`: the engine release the field list describes.
    `source`: a phrase naming where the names came from, for messages.
    """

    def __init__(self, parallel, flat, blocks, release, source, status=None):
        self.parallel = frozenset(parallel)
        self.flat = frozenset(flat)
        self.blocks = {b: frozenset(v) for b, v in blocks.items()}
        self.release, self.source = release, source
        # spec name ("max_num_seqs", "speculative.method") -> its field-list entry
        self.status = dict(status or {})

    def engine_names(self):
        """Every top-level engine key: the flat settings and the nested blocks' names."""
        return self.flat | frozenset(self.blocks)


def vocabulary_from(field_list, deployment_schema):
    """A Vocabulary from a parsed field list and the deployment document's JSON Schema.

    A field that ENGINE_BLOCK_NAMES renames is named only by its engine-block name, because the
    engine block refuses vLLM's name for it. Environment variables are named only through
    ENGINE_BLOCK_NAMES; a spec does not set the launch environment.
    """
    entries = {f"{f['source']}:{f['name']}": f for f in field_list["fields"]}
    renamed = {target: alias for alias, target in ENGINE_BLOCK_NAMES.items() if target in entries}
    status = {}
    for key, entry in entries.items():
        if key in renamed:
            status[renamed[key]] = entry
        elif entry["source"] == "config":
            status[entry["name"]] = entry
    names = set(status)
    flat = {n for n in names if "." not in n}
    blocks = {}
    for n in sorted(names - flat):
        block, _, inner = n.partition(".")
        if block not in flat and "." not in inner:
            blocks.setdefault(block, set()).add(inner)
    try:
        parallel = deployment_schema["$defs"]["deployment.Parallelism"]["properties"]
    except (KeyError, TypeError) as e:
        raise blis_cli.BlisCommandError(
            "`blis schema deployment` has no deployment.Parallelism properties") from e
    engine = field_list["engine"]
    return Vocabulary(parallel, flat, blocks, engine["version"],
                      f"`blis fields --engine {engine['kind']}` ({engine['kind']} "
                      f"{engine['version']}) and `blis schema deployment`", status)


# Statuses whose field changes no decision of the engine. Sweeping one measures the same number
# in every arm, so a sweep over it is refused (VLM-3 statuses; design §5, CFG-5).
INERT = ("proven_inert", "inert_unproven")


def inert_reason(vocab, name):
    """Why sweeping `name` measures nothing, from its field-list entry, or None."""
    entry = vocab.status.get(name)
    if not entry or entry.get("status") not in INERT:
        return None
    ev = entry.get("evidence") or {}
    why = ev.get("proof") or ev.get("note") or ""
    return (f"the field list marks it {entry['status']} ({why}), so every arm of the sweep "
            f"would report the same number; state it in scenario.engine if a run should record it")


def vocabulary(blis):
    """The Vocabulary the binary `blis` publishes. Raises blis_cli.BlisCommandError."""
    return vocabulary_from(blis_cli.fields(blis, ENGINE_KIND), blis_cli.schema(blis, "deployment"))
