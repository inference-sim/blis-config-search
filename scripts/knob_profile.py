#!/usr/bin/env python3
"""What knobs are worth sweeping for ONE model x hardware x workload, read from the sources.

    python3 -I scripts/knob_profile.py --model deepseek-v4.1-flash --hardware b200 \
        --workload agentx-mvp

Prints, per knob: whether it APPLIES to this combination, the values that are legal HERE, and
why. Exists because the three facts that decide a grid live in three different places, and none
of them is the catalog summary a designer tends to read:

  * the catalog's model row      -- arch, experts, top_k, attn_kind, context, quant
  * the model's graph.yaml       -- the declared speculator and its max depth, the KV
                                    bytes_by_cache_dtype table (which dtypes this model prices
                                    at all), sliding window
  * blis-vllm's rules pack       -- the ARCHITECTURE-keyed rules: which cache dtype a request
                                    resolves to on this chip's capability family, which draft
                                    methods the architecture refuses, whether Engram applies
  * the hardware entry           -- ComputeCapability, which selects among those resolutions

A designer reading only the catalog row gets no speculator, no sliding window, no dtype table
and no capability gating -- so they either guess or omit a real axis. This reports them.

NOT a recommender: it says what is legal and applicable, never what will win. Leverage is
measured, and lives in the skill's references/knobs.md.
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_catalog                      # noqa: E402
import validate_experiment as validator   # noqa: E402


def _load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or default
    except (OSError, yaml.YAMLError):
        return default


def _walk_caches(node, out, family=None, current=None):
    """Collect bytes_by_cache_dtype tables, recording the capability family that gates each.

    The table is NOT at the graph root: a model states base caches plus `arch_overrides`, each
    keyed by `capability_family`, and the override's `caches` carry the per-dtype byte counts.
    DeepSeek-V4.1 prices nvfp4_ds_mla only under family 10, so reading the root alone reports no
    dtype table at all -- which is how a B200-only KV axis gets missed.
    """
    if isinstance(node, dict):
        fam = node.get("capability_family", current if current is not None else family)
        for cache in (node.get("caches") or []):
            if isinstance(cache, dict) and cache.get("bytes_by_cache_dtype"):
                for dtype in sorted(cache["bytes_by_cache_dtype"]):
                    out.setdefault(dtype, set()).add(fam)
        for key, value in node.items():
            if key != "caches":
                _walk_caches(value, out, family, fam)
    elif isinstance(node, list):
        for item in node:
            _walk_caches(item, out, family, current)


def graph_facts(catalog, model):
    """Model facts that live in graph.yaml, not in the catalog summary row.

    The catalog row carries no speculator, no sliding window and no per-dtype KV table, so a
    designer reading only the row cannot tell whether drafting is legal, which cache dtypes this
    model prices, or how deep a draft may go.
    """
    path = os.path.join(catalog, "models", model, "graph.yaml")
    doc = _load(path, {}) or {}
    spec = doc.get("speculator") or {}
    out = {
        "architecture": ((doc.get("global") or {}).get("architecture")),
        "speculator": spec.get("method"),
        "max_spec_tokens": spec.get("num_spec_tokens"),
        "kv_dtypes_priced": [],
        "sparse_mla": False,
    }
    # A dtype absent from the graph's bytes_by_cache_dtype tables is not priced for this model,
    # so sweeping it measures the default entry instead. The tables are gated by capability
    # family, so keep that: it is what makes a dtype hardware-specific.
    tables = {}
    _walk_caches(doc, tables, family=(doc.get("base_capability_families") or [None])[0])
    out["kv_dtypes_priced"] = sorted(tables)
    out["kv_dtype_families"] = {d: sorted(f for f in fams if f is not None)
                                for d, fams in sorted(tables.items())}
    out["base_capability_families"] = doc.get("base_capability_families")
    raw = json.dumps(doc) if doc else ""
    out["sparse_mla"] = "sparse_mla" in raw or "indexer" in raw
    cfg = os.path.join(catalog, "models", model, "config.json")
    try:
        with open(cfg, encoding="utf-8") as f:
            conf = json.load(f)
        inner = conf.get("text_config") or conf
        out["sliding_window"] = inner.get("sliding_window")
        # Only a model with architecture-SPECIFIC engine rules states `architecture` in its
        # graph global (DeepSeek-V4/V4.1 do; qwen3-14b, kimi-k2.5 and glm-5.3-flash do not).
        # config.json always carries it, so fall back there: the arch-keyed lookups below need
        # a real name to report "not an Engram architecture" rather than "None is not one".
        if not out["architecture"]:
            archs = conf.get("architectures") or inner.get("architectures") or []
            out["architecture"] = archs[0] if archs else None
            out["architecture_source"] = "config.json"
        else:
            out["architecture_source"] = "graph.yaml global"
    except (OSError, ValueError):
        out["sliding_window"] = None
        out["architecture_source"] = None
    return out


def hardware_facts(catalog, hardware):
    """Hardware facts that gate knob VALUES: capability family and memory."""
    doc = _load(os.path.join(catalog, "hardware", f"{hardware}.yaml"), {}) or {}
    cc = str(doc.get("ComputeCapability") or "")
    family = None
    m = re.match(r"^(\d+)", cc)
    if m:
        family = int(m.group(1))
    return {"compute_capability": cc or None, "capability_family": family,
            "memory_gib": doc.get("MemoryGiB"), "gpus_per_node": doc.get("GPUsPerNode"),
            "tflops_nvfp4": doc.get("TFlopsNVFP4")}


def _pack_source():
    path = os.path.join(SIBLINGS, "blis-vllm", "rules", "v0_32", "pack.go")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


_ARCH_CONSTS = {"ArchDeepseekV41": "DeepseekV41ForCausalLM",
                "ArchDeepseekV4": "DeepseekV4ForCausalLM"}


def arch_rules(architecture):
    """Architecture-keyed rules from blis-vllm's pack, for `architecture`.

    These are the rules a per-model grid most needs and the catalog cannot express:
      * CacheDTypeResolutions -- what `auto`/`fp8` actually RESOLVE to on a given capability
        family, which is why nvfp4_ds_mla is a B200-only axis for DeepSeek-V4.1.
      * the MTP refusal        -- DeepSeek-V4.1 refuses every MTP-family draft method.
      * EngramArchitectures    -- whether engram_config applies at all.
    """
    src = _pack_source()
    out = {"cache_dtype_resolutions": [], "refuses_mtp": False, "engram": False,
            "source": "../blis-vllm rules pack" if src else "unavailable"}
    if not src or not architecture:
        return out
    const = next((k for k, v in _ARCH_CONSTS.items() if v == architecture), None)
    if const:
        # The map is `CacheDTypeResolutions: map[...]{ rules.<Arch>: { {row}, {row}, ... }, }`.
        # Take everything from this architecture's key to the closing `},` of ITS list -- an
        # earlier attempt cut at the first `},` and so saw only the first row, losing the
        # family-9 fp8_ds_mla resolution entirely.
        block = re.search(r"CacheDTypeResolutions:\s*map\[[^\]]*\][^{]*\{(.*?)\n\t\t\},", src,
                          re.S)
        if block and f"rules.{const}:" in block.group(1):
            seg = block.group(1).split(f"rules.{const}:", 1)[1]
            seg = seg.split("\n\t\t\t},", 1)[0]
            for row in re.finditer(
                    r"CapabilityFamily:\s*(\d+),\s*Requested:\s*\[\]string\{([^}]*)\},"
                    r"\s*ResolvesTo:\s*\"([^\"]+)\"(?:,\s*TPDivides:\s*(\d+))?", seg):
                out["cache_dtype_resolutions"].append({
                    "capability_family": int(row.group(1)),
                    "requested": re.findall(r'"([^"]+)"', row.group(2)),
                    "resolves_to": row.group(3),
                    "tp_divides": int(row.group(4)) if row.group(4) else None})
        if const == "ArchDeepseekV41" and "deepseek-v41-refuses-mtp" in src:
            out["refuses_mtp"] = True
    eng = re.search(r"EngramArchitectures:\s*set\(([^)]*)\)", src, re.S)
    if eng:
        names = re.findall(r'"([^"]+)"', eng.group(1))
        consts = re.findall(r"rules\.(Arch\w+)", eng.group(1))
        names += [_ARCH_CONSTS.get(c, c) for c in consts]
        out["engram"] = architecture in names
    return out


def workload_facts(catalog, workload):
    """Whether the workload is a replay, and the traffic facts a grid depends on."""
    doc = _load(os.path.join(catalog, "workloads", f"{workload}.yaml"), {}) or {}
    rep = doc.get("replay")
    if isinstance(rep, dict):
        return {"kind": "replay", "scenario": rep.get("scenario"),
                "category": rep.get("category"), "thinking": rep.get("thinking"),
                "max_context_length": rep.get("max_context_length"),
                "prefix_reuse": True}
    shape = {k: doc.get(k) for k in ("isl_mean", "osl_mean", "isl", "osl") if doc.get(k)}
    return {"kind": "shape", "shape": shape or None, "prefix_reuse": False}


def profile(model, hardware, workload, catalog):
    """The per-knob verdict for this combination: applies, legal values here, and why."""
    cat = build_catalog.build(os.path.join(catalog, os.pardir))
    row = next((m for m in cat.get("models", []) if m.get("name") == model), None)
    if row is None:
        raise SystemExit(f"{model!r} is not in the catalog at {catalog}")
    g = graph_facts(catalog, model)
    hw = hardware_facts(catalog, hardware)
    ar = arch_rules(g.get("architecture"))
    wl = workload_facts(catalog, workload)
    enums, _src = validator.engine_enums()
    moe = row.get("arch") == "MoE"
    fam = hw.get("capability_family")

    knobs = []

    def add(knob, applies, values, why):
        knobs.append({"knob": knob, "applies": applies, "values": values, "why": why})

    # --- always applicable -----------------------------------------------------------------
    add("cudagraph_mode", True, list(enums["cudagraph_mode"]),
        "per-step host launch cost; applies to every model and chip")
    add("max_num_seqs", True, [16, 32, 64, 128, 256, 512],
        "binds downward; INTERACTS with instance count (per-instance session share)")
    add("max_num_batched_tokens", True, [2048, 4096, 8192, 16384, 32768],
        "binds downward; interacts with max_num_seqs (whichever cap binds first)")
    add("gpu_memory_utilization", True, [0.7, 0.8, 0.85, 0.9, 0.95],
        "KV capacity; interacts with tp (KV bytes per GPU)")

    # --- weight precision ------------------------------------------------------------------
    if row.get("quant") in (None, "", "bfloat16", "float16"):
        add("quantization", True, ["fp8", "modelopt_fp4", "mxfp4"],
            f"checkpoint is {row.get('quant') or 'unquantized'}, so the served format is a free "
            f"choice")
    else:
        add("quantization", False, [],
            f"the checkpoint is already {row['quant']}; the graph records that format, so "
            f"restating or changing it misprices the weights")

    # --- KV precision: graph table x architecture resolution x this chip -------------------
    priced = sorted(set(g["kv_dtypes_priced"]))
    here = [d for d, fams in (g.get("kv_dtype_families") or {}).items()
            if not fams or fam in fams]
    elsewhere = [d for d in priced if d not in here]
    gated = [r for r in ar["cache_dtype_resolutions"] if r["capability_family"] == fam]
    if gated:
        legal = sorted({r["resolves_to"] for r in gated})
        why = (f"{g['architecture']} rewrites auto/fp8 on capability family {fam} to "
               f"{', '.join(legal)}")
        tpd = [r["tp_divides"] for r in gated if r.get("tp_divides")]
        if tpd:
            why += (f"; the richest ({legal[-1] if len(legal) > 1 else legal[0]}) additionally "
                    f"needs tp divisible by {tpd[0]}")
        if here:
            why += f". The graph prices {', '.join(here)} on this chip"
        if elsewhere:
            why += (f", and {', '.join(elsewhere)} only on another capability family -- so that "
                    f"is not an axis here")
        add("cache_dtype", True, legal, why)
    elif here:
        add("cache_dtype", True, ["auto"] + here,
            f"the graph prices {', '.join(here)} on this chip (capability family {fam})"
            + (f"; {', '.join(elsewhere)} is priced only on another family, so it is not an "
               f"axis here" if elsewhere else ""))
    elif elsewhere:
        add("cache_dtype", False, [],
            f"the graph prices {', '.join(elsewhere)} only on another capability family, not "
            f"on this chip (family {fam}) -- a KV-precision sweep has nothing to vary here")
    else:
        add("cache_dtype", True, ["auto", "fp8"],
            "no per-dtype KV table in the graph, so the generic set applies")

    # --- expert / MoE knobs ----------------------------------------------------------------
    if moe:
        add("dp + enable_expert_parallel", True, ["dp: [1,2,4,8]", "enable_expert_parallel: [true,false]"],
            f"MoE with {row.get('experts')} experts (top_k {row.get('top_k')}); EP width is "
            f"tp x dp, so EP over width 1 shards nothing")
        add("all2all_backend", True,
            ["allgather_reducescatter", "deepep_high_throughput", "deepep_low_latency",
             "deepep_v2", "pplx"],
            "MoE dispatch; a real lever (33% measured) but ONLY with expert parallelism on")
        add("moe_backend", True, list(enums["moe_backend"])[:8] + ["..."],
            "expert kernel; measured inert on the reference part, so keep a control")
        e = row.get("experts")
        add("eplb", True, [f"num_redundant_experts making ({e} + r) divisible by tp*dp"],
            f"expert-load balancing; with EPLB on the engine ASSERTS that tp x dp divides "
            f"{e} + num_redundant_experts, so a bad value does not start")
    else:
        for k in ("dp + enable_expert_parallel", "all2all_backend", "moe_backend", "eplb"):
            add(k, False, [], f"{model} is DENSE: no experts to shard, dispatch or balance "
                              f"(BLIS refuses dp>1 on a dense model)")

    # --- drafting --------------------------------------------------------------------------
    if g.get("speculator"):
        depths = [k for k in (1, 2, 3, 4, 5, 6, 8) if k <= (g.get("max_spec_tokens") or 0)]
        why = (f"the graph declares speculator {g['speculator']!r} with num_spec_tokens "
               f"{g.get('max_spec_tokens')}, so K must not exceed it")
        if ar["refuses_mtp"]:
            why += f". {g['architecture']} REFUSES every MTP-family method, so only " \
                   f"{g['speculator']!r} is legal"
        add("speculative", True,
            [f"method: [{g['speculator']}]", f"num_spec_tokens: {depths or [1]}"], why)
    else:
        add("speculative", False, [],
            f"{model}'s graph declares NO speculator, so every drafting run is refused")

    # --- sliding window --------------------------------------------------------------------
    if g.get("sliding_window") and g.get("architecture") == "DeepseekV41ForCausalLM":
        add("swa_bounded_replay", True, [True, False],
            f"sliding window {g['sliding_window']}, and only DeepSeek-V4.1 reads this field")
    else:
        add("swa_bounded_replay", False, [],
            "only DeepSeek-V4.1 with a sliding window reads this field")

    # --- engram ----------------------------------------------------------------------------
    if ar["engram"]:
        add("engram_config", True, ["cpu_offload: [true,false]"],
            f"{g['architecture']} is an Engram architecture; the engine refuses the block on "
            f"models without Engram layers")
    else:
        add("engram_config", False, [],
            f"{g.get('architecture')} is not an Engram architecture; the engine refuses the "
            f"block at startup")

    # --- recurrent -------------------------------------------------------------------------
    kind = str(row.get("attn_kind") or "")
    recurrent = "mamba" in kind or "recurrent" in kind
    add("mamba_cache_* ", recurrent, ["auto", "float32", "float16", "bfloat16"] if recurrent else [],
        f"attn_kind is {kind!r}" + ("" if recurrent else ": no recurrent state to size"))

    # --- prefix reuse / routing ------------------------------------------------------------
    add("enable_prefix_caching", wl["prefix_reuse"], [True, False] if wl["prefix_reuse"] else [],
        ("a REPLAY re-sends a growing shared prefix, so the cache has something to hit"
         if wl["prefix_reuse"] else
         f"{workload} is a generated SHAPE: prompts are unique, so this measured "
         f"byte-identical"))
    add("router.endpoint_picker_config", True,
        ["random.yaml", "queue.yaml", "load-aware.yaml", "queue-kv.yaml",
         "prefix-affinity.yaml"],
        "needs >1 instance to mean anything; sharpens as instance count grows"
        + (" -- and prefix-affinity is a real lever on a replay" if wl["prefix_reuse"] else ""))

    # --- context ---------------------------------------------------------------------------
    ctx = row.get("context")
    add("max_model_len", True, [f"<= {ctx:,}" if ctx else "model context"],
        f"catalog context window is {ctx:,}" if ctx else "context window unknown")
    add("long_prefill_token_threshold", True, [256, 512, 1024, 2048, 4096],
        f"must stay <= max_model_len; competes with decode for a per-instance token budget")

    return {"model": model, "hardware": hardware, "workload": workload,
            "catalog_row": row, "graph": g, "hardware_facts": hw, "arch_rules": ar,
            "workload_facts": wl, "knobs": knobs}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Which knobs apply, and with what values, for one model x hardware x workload.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--catalog", default=os.path.join(SIBLINGS, "blis-catalog"))
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    p = profile(args.model, args.hardware, args.workload, args.catalog)
    if args.as_json:
        print(json.dumps(p, indent=2, sort_keys=True))
        return 0
    row, g, hw, wl = p["catalog_row"], p["graph"], p["hardware_facts"], p["workload_facts"]
    print(f"{args.model} on {args.hardware}, {args.workload}")
    print(f"  catalog : arch={row.get('arch')} experts={row.get('experts')} "
          f"top_k={row.get('top_k')} attn={row.get('attn_kind')} "
          f"context={row.get('context')} quant={row.get('quant')}")
    print(f"  graph   : architecture={g.get('architecture')} speculator={g.get('speculator')} "
          f"max_K={g.get('max_spec_tokens')} sliding_window={g.get('sliding_window')} "
          f"kv_dtypes_priced={g.get('kv_dtypes_priced') or '-'}")
    print(f"  hardware: cc={hw.get('compute_capability')} family={hw.get('capability_family')} "
          f"mem={hw.get('memory_gib')} GiB")
    print(f"  workload: {wl.get('kind')}"
          + (f" ({wl.get('scenario')}, category={wl.get('category')}, "
             f"thinking={wl.get('thinking')})" if wl.get("kind") == "replay" else ""))
    print(f"  rules   : {p['arch_rules']['source']}")
    print()
    yes = [k for k in p["knobs"] if k["applies"]]
    no = [k for k in p["knobs"] if not k["applies"]]
    print(f"APPLIES ({len(yes)}):")
    for k in yes:
        vals = ", ".join(str(v) for v in k["values"]) if k["values"] else ""
        print(f"  {k['knob']}")
        if vals:
            print(f"      values: {vals}")
        print(f"      why   : {k['why']}")
    print(f"\nDOES NOT APPLY ({len(no)}) -- sweeping these measures nothing:")
    for k in no:
        print(f"  {k['knob']}: {k['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
