#!/usr/bin/env python3
"""What knobs are worth sweeping for ONE model x hardware x workload, read from the sources.

    python3 -I scripts/knob_profile.py --model deepseek-v4.1-flash --hardware b200 \
        --workload agentx-mvp

Prints, per knob: whether it APPLIES to this combination, the values that are legal HERE, and
why. Exists because the facts that decide a grid live in several places, and none
of them is the catalog summary a designer tends to read:

  * the catalog's model row      -- arch, experts, top_k, attn_kind, context, quant
  * the model's graph.yaml       -- the declared speculator and its max depth, the KV
                                    bytes_by_cache_dtype table (which dtypes this model prices
                                    at all), sliding window
  * the model's model.yaml       -- its drafters: the graph's speculator and each separate
                                    drafter entry, which names the model in its targets
  * the registry's acceptance    -- which drafters have a measured profile for this workload
  * the hardware entry           -- ComputeCapability, which gates the per-dtype KV table

The engine's own architecture rules (which cache dtype `auto` resolves to, which draft methods an
architecture refuses, whether Engram applies) are blis-vllm's, and no simulator command publishes
them yet, so this script does not report them; the engine refuses a setting they forbid.

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
import blis_cli                           # noqa: E402
import run_blis                           # noqa: E402
import build_catalog                      # noqa: E402
import catalog_drafters                   # noqa: E402


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


def _entry(lookup, k):
    """One acceptance entry from `blis acceptance`'s JSON: the rung is the simulator's own step."""
    step = lookup.get("step")
    name, source = run_blis.public_set(lookup.get("set"), lookup.get("source"))
    return {"num_spec_tokens": k, "acceptance_length": lookup.get("acceptance_length"),
            "rung": step, "borrowed": step != "exact", "differed": lookup.get("differed") or [],
            "k_rule": lookup.get("k_rule"), "set": name, "source": source,
            "measured_on": lookup.get("measured_on")}


def drafter_facts(catalog, registry, model, wl, blis=None, hardware=None, engine="vllm"):
    """Each drafter of `model` with whether drafting applies on this workload, and why.

    With `blis` that has `blis acceptance`, a drafter applies when the simulator resolves an
    acceptance for it and the rung is the simulator's step (a borrowed one is flagged). Without it
    (TEMPORARY, inference-sim/inference-sim#2141) the checkout is read through catalog_drafters,
    which sees only exact-key profiles, and the rung is labelled exact_key: the key found, not the
    step the simulator would report.
    """
    category, thinking = wl.get("category"), wl.get("thinking")
    sel = (f"category {category}, thinking {thinking}" if category or thinking
           else "any category and thinking mode (this workload states none; a drafting run "
                "must state them)")
    out = []
    for d in catalog_drafters.drafters(catalog, model):
        row = {**d, "applies": False, "depths": [], "acceptance": []}
        if not d["linked"]:
            row["why"] = f"not a drafter of {model}: {d['why']}"
            out.append(row)
            continue
        found = catalog_drafters.exact_profiles(registry, model, d["draft_model"], d["method"],
                                                category, thinking)
        cap = d.get("max_spec_tokens") or 0
        found = [p for p in found if p["num_spec_tokens"] and (not cap or p["num_spec_tokens"] <= cap)]
        ks = sorted({p["num_spec_tokens"] for p in found}) or ([cap] if cap else [])
        note = ""
        if blis and category and thinking and ks:
            try:
                got = [(k, *blis_cli.acceptance(blis, model, d["draft_model"], d["method"], k, category,
                                                thinking, hardware, engine, catalog, registry))
                       for k in ks]
            except blis_cli.CommandMissing:
                note = ("; this simulator build has no `blis acceptance`, so the rung is only the key "
                        "found in the checkout and a borrowed profile is not seen")
            else:
                ok = [_entry(lk, k) for k, lk, _ in got if lk]
                if ok:
                    row.update(applies=True, depths=[e["num_spec_tokens"] for e in ok], acceptance=ok)
                    row["why"] = f"{d['method']} drafter; the simulator resolves, for {sel}: " + "; ".join(
                        f"K={e['num_spec_tokens']} AL {e['acceptance_length']} via {e['rung']}"
                        + (" (BORROWED" + (f"; differs in {', '.join(e['differed'])}" if e["differed"] else "")
                           + (f"; {e['k_rule']}" if e["k_rule"] else "") + ")" if e["borrowed"] else "")
                        + f" from {e['set']}" for e in ok)
                else:
                    row["why"] = (f"{d['method']} drafter, but the simulator resolves no acceptance for "
                                  f"{sel}: {got[0][2]}")
                out.append(row)
                continue
        if found:
            shown = [run_blis.public_set(p["set"], p.get("source")) for p in found]
            row.update(applies=True, depths=sorted({p["num_spec_tokens"] for p in found}),
                       acceptance=[{**p, "set": n, "source": src, "rung": "exact_key", "borrowed": False}
                                   for p, (n, src) in zip(found, shown)])
            sets = sorted({f"{n} ({src or 'no source recorded'})" for n, src in shown})
            row["why"] = (f"{d['method']} drafter; exact acceptance key at K {row['depths']} for {sel}, "
                          f"from {', '.join(sets)}{note}")
        else:
            row["why"] = (f"{d['method']} drafter, but no acceptance profile has its exact key at "
                          f"{sel}; the simulator's lookup may borrow one at run time (D8), which "
                          f"this script could not ask, so a drafting arm for it is unconfirmed here"
                          f"{note}")
        out.append(row)
    return out


def drafting_arms(drafts):
    """One arm per usable drafter: its method, checkpoint and depths kept together.

    The in-graph speculator names no draft_model. A sweep crosses its lists, so each arm is its own sweep.
    """
    arms = []
    for d in drafts:
        if not d["applies"]:
            continue
        spec = {"method": [d["method"]]}
        if d["draft_model"]:
            spec["draft_model"] = [d["draft_model"]]
        spec["num_spec_tokens"] = list(d["depths"])
        arms.append({"drafter": d["name"], "speculative": spec})
    return arms


def speculative_knob(drafts, model=None):
    """The `speculative` knob row: the per-drafter arms when any applies, else why not."""
    why = "; ".join(f"{d['name']}: {d['why']}" for d in drafts)
    arms = drafting_arms(drafts)
    if arms:
        by = {d["name"]: d for d in drafts}
        values = []
        for a in arms:
            sp = a["speculative"]
            rungs = sorted({e["rung"] for e in by[a["drafter"]]["acceptance"]})
            values.append(f"{a['drafter']}: method: {sp['method']}"
                          + (f", draft_model: {sp['draft_model']}" if "draft_model" in sp
                             else " (in-graph: no draft_model)")
                          + f", num_spec_tokens: {sp['num_spec_tokens']}; acceptance via {', '.join(rungs)}")
        values += ["keep num_spec_tokens 0 as the control, once per sweep",
                   "one sweep per drafter: a sweep crosses its lists, so a checkpoint, its method and its "
                   "depths belong together"]
        return {"knob": "speculative", "applies": True, "values": values, "why": why, "arms": arms}
    if drafts:
        return {"knob": "speculative", "applies": False, "values": [], "why": why, "arms": []}
    return {"knob": "speculative", "applies": False, "values": [], "arms": [],
            "why": f"{model} has no drafter: its graph declares no speculator and no catalog entry "
                   f"drafts for it, so every drafting run is refused"}


def profile(model, hardware, workload, catalog, registry=None, blis=None, engine="vllm"):
    """The per-knob verdict for this combination: applies, legal values here, and why."""
    cat = build_catalog.build(os.path.join(catalog, os.pardir))
    row = next((m for m in cat.get("models", []) if m.get("name") == model), None)
    if row is None:
        raise SystemExit(f"{model!r} is not in the catalog at {catalog}")
    g = graph_facts(catalog, model)
    hw = hardware_facts(catalog, hardware)
    wl = workload_facts(catalog, workload)
    moe = row.get("arch") == "MoE"
    fam = hw.get("capability_family")

    knobs = []

    def add(knob, applies, values, why):
        knobs.append({"knob": knob, "applies": applies, "values": values, "why": why})

    # --- always applicable -----------------------------------------------------------------
    add("cudagraph_mode", True, ["NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE"],
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
    if here:
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
        add("moe_backend", True, [],
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
    # One arm per drafter: the graph's own speculator and each separate drafter entry, applicable
    # when the simulator resolves an acceptance at this workload's selectors (D8).
    drafts = drafter_facts(catalog, registry, model, wl, blis, hardware, engine)
    knobs.append(speculative_knob(drafts, model))

    # --- sliding window --------------------------------------------------------------------
    if g.get("sliding_window") and g.get("architecture") == "DeepseekV41ForCausalLM":
        add("swa_bounded_replay", True, [True, False],
            f"sliding window {g['sliding_window']}, and only DeepSeek-V4.1 reads this field")
    else:
        add("swa_bounded_replay", False, [],
            "only DeepSeek-V4.1 with a sliding window reads this field")

    # --- engram ----------------------------------------------------------------------------

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
            "catalog_row": row, "graph": g, "hardware_facts": hw,
            "workload_facts": wl, "drafters": drafts, "knobs": knobs}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Which knobs apply, and with what values, for one model x hardware x workload.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--catalog", default=os.path.join(SIBLINGS, "blis-substrate", "catalog"))
    ap.add_argument("--registry", default=None,
                    help="the registry/ next to --catalog (default: the catalog's sibling registry/)")
    ap.add_argument("--blis", default=os.environ.get("BLIS"),
                    help="the simulator binary: asked `blis acceptance` for each drafter's rung "
                         "(default $BLIS; without one the rung is only the key found in the checkout)")
    ap.add_argument("--engine", default="vllm", help="the engine the acceptance is looked up for")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    registry = args.registry or os.path.join(args.catalog, os.pardir, "registry")
    p = profile(args.model, args.hardware, args.workload, args.catalog, registry, args.blis, args.engine)
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
    for d in p["drafters"]:
        print(f"  drafter : {d['name']} ({d['kind']}, {d.get('method')}): "
              f"{'drafting applies' if d['applies'] else 'drafting not confirmed'}")
        for a in d["acceptance"]:
            print(f"      K={a['num_spec_tokens']} AL={a['acceptance_length']} rung={a['rung']}"
                  f"{' BORROWED' if a.get('borrowed') else ''} set={a['set']} "
                  f"source={a.get('source') or '-'}")
            if a.get("differed") or a.get("k_rule"):
                print(f"          differs in: {', '.join(a.get('differed') or []) or '-'}"
                      f"{'; ' + a['k_rule'] if a.get('k_rule') else ''}")
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
