#!/usr/bin/env python3
"""Run an Experiment Spec deterministically -> reports/<name>/. NO LLM anywhere in this path.

    python3 -I scripts/run_experiment.py experiments/foo.yaml \
        --blis .build/blis --catalog ../blis-catalog --registry ../blis-registry

    python3 -I scripts/run_experiment.py experiments/foo.yaml --dry-run   # expand only, no BLIS

This is the whole of RFC #21's second half: the LLM designs the spec, and everything after it
is mechanical. Given one spec this script

  1. expands each sweep's grid into candidate configs (a product of the knob lists),
  2. drops combos that do not use EXACTLY `scenario.gpus` -- so every comparison is like-for-like,
  3. writes one EndpointPickerConfig per router-override combination,
  4. runs BLIS at every `scenario.concurrency` level with a fixed seed and the pinned stack,
  5. picks the best config per sweep and overall (most output tok/s per GPU that holds the SLO),
  6. writes reports/<name>/ in the run-directory format report.py and build_gallery.py read.

DETERMINISM IS THE POINT, and it is stronger here than in the agentic search: the grid is a pure
function of the spec, the filter is arithmetic, the order is sorted, and the winner is chosen by
an executable rule rather than a judgement. The same spec and the same pins give the same
report.json -- which is why CI can run this on a merge and publish the result unattended. The
one thing this does NOT do is adapt: load is the spec's fixed concurrency list, with no capacity
probe and no open-loop rate (both out of scope per #21).

Verified: two sweeps of one spec into DIFFERENT output directories produce byte-identical
experiment.json, pareto.json, scenario.json, report.json and spec.yaml, and identical `metrics`
on every ledger row. In runs.jsonl exactly one field is expected to differ -- `wall_s`, how long
that run took -- which is provenance about the machine, not a result. Recorded commands are
stored relative to the run directory (_relativize_cmd) so the ledger does not depend on where
the sweep ran.
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
import sys
import time

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SKILL = os.path.join(ROOT, ".claude", "skills", "config-search", "scripts")
sys.path.insert(0, SKILL)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_blis            # noqa: E402
import topologies          # noqa: E402
import build_catalog       # noqa: E402
import validate_experiment as validator   # noqa: E402

# Knob names in a spec are the SCHEMA's (blis-vllm / blis-schemas) names; run_blis's candidate
# keys are flat. These are the only two that differ, and they differ because run_blis folds
# drafting into a nested `speculative` block itself.
SPEC_TO_CANDIDATE = {"num_spec_tokens": "num_speculative_tokens", "method": "speculative_method"}
# Nested engine blocks that stay nested (unlike `speculative`, which flattens): the runner
# renders them as sub-blocks of the scenario's engine. Sourced from the runner so the two cannot
# disagree about which keys are blocks.
NESTED_ENGINE_BLOCKS = tuple(k for k in run_blis.NESTED_ENGINE_FIELDS if k != "speculative")


def _flag(role, knob):
    """A per-role spec knob as its run_blis candidate key: ("prefill", "tp") -> "prefill_tp"."""
    return f"{role}_{knob}"


def grid(knob_lists):
    """Every combination of a {knob: [values]} mapping, in sorted-key order.

    Sorted keys and list order give a stable product, so run N of a re-run is the same config as
    run N of the first run -- the property the whole pipeline's reproducibility rests on.
    """
    keys = sorted(knob_lists)
    if not keys:
        return [{}]
    pools = [list(knob_lists[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*pools)]


def _engine_candidates(engine):
    """Expand a sweep's `engine:` block, including its nested blocks.

    `speculative` FLATTENS, because run_blis takes drafting as two flat candidate keys and folds
    them back into the nested block itself:
    `speculative: {method: [dspark], num_spec_tokens: [1,3,5]}` -> three candidates carrying
    speculative_method + num_speculative_tokens.

    The other nested blocks (`eplb`, `dbo`, `engram_config`) stay NESTED: the runner passes them
    through to the scenario as sub-blocks, so each combination of their inner fields becomes one
    candidate whose value is a dict -- e.g. `eplb: {enable_eplb: [true], num_redundant_experts:
    [0, 32]}` gives two candidates, `{"eplb": {"enable_eplb": True, "num_redundant_experts": 0}}`
    and the same with 32.
    """
    engine = dict(engine or {})
    spec = engine.pop("speculative", None)
    nested = {k: engine.pop(k) for k in list(engine)
              if k in NESTED_ENGINE_BLOCKS and isinstance(engine[k], dict)}
    out = grid(engine)
    for block, fields in sorted(nested.items()):
        combos = grid(fields)
        out = [{**base, block: combo} for base in out for combo in combos]
    if spec:
        spec_combos = grid({SPEC_TO_CANDIDATE.get(k, k): v for k, v in spec.items()})
        out = [{**b, **s} for b in out for s in spec_combos]
    return out


def expand_sweep(sweep, gpus):
    """(kept, dropped) candidate configs for one sweep.

    `kept` are configs whose GPU total is exactly `gpus`; `dropped` records every rejected combo
    with the arithmetic that rejected it, so a report can state what the grid contained rather
    than only what survived (the same reason pareto.json carries `dropped[]`).
    """
    router_variants = _router_variants(sweep.get("router") or {})
    engine_combos = _engine_candidates(sweep.get("engine"))
    kept, dropped = [], []
    if _is_disaggregated(sweep):
        pf_par = ((sweep.get("prefill") or {}).get("parallel")) or {}
        dc_par = ((sweep.get("decode") or {}).get("parallel")) or {}
        pf_eng = ((sweep.get("prefill") or {}).get("engine")) or {}
        dc_eng = ((sweep.get("decode") or {}).get("engine")) or {}
        for pf in grid(pf_par):
            for dc in grid(dc_par):
                for pi in sweep.get("prefill-instances") or [1]:
                    for di in sweep.get("decode-instances") or [1]:
                        total = (_footprint(pf) * pi) + (_footprint(dc) * di)
                        base = {_flag("prefill", k): v for k, v in pf.items()}
                        base.update({_flag("decode", k): v for k, v in dc.items()})
                        base["prefill_instances"] = pi
                        base["decode_instances"] = di
                        for pfe in grid(pf_eng):
                            for dce in grid(dc_eng):
                                cfg = dict(base)
                                cfg.update({_flag("prefill", k): v for k, v in pfe.items()})
                                cfg.update({_flag("decode", k): v for k, v in dce.items()})
                                for eng in engine_combos:
                                    for router in router_variants:
                                        cand = {**cfg, **eng, **router}
                                        if total == gpus:
                                            kept.append(cand)
                                        else:
                                            dropped.append((cand, total))
    else:
        for par in grid(sweep.get("parallel") or {}):
            for ni in sweep.get("num-instances") or [1]:
                total = _footprint(par) * ni
                for eng in engine_combos:
                    for router in router_variants:
                        cand = {**par, "num_instances": ni, **eng, **router}
                        if total == gpus:
                            kept.append(cand)
                        else:
                            dropped.append((cand, total))
    return kept, dropped


def _footprint(parallel):
    return (int(parallel.get("tp", 1)) * int(parallel.get("pp", 1))
            * int(parallel.get("dp", 1)))


def _is_disaggregated(sweep):
    return bool({"prefill-instances", "decode-instances", "prefill", "decode"} & set(sweep))


# --- router overrides --------------------------------------------------------------------------


def _router_variants(router):
    """Every router configuration a sweep asks for, as candidate fragments.

    With no overrides this is one fragment per listed base file. With overrides it is one per
    (base x override combination), and the generated file is written later by
    materialize_router -- so the expansion stays a pure function and nothing touches the disk
    until a run is actually about to happen.
    """
    bases = router.get("endpoint_picker_config") or []
    if not bases:
        if not (router.get("schedulingProfiles") or router.get("plugins")):
            return [{}]                      # the llm-d default, which is a real arm
        return [{}]
    overrides = _override_combos(router)
    out = []
    for base in bases:
        for ov in overrides:
            frag = {"router_config": base}
            if ov:
                frag["router_overrides"] = ov
            out.append(frag)
    return out


def _override_combos(router):
    """Every combination of the router override trees, flattened to dotted paths.

    `schedulingProfiles.default.queue-scorer.weight: [1,2,4]` becomes the path
    ("schedulingProfiles", "default", "queue-scorer", "weight") with values 1, 2 and 4, so the
    product over all such paths is the arm set. Returns [{}] when there are no overrides.
    """
    paths = {}
    for profile, refs in (router.get("schedulingProfiles") or {}).items():
        for ref, fields in (refs or {}).items():
            for field, values in (fields or {}).items():
                paths[("schedulingProfiles", profile, ref, field)] = list(values)
    for plugin, block in (router.get("plugins") or {}).items():
        for field, values in ((block or {}).get("parameters") or {}).items():
            paths[("plugins", plugin, "parameters", field)] = list(values)
    if not paths:
        return [{}]
    keys = sorted(paths)
    return [dict(zip(keys, combo)) for combo in itertools.product(*[paths[k] for k in keys])]


def apply_overrides(base_doc, overrides):
    """A copy of `base_doc` with each dotted override applied. Raises KeyError on a field the
    base file does not already have.

    Overrides only CHANGE existing fields (RFC #21's stated limitation): adding or removing a
    plugin takes a separate base file. Enforcing that here means a typo'd plugin name fails
    before the run instead of silently producing the base config unchanged -- which would show
    up as a sweep of identical results and be read as "this knob does nothing".
    """
    import copy
    doc = copy.deepcopy(base_doc)
    for path, value in sorted(overrides.items()):
        section = path[0]
        if section == "schedulingProfiles":
            _, profile, ref, field = path
            profiles = doc.get("schedulingProfiles") or []
            target = next((p for p in profiles if p.get("name") == profile), None)
            if target is None:
                raise KeyError(f"schedulingProfiles has no profile named {profile!r} "
                               f"(has: {[p.get('name') for p in profiles]})")
            entry = next((p for p in (target.get("plugins") or [])
                          if p.get("pluginRef") == ref), None)
            if entry is None:
                raise KeyError(f"profile {profile!r} does not reference plugin {ref!r} "
                               f"(references: "
                               f"{[p.get('pluginRef') for p in target.get('plugins') or []]})")
            entry[field] = value
        else:
            _, plugin, _params, field = path
            entry = next((p for p in (doc.get("plugins") or []) if p.get("type") == plugin), None)
            if entry is None:
                raise KeyError(f"the base config declares no plugin of type {plugin!r} "
                               f"(declares: {[p.get('type') for p in doc.get('plugins') or []]})")
            params = entry.setdefault("parameters", {})
            if field not in params:
                raise KeyError(f"plugin {plugin!r} has no parameter {field!r} in the base config "
                               f"(has: {sorted(params)}); an override may only change a field the "
                               f"base file already sets")
            params[field] = value
    return doc


def override_phrase(overrides):
    """A readable description of one override combination, for a config's variant label."""
    parts = []
    for path, value in sorted(overrides.items()):
        if path[0] == "schedulingProfiles":
            parts.append(f"{path[2]} {path[3]} {value}")
        else:
            parts.append(f"{path[1]} {path[3]} {value}")
    return ", ".join(parts)


def materialize_router(cand, routing_dir, out_dir):
    """Resolve a candidate's `router_config` to a real file path, writing an override file if any.

    Returns (config_without_overrides, description). A generated file is named by the HASH of its
    contents, so the same override combination always lands on the same path and a re-run
    rewrites identical bytes -- the files are part of the reproducible output, not scratch.
    """
    cand = dict(cand)
    overrides = cand.pop("router_overrides", None)
    rel = cand.get("router_config")
    if not rel:
        return cand, ""
    base_path = rel if os.path.isabs(rel) else os.path.join(routing_dir, rel)
    if not overrides:
        cand["router_config"] = base_path
        return cand, topologies.routing_arm_name(rel)
    with open(base_path, encoding="utf-8") as f:
        base_doc = yaml.safe_load(f) or {}
    doc = apply_overrides(base_doc, overrides)
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    digest = hashlib.sha1(body.encode()).hexdigest()[:10]
    stem = topologies.routing_arm_name(rel)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}-{digest}.yaml")
    if not os.path.exists(path):
        header = (f"# GENERATED by scripts/run_experiment.py from routing/{rel}.\n"
                  f"# Overrides: {override_phrase(overrides)}\n"
                  f"# Named by content hash, so the same overrides always produce this file.\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + body)
    cand["router_config"] = path
    return cand, f"{stem} ({override_phrase(overrides)})"


# --- labelling ---------------------------------------------------------------------------------


def label_of(cand):
    """The topology label for a candidate -- what a reader scans down a column.

    Colocated reuses topologies.label (`8x TP4xDP2 + EP`); a disaggregated candidate names both
    pools (`3P/5D TP8 + EP`), because instances alone would not say which pool they are.
    """
    if cand.get("prefill_instances") is not None:
        pf, dc = cand["prefill_instances"], cand["decode_instances"]
        tp, dp = cand.get("prefill_tp", 1), cand.get("prefill_dp", 1)
        shape = f"TP{tp}" + (f"xDP{dp}" if dp and dp > 1 else "")
        ep = " + EP" if cand.get("prefill_enable_expert_parallel") else ""
        return f"{pf}P/{dc}D {shape}{ep}"
    return topologies.label({"tp": cand.get("tp", 1), "dp": cand.get("dp", 1),
                             "num_instances": cand.get("num_instances", 1),
                             "enable_expert_parallel": cand.get("enable_expert_parallel")})


# Knobs described in a config's `variant` rather than its label: the engine and routing settings
# that distinguish two runs of the SAME topology. Mirrors topologies.KNOB_NAMES' purpose.
#
# This MUST cover every engine knob a spec can sweep. A swept knob missing from here leaves every
# arm of its sweep labelled "(baseline)", so the report cannot tell them apart -- which is how a
# cudagraph_mode sweep spanning 147 to 263 tok/s/GPU first rendered as four identical rows. The
# list is checked against run_blis.SCENARIO_ENGINE by test_experiment.py so it cannot fall behind
# the runner again.
VARIANT_KEYS = ("num_speculative_tokens", "speculative_method", "router_config",
                # batching / admission
                "max_num_seqs", "max_num_batched_tokens", "long_prefill_token_threshold",
                "scheduling_policy", "async_scheduling",
                # memory / precision
                "gpu_memory_utilization", "cache_dtype", "quantization", "block_size",
                "max_model_len",
                # graph capture
                "cudagraph_mode", "max_cudagraph_capture_size", "cudagraph_capture_sizes",
                # attention / prefix reuse
                "enable_prefix_caching", "disable_cascade_attn", "swa_bounded_replay",
                # MoE dispatch + collectives
                "all2all_backend", "moe_backend", "allreduce_backend",
                "disable_custom_all_reduce",
                # context parallelism
                "dcp_comm_backend", "dcp_q_replicate", "cp_kv_cache_interleave_size",
                # recurrent state caches
                "mamba_cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode",
                # nested blocks (dict-valued; described by _nested_phrase)
                "eplb", "dbo", "engram_config")


def variant_of(cand, router_desc, baseline):
    """How this candidate differs from its sweep's baseline, spelled out in words.

    Measured against the sweep's most common setting per knob rather than against the vLLM
    default, so in a sweep where every arm sets fp8 the variant says what actually varies.
    """
    parts = []
    for key in VARIANT_KEYS:
        if key not in cand:
            continue
        if baseline.get(key) == cand[key]:
            continue
        if key == "router_config":
            parts.append(f"{router_desc} routing")
        elif key == "speculative_method":
            continue                        # named by the draft-token phrase below
        elif isinstance(cand[key], dict):
            parts.append(_nested_phrase(key, cand[key], baseline.get(key) or {}))
        else:
            parts.append(topologies.knob_phrase(key, cand[key]))
    return " · ".join(parts)


def _nested_phrase(block, value, baseline):
    """Describe a nested engine block by the inner fields that DIFFER from the baseline.

    Printing the whole dict would put `{'enable_eplb': True, 'num_redundant_experts': 32}` in a
    label; what distinguishes the arms is usually one inner field, so name that one.
    """
    changed = {k: v for k, v in value.items() if baseline.get(k) != v}
    inner = changed or value
    name = topologies.KNOB_NAMES.get(block, block)
    bits = []
    for k, v in sorted(inner.items()):
        label = topologies.KNOB_NAMES.get(k, k.replace("_", " "))
        if isinstance(v, bool):
            # A block's own enable flag carries the block's name, so "dual-batch overlap:
            # dual-batch overlap off" would say it twice -- just state the block and the state.
            bits.append(f"{'on' if v else 'off'}" if label == name
                        else f"{label} {'on' if v else 'off'}")
        else:
            bits.append(f"{label} {v:,}" if isinstance(v, int) else f"{label} {v}")
    return f"{name} {', '.join(bits)}" if bits else name


def sweep_baseline(cands):
    """The most common value of each variant knob across a sweep -- its implicit baseline."""
    base = {}
    for key in VARIANT_KEYS:
        values = [c[key] for c in cands if key in c]
        if not values:
            continue
        counts = collections.Counter(map(json.dumps, values))
        base[key] = json.loads(counts.most_common(1)[0][0])
    return base


# --- running -----------------------------------------------------------------------------------


def slo_limits(slo):
    """The spec's SLO as milliseconds: {metric: cap_ms}. Durations carry their unit (see #21)."""
    out = {}
    for key, raw in (slo or {}).items():
        ms = validator.parse_duration_ms(raw)
        if ms is not None:
            out[key] = ms
    return out


def floor_rate_from(slo_ms):
    """The per-user token rate an `itl_p99` cap implies: 20 ms/token == 50 tok/s/user.

    The spec states the cap, the ranking needs the rate, and they must be the same constraint
    expressed two ways or a config could pass one and fail the other.
    """
    itl = (slo_ms or {}).get("itl_p99")
    return (1000.0 / itl) if itl else None


def servable(rec, slo_ms):
    """(ok, why_not) for one run against the spec's SLOs.

    Uses run_blis.per_user_rate / worst_itl_p99 -- the acceptance-aware, worst-instance basis the
    report displays -- so a config the sweeper rejects is never one the report shows as passing.
    A declared SLO whose metric is missing counts as a FAILURE, not a silent pass.
    """
    if rec.get("returncode") != 0:
        return False, rec.get("error") or f"rc={rec.get('returncode')}"
    if rec.get("error"):
        return False, rec["error"]
    m = rec.get("metrics") or {}
    if not m.get("tokens_per_sec"):
        return False, "no throughput reported"
    itl_cap = (slo_ms or {}).get("itl_p99")
    if itl_cap:
        worst, missing = run_blis.worst_itl_p99(rec)
        if missing or worst is None:
            return False, "itl_p99 SLO declared but not measured"
        al = run_blis.run_acceptance(rec) or 1.0
        # Per-TOKEN latency: under drafting BLIS logs ITL per verification step, and a step emits
        # AL tokens, so the step interval must be divided by AL to compare against a per-token cap.
        if (worst / al) > itl_cap:
            return False, f"itl_p99 {worst / al:.1f} ms > {itl_cap:g} ms"
    for key, field in (("ttft_p99", "ttft_p99_ms"), ("e2e_p99", "e2e_p99_ms")):
        cap = (slo_ms or {}).get(key)
        if not cap:
            continue
        value = m.get(field)
        if value is None:
            return False, f"{key} SLO declared but not measured"
        if value > cap:
            return False, f"{key} {value:.1f} ms > {cap:g} ms"
    return True, None


def run_row(rec, cand, label, variant, sweep_name, concurrency, gpus):
    """One ledger row's derived fields, on the same bases report.py uses."""
    m = rec.get("metrics") or {}
    tps = m.get("tokens_per_sec")
    pu, _basis, _itl = run_blis.per_user_rate(rec)
    worst, missing = run_blis.worst_itl_p99(rec)
    return {
        "sweep": sweep_name, "label": label, "variant": variant,
        "concurrency": concurrency, "gpus": gpus,
        "tokens_per_sec": round(tps, 1) if tps else None,
        "tokens_per_sec_per_gpu": round(tps / gpus, 2) if tps and gpus else None,
        "tok_per_s_per_user": round(pu, 1) if pu is not None else None,
        "itl_p99_ms": None if missing else (round(worst, 3) if worst is not None else None),
        "itl_p99_ms_aggregate": m.get("itl_p99_ms"),
        "ttft_p99_ms": m.get("ttft_p99_ms"),
        "preemption_count": m.get("preemption_count", 0),
    }


def best_of(rows):
    """The best row: most tok/s per GPU, then most fleet tok/s, then the lowest concurrency.

    An EXECUTABLE rule, not a judgement -- which is what makes the winner reproducible. The
    objective is the lexicon's (tok/s per GPU) so the sweeper, the report and the dashboard all
    agree on what "best" means. The concurrency tiebreak prefers the lighter load that achieves
    the same efficiency, since it leaves more headroom.
    """
    servable_rows = [r for r in rows if r.get("servable")]
    if not servable_rows:
        return None
    return sorted(servable_rows,
                  key=lambda r: (-(r.get("tokens_per_sec_per_gpu") or 0),
                                 -(r.get("tokens_per_sec") or 0),
                                 r.get("concurrency") or 0,
                                 r.get("label") or "", r.get("variant") or ""))[0]


def run_experiment(spec, blis, catalog, registry, out_root, routing_dir=None, dry_run=False,
                   run_timeout=600, max_parallel=1, verbose=True):
    """Expand, filter, run and rank one spec. Returns the assembled report dict."""
    routing_dir = routing_dir or os.path.join(ROOT, "routing")
    sc = spec["scenario"]
    name = spec["name"]
    gpus = int(sc["gpus"])
    seed = int(sc.get("seed", 42))
    model, hardware = sc["model"], sc["hardware"]
    workload = sc["workload"]
    concurrency = list(sc["concurrency"])
    slo_ms = slo_limits(sc.get("slo"))
    shared_engine = dict(sc.get("engine") or {})
    replay = dict(sc.get("replay") or {}) or None

    out_dir = os.path.join(out_root, name)
    scenarios_dir = os.path.join(out_dir, "scenarios")
    generated_routing = os.path.join(out_dir, "routing")
    if not dry_run:
        os.makedirs(scenarios_dir, exist_ok=True)

    entry = _model_entry(model, catalog) if catalog else None
    # A REPLAY workload is fully described by its catalog descriptor -- the harness scenario, the
    # corpus digest, the sending window, the grace period and the context filter are all stated
    # in blis-catalog/workloads/<name>.yaml. So a spec states NONE of them: naming the workload
    # is naming all of it. See resolve_replay.
    replay = resolve_replay(workload, catalog, replay, dry_run=dry_run)

    # DRAFTING ON A REPLAY needs the acceptance selectors, and they are the WORKLOAD's, not a
    # knob: BLIS resolves acceptance from the registry by (model, method, K, thinking, category)
    # and REFUSES a drafting run whose acceptance does not resolve --
    #   "the workload states no thinking mode, which selects the profile; state one"
    # Without this every drafting run in a replay spec failed while the rest of the spec ran, so
    # the report showed an EMPTY drafting sweep rather than an error. The values are read from
    # the catalog's workload entry (agentx-mvp: category coding, thinking on), never invented.
    accept_sel = {}
    if catalog:
        wl = _workload_entry(workload, catalog)
        if wl:
            if wl.get("category"):
                accept_sel["speculative_category"] = wl["category"]
            if wl.get("thinking"):
                accept_sel["speculative_thinking"] = str(wl["thinking"])

    runs, sweep_results, all_rows = [], [], []
    for sweep in spec["sweeps"]:
        sname = sweep["name"]
        kept, dropped = expand_sweep(sweep, gpus)
        baseline = sweep_baseline(kept)
        if verbose:
            print(f"[{name}/{sname}] {len(kept)} configs x {len(concurrency)} loads "
                  f"= {len(kept) * len(concurrency)} runs "
                  f"({len(dropped)} combos dropped off budget)")
        rows, sweep_dropped = [], []
        for tot in sorted({t for _c, t in dropped}):
            sweep_dropped.append({"gpus": tot, "reason":
                                  f"uses {tot} GPUs, not the scenario's {gpus}",
                                  "count": sum(1 for _c, t in dropped if t == tot)})
        # Expand this sweep into a flat work list FIRST, then execute it. The two phases are
        # separate so the output order is the expansion order -- never the order runs happened
        # to finish -- which is what keeps a report reproducible under --max-parallel > 1.
        work = []                               # (idx, cfg, label, variant, c)
        for cand in kept:
            try:
                resolved, router_desc = materialize_router(cand, routing_dir, generated_routing) \
                    if not dry_run else (dict(cand), "")

                if dry_run:
                    resolved.pop("router_overrides", None)
            except (KeyError, OSError, yaml.YAMLError) as e:
                sweep_dropped.append({"config": _jsonable(cand),
                                      "reason": f"router override failed: {e}"})
                continue
            label = label_of(resolved)
            variant = variant_of(cand, router_desc, baseline)
            for c in concurrency:
                cfg = {**shared_engine, **resolved, "concurrency": c}
                # Only a drafting candidate needs them; adding them elsewhere would change a
                # non-drafting run's command line for no reason.
                if cfg.get("speculative_method"):
                    cfg = {**accept_sel, **cfg}
                if dry_run:
                    rows.append({"sweep": sname, "label": label, "variant": variant,
                                 "concurrency": c, "gpus": gpus, "config": _jsonable(cfg),
                                 "servable": None})
                    continue
                work.append((len(work), cfg, label, variant, c))

        def _execute(item):
            """One BLIS run. Pure w.r.t. shared state: run_blis.write_scenario publishes its
            scenario atomically under a content-hashed name (temp + os.replace), so workers
            racing on the same deployment cannot read a half-written file, and the caller owns
            all runs.jsonl I/O."""
            idx, cfg, label, variant, c = item
            nreq = None if replay else max(80, c * 4)
            # `_overrides` is provenance for the report (which catalog settings the spec
            # replaced), not a replay field -- keep it out of the command.
            rec = run_blis.run_one(blis, catalog, registry, scenarios_dir, model, hardware,
                                   sc.get("engine_version", "0.29.0"), workload, nreq, seed,
                                   None, cfg, phase=sname, timeout=run_timeout,
                                   replay={k: v for k, v in replay.items()
                                           if k != "_overrides"} if replay else None)
            rec["sweep"] = sname
            _relativize_cmd(rec, out_dir)
            ok, why = servable(rec, slo_ms)
            row = run_row(rec, cfg, label, variant, sname, c, gpus)
            # The resolved config travels with the row so pareto.json's finalists can carry
            # it: build_gallery reads `finalist.config.tp` / `.num_instances` for the winner
            # card, and showed "tp 0, 0 instances" without it.
            row["config"] = _jsonable(cfg)
            row["servable"] = ok
            if why:
                row["not_servable_because"] = why
            return idx, rec, row

        if work:
            if max_parallel <= 1:
                done = [_execute(it) for it in work]
            else:
                # One BLIS run is single-threaded (a discrete-event loop measured at ~1.1 cores),
                # so the only way to use more cores is more processes. A thread pool is enough:
                # each worker just waits on a subprocess.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
                    # as_completed order would be completion order, which is nondeterministic;
                    # ex.map yields in SUBMISSION order by contract, which is what we want.
                    done = list(ex.map(_execute, work))
            # Belt and braces: ex.map already yields in submission order, but the ledger's
            # reproducibility is the whole point of the deterministic sweeper, so the order is
            # also asserted here rather than inherited from an executor implementation detail.
            done.sort(key=lambda t: t[0])
            for _idx, rec, row in done:
                runs.append(rec)
                rows.append(row)
        all_rows.extend(rows)
        sweep_results.append({
            "name": sname,
            "narrative": (sweep.get("narrative") or "").strip(),
            "configs": len(kept),
            "runs": len(kept) * len(concurrency),
            "dropped": sweep_dropped,
            "best": best_of(rows),
            "rows": rows,
        })

    report = {
        "name": name,
        "description": (spec.get("description") or "").strip(),
        "scenario": {"model": model, "hardware": hardware, "workload": workload, "gpus": gpus,
                     "concurrency": concurrency, "slo": sc.get("slo") or {}, "seed": seed,
                     "engine": shared_engine or None, "replay": replay},
        "slo_ms": slo_ms,
        "interactivity_floor": {"tok_per_s_per_user": floor_rate_from(slo_ms)},
        "objective": "max output tok/s per GPU subject to the scenario's SLOs",
        "sweeps": sweep_results,
        "best_overall": best_of(all_rows),
        "pinned_stack": _pinned_stack(blis, catalog, registry),
        "model_arch": (entry or {}).get("arch"),
        "totals": {"configs": sum(s["configs"] for s in sweep_results),
                   "runs": len(runs) if not dry_run else
                           sum(s["runs"] for s in sweep_results),
                   "servable": sum(1 for r in all_rows if r.get("servable"))},
    }
    if not dry_run:
        _write_run_dir(out_dir, spec, report, runs)
    return report


def _relativize_cmd(rec, out_dir):
    """Rewrite absolute paths in a row's recorded `cmd` as paths relative to the run directory.

    The ledger must not depend on WHERE the sweep ran. Everything else in a report already
    satisfies that (report.py identifies a run by the id inside its data, never by the directory
    path), but `cmd` embedded the absolute --scenarios / --registry / router paths, so running
    one spec twice in two output directories produced ledgers that differed in those strings
    while every measured number was identical. A re-run is supposed to be byte-identical, so a
    diff should mean a changed result -- not a changed working directory.

    The command stays readable and still states exactly which flags and values BLIS received.
    """
    cmd = rec.get("cmd")
    if not cmd:
        return
    bases = [(os.path.abspath(out_dir), "."), (ROOT, ".")]
    out = []
    for arg in cmd:
        text = str(arg)
        for base, prefix in bases:
            if text.startswith(base + os.sep):
                text = os.path.join(prefix, os.path.relpath(text, base))
                break
        out.append(text)
    rec["cmd"] = out


def _jsonable(cfg):
    """A config with paths made relative, so a dry run's output does not embed the checkout path.

    A DISAGGREGATED config also gets flat `tp`/`num_instances` keys derived from its pools. The
    gallery's winner card reads those two names directly, and a P/D config has neither (its
    layout lives under prefill_*/decode_*), so the card would read "tp 0, 0 instances" for a
    disaggregated winner. The flat values describe the prefill pool and the total instance
    count, which is what the card's one-line summary is for; the authoritative per-pool layout
    stays in the same dict beside them.
    """
    out = {}
    for k, v in cfg.items():
        if k == "router_config" and isinstance(v, str):
            out[k] = os.path.relpath(v, ROOT) if os.path.isabs(v) else v
        else:
            out[k] = v
    if out.get("prefill_instances") is not None and "tp" not in out:
        out["tp"] = out.get("prefill_tp", 1)
        out["num_instances"] = int(out["prefill_instances"]) + int(out.get("decode_instances", 0))
    return out


def _model_entry(model, catalog_root):
    """The catalog row for `model` (supplies `arch`), or None."""
    try:
        cat = build_catalog.build(os.path.join(catalog_root, os.pardir))
    except Exception:
        return None
    return next((m for m in cat.get("models", []) if m.get("name") == model), None)


def replay_descriptor(workload, catalog):
    """The catalog's `replay:` block for `workload`, or None when it is a generated SHAPE.

    blis-catalog/workloads/<name>.yaml IS the workload: agentx-mvp states the harness scenario,
    the corpus loader and its pinned HF revision, the sha256, random_seed, benchmark_duration_s,
    grace_period_s, max_context_length, the cache-bust and trajectory settings, and the
    category/thinking pair that selects the acceptance profile. None of that is a per-experiment
    choice, so none of it belongs in a spec.
    """
    if not catalog:
        return None
    path = os.path.join(catalog, "workloads", f"{workload}.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    rep = doc.get("replay")
    return dict(rep) if isinstance(rep, dict) else None


def resolve_replay(workload, catalog, stated, dry_run=False):
    """The replay descriptor to run, built from the CATALOG plus the local corpus path.

    Why the spec states almost nothing: BLIS reads the descriptor itself. Verified against the
    pinned binary -- a `blis replay --workload-spec <catalog>/workloads/agentx-mvp.yaml` with NO
    --corpus-sha256, --benchmark-duration, --grace-period or --max-context-length ran to
    completion, and still refused a corpus whose bytes did not match the descriptor's digest
    ("sha256 50dde578... does not match --corpus-sha256 e39cd2ff..."). The digest is enforced
    from the catalog, not from the spec.

    What a spec legitimately states, because the catalog cannot know it:
      * `corpus` -- where the file sits on THIS machine. The catalog says the corpus is not
        vendored and gives the HF dataset + revision to fetch it from, but not a local path.
      * `corpus_cache` -- an optional parsed-corpus snapshot directory, pure local caching.
      * `grace_period` -- the one run-shape override the search needs: the descriptor's 30s
        default makes most saturated loads report "injected != completed", which the search then
        discards as capacity points even though they are honest saturation data.

    Anything else a spec states is accepted but recorded as an override, so a report can say the
    run did not use the catalog's own settings.
    """
    stated = dict(stated or {})
    desc = replay_descriptor(workload, catalog)
    if desc is None and not stated:
        return None                       # a generated shape workload: nothing to resolve
    out = {"scenario": (desc or {}).get("scenario") or stated.get("scenario") or workload}
    # The digest and the run window come from the descriptor. They are passed explicitly rather
    # than left to --workload-spec so the recorded cmd states what priced the run.
    src = desc or {}
    sha = ((src.get("corpus") or {}).get("sha256") if isinstance(src.get("corpus"), dict)
           else None)
    if sha:
        out["corpus_sha256"] = sha
    if src.get("benchmark_duration_s"):
        out["benchmark_duration"] = f"{src['benchmark_duration_s']}s"
    if src.get("max_context_length"):
        out["max_context_length"] = str(src["max_context_length"])
    if src.get("grace_period_s") is not None:
        out["grace_period"] = f"{src['grace_period_s']}s"
    # The acceptance selectors a drafting replay needs, also the workload's (see below).
    for key in ("category", "thinking"):
        if src.get(key) is not None:
            out[key] = src[key]
    # Local-only fields, and any deliberate override, from the spec.
    overrides = {}
    for key, value in stated.items():
        if key in ("corpus", "corpus_cache"):
            out[key] = value
        elif key == "scenario":
            continue
        else:
            if out.get(key) is not None and str(out[key]) != str(value):
                overrides[key] = {"catalog": out[key], "spec": value}
            out[key] = value
    if overrides:
        out["_overrides"] = overrides
    if not dry_run and not out.get("corpus"):
        raise SystemExit(
            f"workload {workload!r} is a trace REPLAY, so it needs a local corpus file.\n"
            f"The catalog states the corpus is not vendored; fetch it and point the spec at it:\n"
            f"  scenario:\n    replay:\n      corpus: .corpus/traces.jsonl\n"
            f"Everything else (digest, window, grace period, context filter) comes from "
            f"{os.path.join(catalog or '<catalog>', 'workloads', workload + '.yaml')}.")
    return out


def _workload_entry(workload, catalog):
    """The catalog's workload row, which carries the acceptance selectors a drafting replay needs."""
    try:
        cat = build_catalog.build(os.path.join(catalog, os.pardir))
    except Exception:
        return None
    return next((w for w in cat.get("workloads", []) if w.get("name") == workload), None)


def _pinned_stack(blis, catalog, registry):
    """What priced these runs: the intended pins plus the commits/digest that actually ran."""
    try:
        import run_search
        return run_search._pinned_stack(blis, catalog, registry)
    except Exception:
        pins = build_catalog.read_pins()
        return {"blis": pins.get("BLIS_REF", "unpinned"),
                "catalog": pins.get("CATALOG_REF", "unpinned"),
                "registry": pins.get("REGISTRY_REF", "unpinned")}


def _write_run_dir(out_dir, spec, report, runs):
    """Write reports/<name>/ -- the run directory the report and gallery read.

    `spec.yaml` is copied in verbatim: the report states the experiment's narrative beside its
    measured results, and a run directory that carries its own spec can be re-rendered (and
    re-checked against the numbers) without the repo it came from.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "runs.jsonl"), "w", encoding="utf-8") as f:
        for rec in runs:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    _dump(os.path.join(out_dir, "experiment.json"), report)
    with open(os.path.join(out_dir, "spec.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
    _dump(os.path.join(out_dir, "scenario.json"), _scenario_json(spec, report))
    _dump(os.path.join(out_dir, "pareto.json"), _pareto_json(report))
    _dump(os.path.join(out_dir, "report.json"), _report_json(report))


def _report_json(report):
    """The narrative seam report.py renders, assembled MECHANICALLY from the spec + the results.

    The prose here is deliberately thin: it states the spec's own hypothesis for each sweep next
    to what that sweep actually measured, and nothing else. An LLM wrote the hypotheses, in the
    spec, before any run; this path has no model in it and so must not invent analysis. A reader
    gets the prediction, the outcome, and the explorer over every run -- which is the comparison
    the experiment was designed to support.

    Deliberately NOT written here: `operating_points` (report.py derives the table from
    runs.jsonl, and an authored list is how the two surfaces drifted apart before) and a
    `glossary` (the shared lexicon owns the vocabulary).
    """
    sc = report["scenario"]
    winner = report.get("best_overall")
    workload = (sc.get("replay") or {}).get("scenario") or sc["workload"]
    tested = (f"{report['totals']['configs']} configurations over "
              f"{len(report['sweeps'])} sweep{'s' if len(report['sweeps']) != 1 else ''} "
              f"({', '.join(s['name'] for s in report['sweeps'])}), each at "
              f"{len(sc['concurrency'])} load level"
              f"{'s' if len(sc['concurrency']) != 1 else ''}, "
              f"every run on exactly {sc['gpus']} GPUs.")
    if winner:
        name = (f"{winner['label']} ({winner['variant']})" if winner.get("variant")
                else winner["label"])
        found = (f"<strong>{name}</strong>, from the <em>{winner['sweep']}</em> sweep, reached "
                 f"{winner['tokens_per_sec_per_gpu']:g} tok/s/GPU "
                 f"({winner['tokens_per_sec']:g} tok/s across the fleet) at concurrency "
                 f"{winner['concurrency']} while holding the scenario's SLOs.")
        run_html = (f"{name} at concurrency {winner['concurrency']}. "
                    f"{report['totals']['servable']} of {report['totals']['runs']} runs held "
                    f"the SLOs.")
    else:
        found = ("No configuration held the scenario's SLOs at any tested load, so this "
                 "experiment bounds the deployment from below rather than recommending one.")
        run_html = "Nothing in this grid is servable under the stated SLOs."
    sections, num = [], 2
    for s in report["sweeps"]:
        best = s.get("best")
        if best:
            outcome = (f"Measured: best was <strong>{best['label']}"
                       + (f" ({best['variant']})" if best.get("variant") else "")
                       + f"</strong> at {best['tokens_per_sec_per_gpu']:g} tok/s/GPU "
                       f"(concurrency {best['concurrency']}).")
        else:
            outcome = ("Measured: no configuration in this sweep held the SLOs at any "
                       "tested load.")
        prose = [p for p in (s.get("narrative", "").strip(), outcome) if p]
        sections.append({"num": num, "title": f"Sweep: {s['name']}",
                         "prose": prose,
                         "caption_note": f"{s['configs']} configurations, {s['runs']} runs."})
        num += 1
    return {
        "title": f"{sc['model']} on {sc['gpus']}x {sc['hardware']}: {report['name']}",
        "standfirst": (report["description"]
                       or f"A deterministic sweep over {workload} traffic, "
                          f"comparing {len(report['sweeps'])} designs on one GPU budget."),
        "meta": ["BLIS discrete-event simulation", "Experiment Spec (RFC #21)",
                 "deterministic sweep — no LLM in the measurement path",
                 f"seed {sc['seed']}"],
        "hero": [{"h": "What was tested", "html": tested},
                 {"h": "What was found", "html": found},
                 {"h": "What to run", "html": run_html, "run": True}],
        "explore_prose": [
            "Every run in this experiment is in the explorer below: filter by outcome to see "
            "which settings held the SLOs, or by setting to compare one sweep against another.",
            "Each sweep's hypothesis was written before the runs, and is quoted beside that "
            "sweep's measured result below.",
        ],
        "sections": sections,
        "recommendation_num": num,
        "recommendation_prose": [
            ("The table below ranks the measured operating points by output tok/s per GPU, the "
             "objective this experiment selects on, among runs that held the scenario's SLOs."),
        ],
    }


def _dump(path, obj):
    """Write JSON deterministically: sorted keys, fixed indent, trailing newline."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _scenario_json(spec, report):
    """The resolved scenario, in the shape report.py/build_gallery.py already read.

    Writing the EXISTING format (rather than a second one) is what lets an experiment report
    render and appear in the gallery with no changes to those consumers.
    """
    sc = report["scenario"]
    floor = report["interactivity_floor"]
    return {
        "run_id": report["name"],
        "model": sc["model"], "hardware": sc["hardware"], "gpu_budget": sc["gpus"],
        "workload": ({"replay": sc["replay"]} if sc.get("replay")
                     else {"preset": sc["workload"]}),
        "objective": report["objective"],
        "interactivity_floor": ({"tok_per_s_per_user": floor["tok_per_s_per_user"]}
                                if floor.get("tok_per_s_per_user") else {}),
        "slos": sc.get("slo") or {},
        "headroom": 1.0,          # an experiment measures the stated loads; it sizes nothing
        "seed": sc["seed"],
        "pinned_stack": report["pinned_stack"],
        "experiment": {"name": report["name"], "description": report["description"],
                       "sweeps": [{"name": s["name"], "narrative": s["narrative"]}
                                  for s in report["sweeps"]]},
    }


def _pareto_json(report):
    """Finalists + winner in the existing pareto.json shape, one finalist per sweep best."""
    finalists = []
    for s in report["sweeps"]:
        best = s.get("best")
        if not best:
            continue
        finalists.append({
            "label": best["label"], "variant": best.get("variant", ""),
            "sweep": s["name"],
            # `config` and `gpus` are what build_gallery reads for the winner/alternative cards
            # (tp, num_instances, GPU count); omitting them rendered "tp 0, 0 instances".
            "config": best.get("config") or {},
            "gpus": best.get("gpus"),
            "c_star": best["concurrency"], "c_use": best["concurrency"],
            "at_c_use": {"tokens_per_sec": best.get("tokens_per_sec"),
                         "tokens_per_sec_per_gpu": best.get("tokens_per_sec_per_gpu"),
                         "tok_per_s_per_user": best.get("tok_per_s_per_user"),
                         "itl_p99_ms": best.get("itl_p99_ms")},
            "curve": [{"concurrency": r["concurrency"],
                       "tokens_per_sec": r.get("tokens_per_sec"),
                       "tok_per_s_per_user": r.get("tok_per_s_per_user"),
                       "itl_p99_ms": r.get("itl_p99_ms"),
                       "ttft_p99_ms": r.get("ttft_p99_ms"),
                       "preemption_count": r.get("preemption_count", 0)}
                      for r in s["rows"]
                      if r["label"] == best["label"]
                      and r.get("variant", "") == best.get("variant", "")],
        })
    winner = report.get("best_overall")
    dropped = [d for s in report["sweeps"] for d in s["dropped"]]
    out = {"run_id": report["name"], "finalists": finalists, "dropped": dropped}
    if winner:
        name = (f"{winner['label']} ({winner['variant']})" if winner.get("variant")
                else winner["label"])
        out["winner"] = {
            "chosen": name, "label": winner["label"], "variant": winner.get("variant", ""),
            "sweep": winner["sweep"],
            "reason": (f"{name} in sweep '{winner['sweep']}' gives the most tok/s/GPU "
                       f"({winner['tokens_per_sec_per_gpu']:g} tok/s/GPU, "
                       f"{winner['tokens_per_sec']:g} tok/s fleet) while holding the "
                       f"scenario's SLOs, at concurrency {winner['concurrency']}."),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one Experiment Spec deterministically.")
    ap.add_argument("spec", help="experiments/<name>.yaml")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports"))
    ap.add_argument("--blis"); ap.add_argument("--catalog"); ap.add_argument("--registry")
    ap.add_argument("--routing", default=os.path.join(ROOT, "routing"))
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="expand and filter the grids and print the run plan; run no BLIS")
    ap.add_argument("--timeout", type=int, default=600, help="per-run BLIS timeout (seconds)")
    ap.add_argument("--max-parallel", type=int, default=1, dest="max_parallel",
                    help="BLIS runs to execute concurrently (default 1). One run is "
                         "single-threaded (~1.1 cores), so this is the ONLY way to use more "
                         "cores. Output is independent of this value: rows are re-sorted into "
                         "expansion order, so a spec reproduces byte-for-byte at any setting.")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="print the assembled report to stdout")
    args = ap.parse_args(argv)

    spec, err = validator.load(args.spec)
    if err:
        raise SystemExit(f"{args.spec}: {err}")
    errs = validator.validate(spec, args.routing)
    if errs:
        raise SystemExit(f"invalid spec {args.spec}:\n  - " + "\n  - ".join(errs))
    if not args.dry_run and not (args.blis and args.registry):
        raise SystemExit("--blis and --registry are required for a real run "
                         "(use --dry-run to expand the grids only)")

    t0 = time.time()
    report = run_experiment(spec, args.blis, args.catalog, args.registry, args.out,
                            routing_dir=args.routing, dry_run=args.dry_run,
                            run_timeout=args.timeout, max_parallel=args.max_parallel)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        t = report["totals"]
        print(f"\n{report['name']}: {t['configs']} configs, {t['runs']} runs"
              + (f", {t['servable']} servable" if not args.dry_run else " (planned)"))
        winner = report.get("best_overall")
        if winner:
            print(f"best: {winner['label']}"
                  + (f" ({winner['variant']})" if winner.get("variant") else "")
                  + f" in '{winner['sweep']}' -- "
                  f"{winner['tokens_per_sec_per_gpu']:g} tok/s/GPU at c={winner['concurrency']}")
        elif not args.dry_run:
            print("best: none -- no run met the scenario's SLOs")
        if not args.dry_run:
            print(f"wrote {os.path.join(args.out, report['name'])}/ in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
