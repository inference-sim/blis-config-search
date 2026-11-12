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

DETERMINISM IS THE POINT: the grid is a pure
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
import re
import shutil
import sys
import tempfile
import time

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_blis            # noqa: E402
import topologies          # noqa: E402
import blis_cli            # noqa: E402
import build_catalog       # noqa: E402
import pareto              # noqa: E402
import router_configs      # noqa: E402
import validate_experiment as validator   # noqa: E402

# Setting names in a spec are the engine block's names; run_blis's candidate keys are flat. These are the only two that differ, and they differ because run_blis folds
# drafting into a nested `speculative` block itself.
SPEC_TO_CANDIDATE = {"num_spec_tokens": "num_speculative_tokens", "method": "speculative_method"}


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
    # Any other mapping is a nested block (the validator checked its inner names).
    nested = {k: engine.pop(k) for k in list(engine) if isinstance(engine[k], dict)}
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
    sim_combos = grid(sweep.get("simulator") or {})   # simulator switches: [{}] when none
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
                                        for sim in sim_combos:
                                            cand = {**cfg, **eng, **router, **sim}
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
                        for sim in sim_combos:
                            cand = {**par, "num_instances": ni, **eng, **router, **sim}
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
    """Every combination of the router override trees; see router_configs.override_combos."""
    return router_configs.override_combos(router)


def apply_overrides(base_doc, overrides, knobs=None):
    """router_configs.apply_overrides: overrides edit the PARSED file; `knobs` is the index's list
    for a router reference file, None for a local one."""
    return router_configs.apply_overrides(base_doc, overrides, knobs)


override_phrase = router_configs.override_phrase


def materialize_router(cand, routing_dir, out_dir, router=None):
    """Resolve a candidate's `router_config` to a real file path, writing an override file if any.

    Returns (config_without_overrides, description). A base from the router's reference configs
    is staged into `out_dir` byte for byte, so the report holds the exact file each arm ran. A
    generated file is named by the HASH of its contents, so the same override combination always
    lands on the same path and a re-run rewrites identical bytes -- the files are part of the
    reproducible output, not scratch.
    """
    cand = dict(cand)
    overrides = cand.pop("router_overrides", None)
    rel = cand.get("router_config")
    if not rel:
        return cand, ""
    router = router or router_configs.open_checkout()
    source, base_path, knobs = router_configs.resolve(rel, routing_dir, router) \
        if not os.path.isabs(rel) else ("local", rel, None)
    if not overrides:
        if source == "router":
            os.makedirs(out_dir, exist_ok=True)
            staged = os.path.join(out_dir, os.path.basename(rel))
            shutil.copyfile(base_path, staged)
            base_path = staged
        cand["router_config"] = base_path
        return cand, topologies.routing_arm_name(rel)
    with open(base_path, encoding="utf-8") as f:
        base_doc = yaml.safe_load(f) or {}
    doc = apply_overrides(base_doc, overrides, knobs)
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    digest = hashlib.sha1(body.encode()).hexdigest()[:10]
    stem = topologies.routing_arm_name(rel)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}-{digest}.yaml")
    if not os.path.exists(path):
        origin = (f"the router's reference-configs/{rel} at {router.commit or router.want}"
                  if source == "router" else f"routing/{rel}")
        header = (f"# GENERATED by scripts/run_experiment.py from {origin}.\n"
                  f"# Overrides: {override_phrase(overrides)}\n"
                  f"# The base was parsed, edited and written back, so its comments and key order are\n"
                  f"# not kept. Named by a hash of the BYTES below. It is not a semantic hash:\n"
                  f"# reordering keys in the base file renames the generated arm (and only that).\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + body)
    cand["router_config"] = path
    return cand, f"{stem} ({override_phrase(overrides)})"


def router_file_record(cand, resolved_path, routing_dir, out_dir, router):
    """What a row records of its router file: the base (name, source, commit, sha256), the overrides
    applied, and the file the run read (path relative to the run directory, sha256). None when the
    candidate names no router file."""
    rel = cand.get("router_config")
    if not rel or not resolved_path:
        return None
    router = router or router_configs.open_checkout()
    source, base_path, _knobs = router_configs.resolve(rel, routing_dir, router) \
        if not os.path.isabs(rel) else ("local", rel, None)
    inside = os.path.abspath(resolved_path).startswith(os.path.abspath(out_dir) + os.sep)
    rec = {"base": rel, "source": source, "base_sha256": router_configs.sha256_of(base_path),
           "file": os.path.relpath(resolved_path, out_dir if inside else ROOT),
           "sha256": router_configs.sha256_of(resolved_path),
           "overrides": router_configs.override_record(cand.get("router_overrides") or {})}
    if source == "router":
        rec["router_commit"] = router.commit or router.want
    return rec


# --- labelling ---------------------------------------------------------------------------------


def label_of(cand):
    """The topology label for a candidate -- what a reader scans down a column.

    Colocated reuses topologies.label (`8x TP4xPP2xDP2 + EP`). A disaggregated candidate names
    both pools: `3P/5D TP8 + EP` when they share a shape, and `2P TP8 + EP / 3D TP16 + EP` when
    they differ (e.g. a wide-decode sweep), since one pool's shape would otherwise be invisible.
    """
    if cand.get("prefill_instances") is not None:
        def pool_shape(role):
            return topologies.shape(cand.get(f"{role}_tp", 1), cand.get(f"{role}_pp", 1),
                                    cand.get(f"{role}_dp", 1),
                                    cand.get(f"{role}_enable_expert_parallel"))
        pf, dc = cand["prefill_instances"], cand["decode_instances"]
        p_shape, d_shape = pool_shape("prefill"), pool_shape("decode")
        if p_shape == d_shape:
            return f"{pf}P/{dc}D {p_shape}"
        return f"{pf}P {p_shape} / {dc}D {d_shape}"
    return topologies.label({"tp": cand.get("tp", 1), "pp": cand.get("pp", 1),
                             "dp": cand.get("dp", 1),
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
                # simulator switches
                "pd_transfer_contention",
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
                "eplb", "dbo", "engram_config",
                # opened up by the v0.14.0 pin: the engine surface is blis-vllm's
                # engineconfig.Engine now, which carries these (the old blis-schemas
                # 31-field struct did not, so a spec naming one failed every run).
                # Only the priced ones are here -- the eleven that decode but measure
                # byte-identical are refused up front, so no sweep can label them.
                "max_num_active_seqs", "max_num_scheduled_tokens", "frontend")


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
    """The spec's SLO as {metric: cap_ms}; throughput_per_gpu passes through as a tok/s floor."""
    out = {}
    for key, raw in (slo or {}).items():
        if key == "throughput_per_gpu":
            out[key] = float(raw)
            continue
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


def servable(rec, slo_ms, gpus=None):
    """(ok, why_not) for one run against the spec's SLOs.

    Uses run_blis.per_user_rate / worst_itl_p99 -- the worst-instance per-token basis the
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
    floor = (slo_ms or {}).get("throughput_per_gpu")
    if floor:
        if not gpus:
            return False, "throughput_per_gpu SLO declared but the GPU count is unknown"
        per_gpu = m["tokens_per_sec"] / gpus
        if per_gpu < floor:
            return False, f"throughput_per_gpu {per_gpu:.1f} tok/s < {floor:g} tok/s"
    itl_cap = (slo_ms or {}).get("itl_p99")
    if itl_cap:
        worst, missing = run_blis.worst_itl_p99(rec)
        if missing or worst is None:
            return False, "itl_p99 SLO declared but not measured"
        # The worst p99 is the per-request time per output token, already per TOKEN: what speculation
        # gains is in it, so it is compared with the cap as it is (no acceptance-length division; that
        # belonged to the engine's per-step interval, which is no longer the gate).
        if worst > itl_cap:
            return False, f"itl_p99 {worst:.1f} ms > {itl_cap:g} ms"
    for key, field in (("ttft_p99", "ttft_p99_ms"), ("e2e_p99", "e2e_p99_ms")):
        cap = (slo_ms or {}).get(key)
        if not cap:
            continue
        # Worst instance, like the ITL cap above and like report.py's _feasible -- NOT the cluster
        # aggregate, which can sit under the cap while one instance is over it. TTFT is the sharpest
        # case: it is a prefill-pool property, so on a disaggregated run one slow prefill instance
        # is exactly what an aggregate hides (susiejojo review).
        value, missing = run_blis.worst_metric(rec, field)
        if missing or value is None:
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
    # Recorded on the basis `servable` JUDGES on, with the aggregate kept beside it as for ITL: a
    # row showing a passing aggregate next to an infeasible verdict is unreadable.
    worst_ttft, ttft_missing = run_blis.worst_metric(rec, "ttft_p99_ms")
    inter, inter_basis = pareto.interactivity_of(m)
    return {
        "sweep": sweep_name, "label": label, "variant": variant,
        "concurrency": concurrency, "gpus": gpus,
        "tokens_per_sec": round(tps, 1) if tps else None,
        "tokens_per_sec_per_gpu": round(tps / gpus, 2) if tps and gpus else None,
        "tok_per_s_per_user": round(pu, 1) if pu is not None else None,
        "itl_p99_ms": None if missing else (round(worst, 3) if worst is not None else None),
        "itl_p99_ms_aggregate": m.get("itl_p99_ms"),
        "ttft_p99_ms": None if ttft_missing else (round(worst_ttft, 1) if worst_ttft is not None else None),
        "ttft_p99_ms_aggregate": m.get("ttft_p99_ms"),
        "preemption_count": m.get("preemption_count", 0),
        # The Pareto axes (pareto.py): the registry's figures from the Result's measured window.
        "interactivity": round(inter, 3) if inter is not None else None,
        "interactivity_basis": inter_basis,
        "total_tokens_per_sec_per_gpu": _round(m.get("total_tokens_per_sec_per_gpu"), 3),
        "output_tokens_per_sec_per_decode_gpu": _round(m.get("output_tokens_per_sec_per_decode_gpu"), 3),
        "disaggregated": cand.get("prefill_instances") is not None,
    }


def _round(v, n):
    return round(v, n) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


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


def outcome_counts(rows):
    """{outcome: runs} over ledger rows that have one (a dry run has none), sorted by name."""
    return dict(sorted(collections.Counter(r["outcome"] for r in rows if r.get("outcome")).items()))


def _config_key(row):
    """A candidate without its load level: the same deployment at every concurrency."""
    cfg = {k: v for k, v in (row.get("config") or {}).items() if k != "concurrency"}
    return json.dumps([row.get("sweep"), cfg], sort_keys=True)


def refusal_groups(rows):
    """Refused runs grouped by the reason given, most runs first: [{reason, source, issue, runs,
    configs, examples}]. `configs` counts distinct candidates (a candidate refused at every load
    counts once); `examples` are up to three labels so a reader can find them."""
    groups = {}
    for r in rows:
        ref = r.get("refusal")
        if r.get("outcome") != "refused" or not ref:
            continue
        g = groups.setdefault((ref["source"], ref["reason"]), {
            "reason": ref["reason"], "source": ref["source"], "issue": ref.get("issue"),
            "runs": 0, "_configs": set(), "examples": []})
        g["runs"] += 1
        g["_configs"].add(_config_key(r))
        name = f"{r['label']} ({r['variant']})" if r.get("variant") else r["label"]
        if name not in g["examples"] and len(g["examples"]) < 3:
            g["examples"].append(name)
    out = []
    for g in groups.values():
        g["configs"] = len(g.pop("_configs"))
        out.append(g)
    return sorted(out, key=lambda g: (-g["runs"], g["reason"]))


def refused_configs(rows):
    """Distinct candidates refused at one or more load levels."""
    return len({_config_key(r) for r in rows if r.get("outcome") == "refused"})


def ranked_best(rows, objective):
    """The row the report headlines: the scalar rule (best_of) or the head of the Pareto front,
    as the spec's `objective.ranking` says."""
    if objective["ranking"] == "pareto":
        return pareto.best_on_front(rows, objective["throughput"])
    return best_of(rows)


def pareto_report(sweep_results, all_rows, objective):
    """The non-dominated front of every sweep and of all sweeps together, for both throughput
    forms. Always computed, whichever ranking headlines the report."""
    def both(rows):
        return {form: pareto.front(rows, form) for form in pareto.THROUGHPUT_FORMS}
    return {"x": pareto.X_METRIC,
            "y": {form: pareto.THROUGHPUT_FORMS[form][1] for form in pareto.THROUGHPUT_FORMS},
            "interval": None, "throughput": objective["throughput"],
            "overall": both(all_rows),
            "sweeps": [{"name": s["name"], **both(s["rows"])} for s in sweep_results]}


def run_experiment(spec, blis, catalog, registry, out_root, routing_dir=None, dry_run=False,
                   run_timeout=600, max_parallel=1, verbose=True, fields=None, data_override=False,
                   router_cfg=None):
    """Expand, filter, run and rank one spec. Returns the assembled report dict.

    `catalog` and `registry` are where the scripts read files from (a blis-substrate checkout at
    the pinned release). The simulator prices with its embedded data unless `data_override` names
    them to it, which makes the report experimental and refuses a committed reports/ directory
    (CFG-1)."""
    if data_override and not dry_run and _is_committed_reports(out_root):
        raise SystemExit(f"--data-override prices with data other than the pinned release's embedded "
                         f"data, so its report is experimental and never committed: --out {out_root} is the "
                         f"repository's reports/ directory. Choose another --out.")
    routing_dir = routing_dir or os.path.join(ROOT, "routing")
    sc = spec["scenario"]
    name = spec["name"]
    gpus = int(sc["gpus"])
    seed = int(sc.get("seed", 42))
    model, hardware = sc["model"], sc["hardware"]
    workload_spec = sc.get("workload_spec")
    workload = sc.get("workload") or workload_spec_label(workload_spec)
    concurrency = list(sc["concurrency"])
    slo_ms = slo_limits(sc.get("slo"))
    shared_engine = dict(sc.get("engine") or {})
    engine_version = blis_cli.engine_release(blis, run_blis.ENGINE_KIND, fields)
    replay = dict(sc.get("replay") or {}) or None
    # The cluster's inter-node fabric, defaulting to IB so a spec that states none is byte-identical.
    # Only a multi-node run prices cross-node traffic; run_blis ignores it on a single node.
    fabric = (sc.get("cluster") or {}).get("fabric") or run_blis.DEFAULT_FABRIC
    # The node size the specification states, for a chip whose catalog entry fixes none (#96).
    gpus_per_node = (sc.get("cluster") or {}).get("gpus_per_node")

    final_dir = os.path.join(out_root, name)
    # Into the committed reports/, a sweep is staged outside the tree: only the Results can say
    # whether it was priced by a tagged release on its embedded data (CFG-1), and a refusal must
    # leave an existing report byte for byte as it was. Nothing is copied in until the gate passes.
    stage = tempfile.mkdtemp(prefix="sweep-") if (not dry_run and _is_committed_reports(out_root)) else None
    out_dir = os.path.join(stage, name) if stage else final_dir
    scenarios_dir = os.path.join(out_dir, "scenarios")
    generated_routing = os.path.join(out_dir, "routing")
    if not dry_run:
        os.makedirs(scenarios_dir, exist_ok=True)

    # Which simulator is about to price this sweep. Only the CI workflows verified BLIS_SHA, so a
    # contributor sweeping locally against an off-pin binary got real-looking numbers with no
    # warning anywhere in the output (#23). Warned, not refused: see blis_pin_warning.
    # Skipped on a dry run, which prices nothing.
    if not dry_run:
        warn = _blis_pin_warning(blis, catalog, registry)
        # Not gated on `verbose`: a quiet run is exactly the one that would publish off-pin numbers.
        if warn:
            print(f"WARNING: {warn}\n         Every number in this report is priced by that "
                  f"binary, and scenario.json records its digest.", file=sys.stderr, flush=True)

    if router_cfg is None:
        router_cfg = router_configs.open_checkout()
    if router_cfg is not None and router_cfg.problem and not dry_run:
        print(f"WARNING: router configs: {router_cfg.problem}", file=sys.stderr, flush=True)

    entry = _model_entry(model, catalog) if catalog else None
    # A REPLAY workload is fully described by its catalog descriptor -- the harness scenario, the
    # corpus digest, the sending window, the grace period and the context filter are all stated
    # in blis-catalog/workloads/<name>.yaml. So a spec states NONE of them: naming the workload
    # is naming all of it. See resolve_replay.
    if workload_spec:
        workload_doc = load_workload_spec(workload_spec)
        replay = resolve_workload_spec(workload_spec, workload_doc, replay, dry_run=dry_run)
    else:
        replay = resolve_replay(workload, catalog, replay, dry_run=dry_run)

    # DRAFTING ON A REPLAY needs the acceptance selectors, and they are the WORKLOAD's, not a
    # knob: BLIS resolves acceptance from the registry by (model, method, K, thinking, category)
    # and REFUSES a drafting run whose acceptance does not resolve --
    #   "the workload states no thinking mode, which selects the profile; state one"
    # Without this every drafting run in a replay spec failed while the rest of the spec ran, so
    # the report showed an EMPTY drafting sweep rather than an error. The values are read from
    # the catalog's workload entry (agentx-mvp: category coding, thinking on), never invented.
    accept_sel = {}
    if workload_spec:
        # The file is the workload; it states the selectors itself (category, thinking).
        for key, sel in (("category", "speculative_category"), ("thinking", "speculative_thinking")):
            if workload_doc.get(key) is not None:
                accept_sel[sel] = str(workload_doc[key])
    elif catalog:
        wl = _workload_entry(workload, catalog)
        if wl:
            if wl.get("category"):
                accept_sel["speculative_category"] = wl["category"]
            if wl.get("thinking"):
                accept_sel["speculative_thinking"] = str(wl["thinking"])

    objective = pareto.objective_of(spec.get("objective"))
    runs, sweep_results, all_rows = [], [], []
    corpus_warm = False
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
        work = []                               # (idx, cfg, label, variant, c, router_rec)
        for cand in kept:
            try:
                resolved, router_desc = materialize_router(cand, routing_dir, generated_routing,
                                                           router_cfg) \
                    if not dry_run else (dict(cand), "")
                router_rec = (router_file_record(cand, resolved.get("router_config"), routing_dir,
                                                 out_dir, router_cfg) if not dry_run else None)

                if dry_run:
                    resolved.pop("router_overrides", None)
            except (KeyError, OSError, yaml.YAMLError, router_configs.RouterConfigsError) as e:
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
                work.append((len(work), cfg, label, variant, c, router_rec))

        def _execute(item):
            """One BLIS run. Pure w.r.t. shared state: run_blis.write_scenario publishes its
            scenario atomically under a content-hashed name (temp + os.replace), so workers
            racing on the same deployment cannot read a half-written file, and the caller owns
            all runs.jsonl I/O."""
            idx, cfg, label, variant, c, router_rec = item
            nreq = None if replay else max(80, c * 4)
            # `_overrides` and `workload_document` are provenance for the report (which catalog
            # settings the spec replaced; the workload file as read), not replay fields -- keep
            # them out of the command and out of every run record.
            rec = run_blis.run_one(blis, catalog, registry, scenarios_dir, model, hardware,
                                   engine_version, workload, nreq, seed,
                                   None, cfg, phase=sname, timeout=run_timeout,
                                   replay={k: v for k, v in replay.items()
                                           if k not in RECORD_ONLY_REPLAY} if replay else None,
                                   fabric=fabric, gpus_per_node=gpus_per_node, fields=fields,
                                   data_override=data_override)
            rec["sweep"] = sname
            if router_rec:
                rec["router_file"] = router_rec
            _relativize_cmd(rec, out_dir)
            ok, why = servable(rec, slo_ms, gpus)
            row = run_row(rec, cfg, label, variant, sname, c, gpus)
            # The resolved config travels with the row so pareto.json's finalists can carry
            # it: build_gallery reads `finalist.config.tp` / `.num_instances` for the winner
            # card, and showed "tp 0, 0 instances" without it.
            row["config"] = _jsonable(cfg)
            row["servable"] = ok
            if router_rec:
                row["router"] = router_rec
            if why:
                row["not_servable_because"] = why
            outcome, refusal = run_blis.outcome_of(rec)
            row["outcome"] = ("served" if ok else "slo_miss") if outcome == "ran" else outcome
            if refusal:
                row["refusal"] = refusal
            return idx, rec, row

        if work:
            if max_parallel <= 1:
                done = [_execute(it) for it in work]
            else:
                # The corpus snapshot is shared by every run, so build it with one run before the
                # pool; N cold workers would each parse 568 MB and race to write it.
                done = []
                while replay and not corpus_warm and work:
                    done.append(_execute(work[0]))
                    work = work[1:]
                    # A run refused before BLIS starts (returncode None) never loaded the corpus.
                    corpus_warm = done[-1][1].get("returncode") is not None
                # One BLIS run is single-threaded (a discrete-event loop measured at ~1.1 cores),
                # so the only way to use more cores is more processes. A thread pool is enough:
                # each worker just waits on a subprocess.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
                    # as_completed order would be completion order, which is nondeterministic;
                    # ex.map yields in SUBMISSION order by contract, which is what we want.
                    done += list(ex.map(_execute, work))
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
            "best": ranked_best(rows, objective),
            "outcomes": outcome_counts(rows),
            "refusals": refusal_groups(rows),
            "rows": rows,
        })

    report = {
        "name": name,
        "description": (spec.get("description") or "").strip(),
        "scenario": {"model": model, "hardware": hardware, "workload": workload, "gpus": gpus,
                     # A workload named by a file: the file, its own sha256 and the document as read
                     # (null for a catalog workload), so the report can state it is not in the catalog.
                     "workload_spec": ({"path": workload_spec,
                                        "sha256": replay["workload_spec_sha256"],
                                        "catalog": False, "label": WORKLOAD_SPEC_LABEL,
                                        "document": replay["workload_document"],
                                        "recipe": recipe_record(replay["workload_document"], replay)}
                                       if workload_spec else None),
                     "concurrency": concurrency, "slo": sc.get("slo") or {}, "seed": seed,
                     "engine": shared_engine or None, "engine_version": engine_version,
                     # The cluster fabric the sweep assumed, so a report states its interconnect
                     # without reopening the spec. Omitted (null) means the ib-400g default.
                     "cluster": (sc.get("cluster") or None),
                     "replay": ({k: v for k, v in replay.items() if k != "workload_document"}
                                if replay else None)},
        "slo_ms": slo_ms,
        # What every ranked figure is (CFG-4, DEV-8): the metric basis and the scope it ranges over.
        # The cluster figures are the Result's measured window. The worst-instance TTFT, end-to-end
        # and TPOT gates are rebuilt from its profiling-phase rows; the ITL gate is the cluster's,
        # because the decode-step interval is not per request (inference-sim #2087 gives the rest).
        "basis": run_blis.BASIS,
        "scope": "cluster; worst instance for TTFT and end-to-end gates",
        "interactivity_floor": {"tok_per_s_per_user": floor_rate_from(slo_ms)},
        "objective": pareto.describe(objective),
        # Which ranking picked best/best_overall/finalists, and the figures it used.
        "ranking": {"used": objective["ranking"], "throughput": objective["throughput"]},
        "sweeps": sweep_results,
        "best_overall": ranked_best(all_rows, objective),
        "pareto": pareto_report(sweep_results, all_rows, objective),
        "outcomes": outcome_counts(all_rows),
        "refusals": refusal_groups(all_rows),
        "pinned_stack": _pinned_stack(blis, catalog, registry, engine_version,
                                      _priced_with_other_data(runs, data_override), router_cfg),
        "experimental": _priced_with_other_data(runs, data_override),
        "model_arch": (entry or {}).get("arch"),
        "totals": {"configs": sum(s["configs"] for s in sweep_results),
                   "runs": len(runs) if not dry_run else
                           sum(s["runs"] for s in sweep_results),
                   "servable": sum(1 for r in all_rows if r.get("servable")),
                   # Runs the simulator or runner declined, apart from runs that ran and missed an SLO.
                   "refused": sum(1 for r in all_rows if r.get("outcome") == "refused"),
                   "configs_refused": refused_configs(all_rows)},
    }
    if not dry_run:
        # The refusal above covers a requested override. This one covers what the Results say: a
        # report priced with other data, or by an untagged build, is never written to reports/.
        if stage and report["experimental"]:
            shutil.rmtree(stage, ignore_errors=True)       # the staged sweep only; reports/ is untouched
            raise SystemExit(f"{name}: the Results say this sweep was priced with data or a build other than "
                             f"a tagged release's embedded data ({_other_data_reason(runs)}), so its report is "
                             f"experimental and never committed: --out {out_root} is the repository's "
                             f"reports/ directory. Choose another --out, or run the pinned release.")
        _write_run_dir(out_dir, spec, report, runs)
        if stage:
            # The gate passed and the whole report is assembled: replace the report directory as a
            # whole, so a file the new sweep no longer writes does not survive from an older one
            # and the tree depends only on this sweep (the sweeper is deterministic).
            _replace_dir(out_dir, final_dir)
            shutil.rmtree(stage, ignore_errors=True)
    return report


def _relativize_cmd(rec, out_dir):
    """Rewrite absolute paths in a row's recorded `cmd` as paths relative to the run directory.

    The ledger must not depend on WHERE the sweep ran. Everything else in a report already
    satisfies that (report.py identifies a run by the id inside its data, never by the directory
    path), but `cmd` embedded the absolute --scenarios / --registry / router paths, so running
    one spec twice in two output directories produced ledgers that differed in those strings
    while every measured number was identical. A re-run is supposed to be byte-identical, so a
    diff should mean a changed result -- not a changed working directory.

    Each base is matched both as written and as `realpath`, because the two differ whenever the
    run directory or the checkout is reached through a symlink -- `/tmp` resolves to `/private/tmp`
    on macOS, and a container commonly mounts the workspace at a linked path. Matching only
    lexically let the resolved form through as an absolute, host-specific string.

    A path genuinely outside both bases is LEFT ABSOLUTE on purpose: it cannot be made relative to
    the run directory without a `../../..` chain that says less than the absolute path does, and
    the stack it points at is already recorded, resolved, in every row's `pinned_stack`.

    The command stays readable and still states exactly which flags and values BLIS received.
    """
    cmd = rec.get("cmd")
    if not cmd:
        return
    bases = []
    for base in (os.path.abspath(out_dir), ROOT):
        for form in (base, os.path.realpath(base)):
            if form and form not in bases:
                bases.append(form)
    out = []
    for arg in cmd:
        text = str(arg)
        # Compare the argument both as written and resolved, against bases in both forms: the
        # symlink may sit on either side (an arg BLIS was handed by its resolved name, or a base
        # mounted at a linked path), and only one of the four pairings may line up.
        forms = [text]
        if os.path.isabs(text):
            resolved = os.path.realpath(text)
            if resolved != text:
                forms.append(resolved)
        rewritten = None
        for candidate in forms:
            for base in bases:
                if candidate.startswith(base + os.sep):
                    rewritten = os.path.join(".", os.path.relpath(candidate, base))
                    break
            if rewritten is not None:
                break
        out.append(rewritten if rewritten is not None else text)
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


# Replay keys the report records once and no run passes to the simulator.
RECORD_ONLY_REPLAY = ("_overrides", "workload_document")

WORKLOAD_SPEC_LABEL = "custom (not in the catalog)"


def workload_spec_label(path):
    """How a report names a workload that came from a file."""
    return f"{WORKLOAD_SPEC_LABEL}: {os.path.basename(str(path))}"


def _seconds(v):
    """A duration the runner passes as text ("3600s", "30s", "inf") in seconds; inf stays "inf"."""
    if v is None:
        return None
    t = str(v).strip().lower()
    if t in ("inf", ".inf"):
        return "inf"
    m = re.fullmatch(r"([0-9.]+)(ms|s|m|h)?", t)
    if not m:
        return t
    x = float(m.group(1)) * {"ms": 1e-3, "s": 1, "m": 60, "h": 3600, None: 1}[m.group(2)]
    return int(x) if x == int(x) else x


# The recipe fields a spec's replay block can override, and the replay key that does: what BLIS is
# actually run with is the EFFECTIVE replay, not only the workload file.
EFFECTIVE_FIELDS = {"benchmark_duration_s": "benchmark_duration", "grace_period_s": "grace_period"}


def recipe_record(doc, effective=None):
    """The AgentX run recipe a replay is run with, for the report: each recipe field as run, the fields
    that differ from the run's (the file's own values and any override of them by the spec), and one line
    a reader sees. `effective` is the resolved replay (resolve_workload_spec): a spec's replay block may
    override the window or the grace, so the record is derived from it as well as from the document, and
    an override is named. None for a workload that is not an AgentX replay."""
    import validate_experiment as ve
    if not ve.is_agentx(doc):
        return None
    fields = {k: ve._dig(doc, k) for k in ve.AGENTX_RECIPE}
    fields["corpus.loader"] = ve._dig(doc, "corpus.loader")
    overrides = {}
    for field, key in EFFECTIVE_FIELDS.items():
        if effective and effective.get(key) is not None:
            have = _seconds(effective[key])
            if have != fields.get(field):
                overrides[field] = [fields.get(field), have]
                fields[field] = have
    diff = {k: [fields[k], v] for k, v in ve.AGENTX_RECIPE.items() if fields.get(k) is not None and fields[k] != v}
    loader = fields["corpus.loader"]
    if loader is not None and loader != ve.AGENTX_UNCAPPED_LOADER:
        diff["corpus.loader"] = [loader, ve.AGENTX_UNCAPPED_LOADER]
    line = (f"AgentX run recipe: {fields['trace_idle_gap_cap_s']} s idle-gap cap, "
            f"{fields['warmup.requests_per_lane']} warm-up requests per lane, trajectory start "
            f"{fields['trajectory_start.min_ratio']} to {fields['trajectory_start.max_ratio']}, "
            f"{fields['benchmark_duration_s']} s window, {fields['grace_period_s']} s grace, "
            f"corpus {fields['corpus.loader']}")
    if overrides:
        line += "; the spec overrides " + ", ".join(f"{k} ({v[0]} to {v[1]})" for k, v in sorted(overrides.items()))
    if diff:
        line += "; differs from the run's in " + ", ".join(f"{k} ({v[0]} not {v[1]})" for k, v in sorted(diff.items()))
    return {"fields": fields, "overrides": overrides, "differs_from_run": diff, "label": line}


def _spec_path(path):
    """A workload_spec path is relative to the repository, so a spec means the same file anywhere.
    It must stay inside the repository once symlinks and `..` are resolved."""
    full = os.path.realpath(os.path.join(ROOT, path))
    if os.path.isabs(path) or not full.startswith(os.path.realpath(ROOT) + os.sep):
        raise SystemExit(f"scenario.workload_spec: {path!r} is outside the repository")
    return full


def load_workload_spec(path):
    """The replay document in the workload file, as a mapping.

    The file is a blis-schemas Replay document: the replay block itself, as the catalog's
    `replay:` carries it. The simulator reads and checks it (`blis replay --workload-spec`).
    """
    full = _spec_path(path)
    try:
        with open(full, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise SystemExit(f"scenario.workload_spec: {path}: {e}")
    if not isinstance(doc, dict):
        raise SystemExit(f"scenario.workload_spec: {path} is not a mapping")
    return doc


def workload_spec_sha256(path):
    h = hashlib.sha256()
    with open(_spec_path(path), "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def resolve_workload_spec(path, doc, stated, dry_run=False):
    """The replay descriptor for a workload named by file.

    The corpus digest, the window, the grace period and the context filter come from the file, as
    they come from the catalog entry for a named workload. The report records the whole document
    and the file's own sha256, and `_overrides` is empty: the file is the workload.
    """
    stated = dict(stated or {})
    corpus = doc.get("corpus") if isinstance(doc.get("corpus"), dict) else {}
    if not doc.get("scenario"):
        raise SystemExit(f"workload_spec {path!r} states no `scenario` (the harness scenario id)")
    out = {"scenario": doc["scenario"],
           "workload_spec": _spec_path(path),
           "workload_spec_relative": path,
           "workload_spec_sha256": workload_spec_sha256(path),
           "workload_document": doc}
    if corpus.get("sha256"):
        out["corpus_sha256"] = corpus["sha256"]
    if doc.get("benchmark_duration_s"):
        out["benchmark_duration"] = f"{doc['benchmark_duration_s']}s"
    if doc.get("max_context_length"):
        out["max_context_length"] = str(doc["max_context_length"])
    if doc.get("grace_period_s") is not None:
        g = doc["grace_period_s"]
        out["grace_period"] = g if isinstance(g, str) else f"{g}s"
    for key in ("category", "thinking"):
        if doc.get(key) is not None:
            out[key] = doc[key]
    for key in ("corpus", "corpus_cache"):
        if stated.get(key):
            out[key] = stated[key]
    if "grace_period" in stated:
        out["grace_period"] = stated["grace_period"]
    if not dry_run and not out.get("corpus"):
        raise SystemExit(
            f"workload_spec {path!r} is a trace REPLAY, so it needs a local corpus file:\n"
            f"  scenario:\n    replay:\n      corpus: .corpus/traces.jsonl")
    return out


def _workload_entry(workload, catalog):
    """The catalog's workload row, which carries the acceptance selectors a drafting replay needs."""
    try:
        cat = build_catalog.build(os.path.join(catalog, os.pardir))
    except Exception:
        return None
    return next((w for w in cat.get("workloads", []) if w.get("name") == workload), None)


def _git_identity(path):
    """(describe, commit) of a git checkout, or (None, None) when it is not one."""
    import subprocess

    def q(*args):
        try:
            out = subprocess.run(args, cwd=path, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return (out.stdout or "").strip() or None if out.returncode == 0 else None
    if not (path and os.path.isdir(path)):
        return None, None
    return q("git", "describe", "--tags", "--always", "--dirty"), q("git", "rev-parse", "HEAD")


def _file_digest(path):
    """sha256 of a file, or None. The simulator has no version command, so this identifies the build."""
    if not (path and os.path.isfile(path)):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _blis_source_commit(blis):
    """(describe, commit) of the checkout the binary was built in, or (None, None).

    Only a binary sitting in the simulator's own tree (main.go beside it) counts: one copied into
    another repo must not borrow that repo's commit.
    """
    if not (blis and os.path.isfile(blis)):
        return None, None
    src = os.path.dirname(os.path.abspath(blis))
    if not os.path.isfile(os.path.join(src, "main.go")):
        return None, None
    desc, commit = _git_identity(src)
    return (desc, commit) if commit else (None, None)


def _is_committed_reports(out_root):
    return os.path.realpath(out_root) == os.path.realpath(os.path.join(ROOT, "reports"))


def _priced_with_other_data(runs, data_override):
    """True when the sweep asked for other data, or any run's Result says it was priced with data
    other than a tagged release's own embedded data (CFG-1): the report is then experimental,
    whatever the command line said. A Result names its data sources and whether the build is a
    tagged release (`released_data`); an untagged or off-pin build is experimental too."""
    return bool(data_override) or any(
        r.get("data") not in (None, "embedded") or r.get("released_build") is False for r in runs)


def _replace_dir(src, dst):
    """Make `dst` hold exactly what `src` holds. The old `dst`, if any, is moved aside first and
    removed only after the new one is in place, so a failure part way leaves the old report."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    old = None
    if os.path.lexists(dst):
        old = dst + ".old-" + os.path.basename(src)
        os.rename(dst, old)
    try:
        shutil.copytree(src, dst)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        if old:
            os.rename(old, dst)
        raise
    if old:
        shutil.rmtree(old, ignore_errors=True)


def _other_data_reason(runs):
    """One line naming why a sweep's Results are not a tagged release's embedded data."""
    if any(r.get("data") not in (None, "embedded") for r in runs):
        return "a Result names a data source other than embedded"
    return "a Result names a build that is not a tagged release"


def _pinned_stack(blis, catalog, registry, engine_version, data_override=False, router_cfg=None):
    """What priced these runs: the pins.env refs, the engine release the simulator ran, and the
    commits and binary digest that ran. The catalog and registry are the simulator's embedded data
    (the substrate release in pins.env) unless the run overrode them."""
    pins = build_catalog.read_pins()
    out = {"blis": pins.get("BLIS_REF", "unpinned"),
           "kernel": pins.get("KERNEL_REF", "unpinned"),
           "router": pins.get("ROUTER_REF", "unpinned"),
           "substrate": pins.get("SUBSTRATE_REF", "unpinned"),
           "engine_version": engine_version,
           "data": "override" if data_override else "embedded"}
    # The router's reference configs a spec may use: the pin, and the commit actually read.
    if pins.get("ROUTER_CONFIGS_SHA"):
        out["router_configs"] = {"ref": pins.get("ROUTER_CONFIGS_REF"), "pinned": pins["ROUTER_CONFIGS_SHA"],
                                 "resolved": router_cfg.commit if router_cfg is not None else None}
    resolved = {}
    if data_override:
        for name, path in (("catalog", catalog), ("registry", registry)):
            desc, commit = _git_identity(path)
            if desc or commit:
                resolved[name] = {"describe": desc, "commit": commit}
    digest = _file_digest(blis)
    if digest:
        resolved["blis"] = {"sha256": digest}
        desc, commit = _blis_source_commit(blis)
        if commit:
            resolved["blis"]["commit"] = commit
        if desc:
            resolved["blis"]["describe"] = desc
    if resolved:
        out["resolved"] = resolved
    return out


def blis_pin_warning(stack, pins=None):
    """How the simulator that ran differs from BLIS_SHA, or "" when it agrees or cannot be told.

    Compares the SOURCE commit, never the binary digest: one tag builds to different bytes per
    toolchain. A warning, not a refusal, so a local sweep can still test an unreleased simulator;
    CI refuses outright by checking BLIS_SHA on clone (#23).
    """
    blis = ((stack or {}).get("resolved") or {}).get("blis") or {}
    want = (pins if pins is not None else build_catalog.read_pins()).get("BLIS_SHA")
    got = blis.get("commit")
    if not want or not got:
        return ""
    desc = blis.get("describe") or ""
    ref = (stack or {}).get("blis") or "the pinned ref"
    if got == want:
        if desc.endswith("-dirty"):
            return (f"blis: the binary was built from the pinned {ref} ({want[:12]}) with "
                    f"UNCOMMITTED changes in its checkout, so it is not the pinned simulator")
        return ""
    at = f" ({desc})" if desc and desc != got else ""
    return (f"blis: pins.env pins {ref} at {want[:12]}, but the binary was built from "
            f"{got[:12]}{at}")


def _blis_pin_warning(blis, catalog, registry):
    """blis_pin_warning for the binary about to run; never the thing that breaks a sweep."""
    try:
        return blis_pin_warning(_pinned_stack(blis, catalog, registry, None))
    except Exception:
        return ""


def _write_run_dir(out_dir, spec, report, runs):
    """Write reports/<name>/ -- the run directory the report and site read.

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


SCALAR_OBJECTIVE = pareto.describe(pareto.DEFAULT_OBJECTIVE)


def refusal_sentence(groups, limit=3):
    """The refusal groups as one readable sentence fragment, naming each reason."""
    parts = []
    for g in groups[:limit]:
        n = g["configs"]
        parts.append(f"{n} configuration{'s' if n != 1 else ''} ({g['runs']} run"
                     f"{'s' if g['runs'] != 1 else ''}): {g['reason']}")
    if len(groups) > limit:
        parts.append(f"{len(groups) - limit} more reason{'s' if len(groups) - limit != 1 else ''}")
    return "; ".join(parts)


def unplaceable_sentence(report):
    """Why a pareto ranking has no best although runs held the SLOs: their Results lack a figure."""
    rk = report.get("ranking") or {}
    form = rk.get("throughput", "total")
    metric = pareto.THROUGHPUT_FORMS[form][1]
    return (f"{report['totals']['servable']} runs held the scenario's SLOs, but none can be placed on the "
            f"Pareto front: their Results state no {metric} (or no TPOT), which the simulator build that "
            f"ran them does not emit. Re-run with a build that does, or rank with `objective.ranking: scalar`.")


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
    if sc.get("workload_spec"):
        workload = sc["workload_spec"]["label"]
        if (sc["workload_spec"].get("recipe") or {}).get("label"):
            workload += f" ({sc['workload_spec']['recipe']['label']})"
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
                 f"{winner['tokens_per_sec_per_gpu']:g} tok/s/accelerator "
                 f"({winner['tokens_per_sec']:g} tok/s across the fleet) at concurrency "
                 f"{winner['concurrency']} while holding the scenario's SLOs.")
        if report["totals"].get("refused"):
            found += (f" {report['totals']['refused']} runs were refused and are not counted as misses: "
                      + refusal_sentence(report.get("refusals") or []) + ".")
        found += f" Ranking: {report.get('objective', SCALAR_OBJECTIVE)}."
        run_html = (f"{name} at concurrency {winner['concurrency']}. "
                    f"{report['totals']['servable']} of {report['totals']['runs']} runs held "
                    f"the SLOs.")
    elif report["totals"].get("servable"):
        # Runs held the SLOs but the ranking could not place any: say so, never "no configuration held".
        found = unplaceable_sentence(report)
        run_html = "Runs held the SLOs but the Pareto ranking could not place them."
    else:
        refused = report["totals"].get("refused", 0)
        ran = report["totals"]["runs"] - refused
        if refused and not ran:
            found = ("The simulator refused every configuration, so nothing was run and this experiment "
                     "says nothing about the SLOs or about these deployments' speed. Refused: "
                     + refusal_sentence(report.get("refusals") or []) + ".")
            run_html = "Nothing in this grid ran: every configuration was refused."
        elif refused:
            found = (f"No configuration that was run held the scenario's SLOs at any tested load, so the "
                     f"{ran} runs that were made bound the deployment from below rather than recommending one. "
                     f"{refused} further runs were refused and are not evidence of a miss. Refused: "
                     + refusal_sentence(report.get("refusals") or []) + ".")
            run_html = "Nothing that ran in this grid is servable under the stated SLOs."
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
                       + f"</strong> at {best['tokens_per_sec_per_gpu']:g} tok/s/accelerator "
                       f"(concurrency {best['concurrency']}).")
        elif s.get("refusals") and (s.get("outcomes") or {}).get("refused", 0) == s["runs"]:
            outcome = ("Measured: nothing ran; the simulator refused every configuration in this sweep. "
                       "Refused: " + refusal_sentence(s["refusals"]) + ".")
        elif s.get("refusals"):
            outcome = ("Measured: no configuration that ran in this sweep held the SLOs at any tested "
                       "load. Refused: " + refusal_sentence(s["refusals"]) + ".")
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
                 f"seed {sc['seed']}",
                 f"basis: {report['basis'].replace('_', ' ')}",
                 f"scope: {report['scope']}"],
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
            (f"The table below lists the measured operating points. Headline ranking: {report.get('objective', SCALAR_OBJECTIVE)}. "
             f"The Pareto front of every sweep is in pareto.json."),
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
        "workload": ({"replay": sc["replay"],
                      **({"workload_spec": sc["workload_spec"]} if sc.get("workload_spec") else {})}
                     if sc.get("replay") else {"preset": sc["workload"]}),
        "objective": report.get("objective", SCALAR_OBJECTIVE),
        "basis": report["basis"], "scope": report["scope"],
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


def _finalist(report, row, sweep):
    """One finalist in the existing pareto.json shape, built from a ledger row."""
    return {
        "label": row["label"], "variant": row.get("variant", ""),
        "sweep": sweep,
        # `config` and `gpus` are what build_gallery reads for the winner/alternative cards
        # (tp, num_instances, GPU count); omitting them rendered "tp 0, 0 instances".
        "config": row.get("config") or {},
        "gpus": row.get("gpus"),
        "c_star": row["concurrency"], "c_use": row["concurrency"],
        "at_c_use": {"tokens_per_sec": row.get("tokens_per_sec"),
                     "tokens_per_sec_per_gpu": row.get("tokens_per_sec_per_gpu"),
                     "total_tokens_per_sec_per_gpu": row.get("total_tokens_per_sec_per_gpu"),
                     "interactivity": row.get("interactivity"),
                     "tok_per_s_per_user": row.get("tok_per_s_per_user"),
                     "itl_p99_ms": row.get("itl_p99_ms")},
        "curve": [{"concurrency": r["concurrency"],
                   "tokens_per_sec": r.get("tokens_per_sec"),
                   "tok_per_s_per_user": r.get("tok_per_s_per_user"),
                   "itl_p99_ms": r.get("itl_p99_ms"),
                   "ttft_p99_ms": r.get("ttft_p99_ms"),
                   "preemption_count": r.get("preemption_count", 0)}
                  for s in report["sweeps"] if s["name"] == sweep
                  for r in s["rows"]
                  if r["label"] == row["label"]
                  and r.get("variant", "") == row.get("variant", "")],
    }


def _pareto_json(report):
    """pareto.json: finalists and winner (the headline ranking's), the Pareto front of every sweep,
    and how many runs were refused, grouped by reason.

    `ranking` says which rule chose the finalists: `scalar` is one finalist per sweep (the most
    output tok/s per accelerator); `pareto` is the members of the all-sweeps front."""
    ranking = report.get("ranking") or {"used": "scalar"}
    finalists = []
    if ranking["used"] == "pareto":
        form = ranking["throughput"]
        rows = {(r["sweep"], r["label"], r.get("variant", ""), r["concurrency"]): r
                for s in report["sweeps"] for r in s["rows"]}
        for m in (report.get("pareto") or {}).get("overall", {}).get(form, {}).get("members", []):
            row = rows.get((m["sweep"], m["label"], m["variant"], m["concurrency"]))
            if row:
                finalists.append({**_finalist(report, row, m["sweep"]), "front_status": m["status"]})
    else:
        for s in report["sweeps"]:
            best = s.get("best")
            if best:
                finalists.append(_finalist(report, best, s["name"]))
    winner = report.get("best_overall")
    dropped = [d for s in report["sweeps"] for d in s["dropped"]]
    out = {"run_id": report["name"], "finalists": finalists, "dropped": dropped,
           "ranking": {**ranking, "description": report.get("objective", SCALAR_OBJECTIVE)},
           "front": report.get("pareto"),
           "outcomes": report.get("outcomes") or {},
           "refusals": report.get("refusals") or [],
           "sweep_outcomes": [{"name": s["name"], "outcomes": s.get("outcomes") or {},
                               "refusals": s.get("refusals") or []} for s in report["sweeps"]]}
    if winner:
        name = (f"{winner['label']} ({winner['variant']})" if winner.get("variant")
                else winner["label"])
        if ranking["used"] == "pareto":
            reason = (f"{name} in sweep '{winner['sweep']}' is the front member with the most "
                      f"tok/s/accelerator ({winner.get('total_tokens_per_sec_per_gpu') if ranking['throughput'] == 'total' else winner.get('output_tokens_per_sec_per_decode_gpu')} "
                      f"{ranking['throughput']} form) at concurrency {winner['concurrency']}; "
                      f"the front, not this point, is the result.")
        else:
            reason = (f"{name} in sweep '{winner['sweep']}' gives the most tok/s/accelerator "
                      f"({winner['tokens_per_sec_per_gpu']:g} tok/s/accelerator, "
                      f"{winner['tokens_per_sec']:g} tok/s fleet) while holding the "
                      f"scenario's SLOs, at concurrency {winner['concurrency']}.")
        out["winner"] = {"chosen": name, "label": winner["label"],
                         "variant": winner.get("variant", ""), "sweep": winner["sweep"],
                         "reason": reason}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one Experiment Spec deterministically.")
    ap.add_argument("spec", help="experiments/<name>.yaml")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports"))
    ap.add_argument("--blis"); ap.add_argument("--catalog"); ap.add_argument("--registry")
    ap.add_argument("--data-override", action="store_true", dest="data_override",
                    help="EXPERIMENTAL: price with --catalog/--registry instead of the simulator's embedded "
                         "data. The report is marked experimental and may not be written to reports/")
    ap.add_argument("--fields", help="the engine's field list: the pinned blis-vllm release's fields.json, "
                                     "until the binary has `blis fields` (default: ask the binary)")
    ap.add_argument("--routing", default=os.path.join(ROOT, "routing"))
    ap.add_argument("--router-configs", dest="router_configs", default=None,
                    help="checkout of the router repository at the commit pins.env pins, whose "
                         "reference-configs/ hold the EndpointPicker files a spec may name "
                         "(default: $ROUTER_CONFIGS_DIR, else ../llm-d-router)")
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
    if not args.blis:
        raise SystemExit("--blis is required: the simulator's commands supply the setting names, "
                         "the catalog identities and the engine release, even for --dry-run")
    if not args.dry_run and not args.registry:
        raise SystemExit("--registry is required for a real run "
                         "(use --dry-run to expand the grids only)")
    try:
        router_cfg = router_configs.open_checkout(args.router_configs)
    except router_configs.RouterConfigsError as e:
        raise SystemExit(f"{args.spec}: {e}")
    try:
        errs = validator.validate(spec, args.blis, args.routing, args.catalog, args.fields,
                                  router_cfg)
    except blis_cli.BlisCommandError as e:
        raise SystemExit(f"{args.spec}: {e}")
    if errs:
        raise SystemExit(f"invalid spec {args.spec}:\n  - " + "\n  - ".join(errs))

    t0 = time.time()
    report = run_experiment(spec, args.blis, args.catalog, args.registry, args.out,
                            routing_dir=args.routing, dry_run=args.dry_run,
                            run_timeout=args.timeout, max_parallel=args.max_parallel, fields=args.fields,
                            data_override=args.data_override, router_cfg=router_cfg)
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
                  f"{winner['tokens_per_sec_per_gpu']:g} tok/s/accelerator at c={winner['concurrency']}")
        elif not args.dry_run:
            if t["servable"]:
                print("best: none -- " + unplaceable_sentence(report))
            elif t["refused"] and t["refused"] == t["runs"]:
                print("best: none -- the simulator refused every run; nothing was measured")
            else:
                print("best: none -- no run met the scenario's SLOs")
        if not args.dry_run and t["refused"]:
            print(f"refused: {t['refused']} of {t['runs']} runs ({t['configs_refused']} configurations)")
            for g in report["refusals"]:
                print(f"  {g['runs']} runs, {g['configs']} configurations: {g['reason']}")
        if not args.dry_run:
            print(f"wrote {os.path.join(args.out, report['name'])}/ in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
