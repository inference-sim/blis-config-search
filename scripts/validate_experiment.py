#!/usr/bin/env python3
"""Validate an Experiment Spec (RFC #21) BEFORE any BLIS run -- the contract's gatekeeper.

    python3 -I scripts/validate_experiment.py experiments/foo.yaml --blis ./blis
    python3 -I scripts/validate_experiment.py experiments/ --blis ./blis    # every spec
    python3 -I scripts/validate_experiment.py experiments/foo.yaml --blis ./blis --json

Exit 0 when every spec is valid, 1 when one is not, with one `file: path: message` line per
error, and 2 when a simulator command fails. Safe for an untrusted PR: it reads YAML with
safe_load, touches no network, and runs only the simulator's read-only commands.

WHY THIS EXISTS SEPARATELY FROM THE SCHEMA. experiments/schema.json describes STRUCTURE (every
value is a list, a sweep is colocated or disaggregated but not both) for a contributor's EDITOR.
This script never loads it -- the structural rules below are the ENFORCED copy, and
test_experiment.py's SchemaAgreesWithValidator is what stops the two definitions drifting apart.
Three classes of error no schema can reach are exactly the ones that waste a whole sweep:

  1. a setting name that the engine does not have. The names are not ours to invent: an
     `engine:` key is a field of the engine's field list, which `blis fields` prints at the
     release the binary runs (CFG-3), and a `parallel:` key is a field of the deployment
     document, which `blis schema deployment` prints (SUB-19). A typo like `max_num_seq` would
     fail every run, so every name is checked against those two.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_cli  # noqa: E402
import settings  # noqa: E402
import pareto  # noqa: E402
import router_configs  # noqa: E402
import slo as slo_keys  # noqa: E402
import catalog_drafters  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ROUTING_DIR = os.path.join(ROOT, "routing")
# A router file or workload file named with this prefix is beside the spec: ./routing/x.yaml.
SPEC_PREFIX = router_configs.SPEC_PREFIX

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
# The scenario.replay keys the runner passes to BLIS (run_blis.build_replay_cmd); any other key would be
# ignored, so it is refused. A workload file states the scenario, the digest and the context filter
# itself, so beside one those three are not read either.
REPLAY_KEYS = frozenset({"scenario", "corpus", "corpus_sha256", "corpus_cache", "max_context_length",
                         "benchmark_duration", "grace_period", "server_series_interval_s"})
REPLAY_KEYS_WITH_FILE = REPLAY_KEYS - {"scenario", "corpus_sha256", "max_context_length"}
DURATION_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(us|ms|s)$")

# ---------------------------------------------------------------------------------------------
# Knob names. None is kept here. Engine setting names are the engine's own, read from the
# simulator's field list (`blis fields`, CFG-3); pool-layout names are the deployment document's,
# read from its published schema (`blis schema deployment`, SUB-19). Value checks belong to the
# engine, which refuses a bad value at start-up with its own message (X15).

# Engine settings whose value is a list of integers, so a spec sweeps a list of lists. Part of
# the spec's grammar: without it `[1, 2, 4]` could be three plans or one.
ENGINE_INT_LISTS = ("cudagraph_capture_sizes",)

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

def knob_vocabulary(blis, fields=None):
    """The settings.Vocabulary the binary `blis` publishes, reading the engine's field list from the
    file `fields` when one is named. Raises blis_cli.BlisCommandError."""
    return settings.vocabulary(blis, fields)


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
        if key in skip or key in vocabulary:
            continue
        article = "an" if kind == "engine" else "a"
        errs.append(f"{where}.{key}: not {article} {kind} setting in {source}"
                    f"{_suggest(key, vocabulary)}")


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


def _check_int_lists(engine, where, errs):
    """A list-valued setting is swept as a list OF lists; a flat list of ints is almost certainly
    one plan written without the outer brackets."""
    for key in ENGINE_INT_LISTS:
        values = (engine or {}).get(key)
        if not _is_list(values):
            continue
        for v in values:
            if not _is_list(v):
                errs.append(f"{where}.{key}: {key} takes a LIST of sizes, so sweeping it "
                            f"needs a list of lists (write `{key}: [[{v}, ...]]`); got {v!r}")
            elif any(not isinstance(x, int) or isinstance(x, bool) for x in v):
                errs.append(f"{where}.{key}: every size must be an integer, got {v!r}")


def _check_cross_knob(sweep, scenario, where, errs):
    """Constraints that span two settings -- the ones BLIS does not catch.

    Measured against the pinned binary (pins.env): BLIS validates enum membership and simple
    ranges (`cudagraph_mode: BOGUS`, `cache_dtype: fp16` and `max_num_seqs: 0` are all refused
    at startup), but it does NOT check cross-knob agreement -- a `long_prefill_token_threshold`
    of 999,999 against a 32k context runs rc=0. So the 32 rules in blis-vllm's v0_32 pack are
    not all evaluated at run time. They are real vLLM behaviour even so -- the pack transcribes the engine's own startup
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


def _check_nested_block(block, where, fields, errs):
    """A nested engine block, such as `engram_config`: known inner fields, all lists.

    The engine decodes these strictly too, so an invented inner field is as fatal as an invented
    top-level setting. The inner names are the field list's `<block>.<field>` entries.
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


def f_valid(fields):
    """" (valid: a, b, c)" -- kept a helper so the message reads the same everywhere."""
    return " (valid: " + ", ".join(fields) + ")"


def _check_gpu_math_colocated(sweep, gpus, where, errs, node_size=None):
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
        straddling = [fp for fp, _ in ok if node_size and fp > node_size and fp % node_size]
        if len(straddling) == len(ok):
            errs.append(f"{where}: every surviving combo has a per-instance footprint that "
                        f"straddles a node boundary ({sorted(set(straddling))} GPUs on "
                        f"{node_size}-GPU nodes); a multi-node instance must fill whole nodes")
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


def spec_router_file(path):
    """A router file beside the spec, parsed and stripped: (doc, notes); see router_configs.prepare_spec_file.
    Raises RouterConfigsError, OSError or yaml.YAMLError."""
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise router_configs.RouterConfigsError("not an EndpointPickerConfig mapping")
    return router_configs.prepare_spec_file(doc)


def _check_router(router, where, errs, routing_dir=ROUTING_DIR, router_cfg=None, spec_dir=None):
    """Router files must exist, and every override combination must apply to its base file.

    A missing EndpointPickerConfig is the cheapest possible catch: BLIS would fail once per
    run in the sweep, all for the same typo. A file is the local routing/ one, one of the router's
    reference configs, or a ./ file beside the spec (parsed, with the parts the embedded router
    refuses named); the overrides are applied to the parsed file here, exactly as the runner
    will, so a knob the index does not list (or a decider the file cannot take) is rejected before
    any run.
    """
    resolved, beside = {}, {}
    for rel in (router.get("endpoint_picker_config") or []):
        if not isinstance(rel, str):
            continue
        if os.path.isabs(rel) or ".." in rel.split("/"):
            errs.append(f"{where}.endpoint_picker_config: {rel!r} must be a plain file name")
            continue
        try:
            resolved[rel] = router_configs.resolve(rel, routing_dir, router_cfg, spec_dir)
        except router_configs.RouterConfigsError as e:
            errs.append(f"{where}.endpoint_picker_config: {e}")
            continue
        if resolved[rel][0] == "spec":
            try:
                beside[rel] = spec_router_file(resolved[rel][1])[0]
            except (OSError, yaml.YAMLError, router_configs.RouterConfigsError) as e:
                del resolved[rel]
                errs.append(f"{where}.endpoint_picker_config: {rel}: {e}")
    for profile, refs in (router.get("schedulingProfiles") or {}).items():
        for ref, fields in (refs or {}).items():
            _check_value_lists(fields, f"{where}.schedulingProfiles.{profile}.{ref}", errs)
    indexed = any(src == "router" for src, _p, _k in resolved.values())
    for plugin, block in (router.get("plugins") or {}).items():
        known = ROUTER_PLUGIN_PARAMS.get(plugin)
        for section, fields in (block or {}).items():
            if section != "parameters":
                errs.append(f"{where}.plugins.{plugin}.{section}: only `parameters` may be "
                            f"swept inside a plugin")
                continue
            _check_value_lists(fields, f"{where}.plugins.{plugin}.parameters", errs)
            for field in (fields or {}):
                # The router's index is the authority for its files; the table is the local files' subset.
                if known is not None and field not in known and not indexed:
                    errs.append(f"{where}.plugins.{plugin}.parameters.{field}: not a documented "
                                f"parameter of {plugin} (RFC #21 lists: "
                                f"{', '.join(sorted(known))})")
    if len(resolved) == 1 and (router.get("schedulingProfiles") or router.get("plugins")):
        rel, (_src, path, knobs) = next(iter(resolved.items()))
        try:
            if rel in beside:
                base = beside[rel]
            else:
                with open(path, encoding="utf-8") as f:
                    base = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            errs.append(f"{where}.endpoint_picker_config: cannot read {rel}: {e}")
            return
        seen = set()
        for combo in router_configs.override_combos(router):
            try:
                router_configs.apply_overrides(base, combo, knobs)
            except KeyError as e:
                msg = e.args[0] if e.args else str(e)
                if msg not in seen:
                    seen.add(msg)
                    errs.append(f"{where}.router: {rel}: {msg}")


def _node_size(sc, blis, catalog=None):
    """The node size a spec's scenario runs on, or None when it cannot be told (reported elsewhere)."""
    if not isinstance(sc.get("hardware"), str):
        return None
    try:
        fixed = blis_cli.chip_packaging(blis, sc.get("hardware"), catalog)["gpus_per_node"]
    except blis_cli.BlisCommandError:
        return None
    cluster = sc.get("cluster")
    gpn = cluster.get("gpus_per_node") if isinstance(cluster, dict) else None
    return fixed or (gpn if isinstance(gpn, int) and not isinstance(gpn, bool) else None)


def scenario_identity_errors(sc, blis, catalog=None):
    """`model` and `hardware` that the catalog has no entry for, by `blis catalog resolve`.

    EXISTENCE, not feasibility. A name the catalog has no entry for cannot run: BLIS refuses
    every run of the sweep. Whether a KV cache fits stays BLIS's call.
    """
    errs = []
    for key, kind in (("model", "model"), ("hardware", "chip")):
        name = sc.get(key)
        if not name:
            continue
        identity, reason = blis_cli.resolve(blis, catalog=catalog, **{kind: name})
        if identity is None:
            errs.append(f"scenario.{key}: {name!r}: {reason}")
        elif identity != name:
            errs.append(f"scenario.{key}: {name!r} is not a catalog name; the catalog calls it "
                        f"{identity!r}")
    return errs


# What a workload file must state so a report stays reproducible (CFG-1, CFG-2): where the
# corpus comes from, and its digest. The simulator checks the digest of the bytes it replays.
SPEC_SOURCES = ("huggingface", "url")


# The AgentX run recipe (workloads/agentx-run-recipe.yaml): the fields an AgentX replay descriptor must
# state, each a (path in the document, what the run recorded) pair. A descriptor missing one is
# refused: BLIS then replays the scenario's default for it (no per-trajectory idle cap, no warm-up), a
# different request pattern from the run's, with nothing in the report to say so.
AGENTX_RECIPE = {
    "benchmark_duration_s": 3600,
    "grace_period_s": 30,
    "trace_idle_gap_cap_s": 300,
    "warmup.requests_per_lane": 10,
    "trajectory_start.min_ratio": 0.25,
    "trajectory_start.max_ratio": 0.75,
}
AGENTX_UNCAPPED_LOADER = "semianalysis_cc_traces_weka_062126"


def _dig(doc, dotted):
    for k in dotted.split("."):
        if not isinstance(doc, dict) or k not in doc:
            return None
        doc = doc[k]
    return doc


def is_agentx(doc):
    return str((doc or {}).get("scenario") or "").startswith("inferencex-agentx")


def recipe_missing(doc):
    """The AgentX recipe fields a replay descriptor does not state, [] when complete or not AgentX."""
    if not is_agentx(doc):
        return []
    missing = [k for k in AGENTX_RECIPE if _dig(doc, k) is None]
    if _dig(doc, "warmup.mode") not in (None, "cache_pressure"):
        missing.append("warmup.mode: cache_pressure")
    loader = _dig(doc, "corpus.loader")
    if loader is None:
        missing.append("corpus.loader")
    return missing


def recipe_differences(doc):
    """{field: [stated, the run's]} for each stated recipe field whose value is not the run's, and the
    corpus when it is not the uncapped one. A sweep may differ (it is a different experiment); the report
    says so."""
    if not is_agentx(doc):
        return {}
    out = {k: [_dig(doc, k), v] for k, v in AGENTX_RECIPE.items()
           if _dig(doc, k) is not None and _dig(doc, k) != v}
    loader = _dig(doc, "corpus.loader")
    if loader is not None and loader != AGENTX_UNCAPPED_LOADER:
        out["corpus.loader"] = [loader, AGENTX_UNCAPPED_LOADER]
    return out


def workload_spec_errors(path, root=ROOT, spec_dir=None):
    """Why the workload file at `path` cannot be a workload, else []. The file is read as the replay
    block of a catalog workload entry: `corpus.sha256` and `corpus.source` (a Hugging Face dataset
    and revision, or a URL) are required, so every report names the bytes it replayed. A ./ path is
    a file beside the spec, in `spec_dir`."""
    import yaml
    beside = path.startswith(SPEC_PREFIX)
    if beside and not spec_dir:
        return [f"scenario.workload_spec: {path!r} is beside the spec, but the spec's directory is not known"]
    root = spec_dir if beside else root
    where = "the spec's directory" if beside else "this repository"
    full = os.path.realpath(os.path.join(root, path))
    if not os.path.isabs(path) and not full.startswith(os.path.realpath(root) + os.sep):
        return [f"scenario.workload_spec: {path!r} is outside {where}"]
    if os.path.isabs(path):
        return [f"scenario.workload_spec: {path!r} is an absolute path, which pins the spec to one "
                f"machine -- use a path inside this repository, such as workloads/muse-cb-1m.yaml, "
                f"or a ./ path beside the spec"]
    if not os.path.isfile(full):
        return [f"scenario.workload_spec: {path!r}: no such file in {where}"]
    try:
        with open(full, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        return [f"scenario.workload_spec: {path!r}: {e}"]
    if not isinstance(doc, dict):
        return [f"scenario.workload_spec: {path!r} is not a mapping"]
    errs = []
    if not doc.get("scenario"):
        errs.append(f"scenario.workload_spec: {path!r}: `scenario` (the harness scenario id) is required")
    corpus = doc.get("corpus")
    if not isinstance(corpus, dict):
        return [f"scenario.workload_spec: {path!r} has no `corpus` mapping (source and sha256)"]
    sha = corpus.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        errs.append(f"scenario.workload_spec: {path!r}: corpus.sha256 is required, 64 lowercase hex "
                    f"digits -- a report must name the bytes it replayed")
    gaps = recipe_missing(doc)
    if gaps:
        errs.append(f"scenario.workload_spec: {path!r}: an AgentX replay must state the whole run recipe "
                    f"(workloads/agentx-run-recipe.yaml); missing {', '.join(gaps)}")
    src = corpus.get("source")
    if not isinstance(src, dict) or not any(src.get(k) for k in SPEC_SOURCES):
        errs.append(f"scenario.workload_spec: {path!r}: corpus.source is required, a Hugging Face "
                    f"dataset and revision (huggingface, revision) or a url")
    elif src.get("huggingface") and not src.get("revision"):
        errs.append(f"scenario.workload_spec: {path!r}: corpus.source.revision is required with "
                    f"huggingface -- a dataset's default branch moves")
    return errs


def workload_errors(sc, blis=None, root=ROOT, spec_dir=None):
    """`workload` (a catalog name) or `workload_spec` (a file), one of them and never both."""
    named, filed = sc.get("workload"), sc.get("workload_spec")
    if named and filed:
        return ["scenario: state `workload` or `workload_spec`, never both -- a workload comes from "
                "the catalog or from a file"]
    if not named and not filed:
        return ["scenario.workload: required (a catalog name), or `workload_spec` (a file in this "
                "repository)"]
    if filed:
        if not isinstance(filed, str):
            return ["scenario.workload_spec: must be a path string"]
        return workload_spec_errors(filed, root, spec_dir)
    if str(named).startswith("agentx"):
        return [f"scenario.workload: {named!r} is the catalog's AgentX descriptor, which states no per-trajectory "
                f"idle-gap cap and no warm-up: name `workload_spec: workloads/agentx-run-recipe.yaml` (the full "
                f"run recipe) instead"]
    return []


def replay_errors(sc):
    """A `scenario.replay` key the runner never passes to BLIS (#197), and
    `server_series_interval_s`, when stated: a number of seconds above zero."""
    rep = sc.get("replay")
    if not isinstance(rep, dict):
        return []
    passed = REPLAY_KEYS_WITH_FILE if sc.get("workload_spec") else REPLAY_KEYS
    errs = [f"scenario.replay.{key}: never reaches BLIS (the runner passes only "
            f"{', '.join(sorted(passed))}), so the run would ignore it. The replay's pacing "
            f"(trace_idle_gap_cap_s, warmup, trajectory_start) is the workload file's: name a file "
            f"with scenario.workload_spec, which may be a ./ path beside the spec"
            for key in sorted(set(rep) - passed)]
    if "server_series_interval_s" not in rep:
        return errs
    v = rep["server_series_interval_s"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
        errs.append("scenario.replay.server_series_interval_s: must be a number of seconds above 0; "
                    "leave it out to collect no series")
    return errs


def cluster_errors(sc, blis=None, catalog=None):
    """`scenario.cluster` structure: a mapping of `fabric` (a non-empty string) and
    `gpus_per_node` (a positive integer).

    The fabric's name is not checked here: no simulator command resolves a network name yet, and
    this repository reads no catalog file (design §5). BLIS refuses an unknown fabric on every
    run. Omitted means the runner's ib-400g default.
    """
    cluster = sc.get("cluster")
    errs = []
    if cluster is not None:
        if not isinstance(cluster, dict) or not cluster:
            errs.append("scenario.cluster: must be a non-empty mapping, e.g. {fabric: roce-200g}")
            return errs
        for key in cluster:
            if key not in ("fabric", "gpus_per_node"):
                errs.append(f"scenario.cluster.{key}: unknown key (the cluster knobs are `fabric` "
                            f"and `gpus_per_node`)")
        gpn = cluster.get("gpus_per_node")
        if "gpus_per_node" in cluster and (isinstance(gpn, bool) or not isinstance(gpn, int) or gpn < 1):
            errs.append("scenario.cluster.gpus_per_node: must be a positive integer")
        fabric = cluster.get("fabric")
        if "fabric" in cluster and (not isinstance(fabric, str) or not fabric):
            errs.append("scenario.cluster.fabric: must be a non-empty string naming a networks/ class "
                        "(ib-400g, roce-200g, ethernet-100gbe)")
    # The node size: the catalog's where it fixes one, else the specification's (#96).
    gpn = (cluster or {}).get("gpus_per_node") if isinstance(cluster, dict) else None
    if blis is not None and sc.get("hardware") and not errs:
        try:
            fixed = blis_cli.chip_packaging(blis, sc["hardware"], catalog)["gpus_per_node"]
        except blis_cli.BlisCommandError:
            fixed = None          # an unknown chip is reported by scenario_identity_errors
        else:
            if fixed and gpn is not None and gpn != fixed:
                errs.append(f"scenario.cluster.gpus_per_node: {gpn} contradicts the catalog, which fixes "
                            f"{sc['hardware']} at {fixed}; omit it")
            elif not fixed and gpn is None:
                errs.append(f"scenario.cluster.gpus_per_node: required, because the catalog fixes no node "
                            f"size for {sc['hardware']} (an 8-GPU HGX server: 8)")
    return errs
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
    # (A workload named by a file states these in the file, and the spec still must not.) A key the
    # runner never passes is refused by replay_errors instead.
    passed = REPLAY_KEYS_WITH_FILE if (spec.get("scenario") or {}).get("workload_spec") else REPLAY_KEYS
    for key in sorted(set(rep) & CATALOG_OWNED_REPLAY_FIELDS & passed):
        notes.append(
            f"scenario.replay.{key}: already stated by the workload's descriptor, which "
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
        router = sw.get("router") if isinstance(sw.get("router"), dict) else {}
        bases = router.get("endpoint_picker_config")
        for rel in (bases if path and isinstance(bases, list) else []):
            if isinstance(rel, str) and rel.startswith(SPEC_PREFIX):
                try:
                    found = router_configs.resolve(rel, None, None, os.path.dirname(os.path.abspath(path)))
                    stripped = spec_router_file(found[1])[1]
                except (OSError, yaml.YAMLError, router_configs.RouterConfigsError):
                    continue            # validate() reports it
                notes += [f"{where}.router.endpoint_picker_config: {rel}: {n}" for n in stripped]
        text = " ".join(str(sw.get("narrative") or "").split())
        if len(text) > NARRATIVE_CHARS:
            # The REASON is printed once per run by main(), not once per sweep: repeating it 51
            # times buries the file names it is meant to point at.
            notes.append(f"{where}.narrative: {len(text)} characters")
    return notes


def runner_refusals(sw, sc, vocab, node_size=None, pool_offload=False):
    """(arms, {reason: arms}) for a sweep: how many candidates it expands to and, for those the
    runner would refuse before any simulator run, the reason (run_blis.classify_config's, which
    mirrors the engine's own pool rules). Arms the engine alone can refuse (a model-dependent
    cache group, say) are only known by running them, and are reported as refused with the
    engine's reason."""
    import collections
    import run_blis
    import run_experiment
    kept, _dropped = run_experiment.expand_sweep(sw, sc["gpus"])
    reasons = collections.Counter()
    for cand in kept:
        cand = {k: v for k, v in cand.items() if k not in ("router_config", "router_overrides")}
        try:
            run_blis.classify_config(cand, vocab, node_size, pool_offload)
        except run_blis.ConfigError as e:
            reasons[str(e)] += 1
    return len(kept), dict(reasons)


def refusal_notes(spec, blis, fields=None, catalog=None):
    """Notes for sweeps where some, but not all, arms are refused by the runner: they run as
    `refused` rows, not as failures to meet an SLO, and the report counts them by reason."""
    if not isinstance(spec, dict) or not isinstance(spec.get("sweeps"), list):
        return []
    sc = spec.get("scenario")
    if not isinstance(sc, dict) or not isinstance(sc.get("gpus"), int):
        return []
    vocab = knob_vocabulary(blis, fields)
    notes = []
    for i, sw in enumerate(spec["sweeps"]):
        if not isinstance(sw, dict):
            continue
        try:
            n, reasons = runner_refusals(sw, sc, vocab, _node_size(sc, blis, catalog),
                                      blis_cli.supports_pool_offload(blis))
        except Exception:
            continue                    # a malformed sweep is validate()'s to report
        bad = sum(reasons.values())
        if 0 < bad < n:
            where = f"sweeps[{i}]({sw.get('name', '?')})"
            top = max(reasons, key=reasons.get)
            notes.append(f"{where}: {bad} of {n} arms are refused by the runner and will be reported "
                         f"as refused: {top}")
    return notes


def objective_errors(obj):
    """Errors in the optional top-level `objective:` block (see pareto.py)."""
    if obj is None:
        return []
    if not isinstance(obj, dict) or not obj:
        return ["objective: must be a non-empty mapping"]
    errs = []
    for key in obj:
        if key == "tie_tolerance":
            errs.append("objective.tie_tolerance: removed. A fixed tolerance cannot say two deployments "
                        "are not distinguished; that needs paired replications with an interval on the "
                        "difference (design DEV-11), so points are compared exactly until then")
        elif key not in pareto.DEFAULT_OBJECTIVE:
            errs.append(f"objective.{key}: unknown key (allowed: {', '.join(sorted(pareto.DEFAULT_OBJECTIVE))})")
    if obj.get("ranking") is not None and obj["ranking"] not in pareto.RANKINGS:
        errs.append(f"objective.ranking: {obj['ranking']!r} must be one of {', '.join(pareto.RANKINGS)}")
    if obj.get("throughput") is not None and obj["throughput"] not in pareto.THROUGHPUT_FORMS:
        errs.append(f"objective.throughput: {obj['throughput']!r} must be one of "
                    f"{', '.join(pareto.THROUGHPUT_FORMS)}")
    return errs


def simulator_errors(sw, where):
    """Errors in a sweep's `simulator:` block: simulator switches, each a list of booleans. The
    simulator refuses --pd-transfer-contention on a run that is not disaggregated, so a colocated
    sweep may not turn it on."""
    block = sw.get("simulator")
    if block is None:
        return []
    import run_blis
    if not isinstance(block, dict) or not block:
        return [f"{where}.simulator: must be a non-empty mapping"]
    errs = []
    for key, vals in block.items():
        if key not in run_blis.CLI_SWITCHES:
            errs.append(f"{where}.simulator.{key}: unknown simulator option "
                        f"(allowed: {', '.join(sorted(run_blis.CLI_SWITCHES))})")
        elif not _is_list(vals) or not vals or any(not isinstance(v, bool) for v in vals):
            errs.append(f"{where}.simulator.{key}: must be a non-empty list of true/false")
        elif key == "pd_transfer_contention" and True in vals and _sweep_is_colocated(sw):
            errs.append(f"{where}.simulator.pd_transfer_contention: true needs a disaggregated sweep "
                        f"(prefill/decode pools); the simulator refuses the flag on a colocated run")
    return errs


_SIZE_RE = None
_SIZE_UNITS = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40,
               "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}


def parse_size(v):
    """Bytes of a size: 0, a positive integer of bytes, or a number with a unit ("512GiB"). None if it
    is none of these (a bare number above zero has no unit and is refused as ambiguous only when it
    is a string; an integer is bytes)."""
    import re
    global _SIZE_RE
    if _SIZE_RE is None:
        _SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(" + "|".join(_SIZE_UNITS) + r")\s*$")
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    m = _SIZE_RE.match(v) if isinstance(v, str) else None
    return int(float(m.group(1)) * _SIZE_UNITS[m.group(2)]) if m else None


# The storage device classes the pinned registry prices, the fallback when no catalog checkout is
# present to read `devices/storage.yaml` from. cpu_dram is the primary tier (stated with cpu_bytes),
# so a `tiers` entry names one of the others.
PRICED_STORAGE_DEVICES = ("cpu_dram", "nvme_gen3", "nvme_gen4", "nvme_gen4_raid0_8x", "sata_ssd", "s3")


def _storage_devices(catalog):
    """The storage device names the catalog lists (`devices/storage.yaml`), or the priced fallback
    when no catalog directory is given or the file cannot be read. A missing checkout never fails
    validation: the fallback is the set the registry prices."""
    if catalog:
        path = os.path.join(catalog, "devices", "storage.yaml")
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except OSError:
            data = None
        if isinstance(data, dict) and data:
            return set(data)
    return set(PRICED_STORAGE_DEVICES)


def offload_errors(sw, where, pool_offload=False, catalog=None):
    """Errors in a sweep's `offload:` blocks: the KV tier hierarchy, off as the control.

    cpu_bytes: sizes with units, 0 for no tier; eviction_policy: names the engine validates;
    tiers: extra fs/obj device tiers below the CPU tier; p2p: a p2p secondary tier. On a colocated
    sweep the block is the pool's tiers. On a disaggregated one it is the default for both pools,
    and `prefill.offload` / `decode.offload` state one pool's own (replacing the default for it;
    `cpu_bytes: [0]` is a pool with no tier). That needs a simulator whose deployment schema lets a
    pool state its offload (`pool_offload`, blis-substrate#307).
    """
    errs = []
    blocks = [("offload", sw.get("offload"))]
    for role in ("prefill", "decode"):
        if isinstance(sw.get(role), dict):
            blocks.append((f"{role}.offload", sw[role].get("offload")))
    devices = _storage_devices(catalog)
    for name, block in blocks:
        if block is None:
            continue
        errs.extend(_offload_block_errors(block, f"{where}.{name}", devices))
        if _sweep_is_colocated(sw):
            continue
        if not pool_offload:
            errs.append(f"{where}.{name}: this simulator's deployment schema has no per-pool offload "
                        f"(blis-substrate#307), so a disaggregated sweep cannot state a tier")
        elif isinstance(block, dict) and ("tiers" in block or "p2p" in block):
            # A disaggregated pool's secondary (fs/obj) tier or p2p option renders into the pool's
            # engine `kv_offload` alongside a MultiConnector (the pool's own NIXL + CPU-tier pair,
            # run_blis._pd_scenario_yaml); the pinned simulator's splitOffload does not move either
            # key into the offloading child's own kv_connector_extra_config, so vLLM refuses the pair
            # outright -- verified against the pinned binary (v0.19.0, e7597a8c): "vllm: kv_offload.
            # secondary_tiers beside a MultiConnector: the offloading child's own kv_connector_extra_
            # config states it as vLLM does; state it there once" (engines/vllm/core/vllm/
            # multiconnector.go:217). Not fixable from the runner side (it cannot place the key in the
            # child's own config without duplicating the simulator's own derived offload values), so a
            # disaggregated sweep's tier is cpu_bytes only (the primary CPU tier, which works) until the
            # simulator closes this (blis-config-search#94).
            errs.append(f"{where}.{name}: a disaggregated pool's tiers/p2p option fails at runtime "
                        f"against a MultiConnector (the pinned simulator's splitOffload leaves "
                        f"secondary_tiers at the top level instead of the offloading child's own "
                        f"kv_connector_extra_config; engines/vllm/core/vllm/multiconnector.go:217), "
                        f"so a disaggregated sweep's tier is cpu_bytes only (blis-config-search#94)")
    return errs


def _offload_tiers_errors(vals, where, devices):
    """Errors in a sweep's `offload.tiers`: a fixed list (not swept) of {tier: <device>, bytes}
    hierarchy entries below the primary CPU tier. Each device is a catalog storage class other
    than cpu_dram (the primary tier, stated with cpu_bytes)."""
    if not _is_list(vals) or not vals:
        return [f"{where}: must be a non-empty list of {{tier: <device>, bytes: <size>}} entries"]
    errs = []
    for i, t in enumerate(vals):
        at = f"{where}[{i}]"
        if not isinstance(t, dict) or set(t) - {"tier", "bytes"} or "tier" not in t or "bytes" not in t:
            errs.append(f"{at}: must be a mapping of exactly tier and bytes")
            continue
        dev = t["tier"]
        if dev == "cpu_dram":
            errs.append(f"{at}.tier: cpu_dram is the primary tier; state its size with cpu_bytes, "
                        f"not as a tiers entry")
        elif not isinstance(dev, str) or dev not in devices:
            errs.append(f"{at}.tier: {dev!r} is not a catalog storage device "
                        f"(known: {', '.join(sorted(devices))})")
        if parse_size(t["bytes"]) in (None, 0):
            errs.append(f"{at}.bytes: {t['bytes']!r} is not a non-zero size: use a number with a unit "
                        f"such as 2TiB")
    return errs


def _offload_block_errors(block, where, devices=None):
    if devices is None:
        devices = set(PRICED_STORAGE_DEVICES)
    if not isinstance(block, dict) or not block:
        return [f"{where}: must be a non-empty mapping"]
    errs = []
    for key, vals in block.items():
        if key not in ("cpu_bytes", "eviction_policy", "tiers", "p2p"):
            errs.append(f"{where}.{key}: unknown offload option "
                        f"(allowed: cpu_bytes, eviction_policy, tiers, p2p)")
        elif key == "tiers":
            errs.extend(_offload_tiers_errors(vals, f"{where}.tiers", devices))
        elif key == "p2p":
            if not isinstance(vals, bool):
                errs.append(f"{where}.p2p: must be true or false (a p2p secondary tier, on or off)")
        elif not _is_list(vals) or not vals:
            errs.append(f"{where}.{key}: must be a non-empty list")
        elif key == "cpu_bytes":
            for v in vals:
                if parse_size(v) is None:
                    errs.append(f"{where}.cpu_bytes: {v!r} is not a size: use 0 or a number with a unit "
                                f"such as 512GiB")
        elif any(not isinstance(v, str) or not v for v in vals):
            errs.append(f"{where}.eviction_policy: must be a list of policy names")
    if "eviction_policy" in block and "cpu_bytes" not in block:
        errs.append(f"{where}.eviction_policy: needs cpu_bytes (a policy of no tier states nothing)")
    # A fs/obj tier or a p2p tier sits below the primary CPU tier, so it needs one: the engine puts
    # cpu_dram first and refuses secondary tiers without a primary (engines/vllm build/offload.go).
    if ("tiers" in block or "p2p" in block) and "cpu_bytes" not in block:
        errs.append(f"{where}: tiers and p2p sit below the CPU tier, so they need cpu_bytes "
                    f"(the primary tier)")
    return errs


# The keys of the acceptance and routing scenario pins, passed through to BLIS whole (it validates
# their values: acceptance.Pin/acceptance.Value, routing.Pin/routing.Value, at the pinned release).
# config-search checks only the key names, so a typo fails here rather than per-run in a whole sweep.
_ACCEPTANCE_PIN_KEYS = ("profiles", "value", "fallback_category", "exact_only")
_ACCEPTANCE_VALUE_KEYS = ("num_spec_tokens", "acceptance_length", "acceptance_rate",
                          "per_position_rates", "source", "realisation")
_ROUTING_PIN_KEYS = ("profiles", "value", "fallback_category")
_ROUTING_VALUE_KEYS = ("token_imbalance", "expert_imbalance", "source")


def _pin_errors(pin, where, pin_keys, value_keys):
    """Key-name errors in a scenario acceptance/routing pin (a mapping named by `where`). The pin
    either names a registry profile set (`profiles`) or states a value inline (`value`); BLIS
    checks that choice and every value, so config-search checks only that the keys are known."""
    if not isinstance(pin, dict) or not pin:
        return [f"{where}: must be a non-empty mapping (a profile set name or an inline value)"]
    errs = []
    for key in pin:
        if key not in pin_keys:
            errs.append(f"{where}.{key}: unknown key (allowed: {', '.join(pin_keys)})")
    value = pin.get("value")
    if value is not None:
        if not isinstance(value, dict) or not value:
            errs.append(f"{where}.value: must be a non-empty mapping")
        else:
            for key in value:
                if key not in value_keys:
                    errs.append(f"{where}.value.{key}: unknown key "
                                f"(allowed: {', '.join(value_keys)})")
    return errs


def _registry_profile_sets(registry, subdir):
    """Every profile-set document in <registry>/<subdir>/*.yaml ({"routing", "acceptance"}), each
    {"name", "profiles", ...} (the file's own fields passed through). A file that cannot be read or
    parsed is skipped -- the same leniency catalog_drafters.exact_profiles gives a checkout that is
    not there, since this is an optional, best-effort check (registry may be None)."""
    d = os.path.join(registry or "", subdir)
    try:
        files = sorted(n for n in os.listdir(d) if n.endswith((".yaml", ".yml")))
    except OSError:
        return []
    out = []
    for fn in files:
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(doc, dict) and doc.get("name"):
            out.append(doc)
    return out


def _profile_name_errors(sc, where, pin, subdir, registry):
    """`pin.profiles` (a registry set name) must name a set under <registry>/<subdir>/ that carries
    at least one profile for the scenario's model -- else every lookup at it fails or falls back
    silently (acceptance.resolve.go / routing.Resolve: a request whose ScenarioSet name does not
    match the set's own name is treated as naming no set). [] when there is no registry checkout to
    check against (--registry, SUBSTRATE_REGISTRY): BLIS itself refuses an unknown name at run
    time, just not until the whole sweep has already run."""
    if registry is None:
        return []
    name = pin.get("profiles") if isinstance(pin, dict) else None
    if not isinstance(name, str) or not name:
        return []
    sets = _registry_profile_sets(registry, subdir)
    by_name = {s.get("name"): s for s in sets}
    if name not in by_name:
        known = ", ".join(sorted(n for n in by_name if n)) or "none"
        return [f"{where}.profiles: {name!r} is not a {subdir} profile set in the registry "
                f"(known: {known})"]
    model = sc.get("model")
    if not isinstance(model, str) or not model:
        return []
    profiles = by_name[name].get("profiles") or []
    models = {p.get("model") for p in profiles if isinstance(p, dict)}
    if model not in models:
        return [f"{where}.profiles: {name!r} has no profile for model {model!r} "
                f"(models in that set: {', '.join(sorted(m for m in models if m)) or 'none'})"]
    return []


def scenario_pin_errors(sc, registry=None):
    """Errors in the scenario's acceptance/routing pins (#201). Both are optional and pass through
    to BLIS unchanged; routing has no CLI alternative (the flags were removed in v0.12.0).

    `registry` (a registry checkout directory, --registry/SUBSTRATE_REGISTRY) additionally checks a
    `routing` pin's `profiles` name against the registry's routing profile sets: a name that is not
    there, or carries no profile for the scenario's model, otherwise only fails per-run once a whole
    sweep has already run (acceptance is not checked here -- out of scope, see the PR review this
    closes). None skips that check."""
    errs = []
    if sc.get("acceptance") is not None:
        errs.extend(_pin_errors(sc["acceptance"], "scenario.acceptance",
                                _ACCEPTANCE_PIN_KEYS, _ACCEPTANCE_VALUE_KEYS))
    if sc.get("routing") is not None:
        errs.extend(_pin_errors(sc["routing"], "scenario.routing",
                                _ROUTING_PIN_KEYS, _ROUTING_VALUE_KEYS))
        if isinstance(sc["routing"], dict):
            errs.extend(_profile_name_errors(sc, "scenario.routing", sc["routing"], "routing", registry))
    return errs


def _acceptance_k_mismatch_errors(spec):
    """scenario.acceptance.value.num_spec_tokens, an inline K, must agree with every sweep's own
    swept `engine.speculative.num_spec_tokens` list when both are stated -- drafting's K is a
    whole-run property (the engine refuses pools, and by extension a run, that disagree on the draft
    width), so a swept K the inline acceptance value was not measured at prices those runs with the
    wrong curve, silently."""
    sc = spec.get("scenario") or {}
    acc = sc.get("acceptance")
    if not isinstance(acc, dict):
        return []
    value = acc.get("value")
    if not isinstance(value, dict):
        return []
    k = value.get("num_spec_tokens")
    if not isinstance(k, int) or isinstance(k, bool):
        return []
    errs = []
    for i, sw in enumerate(spec.get("sweeps") or []):
        if not isinstance(sw, dict):
            continue
        eng = sw.get("engine")
        ks = ((eng or {}).get("speculative") or {}).get("num_spec_tokens") if isinstance(eng, dict) else None
        if not _is_list(ks):
            continue
        mismatched = sorted({v for v in ks if isinstance(v, int) and not isinstance(v, bool) and v != k})
        if mismatched:
            errs.append(f"sweeps[{i}].engine.speculative.num_spec_tokens: {mismatched} differs from "
                        f"scenario.acceptance.value.num_spec_tokens ({k}); drafting's K is a "
                        f"whole-run property, so a swept value the inline acceptance was not "
                        f"measured at prices those runs with the wrong curve")
    return errs


def offload_policy_errors(spec, blis, fields=None, catalog=None):
    """Ask the engine whether it models each eviction policy a spec names (`blis fields --scenario` opens it),
    on a probe scenario of the spec's own model and hardware. The policy names belong to the engine (CFG-3).

    The probe's packaging is the catalog's where it fixes one (and then the scenario states nothing,
    as a run's does), else the specification's. A spec that gives the probe no packaging is
    refused here rather than probed on an assumed eight GPUs a node (#154), because a packaging
    refusal would hide the policy refusal this check exists to find."""
    import blis_cli
    import run_blis
    sc = spec.get("scenario") or {}
    errs = []
    node_size = states_node_size = None
    for i, sw in enumerate(spec.get("sweeps") or []):
        if not isinstance(sw, dict):
            continue
        blocks = [sw.get("offload")] + [(sw.get(r) or {}).get("offload") if isinstance(sw.get(r), dict) else None
                                       for r in ("prefill", "decode")]
        policies = [p for b in blocks if isinstance(b, dict) and _is_list(b.get("eviction_policy"))
                    for p in b["eviction_policy"]]
        if not policies:
            continue
        block = {"eviction_policy": policies}
        if node_size is None:
            cluster = sc.get("cluster")
            stated = cluster.get("gpus_per_node") if isinstance(cluster, dict) else None
            try:
                node_size, states_node_size = run_blis.cluster_node_size(
                    blis, catalog, sc.get("hardware"), stated)
            except (run_blis.ConfigError, blis_cli.BlisCommandError) as e:
                return [f"sweeps[{i}].offload.eviction_policy: cannot probe the engine without the "
                        f"hardware's packaging: {e}"]
        tps = ((sw.get("parallel") or {}).get("tp")
               or (((sw.get("prefill") or {}).get("parallel") or {}).get("tp")) or [1])
        # The largest tp: a smaller one may not hold the model, and the engine would refuse that, not the policy.
        tp = max([t for t in tps if isinstance(t, int) and not isinstance(t, bool)] or [1]) if _is_list(tps) else 1
        for policy in dict.fromkeys(p for p in block["eviction_policy"] if isinstance(p, str)):
            parallel = {"tp": tp, "pp": 1, "dp": 1, "offload": {"cpu_bytes": 2**30, "eviction_policy": policy}}
            body = run_blis.scenario_yaml(sc.get("model"), sc.get("hardware"), parallel,
                                          run_blis.engine_defaults(), "policy-probe",
                                          node_size=node_size, states_node_size=states_node_size)
            line = blis_cli.engine_refusal(blis, body, word="eviction_policy")
            if line:
                errs.append(f"sweeps[{i}].offload.eviction_policy: the engine refuses {policy!r}: {line}")
    return errs


def _sweep_is_colocated(sw):
    return not ({"prefill-instances", "decode-instances", "prefill", "decode"} & set(sw))


def _ask_field_list(blis, catalog, spec):
    """Ask the binary for the field list on a probe of the spec's model and hardware (#158): None when
    it answers or there is nothing to probe, else the problem, so a spec the build refuses is an
    error that validate returns, not a crash."""
    sc = spec.get("scenario") if isinstance(spec, dict) else None
    if not (isinstance(sc, dict) and all(isinstance(sc.get(k), str) and sc[k] for k in ("model", "hardware"))):
        return None   # nothing to probe: knob_vocabulary reuses a list already asked, or says why it has none
    if not isinstance(sc.get("engine") or {}, dict):
        return None   # validate reports the malformed engine block
    import run_blis
    try:
        run_blis.ask_field_list(blis, catalog, sc)
    except (blis_cli.BlisCommandError, run_blis.ConfigError) as e:
        return f"the field list: {e}"
    return None


def validate(spec, blis, routing_dir=ROUTING_DIR, catalog=None, fields=None, router_cfg=None,
             spec_dir=None, registry=None):
    """Every error in `spec`, as readable `path: message` strings. Empty means valid.

    `router_cfg` is the router's reference configs (router_configs.open_checkout) a spec may name;
    None reads them from `blis router-configs --json`. `spec_dir` is the spec's directory, where a
    ./ router file or workload file is read from.

    `blis` is the simulator binary whose commands supply the names (`blis fields`, `blis
    schema deployment`) and the catalog identities (`blis catalog resolve`). `catalog` names a
    catalog directory; None means the catalog the binary embeds. `registry` names a registry
    checkout (--registry/SUBSTRATE_REGISTRY) that additionally checks a `scenario.routing` pin's
    profile set name and model coverage; None skips that one check.

    Collects ALL errors rather than stopping at the first: a contributor fixing a spec by hand
    should see every problem in one run, not discover them one CI round-trip at a time.
    """
    errs = []
    if not isinstance(spec, dict):
        return ["spec must be a mapping"]
    if router_cfg is None:
        try:
            router_cfg = router_configs.open_checkout(blis=blis)
        except router_configs.RouterConfigsError as e:
            return [f"router configs: {e}"]
    kind = settings.spec_engine_kind(spec)
    if kind not in settings.ENGINE_KINDS:
        return [f"scenario.engine_kind: {kind!r} is none of {', '.join(settings.ENGINE_KINDS)}"]
    settings.use_engine(kind)
    problem = None if fields else _ask_field_list(blis, catalog, spec)
    try:
        vocab = knob_vocabulary(blis, fields)
    except blis_cli.BlisCommandError as e:
        return [problem or f"the field list: {e}"]
    pool_offload = blis_cli.supports_pool_offload(blis)
    par_vocab, eng_vocab, source = vocab.parallel, vocab.engine_names(), vocab.source

    name = spec.get("name")
    if not name:
        errs.append("name: required")
    elif not NAME_RE.match(str(name)):
        errs.append(f"name: {name!r} must be lowercase letters, digits and dashes "
                    f"(it is also the reports/<name>/ directory)")

    known_top = {"name", "description", "scenario", "sweeps", "objective"}
    for key in spec:
        if key not in known_top:
            errs.append(f"{key}: unknown top-level key (allowed: {', '.join(sorted(known_top))})")

    errs.extend(objective_errors(spec.get("objective")))
    sc = spec.get("scenario")
    if not isinstance(sc, dict):
        errs.append("scenario: required mapping (model, hardware, workload or workload_spec, gpus, concurrency)")
        sc = {}
    else:
        known_sc = {"model", "hardware", "workload", "workload_spec", "gpus", "slo", "concurrency",
                    "seed", "engine", "engine_kind", "replay", "cluster", "acceptance", "routing"}
        for key in sc:
            if key not in known_sc:
                errs.append(f"scenario.{key}: unknown key "
                            f"(allowed: {', '.join(sorted(known_sc))})")
        for key in ("model", "hardware"):
            if not sc.get(key):
                errs.append(f"scenario.{key}: required")
        errs.extend(workload_errors(sc, blis, spec_dir=spec_dir))
        # Present AND real: a name the catalog does not define cannot run anywhere.
        errs.extend(scenario_identity_errors(sc, blis, catalog))
        errs.extend(cluster_errors(sc, blis, catalog))
        errs.extend(replay_errors(sc))
        errs.extend(scenario_pin_errors(sc, registry))
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
                # The statistics are the simulator's metric registry's, never a list kept here.
                try:
                    registry = blis_cli.metric_registry(blis)
                except blis_cli.BlisCommandError as e:
                    errs.append(f"scenario.slo: the metric registry: {e}")
                else:
                    errs.extend(slo_keys.key_errors(slo, registry, parse_duration_ms))
        # scenario.engine holds settings every run carries, so its values are SINGLE, not lists.
        # Getting this backwards is easy, and a list here would be written into the scenario
        # verbatim and refused by the engine, so say which way round it goes.
        shared = sc.get("engine") or {}
        if not isinstance(shared, dict):
            errs.append(f"scenario.engine: must be a mapping of setting to value, not {shared!r}")
            shared = {}
        for key, val in shared.items():
            if _is_list(val) and key not in ENGINE_INT_LISTS:
                errs.append(f"scenario.engine.{key}: a shared setting takes ONE value, not a "
                            f"list (sweep it under a sweep's `engine:` instead)")
        _check_knob_names(shared, "scenario.engine", eng_vocab, "engine", errs,
                          source)
        for block, inner in vocab.blocks.items():
            val = shared.get(block)
            if isinstance(val, dict):
                for field in val:
                    if field not in inner:
                        errs.append(f"scenario.engine.{block}.{field}: not a field of this "
                                    f"block{_suggest(field, inner) or f_valid(sorted(inner))}")
        seed = sc.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
            errs.append("scenario.seed: must be a non-negative integer")

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
                    "prefill-instances", "decode-instances", "prefill", "decode", "simulator",
                    "offload"}
        errs.extend(offload_errors(sw, where, pool_offload, catalog))
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
        _check_engine(sw.get("engine") or {}, f"{where}.engine", vocab, errs)
        errs.extend(draft_model_errors(sw.get("engine") or {}, f"{where}.engine", sc.get("model"),
                                       blis, catalog))
        for role in ("prefill", "decode"):
            block = sw.get(role)
            if not isinstance(block, dict):
                if block is not None:
                    errs.append(f"{where}.{role}: must be a mapping with parallel/engine/offload")
                continue
            for key in block:
                if key not in ("parallel", "engine", "offload"):
                    errs.append(f"{where}.{role}.{key}: only `parallel`, `engine` and `offload` may be "
                                f"stated per role")
            _check_value_lists(block.get("parallel") or {}, f"{where}.{role}.parallel", errs)
            _check_knob_names(block.get("parallel") or {}, f"{where}.{role}.parallel", par_vocab,
                              "parallel", errs, source)
            _check_engine(block.get("engine") or {}, f"{where}.{role}.engine", vocab, errs)
            if "speculative" in (block.get("engine") or {}):
                errs.append(f"{where}.{role}.engine.speculative: drafting is one configuration per run, "
                            f"not per pool -- the engine refuses pools that disagree on the draft width "
                            f"or the method, so state `speculative` once in the sweep's `engine` block")

        errs.extend(simulator_errors(sw, where))
        router = sw.get("router")
        if router is not None:
            if not isinstance(router, dict) or not router:
                errs.append(f"{where}.router: must be a non-empty mapping")
            else:
                for key in router:
                    if key == "decider":
                        errs.append(f"{where}.router.decider: the decider is not a knob. Swapping it is "
                                    f"choosing a complete policy variant: list that policy's file in "
                                    f"endpoint_picker_config (for the prefix-based decider, the router's "
                                    f"pd-prefix-based.yaml)")
                    elif key not in ("endpoint_picker_config", "schedulingProfiles", "plugins"):
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
                _check_router(router, f"{where}.router", errs, routing_dir, router_cfg, spec_dir)

        if isinstance(sc.get("gpus"), int) and not isinstance(sc.get("gpus"), bool):
            if disagg:
                _check_gpu_math_pd(sw, sc["gpus"], where, errs)
            elif colocated:
                _check_gpu_math_colocated(sw, sc["gpus"], where, errs, _node_size(sc, blis, catalog))
        # Cross-knob and model-aware constraints LAST: they read the sweep as a whole, so they
        # run once the individual knobs are known to be well-formed.
        if isinstance(sc, dict):
            _check_cross_knob(sw, sc, where, errs)
        # A sweep whose every arm the runner refuses expands to no runs worth making: say so now.
        if not errs and isinstance(sc.get("gpus"), int) and not isinstance(sc.get("gpus"), bool):
            n, reasons = runner_refusals(sw, sc, vocab, _node_size(sc, blis, catalog),
                                      blis_cli.supports_pool_offload(blis))
            if n and sum(reasons.values()) == n:
                top = max(reasons, key=reasons.get)
                errs.append(f"{where}: the runner refuses every arm ({n}): {top}")
    errs.extend(_acceptance_k_mismatch_errors(spec))
    if not errs:
        errs.extend(offload_policy_errors(spec, blis, fields, catalog))
    return errs


def _check_engine(engine, where, vocab, errs):
    """Engine settings, with the nested blocks (`speculative`, `engram_config`, ...) as such."""
    _check_value_lists(engine, where, errs, allow_nested=tuple(vocab.blocks) + ENGINE_INT_LISTS)
    _check_knob_names(engine, where, vocab.engine_names(), "engine", errs, vocab.source)
    _check_int_lists(engine, where, errs)
    for block, fields in vocab.blocks.items():
        if block in (engine or {}):
            _check_nested_block(engine[block], f"{where}.{block}", sorted(fields), errs)
    # A swept setting the field list marks inert measures the same number in every arm.
    for key, val in (engine or {}).items():
        names = [f"{key}.{f}" for f in val] if isinstance(val, dict) else [key]
        for name in names:
            why = settings.inert_reason(vocab, name)
            if why:
                errs.append(f"{where}.{name}: {why}")
    spec = (engine or {}).get("speculative")
    # BOTH are required: BLIS refuses a drafting scenario whose acceptance does not resolve, and
    # acceptance is selected by the method AND the depth.
    if isinstance(spec, dict):
        for key in ("method", "num_spec_tokens"):
            if key not in spec:
                errs.append(f"{where}.speculative: needs both method and num_spec_tokens "
                            f"(missing {key})")


def draft_model_errors(engine, where, model, blis, catalog=None):
    """Each `speculative.draft_model` value that is not a catalog drafter of `model`.

    The name must resolve in the catalog (`blis catalog resolve`), and the link must hold both
    ways: `model` lists it under drafters and it names `model` in its targets. TEMPORARY
    (inference-sim/inference-sim#2141): the link is read from a catalog checkout (`catalog`, else
    $SUBSTRATE_CATALOG) because `blis catalog resolve` does not list drafters yet.
    """
    spec = (engine or {}).get("speculative")
    names = spec.get("draft_model") if isinstance(spec, dict) else None
    if not isinstance(names, list) or not model:
        return []
    at = f"{where}.speculative.draft_model"
    checkout = catalog or os.environ.get("SUBSTRATE_CATALOG")
    linked = None
    errs = []
    for name in names:
        if not isinstance(name, str) or not name:
            errs.append(f"{at}: {name!r} is not a catalog model name")
            continue
        if name == catalog_drafters.IN_GRAPH:
            errs.append(f"{at}: {name!r} is the model's own speculator; leave draft_model out to "
                        f"draft with it")
            continue
        identity, reason = blis_cli.resolve(blis, model=name, catalog=catalog)
        if identity is None:
            errs.append(f"{at}: {name!r}: {reason}")
            continue
        if identity != name:
            errs.append(f"{at}: {name!r} is not a catalog name; the catalog calls it {identity!r}")
            continue
        if not checkout or not os.path.isdir(os.path.join(checkout, "models")):
            errs.append(f"{at}: {name!r}: whether it drafts for {model} is read from a catalog "
                        f"checkout until `blis catalog resolve` lists drafters "
                        f"(inference-sim/inference-sim#2141); pass --catalog or set "
                        f"SUBSTRATE_CATALOG")
            continue
        if linked is None:
            linked = {d["draft_model"]: d for d in catalog_drafters.drafters(checkout, model)
                      if d["kind"] == "separate"}
        d = linked.get(name)
        if d is None:
            known = sorted(n for n, x in linked.items() if x["linked"])
            errs.append(f"{at}: {name!r} is not a drafter of {model} (its separate drafters: "
                        f"{', '.join(known) or 'none'})")
        elif not d["linked"]:
            errs.append(f"{at}: {name!r} is not a drafter of {model}: {d['why']}")
        else:
            errs.extend(_drafter_pairing_errors(spec, at, name, d))
    return errs


def _drafter_pairing_errors(spec, at, name, d):
    """A sweep crosses method x draft_model x num_spec_tokens, so a checkpoint must be paired with
    its own method: one sweep per drafter. The graph's num_spec_tokens is not a depth limit for an
    external drafter (the registry measures EAGLE3 and DFlash drafters above it), so K is not checked."""
    errs = []
    methods = spec.get("method")
    stray = sorted({m for m in methods if m != d["method"]}) if isinstance(methods, list) else []
    if stray:
        errs.append(f"{at}: {name!r} drafts with method {d['method']}, but the sweep also states "
                    f"{', '.join(map(str, stray))} (one sweep per drafter: a sweep crosses its lists)")
    return errs


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
    ap.add_argument("--router-configs", dest="router_configs", default=None,
                    help="developer override: a router checkout whose reference-configs/ replace the "
                         "ones the simulator embeds (default: $ROUTER_CONFIGS_DIR, else "
                         "`blis router-configs --json`)")
    ap.add_argument("--blis", default=os.environ.get("BLIS"),
                    help="the simulator binary whose commands supply setting names and catalog "
                         "identities (default: $BLIS)")
    ap.add_argument("--fields",
                    help="the engine's field list: the pinned blis-vllm release's fields.json (scripts/fetch_fields.py). "
                         "Default: ask the binary (`blis fields --scenario` on a probe of the spec's model and hardware)")
    ap.add_argument("--catalog",
                    help="a catalog directory for `blis catalog resolve` (default: the catalog "
                         "the binary embeds)")
    ap.add_argument("--registry", default=os.environ.get("SUBSTRATE_REGISTRY"),
                    help="a registry checkout (substrate/registry layout) to check a "
                         "scenario.routing pin's profile set name and model coverage against "
                         "(default: $SUBSTRATE_REGISTRY; omit to skip that check)")
    args = ap.parse_args(argv)

    paths = spec_paths(args.target)
    if not paths:
        print(f"no spec files under {args.target}", file=sys.stderr)
        return 1
    first, _ = load(paths[0])
    if len(paths) == 1:
        if isinstance(first, dict) and settings.spec_engine_kind(first) in settings.ENGINE_KINDS:
            settings.use_engine(settings.spec_engine_kind(first))
    try:
        if not args.fields:
            # The first spec whose probe the binary accepts; validate() reports each one it refuses.
            for path in paths:
                doc, _ = load(path)
                if (isinstance(doc, dict) and settings.spec_engine_kind(doc) == settings.ENGINE_KIND
                        and not _ask_field_list(args.blis, args.catalog, doc)):
                    break
        vocab = knob_vocabulary(args.blis, args.fields)
    except blis_cli.BlisCommandError as e:
        print(f"validate_experiment: {e}", file=sys.stderr)
        return 2
    try:
        router_cfg = router_configs.open_checkout(args.router_configs, blis=args.blis)
    except router_configs.RouterConfigsError as e:
        print(f"validate_experiment: {e}", file=sys.stderr)
        return 2
    if router_cfg is not None and router_cfg.problem:
        print(f"WARNING: router configs: {router_cfg.problem}", file=sys.stderr)
    report, notes, failed = {}, {}, 0
    for path in paths:
        # Inside the repo a relative path is what a contributor recognises; outside it (a spec
        # under /tmp, say) `relpath` climbs out with ../.. noise, so keep the path as given.
        rel = os.path.relpath(path, ROOT)
        if rel.startswith(os.pardir):
            rel = path
        spec, err = load(path)
        try:
            errs = [err] if err else validate(spec, args.blis, args.routing, args.catalog, args.fields, router_cfg,
                                              os.path.dirname(os.path.abspath(path)), registry=args.registry)
        except blis_cli.BlisCommandError as e:
            print(f"validate_experiment: {rel}: {e}", file=sys.stderr)
            return 2
        report[rel] = errs
        notes[rel] = [] if err else advisories(spec, path) + (
            [] if errs else refusal_notes(spec, args.blis, args.fields, args.catalog))
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
        print(f"\n{len(paths) - failed}/{len(paths)} valid (setting names checked against "
              f"{vocab.source}; model and hardware checked with `blis catalog resolve`)")
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
