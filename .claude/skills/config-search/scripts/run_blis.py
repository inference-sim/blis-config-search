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
# Pinned stack: BLIS v0.10.0 @ 0d092d4 (kernel v0.1.0 in go.mod), blis-catalog 0.2.1,
# blis-registry v0.1.1 (holds the 6 coefficient sets below, incl cost-model-memory).
import argparse, hashlib, json, os, subprocess, sys, tempfile, time

COEFFICIENTS = ["cost-model-primitives", "cost-model-collectives", "cost-model-host-overheads",
                "cost-model-attention", "cost-model-recurrent", "cost-model-memory"]

# --- candidate-key classification (allowlist; anything else is rejected, never silently ignored) ---
SCENARIO_PARALLEL = ("tp", "pp", "dp", "enable_expert_parallel")       # -> scenario Deployment.parallel
SCENARIO_ENGINE = ("quantization", "cache_dtype", "block_size", "max_num_batched_tokens",
                   "max_num_seqs", "max_model_len", "cudagraph_mode", "gpu_memory_utilization")
# Speculative decoding IS priced by the kernel (v0.12.0 + registry v0.2.0) -- the earlier runner
# refused it as "not yet validated on the kernel backend", which was true of v0.10.0 and is stale.
# It is a nested `speculative: {method, num_spec_tokens}` block on the pool's engine
# (blis-schemas v0.5.0 Deployment.Pool.Engine.Speculative), so the two candidate keys below are
# folded into that block rather than emitted as flat engine fields.
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
# DEPLOYMENT-level `router:` block (blis-schemas v0.5.0 spec/deployment/deployment.go:49) -- see
# ROUTING_MOVED below for why this replaced the former --routing-policy flag.
SCENARIO_ROUTER = ("router_config",)
PARALLEL_DEFAULTS = {"tp": 1, "pp": 1, "dp": 1, "enable_expert_parallel": False}
GPUS_PER_NODE = 8   # single-node H100; a per-instance footprint beyond this spans nodes (needs a fabric)
# Cross-node cost resolves from the fabric (a catalog networks/ entry), required once nodes > 1.
DEFAULT_FABRIC = "ib-400g"   # InfiniBand NDR 400 Gb/s; the standard H100 multi-node interconnect
# NON-PRECISION engine defaults only. quantization/cache_dtype are intentionally NOT defaulted: an
# empty precision lets the checkpoint format govern (e.g. qwen3-14b stays BF16), so forcing FP8 here
# would silently mis-price every non-FP8 model. `block_size` is also absent here because its right
# value depends on the MODEL -- see default_block_size(), which fills it in per run.
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
# Deliberately unsupported in this runner -> fail loudly with the reason (do not silently drop).
UNSUPPORTED = {
    "routing_policy": ROUTING_MOVED,
    "routing_scorers": ROUTING_MOVED,
    "admission_policy": ROUTING_MOVED,
    "prefill_routing_scorers": ROUTING_MOVED,
    "decode_routing_scorers": ROUTING_MOVED,
    "prefill_instances": "PD disaggregation is not supported yet (single colocated pool only)",
    "decode_instances": "PD disaggregation is not supported yet (single colocated pool only)",
    "prefill_tp": "per-role PD TP is not supported yet (single colocated pool only)",
    "decode_tp": "per-role PD TP is not supported yet (single colocated pool only)",
    "long_prefill_token_threshold": "not a kernel scenario field (chunked-prefill threshold is engine-internal)",
}


class ConfigError(ValueError):
    pass


def classify_config(config):
    """Split a candidate into (parallel, engine, cli, router); raise ConfigError on unknown keys.

    The speculative keys are folded into engine["speculative"], the nested block the schema wants,
    and the acceptance selectors go to the CLI.
    """
    parallel, engine, cli, router = dict(PARALLEL_DEFAULTS), dict(ENGINE_DEFAULTS), {}, {}
    spec = {}
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
        elif k in CLI_FLAGS:
            cli[k] = v
        elif k in UNSUPPORTED:
            raise ConfigError(f"{k!r}: {UNSUPPORTED[k]}")
        else:
            supported = (sorted(SCENARIO_PARALLEL) + sorted(SCENARIO_ENGINE)
                         + sorted(SCENARIO_SPECULATIVE) + sorted(SPECULATIVE_CLI)
                         + sorted(SCENARIO_ROUTER) + sorted(CLI_FLAGS))
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
    if parallel["pp"] != 1:
        raise ConfigError("pp>1: the pinned kernel does not price pipeline parallelism")
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
    return parallel, engine, cli, router


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


def parse_metrics(stdout):
    cands = _all_metrics(stdout)
    for o in cands:
        if o.get("instance_id") == "cluster":
            return o
    return cands[0] if cands else {}


def parse_all_instances(stdout):
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


def scenario_yaml(model, hardware, engine_version, parallel, engine, name, num_instances=1,
                  fabric=DEFAULT_FABRIC, router=None):
    """Render a two-doc Scenario+Deployment YAML (no external yaml dep). Omitted engine keys (e.g.
    quantization when unset) let the checkpoint/engine govern."""
    def block(d, indent):
        """Render a flat mapping, plus the one NESTED value the engine takes (`speculative`).

        json.dumps on a dict would emit JSON-flow style, which the strict loader rejects for this
        field, so a dict value is rendered as an indented sub-block instead.
        """
        pad = " " * indent
        out = []
        for k, v in d.items():
            if isinstance(v, dict):
                out.append(f"{pad}{k}:")
                out.extend(f"{pad}  {kk}: {json.dumps(vv)}" for kk, vv in v.items())
            else:
                out.append(f"{pad}{k}: {json.dumps(v)}")
        return "\n".join(out)
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
    """The CLI flag for a cli-slot candidate key: a deployment knob or an acceptance selector."""
    try:
        return CLI_FLAGS[key]
    except KeyError:
        return SPECULATIVE_CLI[key]


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
    fp = footprint_gpus(parallel)
    # Fill block_size per MODEL when the candidate did not pin one (see default_block_size): the
    # adapter requires a value, and DSA models require it to be a multiple of 64.
    engine.setdefault("block_size", default_block_size(catalog, model, hardware))
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
        "deployment": {"parallel": parallel, "engine": engine, "num_instances": num_instances,
                       "footprint_gpus_per_instance": fp, "gpus": fp * num_instances,
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
