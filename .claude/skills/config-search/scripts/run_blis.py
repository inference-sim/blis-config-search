#!/usr/bin/env python3
# run_blis.py -- the ONE place config-search invokes BLIS and records a single run.
#
#   python3 -I scripts/run_blis.py ...        (importable: sweep_pool.py reuses run_one())
#
# BLIS is driven on the **blis-latency-kernel** backend (the accurate per-operator model). The kernel
# reads the deployment (model, hardware, tp/pp/dp/ep, engine knobs) from a committed SCENARIO, and
# REFUSES --model/--hardware/--tp/--dp/--enable-expert-parallel on the CLI. So a candidate's
# per-instance topology is written into a generated Scenario+Deployment YAML; only the things the
# scenario does not state -- workload, concurrency, instance counts, seed -- stay on the CLI.
#
# Pinned stack: BLIS PR #1851 @ 99bec1b6 (kernel v0.1.0 in go.mod), blis-catalog 0.2.1,
# blis-registry v0.1.1 (holds the 6 coefficient sets below, incl cost-model-memory).
#
# `blis run` prints a "=== Simulation Metrics ===" header then one JSON object per instance (+ a
# "cluster" aggregate); parse_metrics() extracts the cluster object, parse_all_instances() keeps the
# per-instance ones (needed for idle-pool / imbalance diagnosis, disc #3 Regime 2).
import argparse, hashlib, json, os, subprocess, sys, time

# Coefficient sets a scenario loads from blis-registry v0.1.1 (cost-model-memory is required for
# cudagraph memory occupancy; it is in v0.1.1 but not every older clone).
COEFFICIENTS = ["cost-model-primitives", "cost-model-collectives", "cost-model-host-overheads",
                "cost-model-attention", "cost-model-recurrent", "cost-model-memory"]

# config keys describing ONE instance's deployment -> written into the scenario (NOT CLI flags).
PARALLEL_KEYS = ("tp", "pp", "dp", "enable_expert_parallel")
ENGINE_KEYS = ("quantization", "cache_dtype", "block_size", "max_num_batched_tokens",
               "max_num_seqs", "max_model_len", "cudagraph_mode", "gpu_memory_utilization")
PARALLEL_DEFAULTS = {"tp": 1, "pp": 1, "dp": 1, "enable_expert_parallel": False}
ENGINE_DEFAULTS = {"quantization": "fp8", "cache_dtype": "fp8", "block_size": 16,
                   "max_num_batched_tokens": 8192, "max_num_seqs": 256, "max_model_len": 32768,
                   "cudagraph_mode": "PIECEWISE", "gpu_memory_utilization": 0.9}
# config keys that stay on the CLI (instance count + load); the kernel takes these, not the scenario.
CLI_INT_FLAGS = {"num_instances": "--num-instances", "concurrency": "--concurrency",
                 "prefill_instances": "--prefill-instances", "decode_instances": "--decode-instances"}


def _iter_json_objects(text):
    """Yield each top-level JSON object in text (string/escape-aware via raw_decode), tolerating prose."""
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
    """Collect every dict carrying 'tokens_per_sec', recursing into nested dicts/lists."""
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
    """The cluster-aggregate metrics object (or the first metrics dict) from `blis run` stdout."""
    cands = _all_metrics(stdout)
    for o in cands:
        if o.get("instance_id") == "cluster":
            return o
    return cands[0] if cands else {}


def parse_all_instances(stdout):
    """Per-instance metrics objects (excludes the cluster aggregate) -- for idle-pool/imbalance checks."""
    return [o for o in _all_metrics(stdout) if o.get("instance_id") not in (None, "cluster")]


def _completion_issue(m):
    """Why a run is NOT a usable capacity/ranking point (incomplete / overloaded), else None.

    p99 latencies describe only the COMPLETED requests, so an overloaded point -- one that queued,
    dropped, timed out, or left requests running/unserved -- can appear to satisfy the floor and be
    picked as C*. (length_capped_requests is normal output truncation, not an overload signal.)
    """
    inj, comp = m.get("injected_requests"), m.get("completed_requests")
    if inj is not None and comp is not None and inj != comp:
        return f"incomplete: injected={inj} != completed={comp}"
    for f in ("still_queued", "still_running", "dropped_unservable", "timed_out_requests"):
        v = m.get(f)
        if v:
            return f"overloaded: {f}={v}"
    return None


def _deployment(config):
    """Split a candidate config into the scenario's parallel/engine blocks (defaults filled in)."""
    parallel = dict(PARALLEL_DEFAULTS)
    engine = dict(ENGINE_DEFAULTS)
    for k, v in config.items():
        if k in PARALLEL_KEYS:
            parallel[k] = v
        elif k in ENGINE_KEYS:
            engine[k] = v
    return parallel, engine


def scenario_yaml(model, hardware, engine_version, parallel, engine, name):
    """Render a two-doc Scenario+Deployment YAML the kernel backend prices (no external yaml dep)."""
    def block(d, indent):
        pad = " " * indent
        out = []
        for k, v in d.items():
            out.append(f"{pad}{k}: {json.dumps(v)}")  # json.dumps -> true/false, quoted strings
        return "\n".join(out)
    nodes = max(1, -(-parallel["tp"] // 8))  # ceil(tp/8); single-node H100 is 8 GPUs
    return f"""kind: Scenario
name: {name}
engine_version: "{engine_version}"
model: {model}
coefficients: [{", ".join(COEFFICIENTS)}]
cluster:
  hardware: {hardware}
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
{block(engine, 6)}
"""


def write_scenario(scenarios_dir, model, hardware, engine_version, parallel, engine):
    """Write (idempotently) a scenario file for this per-instance deployment; return its file name."""
    key = json.dumps([model, hardware, engine_version, parallel, engine], sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    name = f"{model}-{hardware}-{h}"
    os.makedirs(scenarios_dir, exist_ok=True)
    path = os.path.join(scenarios_dir, name + ".yaml")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(scenario_yaml(model, hardware, engine_version, parallel, engine, name))
    return name + ".yaml"


def resolve_defaults(blis, defaults):
    if defaults:
        return defaults
    beside = os.path.join(os.path.dirname(os.path.abspath(blis)), "defaults.yaml")
    return beside if os.path.exists(beside) else None


def build_cmd(blis, catalog, registry, scenarios_dir, scenario_name, workload, num_requests, seed,
              horizon, defaults, cli_knobs):
    cmd = [blis, "run", "--latency-model", "blis-latency-kernel",
           "--scenario", scenario_name, "--scenarios", scenarios_dir, "--registry", registry]
    if catalog:
        cmd += ["--catalog", catalog]
    if resolve_defaults(blis, defaults):
        cmd += ["--defaults-filepath", resolve_defaults(blis, defaults)]
    if workload:
        cmd += ["--workload", workload]
    if num_requests is not None:
        cmd += ["--num-requests", str(num_requests)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if horizon is not None:
        cmd += ["--horizon", str(horizon)]
    for k, flag in CLI_INT_FLAGS.items():
        if k in cli_knobs:
            cmd += [flag, str(cli_knobs[k])]
    return cmd


def run_one(blis, catalog, registry, scenarios_dir, model, hardware, engine_version,
            workload, num_requests, seed, horizon, config, phase="probe", timeout=120, defaults=None):
    """Run one kernel-priced BLIS config and return a record dict. No runs.jsonl I/O (caller writes)."""
    parallel, engine = _deployment(config)
    cli_knobs = {k: config[k] for k in CLI_INT_FLAGS if k in config}
    num_instances = cli_knobs.get("num_instances", 1)
    scenario_name = write_scenario(scenarios_dir, model, hardware, engine_version, parallel, engine)
    cmd = build_cmd(blis, catalog, registry, scenarios_dir, scenario_name, workload,
                    num_requests, seed, horizon, defaults, cli_knobs)
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
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
    return {
        "config": config,
        "concurrency": config.get("concurrency"),
        "config_hash": config_hash,
        "phase": phase,
        # resolved deployment so the report need not parse the scenario YAML (per-GPU denominator etc.)
        "deployment": {"parallel": parallel, "engine": engine, "num_instances": num_instances,
                       "gpus": parallel["tp"] * num_instances, "scenario": scenario_name},
        "cmd": cmd,
        "returncode": rc,
        "wall_s": round(time.time() - t0, 3),
        "metrics": metrics,
        "metrics_all_instances": parse_all_instances(stdout) if rc == 0 else [],
        "error": error,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def add_common_args(p):
    """Shared args for run_blis and sweep_pool so the kernel invocation surface stays single-sourced."""
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


def require_env(args):
    if not args.catalog:
        sys.exit("no catalog: pass --catalog or set BLIS_CATALOG (NS-6)")
    if not args.registry:
        sys.exit("no registry: pass --registry or set BLIS_REGISTRY (blis-latency-kernel needs it)")


def main():
    p = argparse.ArgumentParser(description="Run one kernel-priced BLIS config; print its record")
    add_common_args(p)
    p.add_argument("--config", required=True, help='JSON knob dict, e.g. {"tp":8,"enable_expert_parallel":true,"num_instances":2,"concurrency":32}')
    p.add_argument("--out", help="optional runs.jsonl to append to (single-run use only)")
    args = p.parse_args()
    require_env(args)
    rec = run_one(args.blis, args.catalog, args.registry, args.scenarios, args.model, args.hardware,
                  args.engine_version, args.workload, args.num_requests, args.seed, args.horizon,
                  json.loads(args.config), args.phase, args.timeout, args.defaults)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
