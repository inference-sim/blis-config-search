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
CLI_FLAGS = {"num_instances": "--num-instances", "concurrency": "--concurrency",
             "scheduler": "--scheduler", "routing_policy": "--routing-policy",
             "routing_scorers": "--routing-scorers", "admission_policy": "--admission-policy",
             "preemption_policy": "--preemption-policy"}
PARALLEL_DEFAULTS = {"tp": 1, "pp": 1, "dp": 1, "enable_expert_parallel": False}
GPUS_PER_NODE = 8   # single-node H100; a per-instance footprint beyond this spans nodes (needs a fabric)
# NON-PRECISION engine defaults only. quantization/cache_dtype are intentionally NOT defaulted: an
# empty precision lets the checkpoint format govern (e.g. qwen3-14b stays BF16), so forcing FP8 here
# would silently mis-price every non-FP8 model.
ENGINE_DEFAULTS = {"block_size": 16, "max_num_batched_tokens": 8192, "max_num_seqs": 256,
                   "max_model_len": 32768, "cudagraph_mode": "PIECEWISE", "gpu_memory_utilization": 0.9}
# Deliberately unsupported in this runner -> fail loudly with the reason (do not silently drop).
UNSUPPORTED = {
    "prefill_instances": "PD disaggregation is not supported yet (single colocated pool only)",
    "decode_instances": "PD disaggregation is not supported yet (single colocated pool only)",
    "prefill_tp": "per-role PD TP is not supported yet (single colocated pool only)",
    "decode_tp": "per-role PD TP is not supported yet (single colocated pool only)",
    "num_speculative_tokens": "speculative decoding is not yet validated on the kernel backend",
    "speculative_acceptance_rate": "speculative decoding is not yet validated on the kernel backend",
    "speculative_method": "speculative decoding is not yet validated on the kernel backend",
    "long_prefill_token_threshold": "not a kernel scenario field (chunked-prefill threshold is engine-internal)",
}


class ConfigError(ValueError):
    pass


def classify_config(config):
    """Split a candidate into (parallel, engine, cli); raise ConfigError on unknown/unsupported keys."""
    parallel, engine, cli = dict(PARALLEL_DEFAULTS), dict(ENGINE_DEFAULTS), {}
    for k, v in config.items():
        if k in SCENARIO_PARALLEL:
            parallel[k] = v
        elif k in SCENARIO_ENGINE:
            engine[k] = v
        elif k in CLI_FLAGS:
            cli[k] = v
        elif k in UNSUPPORTED:
            raise ConfigError(f"{k!r}: {UNSUPPORTED[k]}")
        else:
            supported = sorted(SCENARIO_PARALLEL) + sorted(SCENARIO_ENGINE) + sorted(CLI_FLAGS)
            raise ConfigError(f"{k!r}: unknown key; supported = {supported}")
    if parallel["pp"] != 1:
        raise ConfigError("pp>1: the pinned kernel does not price pipeline parallelism")
    fp = footprint_gpus(parallel)
    if fp > GPUS_PER_NODE:
        raise ConfigError(
            f"per-instance footprint tp*pp*dp={fp} > {GPUS_PER_NODE} (one node): a multi-node single "
            f"instance needs a cluster fabric, not supported yet -- use num_instances to scale out instead")
    return parallel, engine, cli


def footprint_gpus(parallel):
    """GPUs one instance occupies = tp x pp x dp (EP is within the TP ranks, not an extra factor)."""
    return parallel["tp"] * parallel["pp"] * parallel["dp"]


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


def scenario_yaml(model, hardware, engine_version, parallel, engine, name):
    """Render a two-doc Scenario+Deployment YAML (no external yaml dep). Omitted engine keys (e.g.
    quantization when unset) let the checkpoint/engine govern."""
    def block(d, indent):
        pad = " " * indent
        return "\n".join(f"{pad}{k}: {json.dumps(v)}" for k, v in d.items())
    nodes = max(1, -(-footprint_gpus(parallel) // 8))   # ceil(footprint/8); single-node H100 = 8 GPUs
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
    """Write a scenario ATOMICALLY (temp + os.replace) so concurrent pool workers never read a
    half-written file for the same hashed deployment; return its file name."""
    key = json.dumps([model, hardware, engine_version, parallel, engine], sort_keys=True)
    name = f"{model}-{hardware}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    os.makedirs(scenarios_dir, exist_ok=True)
    path = os.path.join(scenarios_dir, name + ".yaml")
    if not os.path.exists(path):
        body = scenario_yaml(model, hardware, engine_version, parallel, engine, name)
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


def build_cmd(blis, catalog, registry, scenarios_dir, scenario_name, workload, num_requests, seed,
              horizon, defaults, cli):
    # BLIS v0.10.0 (the pinned release) selects the kernel cost-model backend from the
    # scenario's `coefficients:` list (written into the generated scenario), not a CLI flag.
    # The old `--latency-model blis-latency-kernel` flag was removed in the release and is
    # rejected ("unknown flag"), so it must not be passed.
    cmd = [blis, "run",
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
    for k, v in cli.items():
        cmd += [CLI_FLAGS[k], str(v)]
    return cmd


def _record_shell(config, phase):
    return {"config": config, "concurrency": config.get("concurrency"),
            "config_hash": hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
            "phase": phase}


def run_one(blis, catalog, registry, scenarios_dir, model, hardware, engine_version,
            workload, num_requests, seed, horizon, config, phase="probe", timeout=120, defaults=None):
    """Run one kernel-priced BLIS config and return a record dict. No runs.jsonl I/O (caller writes)."""
    rec = _record_shell(config, phase)
    try:
        parallel, engine, cli = classify_config(config)
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
    scenario_name = write_scenario(scenarios_dir, model, hardware, engine_version, parallel, engine)
    cmd = build_cmd(blis, catalog, registry, scenarios_dir, scenario_name, workload,
                    num_requests, seed, horizon, defaults, cli)
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
        "deployment": {"parallel": parallel, "engine": engine, "num_instances": num_instances,
                       "footprint_gpus_per_instance": fp, "gpus": fp * num_instances,
                       "scenario": scenario_name},
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
    except ValueError as e:
        sys.exit(str(e))
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
