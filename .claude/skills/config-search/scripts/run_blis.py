#!/usr/bin/env python3
# run_blis.py -- the ONE place config-search invokes BLIS and records a single run.
#
# Run it isolated:  python3 -I scripts/run_blis.py ...
# Importable: sweep_pool.py imports run_one()/build_cmd()/parse_metrics() for batch parallelism.
#
# It maps a config (knobs) + scenario-fixed args to pr-1851 `blis run` flags, runs the pinned
# binary, parses the JSON metrics object from stdout, and returns/records one JSON record.
#
# pr-1851 `blis run` prints a "=== Simulation Metrics ===" header then one JSON object per instance
# (plus a "cluster" aggregate); parse_metrics() extracts the cluster object. Confirmed against a live
# qwen3-14b run (fields: tokens_per_sec, responses_per_sec, ttft/itl/e2e *_p99_ms,
# scheduling_delay_p99_ms, preemption_count, ...). Re-confirm if the output format changes.
import argparse, hashlib, json, os, subprocess, sys, time

# config-knob -> blis flag. Extend as the search space grows; unknown keys pass as --<key>.
FLAG = {
    "tp": "--tp", "num_instances": "--num-instances", "dp": "--dp",
    "prefill_instances": "--prefill-instances", "decode_instances": "--decode-instances",
    "prefill_tp": "--prefill-tp", "decode_tp": "--decode-tp",
    "concurrency": "--concurrency",
    "max_num_seqs": "--max-num-seqs", "max_num_batched_tokens": "--max-num-batched-tokens",
    "long_prefill_token_threshold": "--long-prefill-token-threshold",
    "block_size_in_tokens": "--block-size-in-tokens",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "kv_cache_dtype": "--kv-cache-dtype", "total_kv_blocks": "--total-kv-blocks",
    "scheduler": "--scheduler", "routing_policy": "--routing-policy",
    "routing_scorers": "--routing-scorers", "admission_policy": "--admission-policy",
    "preemption_policy": "--preemption-policy",
    "num_speculative_tokens": "--num-speculative-tokens",
    "speculative_acceptance_rate": "--speculative-acceptance-rate",
    "speculative_method": "--speculative-method",
}
BOOL_FLAGS = {"enable_expert_parallel": "--enable-expert-parallel"}


def _iter_json_objects(text):
    """Yield each top-level JSON object in text, tolerating surrounding prose.

    Uses JSONDecoder.raw_decode (string/escape-aware), so braces inside string values do not
    desync parsing -- a hand-rolled brace counter would mis-match on an error string or a path
    token containing '{' or '}'.
    """
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
            except ValueError:
                i += 1          # not a valid object here (stray brace in prose); keep scanning
                continue
            if isinstance(obj, dict):
                yield obj
            i = end
        else:
            i += 1


def _find_metrics_dicts(obj, out):
    """Collect every dict carrying 'tokens_per_sec', recursing into nested dicts/lists so a future
    enveloped format ({"result": {...cluster...}}) is still found, not just top-level objects."""
    if isinstance(obj, dict):
        if "tokens_per_sec" in obj:
            out.append(obj)
        for v in obj.values():
            _find_metrics_dicts(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _find_metrics_dicts(v, out)
    return out


def parse_metrics(stdout):
    """Return the cluster-aggregate metrics object (or the first metrics dict) from `blis run` stdout."""
    cands = []
    for o in _iter_json_objects(stdout):
        _find_metrics_dicts(o, cands)
    if not cands:
        return {}
    for o in cands:
        if o.get("instance_id") == "cluster":
            return o
    return cands[0]


def build_cmd(blis, catalog, model, hardware, workload, num_requests, seed, horizon, config, defaults):
    cmd = [blis, "run", "--model", model, "--hardware", hardware]
    if catalog:
        cmd += ["--catalog", catalog]
    if defaults:
        cmd += ["--defaults-filepath", defaults]
    if workload:
        cmd += ["--workload", workload]
    if num_requests is not None:
        cmd += ["--num-requests", str(num_requests)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if horizon is not None:
        cmd += ["--horizon", str(horizon)]
    for k, v in config.items():
        flag = FLAG.get(k) or BOOL_FLAGS.get(k) or f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            # presence-only: emit the flag when True, omit entirely when False -- for ANY boolean
            # knob, not only those pre-listed in BOOL_FLAGS (else `foo: false` -> `--foo False`).
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    return cmd


def resolve_defaults(blis, defaults):
    """Locate defaults.yaml so the run does not depend on cwd: explicit arg, else next to the binary."""
    if defaults:
        return defaults
    beside = os.path.join(os.path.dirname(os.path.abspath(blis)), "defaults.yaml")
    return beside if os.path.exists(beside) else None


def run_one(blis, catalog, model, hardware, workload, num_requests, seed, horizon, config,
            phase="probe", timeout=120, defaults=None):
    """Run one BLIS config and return a record dict. No file I/O (caller writes runs.jsonl)."""
    cmd = build_cmd(blis, catalog, model, hardware, workload, num_requests, seed, horizon, config,
                    resolve_defaults(blis, defaults))
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
        # rc=0 but nothing parseable -> a format/parse failure, NOT a clean zero-throughput run.
        # Flag it so sweep_pool and the search never treat it as a healthy result.
        error = "rc=0 but no cluster metrics parsed from stdout (output format changed?)"
    else:
        error = None
    return {
        "config": config,
        "concurrency": config.get("concurrency"),
        "config_hash": config_hash,
        "phase": phase,
        "cmd": cmd,
        "returncode": rc,
        "wall_s": round(time.time() - t0, 3),
        "metrics": metrics,
        "error": error,
        "stdout_tail": stdout[-2000:],
    }


def add_common_args(p):
    """Shared run-config flags for both run_blis and sweep_pool, so the surface stays single-sourced."""
    p.add_argument("--blis", default="./blis")
    p.add_argument("--catalog", default=os.environ.get("BLIS_CATALOG"))
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--workload")
    p.add_argument("--num-requests", type=int, dest="num_requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int)
    p.add_argument("--defaults", help="path to defaults.yaml (default: next to the blis binary)")
    p.add_argument("--phase", default="probe")
    p.add_argument("--timeout", type=int, default=120)


def require_catalog(catalog):
    if not catalog:
        sys.exit("no catalog: pass --catalog or set BLIS_CATALOG (NS-6)")


def main():
    p = argparse.ArgumentParser(description="Run one BLIS config; print its record, optionally append to runs.jsonl")
    add_common_args(p)
    p.add_argument("--config", required=True, help='JSON knob dict, e.g. {"tp":8,"num_instances":2,"concurrency":64}')
    p.add_argument("--out", help="optional runs.jsonl to append to (single-run use only)")
    args = p.parse_args()
    require_catalog(args.catalog)
    rec = run_one(args.blis, args.catalog, args.model, args.hardware, args.workload,
                  args.num_requests, args.seed, args.horizon, json.loads(args.config),
                  args.phase, args.timeout, args.defaults)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
