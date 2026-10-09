#!/usr/bin/env python3
# sweep_pool.py -- run a BATCH of BLIS configs in parallel, bounded by a HOST-ADAPTIVE cap.
#
#   python3 -I scripts/sweep_pool.py --model qwen3-14b --hardware H100 --workload chatbot \
#       --batch batch.json --out runs.jsonl
#
# The cap is min(core_cap, memory_cap) from the host's AVAILABLE (not installed) resources at
# runtime, so the search uses the machine it is on without assuming it owns it:
#   core_cap   = floor(available_cpus * util_fraction)   # available_cpus respects cgroup/affinity
#   memory_cap = floor(MemAvailable * mem_budget_fraction / peak_rss_per_run)
# BLIS is ~1 core and ~0.1-0.6 GiB RSS per run, so on a big host the core cap binds and on a small /
# shared / containerized host the memory cap binds -- neither is hard-coded. --max-parallel overrides.
#
# The search decision order stays deterministic: records are returned sorted by the batch's input
# order, independent of which worker finished first (keeps the recommendation reproducible, INV-6).
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_blis  # noqa: E402  -- single source of truth for invoking BLIS + parsing metrics


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))          # respects cpuset/affinity (not just nproc)
    except AttributeError:
        return os.cpu_count() or 1


def mem_available_mib():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024   # kB -> MiB
    except OSError:
        pass
    return None


def compute_cap(util_fraction, mem_budget_fraction, peak_rss_mib, max_parallel):
    cpus = available_cpus()
    core_cap = max(1, int(cpus * util_fraction))
    mem = mem_available_mib()
    if mem is not None:
        mem_cap = max(1, int(mem * mem_budget_fraction / peak_rss_mib))
    else:
        mem_cap = core_cap  # unknown free memory -> let cores decide
    cap = min(core_cap, mem_cap)
    bound = "cores" if core_cap <= mem_cap else "memory"
    if max_parallel:
        cap = min(cap, max_parallel)
        if cap == max_parallel:
            bound = "explicit --max-parallel"
    return cap, bound, {"available_cpus": cpus, "mem_available_mib": mem,
                        "core_cap": core_cap, "mem_cap": mem_cap}


def main():
    p = argparse.ArgumentParser(description="Run a batch of BLIS configs in parallel under a host-adaptive cap")
    p.add_argument("--blis", default="./blis")
    p.add_argument("--catalog", default=os.environ.get("BLIS_CATALOG"))
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--workload")
    p.add_argument("--num-requests", type=int, dest="num_requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int)
    p.add_argument("--phase", default="probe")
    p.add_argument("--defaults", help="path to defaults.yaml (default: next to the blis binary)")
    p.add_argument("--batch", required=True, help="JSON file: array of config dicts (or '-' for stdin)")
    p.add_argument("--out", required=True, help="runs.jsonl to append to (written single-threaded)")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-parallel", type=int, dest="max_parallel")
    p.add_argument("--util-fraction", type=float, default=0.8)
    p.add_argument("--mem-budget-fraction", type=float, default=0.8)
    p.add_argument("--peak-rss-mib", type=float, default=1024.0, help="conservative BLIS RSS/run")
    args = p.parse_args()
    if not args.catalog:
        sys.exit("no catalog: pass --catalog or set BLIS_CATALOG (NS-6)")

    src = sys.stdin if args.batch == "-" else open(args.batch)
    configs = json.load(src)
    if not isinstance(configs, list):
        sys.exit("--batch must be a JSON array of config dicts")

    cap, bound, info = compute_cap(args.util_fraction, args.mem_budget_fraction,
                                   args.peak_rss_mib, args.max_parallel)
    print(json.dumps({"parallel_cap": cap, "bound_by": bound, "n_configs": len(configs), **info}),
          file=sys.stderr)

    t0 = time.time()
    results = [None] * len(configs)
    with ThreadPoolExecutor(max_workers=cap) as ex:
        futs = {ex.submit(run_blis.run_one, args.blis, args.catalog, args.model, args.hardware,
                          args.workload, args.num_requests, args.seed, args.horizon, cfg,
                          args.phase, args.timeout, args.defaults): i for i, cfg in enumerate(configs)}
        for fut, i in futs.items():
            results[i] = fut.result()

    # single-threaded write in stable (input) order -> deterministic ledger
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    wall = time.time() - t0
    ok = sum(1 for r in results if r["returncode"] == 0)
    sum_run_s = sum(r["wall_s"] for r in results)
    print(json.dumps({
        "parallel_cap": cap, "bound_by": bound, "n_configs": len(configs), "ok": ok,
        "wall_s": round(wall, 2), "sum_serial_s": round(sum_run_s, 2),
        "speedup_x": round(sum_run_s / wall, 2) if wall > 0 else None,
    }))


if __name__ == "__main__":
    main()
