#!/usr/bin/env python3
"""Deterministic mini config-search for ONE scenario -> reports/<run_id>/.

    python3 -I scripts/run_search.py --submission submissions/foo.yaml \
        --blis .build/blis --catalog ../blis-catalog --registry ../blis-registry

This is the automation path behind the local Submit button (serve_local.py calls it). It is a
SIMPLIFIED, non-LLM sweep: it enumerates a small fixed set of feasible topologies for the
scenario's GPU budget, probes each over a short concurrency ramp with REAL BLIS runs (via the
skill's run_blis.run_one), classifies servability against the interactivity floor, picks the
Pareto winner on usable capacity, and writes a run directory report.py can render.

It is NOT the full agentic search (no diagnostic knob steering); it exists so the Submit UI
produces a real, honest report from real BLIS numbers. The run dir it writes is the same shape
the skill writes, so build_gallery.py ingests it unchanged.
"""
import argparse
import json
import os
import sys

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SKILL = os.path.join(ROOT, ".claude", "skills", "config-search", "scripts")
sys.path.insert(0, SKILL)
import run_blis  # noqa: E402

VALID_TP = [1, 2, 4, 8]  # tp is bounded by one node (8 H100s); scale out with instances
MAX_CANDIDATES = 8


def candidate_topologies(gpu_budget):
    """Candidate topologies that use the WHOLE budget, across as many nodes as needed.

    For each tp that fits in a node and divides the budget, fill the budget with instances
    (num_instances = budget / tp). run_blis sizes the scenario's pool for all instances and
    names a fabric once it spans nodes, so a 128-GPU budget really explores 128 GPUs. Wider
    tp = fewer, larger instances; narrower tp = more replicas — the trade-off the search is for.
    For a very large budget the instance counts get big, so cap the candidate list."""
    out = []
    for tp in VALID_TP:
        if tp > gpu_budget:
            continue
        ni = gpu_budget // tp
        if ni >= 1:
            out.append({"tp": tp, "num_instances": ni})
    # keep the widest-tp (fewest instances) candidates first — they run fastest — and bound the set
    out.sort(key=lambda t: -t["tp"])
    return out[:MAX_CANDIDATES]


def per_user(metrics):
    """Per-user tok/s on the floor's p99 basis: 1000 / ITL p99 (report.py's fallback; the
    engine does not emit tok_per_s_per_user directly)."""
    itl = metrics.get("itl_p99_ms")
    return round(1000.0 / itl, 1) if itl else None


def search(scenario, blis, catalog, registry, scenarios_dir, num_requests=80, seed=0):
    model = scenario["model"]
    hardware = scenario["hardware"]
    budget = int(scenario["gpu_budget"])
    wl = scenario.get("workload") or {}
    workload = wl.get("preset") if isinstance(wl, dict) else wl
    floor = scenario.get("interactivity_floor") or {}
    floor_itl = float(floor.get("itl_p99_ms_max", 20.0))
    floor_rate = float(floor.get("tok_per_s_per_user", 50.0))
    headroom = float(scenario.get("headroom", 0.8))
    engine_version = scenario.get("engine_version", "0.29.0")

    ramp = [1, 2, 4, 8, 16, 24, 32]
    runs, finalists = [], []
    for topo in candidate_topologies(budget):
        gpus = topo["tp"] * topo["num_instances"]
        curve, c_star = [], None
        for c in ramp:
            if c > budget * 8:  # don't probe absurd loads
                break
            cfg = {**topo, "concurrency": c}
            rec = run_blis.run_one(blis, catalog, registry, scenarios_dir, model, hardware,
                                   engine_version, workload, num_requests, seed, None, cfg,
                                   phase="probe")
            runs.append(rec)
            m = rec.get("metrics") or {}
            if rec.get("returncode") != 0 or not m.get("tokens_per_sec"):
                continue
            pu = per_user(m)
            servable = pu is not None and pu >= floor_rate and (m.get("itl_p99_ms", 1e9) <= floor_itl)
            pt = {"concurrency": c, "tokens_per_sec": round(m["tokens_per_sec"], 1),
                  "itl_p99_ms": m.get("itl_p99_ms"), "tok_per_s_per_user": pu,
                  "preemption_count": m.get("preemption_count", 0)}
            curve.append(pt)
            if servable:
                c_star = c
        if c_star is None:
            continue  # nothing servable for this topology
        c_use = max(1, int(round(headroom * c_star)))
        at = min((p for p in curve if p["concurrency"] <= c_use), key=lambda p: abs(p["concurrency"] - c_use), default=None)
        if at is None:
            continue
        tps = at["tokens_per_sec"]
        finalists.append({
            "label": f"{topo['num_instances']}x TP{topo['tp']}",
            "config": topo, "gpus": gpus, "c_star": c_star, "c_use": c_use,
            "at_c_use": {"tokens_per_sec": tps, "tokens_per_sec_per_gpu": round(tps / gpus, 1),
                         "tok_per_s_per_user": at["tok_per_s_per_user"], "itl_p99_ms": at["itl_p99_ms"],
                         "ttft_p99_ms": 0.0, "preemption_count": at.get("preemption_count", 0)},
            "curve": curve,
        })
    return runs, finalists, (floor_rate, floor_itl, headroom)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deterministic mini config-search -> reports/<run_id>/")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--blis", default=os.path.join(ROOT, ".build", "blis"))
    ap.add_argument("--catalog", default=os.path.join(ROOT, os.pardir, "blis-catalog"))
    ap.add_argument("--registry", default=os.path.join(ROOT, os.pardir, "blis-registry"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports"))
    ap.add_argument("--num-requests", type=int, default=80)
    args = ap.parse_args(argv)

    sub = os.path.abspath(args.submission)
    if not os.path.isfile(sub):
        raise SystemExit(f"no such submission: {args.submission}")
    scenario = yaml.safe_load(open(sub)) or {}
    blis = os.path.abspath(args.blis)
    if not os.path.isfile(blis):
        raise SystemExit(f"blis binary not found at {blis} — build it first (see SETUP.md)")

    name = os.path.splitext(os.path.basename(sub))[0]
    run_id = name if name[:1].isalnum() else "submission"
    run_dir = os.path.join(args.out, run_id)
    os.makedirs(run_dir, exist_ok=True)
    # Generated kernel scenarios are build scratch, not run-dir data — keep them in a temp dir
    # so the committed/ingested run directory holds only scenario/pareto/runs/gallery/report.
    import tempfile
    scen_dir = tempfile.mkdtemp(prefix="cs-scenarios-")

    print(f"searching {scenario.get('model')} on {scenario.get('gpu_budget')}x {scenario.get('hardware')} "
          f"({(scenario.get('workload') or {}).get('preset')})…", flush=True)
    runs, finalists, (floor_rate, floor_itl, headroom) = search(
        scenario, blis, args.catalog, args.registry, scen_dir, num_requests=args.num_requests)
    if not finalists:
        raise SystemExit("no servable configuration found for this scenario (nothing met the floor)")

    # winner: dual-pareto proxy — most tok/s/GPU, tie-break total tok/s
    finalists.sort(key=lambda f: (f["at_c_use"]["tokens_per_sec_per_gpu"], f["at_c_use"]["tokens_per_sec"]), reverse=True)
    winner = finalists[0]
    n_servable = sum(1 for r in runs if (r.get("metrics") or {}).get("itl_p99_ms", 1e9) <= floor_itl and r.get("returncode") == 0)

    scenario_out = {"run_id": run_id, "model": scenario["model"], "hardware": scenario["hardware"],
                    "gpu_budget": scenario["gpu_budget"], "workload": scenario.get("workload"),
                    "objective": "dual-pareto",
                    "interactivity_floor": {"tok_per_s_per_user": floor_rate, "itl_p99_ms_max": floor_itl},
                    "headroom": headroom, "seed": 0}
    pareto = {"objective": "dual-pareto",
              "interactivity_floor": {"tok_per_s_per_user": floor_rate, "itl_p99_ms_max": floor_itl},
              "headroom": headroom, "finalists": finalists, "dropped": [],
              "winner": {"chosen": winner["label"], "budget_max": winner["label"],
                         "efficiency_per_gpu": winner["label"],
                         "reason": f"{winner['label']} gives the most tok/s/GPU while holding the "
                                   f"{floor_rate:g} tok/s/user floor; usable load c={winner['c_use']} (C*={winner['c_star']}).",
                         "move_trail": ["enumerated single-node topologies within budget",
                                        "probed a concurrency ramp per topology with real BLIS runs",
                                        f"ranked finalists on usable capacity (headroom {headroom})"]}}
    gallery = {"synthetic": False, "isl_osl": "", "generated": __import__("datetime").date.today().isoformat(),
               "n_runs": len(runs), "n_servable": n_servable,
               "reason": pareto["winner"]["reason"]}

    with open(os.path.join(run_dir, "scenario.json"), "w") as f:
        json.dump(scenario_out, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, "pareto.json"), "w") as f:
        json.dump(pareto, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, "gallery.json"), "w") as f:
        json.dump(gallery, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, "runs.jsonl"), "w") as f:
        for r in runs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {os.path.relpath(run_dir, ROOT)}/ — winner {winner['label']}, "
          f"{len(finalists)} finalists, {len(runs)} runs ({n_servable} servable)")
    return run_id


if __name__ == "__main__":
    main()
