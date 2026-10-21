#!/usr/bin/env python3
"""Generate run directories under reports/ for the demo gallery.

The ONE real scenario (glm53flash-h100-chatbot) already lives in reports/ as a full
run dir copied from the committed sample-run fixture. This script synthesizes the
remaining CLEARLY-MOCK scenarios so the gallery's cross-filter and the per-card report
links are all live and browsable end-to-end.

Each synthesized run dir gets:
  scenario.json  — resolved Scenario (model/hw/budget/workload/floor/headroom/run_id)
  pareto.json    — finalists + winner, in the shape report.py consumes
  report.json    — a short brain-style narrative (standfirst + a hero + recommendation)
  gallery.json   — display sidecar the gallery build reads (synthetic flag, isl_osl,
                   generated date, run counts, one-line reason)
  MOCK           — a marker file; its presence is a second, filesystem-level signal that
                   this directory is illustrative, not a real BLIS run.

Run from the repo root:  python3 -I scripts/gen_demo_reports.py
It only writes under reports/<run_id>/ and never runs BLIS.
"""
import json
import os

REPORTS = os.path.join(os.path.dirname(__file__), os.pardir, "reports")

# The mock scenarios, mirroring design/mockups/data.js (the 5 synthetic rows). The real
# glm53flash-h100-chatbot row is NOT here — it is the committed fixture in reports/.
MOCKS = [
    {
        "run_id": "qwen3-14b-h100-chatbot-0002", "model": "qwen3-14b", "hardware": "h100",
        "gpu_budget": 8, "workload": "chatbot", "isl_osl": "ShareGPT",
        "floor_rate": 50, "floor_itl": 20.0, "headroom": 0.8,
        "n_runs": 22, "n_servable": 14, "generated": "2026-10-19",
        "winner": "4x TP2",
        "reason": "Mock scenario — narrowest TP that still fits KV spreads load across four replicas.",
        "finalists": [
            {"label": "2x TP4", "tp": 4, "ni": 2, "gpus": 8, "c_use": 14, "tps": 1190.0, "tps_per_gpu": 148.8, "per_user": 88.0, "itl_p99": 11.4},
            {"label": "4x TP2", "tp": 2, "ni": 4, "gpus": 8, "c_use": 24, "tps": 1842.0, "tps_per_gpu": 230.3, "per_user": 68.0, "itl_p99": 14.7},
            {"label": "8x TP1", "tp": 1, "ni": 8, "gpus": 8, "c_use": 20, "tps": 1610.0, "tps_per_gpu": 201.3, "per_user": 59.0, "itl_p99": 17.2},
        ],
    },
    {
        "run_id": "glm53flash-h200-multidoc-0003", "model": "glm-5.3-flash", "hardware": "h200",
        "gpu_budget": 8, "workload": "multidoc", "isl_osl": "10k / 1.5k",
        "floor_rate": 50, "floor_itl": 20.0, "headroom": 0.8,
        "n_runs": 18, "n_servable": 9, "generated": "2026-10-19",
        "winner": "2x TP4 + EP",
        "reason": "Mock scenario — H200 KV headroom lets two TP4 instances share the long-context load.",
        "finalists": [
            {"label": "1x TP8 + EP", "tp": 8, "ni": 1, "gpus": 8, "c_use": 6, "tps": 540.0, "tps_per_gpu": 67.5, "per_user": 108.0, "itl_p99": 9.0},
            {"label": "2x TP4 + EP", "tp": 4, "ni": 2, "gpus": 8, "c_use": 12, "tps": 940.0, "tps_per_gpu": 117.5, "per_user": 72.0, "itl_p99": 13.0},
        ],
    },
    {
        "run_id": "qwen3-14b-h200-summarization-0004", "model": "qwen3-14b", "hardware": "h200",
        "gpu_budget": 4, "workload": "summarization", "isl_osl": "16k / 256",
        "floor_rate": 40, "floor_itl": 25.0, "headroom": 0.8,
        "n_runs": 15, "n_servable": 11, "generated": "2026-10-20",
        "winner": "2x TP2",
        "reason": "Mock scenario — long-input summarization is prefill-heavy; replicas win on throughput.",
        "finalists": [
            {"label": "1x TP4", "tp": 4, "ni": 1, "gpus": 4, "c_use": 9, "tps": 610.0, "tps_per_gpu": 152.5, "per_user": 71.0, "itl_p99": 13.8},
            {"label": "2x TP2", "tp": 2, "ni": 2, "gpus": 4, "c_use": 18, "tps": 1060.0, "tps_per_gpu": 265.0, "per_user": 52.0, "itl_p99": 18.9},
        ],
    },
    {
        "run_id": "glm53flash-h100-multidoc-0005", "model": "glm-5.3-flash", "hardware": "h100",
        "gpu_budget": 32, "workload": "multidoc", "isl_osl": "10k / 1.5k",
        "floor_rate": 50, "floor_itl": 20.0, "headroom": 0.8,
        "n_runs": 34, "n_servable": 20, "generated": "2026-10-20",
        "winner": "8x TP4 + EP",
        "reason": "Mock scenario — doubling the budget scales the 4× TP4 winner linearly to eight instances.",
        "finalists": [
            {"label": "4x TP8 + EP", "tp": 8, "ni": 4, "gpus": 32, "c_use": 20, "tps": 1960.0, "tps_per_gpu": 61.3, "per_user": 99.0, "itl_p99": 10.1},
            {"label": "8x TP4 + EP", "tp": 4, "ni": 8, "gpus": 32, "c_use": 32, "tps": 2510.0, "tps_per_gpu": 78.4, "per_user": 71.0, "itl_p99": 13.9},
        ],
    },
    {
        "run_id": "qwen3-14b-h100-contentgen-0006", "model": "qwen3-14b", "hardware": "h100",
        "gpu_budget": 16, "workload": "contentgen", "isl_osl": "1k / 1k",
        "floor_rate": 60, "floor_itl": 16.0, "headroom": 0.8,
        "n_runs": 26, "n_servable": 12, "generated": "2026-10-20",
        "winner": "8x TP2",
        "reason": "Mock scenario — balanced 1k/1k content generation rewards many narrow replicas under a tight floor.",
        "finalists": [
            {"label": "4x TP4", "tp": 4, "ni": 4, "gpus": 16, "c_use": 24, "tps": 2180.0, "tps_per_gpu": 136.3, "per_user": 78.0, "itl_p99": 12.1},
            {"label": "8x TP2", "tp": 2, "ni": 8, "gpus": 16, "c_use": 40, "tps": 3120.0, "tps_per_gpu": 195.0, "per_user": 64.0, "itl_p99": 15.2},
        ],
    },
    {
        "run_id": "llama31-70b-h200-summarization-0007", "model": "llama-3.1-70b-instruct", "hardware": "h200",
        "gpu_budget": 16, "workload": "summarization", "isl_osl": "4k / 512",
        "floor_rate": 45, "floor_itl": 22.0, "headroom": 0.8,
        "n_runs": 24, "n_servable": 13, "generated": "2026-10-21",
        "winner": "4x TP4",
        "reason": "Mock scenario — a 70B dense model on H200 fits TP4; four replicas carry the summarization load before the ITL ceiling.",
        "finalists": [
            {"label": "2x TP8", "tp": 8, "ni": 2, "gpus": 16, "c_use": 10, "tps": 820.0, "tps_per_gpu": 51.3, "per_user": 96.0, "itl_p99": 11.0},
            {"label": "4x TP4", "tp": 4, "ni": 4, "gpus": 16, "c_use": 18, "tps": 1380.0, "tps_per_gpu": 86.3, "per_user": 58.0, "itl_p99": 18.5},
        ],
    },
    {
        "run_id": "deepseek-v3-h200-chatbot-0008", "model": "deepseek-v3", "hardware": "h200",
        "gpu_budget": 16, "workload": "chatbot", "isl_osl": "ShareGPT",
        "floor_rate": 50, "floor_itl": 20.0, "headroom": 0.8,
        "n_runs": 30, "n_servable": 16, "generated": "2026-10-21",
        "winner": "2x TP8 + EP",
        "reason": "Mock scenario — a 256-expert MoE wants expert parallelism; two TP8+EP instances beat four narrow replicas on both throughput and per-user rate.",
        "finalists": [
            {"label": "1x TP8 + EP", "tp": 8, "ni": 1, "gpus": 8, "c_use": 8, "tps": 760.0, "tps_per_gpu": 95.0, "per_user": 104.0, "itl_p99": 9.6},
            {"label": "2x TP8 + EP", "tp": 8, "ni": 2, "gpus": 16, "c_use": 16, "tps": 1480.0, "tps_per_gpu": 92.5, "per_user": 82.0, "itl_p99": 12.4},
        ],
    },
]


def scenario_json(m):
    return {
        "_source": "SYNTHETIC mock scenario for the gallery demo — NOT a BLIS run",
        "run_id": m["run_id"], "model": m["model"], "hardware": m["hardware"],
        "gpu_budget": m["gpu_budget"], "workload": {"preset": m["workload"]},
        "objective": "dual-pareto",
        "interactivity_floor": {"tok_per_s_per_user": m["floor_rate"], "itl_p99_ms_max": m["floor_itl"]},
        "headroom": m["headroom"], "slos": {}, "seed": 0,
        "latency_model": "blis-latency-kernel",
    }


def pareto_json(m):
    finalists = []
    for f in m["finalists"]:
        finalists.append({
            "label": f["label"],
            "config": {"tp": f["tp"], "num_instances": f["ni"]},
            "gpus": f["gpus"], "c_star": round(f["c_use"] / m["headroom"]), "c_use": f["c_use"],
            "at_c_use": {
                "tokens_per_sec": f["tps"], "tokens_per_sec_per_gpu": f["tps_per_gpu"],
                "tok_per_s_per_user": f["per_user"], "itl_p99_ms": f["itl_p99"],
                "ttft_p99_ms": 0.0, "preemption_count": 0,
            },
            "curve": [{"concurrency": f["c_use"], "tokens_per_sec": f["tps"],
                       "itl_p99_ms": f["itl_p99"], "tok_per_s_per_user": f["per_user"],
                       "preemption_count": 0}],
        })
    return {
        "_source": "SYNTHETIC mock — illustrative, NOT a BLIS measurement",
        "objective": "dual-pareto",
        "interactivity_floor": {"tok_per_s_per_user": m["floor_rate"], "itl_p99_ms_max": m["floor_itl"]},
        "headroom": m["headroom"], "finalists": finalists, "dropped": [],
        "winner": {"chosen": m["winner"], "budget_max": m["winner"],
                   "efficiency_per_gpu": m["winner"], "reason": m["reason"], "move_trail": []},
    }


def report_json(m):
    w = next(f for f in m["finalists"] if f["label"] == m["winner"])
    return {
        "_source": "SYNTHETIC mock narrative for the gallery demo",
        "title": f"Serving {m['model']} on {m['gpu_budget']}× {m['hardware'].upper()} (MOCK)",
        "standfirst": f"Illustrative placeholder scenario — {m['reason']}",
        "meta": ["MOCK — not a BLIS measurement",
                 f"{m['model']} · {m['gpu_budget']}× {m['hardware'].upper()} · {m['workload']}"],
        "hero": [
            {"h": "What this is", "html": "A <b>mock</b> scenario included so the gallery's filtering and report links are browsable end to end. The numbers are illustrative, not simulator output."},
            {"h": "What to run", "html": f"<b>{m['winner']}</b> — {w['tps']:.0f} tok/s ({w['tps_per_gpu']:.0f}/GPU), {w['per_user']:.0f} tok/s/user at c={w['c_use']}.", "run": True},
        ],
        "recommendation_prose": [m["reason"], "These are placeholder numbers for a design demo — do not deploy from them."],
        "operating_points": [
            {"conc": w["c_use"], "flags": {"tp": w["tp"], "ni": w["ni"]},
             "pu": w["per_user"], "cl": w["tps"], "itl": w["itl_p99"], "hit": None}
        ],
    }


def gallery_json(m, runs=None):
    """Display sidecar the gallery build reads for fields not in scenario/pareto.
    n_runs / n_servable are taken from the synthesized sweep when present, so the card's
    "N/M servable" count matches what the detail explorer actually shows."""
    if runs is not None:
        n_runs = len(runs)
        n_serv = sum(1 for r in runs if r["metrics"]["itl_p99_ms"] <= m["floor_itl"])
    else:
        n_runs, n_serv = m["n_runs"], m["n_servable"]
    return {"synthetic": True, "isl_osl": m["isl_osl"], "generated": m["generated"],
            "n_runs": n_runs, "n_servable": n_serv, "reason": m["reason"]}


def runs_jsonl(m):
    """Synthesize a plausible multi-point sweep per finalist so the scenario-detail explorer
    (which cross-filters per-run RECORDS) is populated for mock scenarios too. Each finalist
    is swept over a concurrency ramp; per-user rate holds near its c_use value, then falls off
    a cliff past c_star, matching the interactivity-cliff story the real data shows. Clearly
    mock — these rows live only in a MOCK-marked run dir and never claim to be BLIS output."""
    floor_itl = m["floor_itl"]
    rows = []
    for f in m["finalists"]:
        c_use = f["c_use"]
        c_star = round(c_use / m["headroom"])
        # concurrency ramp: a few below c_use, c_use, c_star, and two past the cliff
        ramp = sorted(set([1, max(2, c_use // 2), c_use, c_star,
                           round(c_star * 1.2), round(c_star * 1.5)]))
        for c in ramp:
            frac = c / c_star if c_star else 1.0
            # tokens/s rises with load then saturates; per-user holds then collapses past c_star
            tps = round(f["tps"] * min(1.08, frac if frac < 1 else 1.0 + 0.08 * (1 - min(1, (c - c_star) / c_star))), 1)
            if c <= c_star:
                itl = round(f["itl_p99"] * (0.7 + 0.3 * frac), 3)
                per_user = round(f["per_user"] * (1.15 - 0.15 * frac), 1)
            else:  # past the cliff: ITL blows up, per-user collapses
                over = (c - c_star) / c_star
                itl = round(f["itl_p99"] * (1.0 + 6 * over), 3)
                per_user = round(max(5.0, f["per_user"] * (1.0 - 1.8 * over)), 1)
            servable = itl <= floor_itl
            rows.append({
                "config": {"tp": f["tp"], "num_instances": f["ni"], "concurrency": c},
                "concurrency": c,
                "config_hash": f"{m['run_id'][:6]}-{f['label'].replace(' ', '')}-{c}",
                "phase": "probe" if c < c_use else ("finalist" if c <= c_star else "overload"),
                "deployment": {
                    "parallel": {"tp": f["tp"], "pp": 1, "dp": 1, "enable_expert_parallel": "EP" in f["label"]},
                    "engine": {"block_size": 16, "max_num_batched_tokens": 8192, "max_num_seqs": 256,
                               "max_model_len": 32768, "gpu_memory_utilization": 0.9},
                    "num_instances": f["ni"], "gpus": f["gpus"],
                    "scenario": f"{m['model']}-{m['hardware']}-mock.yaml",
                },
                "returncode": 0,
                "metrics": {
                    "instance_id": "cluster", "completed_requests": 200,
                    "tokens_per_sec": tps, "itl_p99_ms": itl, "itl_mean_ms": round(itl * 0.85, 3),
                    "tok_per_s_per_user": per_user,
                    "ttft_p99_ms": round(200 + 30 * c, 1), "e2e_p95_ms": round(2000 + 40 * c, 1),
                    "preemption_count": 0,
                },
                "tok_per_s_per_user": per_user,
                "error": None,
            })
    return rows


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    reports = os.path.abspath(REPORTS)
    os.makedirs(reports, exist_ok=True)
    # Sidecar for the REAL fixture so the build has its display fields too.
    real = os.path.join(reports, "glm53flash-h100-chatbot-kernel-0001")
    if os.path.isdir(real):
        write_json(os.path.join(real, "gallery.json"),
                   {"synthetic": False, "isl_osl": "ShareGPT", "generated": "2026-10-18",
                    "n_runs": 27, "n_servable": 17,
                    "reason": "Most output tok/s on the budget while holding ≥50 tok/s/user on every instance."})
    for m in MOCKS:
        d = os.path.join(reports, m["run_id"])
        os.makedirs(d, exist_ok=True)
        write_json(os.path.join(d, "scenario.json"), scenario_json(m))
        write_json(os.path.join(d, "pareto.json"), pareto_json(m))
        write_json(os.path.join(d, "report.json"), report_json(m))
        runs = runs_jsonl(m)
        write_json(os.path.join(d, "gallery.json"), gallery_json(m, runs))
        # runs.jsonl: synthesized sweep so the scenario-detail explorer is populated
        with open(os.path.join(d, "runs.jsonl"), "w", encoding="utf-8") as f:
            for row in runs:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        open(os.path.join(d, "MOCK"), "w").close()
        print(f"wrote {m['run_id']}/ (mock)")


if __name__ == "__main__":
    main()
