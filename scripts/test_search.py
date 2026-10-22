#!/usr/bin/env python3
"""Hermetic tests for run_search.py's search loop (no BLIS — the runner is stubbed).

    python3 -I scripts/test_search.py

Covers the three defects that made the deepseek-v4-pro report recommend the weaker shape:
candidate enumeration that never varied EP/dp, a concurrency ramp that saturated against a
fixed --num-requests, and failed shapes vanishing instead of landing in pareto.json's dropped[].
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_search  # noqa: E402

PASS = 0
FAIL = 0

MOE = {"name": "deepseek-v4-pro", "arch": "MoE", "experts": 384}


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def scenario(budget=256, model="deepseek-v4-pro", hardware="h200"):
    return {"model": model, "hardware": hardware, "gpu_budget": budget,
            "workload": {"preset": "chatbot"}, "objective": "dual-pareto",
            "interactivity_floor": {"tok_per_s_per_user": 50.0, "itl_p99_ms_max": 20.0},
            "headroom": 0.8, "seed": 0}


class FakeRunner:
    """Stands in for run_blis.run_one. Prices a run from its topology so the search sees a
    realistic ordering: EP beats non-EP, wider tp beats narrower, and throughput rises with
    concurrency until a per-shape ceiling. Footprints below 8 fail on memory, as measured."""

    def __init__(self, fail_small_footprint=True):
        self.calls = []
        self.fail_small_footprint = fail_small_footprint

    def __call__(self, blis, catalog, registry, scenarios_dir, model, hardware, engine_version,
                 workload, num_requests, seed, horizon, config, phase="probe"):
        self.calls.append(dict(config, num_requests=num_requests))
        tp, dp = config.get("tp", 1), config.get("dp", 1)
        ni, conc = config.get("num_instances", 1), config.get("concurrency", 1)
        ep = bool(config.get("enable_expert_parallel"))
        rec = {"config": dict(config), "concurrency": conc, "phase": phase,
               "deployment": {"parallel": {"tp": tp, "dp": dp, "pp": 1,
                                           "enable_expert_parallel": ep},
                              "engine": {"max_num_seqs": 256, "max_num_batched_tokens": 8192},
                              "num_instances": ni, "gpus": tp * dp * ni,
                              "footprint_gpus_per_instance": tp * dp}}
        if self.fail_small_footprint and tp * dp < 8:
            rec.update({"returncode": 1, "metrics": {}, "metrics_all_instances": [],
                        "error": "kernelmodel: the kernel reports 191.2 GiB of per-rank fixed "
                                 "occupancy on a 141.0 GiB device (expert-parallel width 1)"})
            return rec
        # throughput: rises with load but saturates at the shape's ceiling; EP lifts the ceiling.
        ceiling = 60.0 * tp * (1.25 if ep else 1.0)
        tps = min(conc * 30.0, ceiling * ni / 8.0)
        itl = 8.0 + conc / max(1.0, tp * (1.3 if ep else 1.0))
        rec.update({"returncode": 0, "error": None, "metrics_all_instances": [],
                    "metrics": {"tokens_per_sec": tps, "itl_p99_ms": itl, "preemption_count": 0,
                                "ttft_p99_ms": 100.0, "injected_requests": num_requests,
                                "completed_requests": num_requests}})
        return rec


def run(sc, runner):
    """Call search() with the runner stubbed, tolerating either return arity."""
    import run_blis
    orig = run_blis.run_one
    run_blis.run_one = runner
    try:
        return run_search.search(sc, "/nonexistent/blis", "/cat", "/reg", "/tmp/scen",
                                 num_requests=80, seed=0, model_entry=MOE)
    finally:
        run_blis.run_one = orig


def test_explores_ep():
    print("search explores expert parallelism")
    r = FakeRunner()
    run(scenario(), r)
    eps = {bool(c.get("enable_expert_parallel")) for c in r.calls}
    check(True in eps, "at least one EP run was issued")
    check(False in eps, "the non-EP control was also run")
    check(any(c.get("dp", 1) > 1 for c in r.calls), "dp was varied, not pinned to 1")
    shapes = {(c.get("tp"), c.get("dp", 1), bool(c.get("enable_expert_parallel")))
              for c in r.calls}
    check((8, 1, True) in shapes, "TP8 + EP probed (the measured winner)")


def test_ep_wins_when_it_is_better():
    print("the better EP arm becomes the winner")
    r = FakeRunner()
    runs, finalists, _ = run(scenario(), r)
    check(finalists, "some finalists survived")
    labels = [f["label"] for f in finalists]
    check(any("EP" in l for l in labels), f"an EP shape is a finalist (got {labels})")
    best = max(finalists, key=lambda f: f["at_c_use"]["tokens_per_sec_per_gpu"])
    check("EP" in best["label"], f"the best-per-GPU finalist is an EP shape (got {best['label']})")
    check(len(finalists) > 1, f"more than one finalist (got {len(finalists)})")


def test_ramp_does_not_saturate_on_num_requests():
    print("ramp scales requests with concurrency")
    r = FakeRunner()
    run(scenario(), r)
    # A fixed --num-requests caps the ramp: measured, c=128 and c=256 returned byte-identical
    # throughput because the request pool ran dry. Every probe must offer more requests than
    # sessions or the point measures the pool, not the shape.
    bad = [c for c in r.calls if c["num_requests"] <= c.get("concurrency", 1)]
    check(not bad, f"no probe has num_requests <= concurrency (offenders: {bad[:3]})")
    concs = sorted({c.get("concurrency") for c in r.calls})
    check(max(concs) > 32, f"the ramp reaches past the old fixed ceiling of 32 (got {max(concs)})")


def test_failed_shapes_are_reported_as_dropped():
    print("failed shapes land in dropped[]")
    r = FakeRunner()
    runs, finalists, extra = run(scenario(), r)
    dropped = extra[-1] if isinstance(extra, tuple) else None
    check(isinstance(dropped, list), "search returns a dropped[] list")
    if isinstance(dropped, list):
        check(dropped, "the memory-infeasible footprints were recorded, not silently absent")
        check(all("label" in d and "reason" in d for d in dropped),
              "each dropped entry carries a label + reason")
        check(any("141.0 GiB" in (d.get("reason") or "") for d in dropped),
              "the kernel's own error text is preserved in the reason")


class TooBigForOneNode(FakeRunner):
    """A model whose weights do not fit in one node at any tp — kimi-k3's real failure mode. Only
    a footprint of 32+ GPUs runs; everything narrower reports impossible per-rank occupancy."""

    def __call__(self, *a, **kw):
        rec = super().__call__(*a, **kw)
        cfg = rec["config"]
        tp, dp = cfg.get("tp", 1), cfg.get("dp", 1)
        if tp * dp < 32:
            rec.update({"returncode": 1, "metrics": {}, "metrics_all_instances": [],
                        "error": "kernelmodel: the kernel reports 332.3 GiB of per-rank fixed "
                                 "occupancy on a 141.0 GiB device (tp=%d, expert-parallel "
                                 "width %d)" % (tp, tp * dp)})
        return rec


def test_escalates_to_multinode_when_nothing_fits_one_node():
    print("escalates to multi-node footprints")
    r = TooBigForOneNode(fail_small_footprint=False)
    runs, finalists, extra = run(scenario(budget=64), r)
    fps = {c.get("tp", 1) * c.get("dp", 1) for c in r.calls}
    check(any(f > 8 for f in fps), f"tried a multi-node footprint (saw {sorted(fps)})")
    check(32 in fps, "reached the 32-GPU footprint the model needs")
    check(finalists, "found a servable multi-node shape instead of giving up")
    if finalists:
        check(all(f["config"]["tp"] * f["config"].get("dp", 1) >= 32 for f in finalists),
              "finalists are the shapes that actually fit")


def test_does_not_escalate_when_one_node_works():
    print("no escalation when a single node fits")
    r = FakeRunner()
    run(scenario(budget=64), r)
    fps = {c.get("tp", 1) * c.get("dp", 1) for c in r.calls}
    check(all(f <= 8 for f in fps),
          f"stays single-node when single-node shapes are servable (saw {sorted(fps)})")


def test_refines_knobs_on_the_best_topologies():
    print("knob refinement on finalists")
    r = FakeRunner()
    runs, finalists, _ = run(scenario(), r)
    swept = {k for c in r.calls for k in c
             if k in ("routing_policy", "max_num_seqs", "max_num_batched_tokens",
                      "gpu_memory_utilization")}
    check(swept, f"engine/routing knobs were swept at all (got {sorted(swept)})")
    check("routing_policy" in swept, "routing policy — the biggest measured lever — was swept")
    # refinement must be TARGETED: a full cross product over every shape and load would be
    # thousands of runs. Only the best topologies get refined.
    refined = {(c.get("tp"), c.get("dp", 1)) for c in r.calls
               if any(k in c for k in ("routing_policy", "max_num_seqs"))}
    allshapes = {(c.get("tp"), c.get("dp", 1)) for c in r.calls}
    check(len(refined) < len(allshapes),
          f"refined {len(refined)} of {len(allshapes)} shapes, not all of them")
    check(len(r.calls) < 1200, f"total runs stays bounded (got {len(r.calls)})")


def test_labels_are_topology_and_variant_is_separate():
    print("label is topology; knobs live in variant")
    r = FakeRunner()
    runs, finalists, _ = run(scenario(), r)
    # The label is what a reader scans — GPUs x TP x EP x instances. Knobs are NOT jammed into it
    # (that produced "32x TP8 + EP · ll · b8k · seq256" with invented abbreviations).
    for f in finalists:
        check("·" not in f["label"], f"label is pure topology: {f['label']!r}")
        check("variant" in f, f"finalist carries a variant field: {f['label']}")
    # a knob-refined finalist describes its delta in words, spelled out
    refined = [f for f in finalists if f.get("variant")]
    check(refined, "some finalists are knob variants")
    for f in refined:
        v = f["variant"]
        check(not any(t in v for t in ("b2k", "b8k", "seq6", "mem0")),
              f"variant avoids invented abbreviations: {v!r}")
        check(any(w in v for w in ("routing", "batch tokens", "max seqs", "gpu mem util")),
              f"variant names the knob in words: {v!r}")
    # label + variant together must still identify a run uniquely
    keys = [(f["label"], f.get("variant", "")) for f in finalists]
    check(len(keys) == len(set(keys)), f"label+variant is unique: {len(keys)} vs {len(set(keys))}")
    # the all-defaults run in a refined group has an EMPTY variant
    check(any(f.get("variant") == "" for f in finalists), "a baseline finalist has no variant")


def test_winner_reason_names_the_knobs_that_won():
    print("winner reason names its knobs")
    r = FakeRunner()
    runs, finalists, _ = run(scenario(), r)
    # pareto.json's winner is identified by LABEL, and several finalists legitimately share one
    # once knobs are swept. A reason that says only "64x TP4xDP2 + EP gives the most tok/s/GPU"
    # does not say WHICH of the five same-named configs, nor that routing was the reason.
    win = finalists[0]
    reason = run_search.winner_reason(win, 50.0, 0.8)
    check(win["label"] in reason, "reason names the topology")
    if win.get("variant"):
        check(win["variant"] in reason,
              f"reason names the knob delta {win['variant']!r}: {reason}")
    # and a baseline winner should say it needs no knob changes, not stay silent
    base = next((f for f in finalists if not f.get("variant")), None)
    if base:
        rb = run_search.winner_reason(base, 50.0, 0.8)
        check("default" in rb.lower() or "no knob" in rb.lower(),
              f"a baseline winner says it is at defaults: {rb}")


def test_winner_prefers_the_simpler_config_on_a_tie():
    print("ties prefer the simpler config")
    # qwen3-14b: "32x TP8 · seq64" beat plain "32x TP8" by 0.3 tok/s out of 39,801 (0.0008%) —
    # noise, but it promoted a knob-tweaked config over the clean default and made the headline
    # recommendation look like max_num_seqs=64 mattered. Within tolerance, fewer knobs wins.
    a = {"label": "32x TP8 · seq64", "config": {"tp": 8, "num_instances": 32, "max_num_seqs": 64},
         "at_c_use": {"tokens_per_sec_per_gpu": 155.5, "tokens_per_sec": 39801.3}}
    b = {"label": "32x TP8", "config": {"tp": 8, "num_instances": 32},
         "at_c_use": {"tokens_per_sec_per_gpu": 155.5, "tokens_per_sec": 39801.0}}
    check(run_search.rank_finalists([a, b])[0] is b,
          "a 0.0008% edge does not beat a simpler config")
    # but a REAL difference still wins, however many knobs it carries
    c = {"label": "32x TP8 · ll", "config": {"tp": 8, "num_instances": 32,
                                             "routing_policy": "least-loaded"},
         "at_c_use": {"tokens_per_sec_per_gpu": 67.0, "tokens_per_sec": 34307.6}}
    d = {"label": "32x TP8", "config": {"tp": 8, "num_instances": 32},
         "at_c_use": {"tokens_per_sec_per_gpu": 36.4, "tokens_per_sec": 18657.3}}
    check(run_search.rank_finalists([d, c])[0] is c,
          "an 84% improvement wins even though it sets a knob")


class OverloadedAtHighLoad(FakeRunner):
    """Every probe returns rc=0 WITH a tokens_per_sec, but past a load it also sets rec["error"]
    the way run_blis._completion_issue does for an overloaded/incomplete run. Such a run must not
    count as a capacity point: report.py:_feasible rejects it, so if the search accepted it the
    report would contradict its own recommendation."""

    def __call__(self, *a, **kw):
        rec = super().__call__(*a, **kw)
        if rec.get("returncode") == 0 and rec["config"].get("concurrency", 0) > 64:
            rec["error"] = "overloaded: dropped_unservable=12"
        return rec


def test_overloaded_runs_do_not_set_capacity():
    print("a tokens-bearing but errored run is not a capacity point")
    r = OverloadedAtHighLoad()
    runs, finalists, _ = run(scenario(), r)
    # the errored (overloaded) runs must never become the operating point of any finalist
    for f in finalists:
        check(f["c_star"] <= 64,
              f"c_star stays at a load with no completion error (got {f['c_star']})")
        for pt in f.get("curve", []):
            # the kept curve points are the clean ones; errored loads are excluded, so no
            # curve point sits above the overload threshold
            check(pt["concurrency"] <= 64,
                  f"curve excludes overloaded loads (saw c={pt['concurrency']})")


def test_dense_model_is_unchanged():
    print("dense models keep the narrow grid")
    r = FakeRunner(fail_small_footprint=False)
    import run_blis
    orig = run_blis.run_one
    run_blis.run_one = r
    try:
        run_search.search(scenario(budget=16, model="qwen3-14b", hardware="h100"),
                          "/nonexistent/blis", "/cat", "/reg", "/tmp/scen", num_requests=80,
                          seed=0, model_entry={"name": "qwen3-14b", "arch": "dense"})
    finally:
        run_blis.run_one = orig
    check(all(c.get("dp", 1) == 1 for c in r.calls), "dense: never dp>1 (BLIS refuses it)")
    check(all(not c.get("enable_expert_parallel") for c in r.calls), "dense: never expert-parallel")


if __name__ == "__main__":
    test_explores_ep()
    test_ep_wins_when_it_is_better()
    test_ramp_does_not_saturate_on_num_requests()
    test_failed_shapes_are_reported_as_dropped()
    test_escalates_to_multinode_when_nothing_fits_one_node()
    test_does_not_escalate_when_one_node_works()
    test_refines_knobs_on_the_best_topologies()
    test_labels_are_topology_and_variant_is_separate()
    test_winner_reason_names_the_knobs_that_won()
    test_winner_prefers_the_simpler_config_on_a_tie()
    test_overloaded_runs_do_not_set_capacity()
    test_dense_model_is_unchanged()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
