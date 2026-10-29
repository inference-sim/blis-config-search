#!/usr/bin/env python3
"""Hermetic tests for run_search.py's search loop (no BLIS — the runner is stubbed).

    python3 -I scripts/test_search.py

Covers the three defects that made the deepseek-v4-pro report recommend the weaker shape:
candidate enumeration that never varied EP/dp, a concurrency ramp that saturated against a
fixed --num-requests, and failed shapes vanishing instead of landing in pareto.json's dropped[].
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_search  # noqa: E402
import topologies  # noqa: E402
from run_search import _workload_label  # noqa: E402

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
                 workload, num_requests, seed, horizon, config, phase="probe", timeout=120,
                 defaults=None, replay=None):
        self.calls.append(dict(config, num_requests=num_requests))
        # Record the seed EVERY run was issued with, so a test can assert the submission's seed
        # reached BLIS rather than the search's default (namasl review #1).
        self.seeds = getattr(self, "seeds", [])
        self.seeds.append(seed)
        self.phases = getattr(self, "phases", [])
        self.phases.append(phase)
        self.replays = getattr(self, "replays", [])
        self.replays.append(replay)
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


def run(sc, runner, max_parallel=1, want_coverage=False):
    """Call search() with the runner stubbed.

    Returns (runs, finalists, extra) by default -- the shape every caller here unpacks -- and the
    4-tuple including `coverage` when asked. The old docstring said "tolerating either return
    arity" while every call site unpacked exactly three, so a change to search()'s contract was
    invisible to these tests (namasl review)."""
    import run_blis
    orig = run_blis.run_one
    run_blis.run_one = runner
    try:
        out = run_search.search(sc, "/nonexistent/blis", "/cat", "/reg", "/tmp/scen",
                                num_requests=80, seed=0, model_entry=MOE,
                                max_parallel=max_parallel)
    finally:
        run_blis.run_one = orig
    runs, finalists, extra, coverage = out
    return (runs, finalists, extra, coverage) if want_coverage else (runs, finalists, extra)


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
    runs, finalists, extra, coverage = run(scenario(budget=64), r, want_coverage=True)
    fps = {c.get("tp", 1) * c.get("dp", 1) for c in r.calls}
    check(any(f > 8 for f in fps), f"tried a multi-node footprint (saw {sorted(fps)})")
    check(32 in fps, "reached the 32-GPU footprint the model needs")
    check(finalists, "found a servable multi-node shape instead of giving up")
    if finalists:
        check(all(f["config"]["tp"] * f["config"].get("dp", 1) >= 32 for f in finalists),
              "finalists are the shapes that actually fit")

    # COVERAGE MUST FOLLOW THE ESCALATION. It was assigned only while still None, so it always
    # described the first max_footprint=8 tier: here it claimed "12 of 19 valid layouts" while 28
    # distinct topologies had been probed out of 35 valid at the winning cap, which would make the
    # hero contradict its own move trail (namasl review).
    labels = {f["label"] for f in finalists} | {d["label"] for d in extra[3]}
    check(coverage is not None, "escalation still records coverage")
    if coverage:
        check(coverage["probed"] == len(labels),
              f"probed ({coverage['probed']}) == distinct topologies actually tried ({len(labels)})")
        check(coverage["probed"] > MAX_SINGLE_NODE_PROBED,
              f"and it exceeds the first tier's {MAX_SINGLE_NODE_PROBED} (got {coverage['probed']})")
        check(coverage["valid_layouts"] >= coverage["probed"],
              f"the universe is at least what was probed ({coverage})")
        check(len(coverage["omitted"]) == coverage["valid_layouts"] - coverage["probed"],
              "the omitted list accounts for every unprobed layout at the widest cap reached")
        caps = coverage.get("footprint_caps_attempted") or []
        check(caps and max(caps) > 8,
              f"the attempted footprint caps show the escalation (got {caps})")
        check(all(c in run_search.footprint_tiers(64) for c in caps),
              "the recorded caps are real tiers")


MAX_SINGLE_NODE_PROBED = 12          # what a non-escalating 64-GPU search probes (= MAX_CANDIDATES)


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
             if k in ("router_config", "max_num_seqs", "max_num_batched_tokens",
                      "gpu_memory_utilization")}
    check(swept, f"engine/routing knobs were swept at all (got {sorted(swept)})")
    # Routing is still swept, now as an llm-d EPP config file (v0.12.0 removed --routing-policy).
    check("router_config" in swept, "the routing axis was swept (as an EPP config)")
    # refinement must be TARGETED: a full cross product over every shape and load would be
    # thousands of runs. Only the best topologies get refined.
    refined = {(c.get("tp"), c.get("dp", 1)) for c in r.calls
               if any(k in c for k in ("router_config", "max_num_seqs"))}
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
        check(any(w in v for w in ("routing", "batch tokens", "max seqs", "gpu mem util",
                                   "draft token", "drafting", "drafter")),
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
                                             "router_config": "routing/queue-kv.yaml"},
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


def test_replay_workload_takes_the_replay_path():
    print("replay workload drives blis replay, on a lane ramp")
    sc = scenario(budget=32, model="deepseek-v4.1-flash", hardware="h100")
    rep = {"scenario": "inferencex-agentx-mvp", "corpus": "/tmp/traces.jsonl",
           "corpus_sha256": "deadbeef", "max_context_length": "max_model_len"}
    sc["workload"] = {"replay": rep}
    sc["engine"] = {"cache_dtype": "fp8_ds_mla", "max_model_len": 262144}
    r = FakeRunner()
    runs, finalists, _ = run(sc, r)
    # Every run must carry the replay descriptor, or it silently ran a generated workload.
    check(all(x is not None for x in r.replays),
          "every run was handed the replay descriptor")
    # A replay has no request count: it is bounded by its sending window.
    check(all(c.get("num_requests") is None for c in r.calls),
          "no --num-requests is offered for a replay")
    # The scenario's engine knobs must reach every run: at the 32768 default the corpus filter
    # drops EVERY trace, so an unraised max_model_len would make the whole search vacuous.
    check(all(c.get("max_model_len") == 262144 for c in r.calls),
          "the scenario's engine knobs reach every run")
    check(all(c.get("cache_dtype") == "fp8_ds_mla" for c in r.calls),
          "the required cache dtype reaches every run")
    # Lanes, not requests: the ladder is still far shorter than the per-GPU one, but it must run
    # PAST the knee so C* is bracketed. The old cap of 128 was not enough on a fast part -- the
    # b200 winner was still servable at the final rung, so C* was the highest load TESTED and the
    # report called it a measured ceiling anyway (namasl review).
    lanes = sorted({c["concurrency"] for c in r.calls})
    ramp = run_search.concurrency_ramp(32, replay={"scenario": "x"})
    check(max(ramp) >= 512, f"the lane ladder reaches past the knee to bracket C* (got {ramp})")
    check(len(ramp) <= 12, f"and stays a short ladder, not a per-GPU one (got {len(ramp)} rungs)")
    check(max(lanes) <= max(ramp), f"no run exceeds the ladder (got {lanes})")
    check(_workload_label(sc) == "inferencex-agentx-mvp",
          "a replay is named by its scenario, not a missing preset")


def test_shape_workload_passes_no_replay():
    print("a shape workload still takes the generated path")
    r = FakeRunner()
    run(scenario(budget=32), r)
    check(all(x is None for x in r.replays), "no replay descriptor on a shape workload")
    check(all(c.get("num_requests") for c in r.calls), "requests are offered for a shape")


class SlowUnevenRunner(FakeRunner):
    """Finishes candidates in a deliberately different order than they were submitted, so a
    fold that depended on completion order would reorder the ledger and could change the
    winner."""

    def __call__(self, *a, **kw):
        import random
        import time
        config = kw.get("config") or a[11]
        # Narrow-tp shapes "finish" last; wide ones first -- the reverse of submission order.
        time.sleep(0.004 * (config.get("tp", 1) % 5) + random.random() * 0.002)
        return super().__call__(*a, **kw)


def test_parallel_probing_matches_sequential_exactly():
    print("parallel probing is byte-identical to sequential")
    sc = scenario(budget=64)
    seq_runs, seq_fin, seq_rest = run(sc, SlowUnevenRunner(), max_parallel=1)
    par_runs, par_fin, par_rest = run(sc, SlowUnevenRunner(), max_parallel=6)
    # The ledger itself must match in ORDER, not merely as a set: report.py and the gallery read
    # runs.jsonl positionally, and "stable input order" is the determinism claim (INV-6).
    check(json.dumps(seq_runs, sort_keys=True) == json.dumps(par_runs, sort_keys=True),
          f"runs.jsonl is identical and in the same order "
          f"({len(seq_runs)} vs {len(par_runs)} runs)")
    check(json.dumps(seq_fin, sort_keys=True) == json.dumps(par_fin, sort_keys=True),
          "finalists are identical in the same order")
    check(json.dumps(seq_rest[3], sort_keys=True) == json.dumps(par_rest[3], sort_keys=True),
          "dropped[] is identical in the same order")
    w_seq = run_search.rank_finalists(seq_fin)[0]
    w_par = run_search.rank_finalists(par_fin)[0]
    check(run_search.winner_name(w_seq) == run_search.winner_name(w_par),
          f"the same winner is chosen ({run_search.winner_name(w_seq)})")


def test_c_use_is_measured_not_substituted():
    print("c_use is measured, not read off the nearest lower rung")
    r = FakeRunner()
    runs, finalists, _ = run(scenario(budget=64), r)
    probed = {c["concurrency"] for c in r.calls}
    for f in finalists:
        # The headline numbers must come from a run AT c_use. The ramp doubles, so c_use
        # (0.8 x c_star) almost never lands on a rung; substituting the rung below understated
        # throughput by ~37% on every doubling-ramp finalist.
        check(f["at_c_use"]["measured_at_concurrency"] == f["c_use"],
              f"{f['label']}: measured at c={f['at_c_use']['measured_at_concurrency']}, "
              f"claims c_use={f['c_use']}")
        check(f["c_use"] in probed, f"{f['label']}: a run exists at c_use={f['c_use']}")
        # And the curve must carry that point, so the figures draw what the text cites.
        check(any(p["concurrency"] == f["c_use"] for p in f["curve"]),
              f"{f['label']}: the curve includes c_use={f['c_use']}")


def test_c_use_run_is_labelled_as_its_own_phase():
    print("the c_use run is a distinct phase in the ledger")
    r = FakeRunner()
    runs, finalists, _ = run(scenario(budget=64), r)
    phases = {x.get("phase") for x in runs}
    check("c_use" in phases, f"c_use runs are labelled (phases seen: {sorted(p for p in phases if p)})")


def test_submission_seed_reaches_every_run():
    """The submission's seed must be the seed BLIS is given, on EVERY phase.

    main() used to call search() without forwarding scenario["seed"], so the search ran on its
    default seed=0 while scenario.json recorded the submission's seed. A later pass that read the
    recorded seed (the retired drafting backfill) then replayed DIFFERENT traffic than the one it was
    compared against -- AgentX picks each lane's trajectory start from the seed, so the committed
    "matched-load" tables compared two different workloads (namasl review #1)."""
    print("the submission's seed reaches every run, and is recorded truthfully")
    import run_blis
    sc = scenario(budget=64)
    sc["seed"] = 42                                     # NONZERO: 0 would hide the bug
    r = FakeRunner()
    orig = run_blis.run_one
    run_blis.run_one = r
    try:
        run_search.search(sc, "/nonexistent/blis", "/cat", "/reg", "/tmp/scen",
                          num_requests=80, seed=run_search.scenario_seed(sc),
                          model_entry=MOE)
    finally:
        run_blis.run_one = orig
    seeds = set(r.seeds)
    check(seeds == {42}, f"every run used the submission's seed 42 (saw {sorted(seeds)})")
    # and specifically on the later phases, which is where the split showed up in the ledgers
    by_phase = {}
    for ph, sd in zip(r.phases, r.seeds):
        by_phase.setdefault(ph, set()).add(sd)
    for ph, sds in sorted(by_phase.items()):
        check(sds == {42}, f"phase {ph!r} ran at seed 42 (saw {sorted(sds)})")
    check(run_search.scenario_seed({"seed": 42}) == 42, "scenario_seed reads an explicit seed")
    check(run_search.scenario_seed({}) == 0, "scenario_seed defaults to 0 when unstated")
    check(run_search.scenario_seed({"seed": None}) == 0, "a null seed is 0, not a crash")


def test_per_user_is_acceptance_aware_in_the_search():
    """The search must score a drafting arm on the per-TOKEN rate.

    per_user() returned a raw 1000/ITL, but under drafting BLIS reports a VERIFICATION-STEP
    latency, so the search understated every drafting arm by the whole acceptance factor and could
    reject a configuration report.py then displayed as comfortably above the floor
    (namasl review #2)."""
    print("the search scores drafting arms on the acceptance-scaled per-token rate")
    drafting = {"metrics": {"itl_p99_ms": 20.0},
                "deployment": {"engine": {"speculative": {"num_spec_tokens": 3}},
                               "acceptance": {"acceptance_length": 3.0}},
                "config": {"num_speculative_tokens": 3}}
    check(run_search.per_user(drafting) == 150.0,
          f"AL-scaled: 3.0 x 1000/20 = 150, got {run_search.per_user(drafting)}")
    withheld = {"metrics": {"itl_p99_ms": 20.0}, "config": {"num_speculative_tokens": 3}}
    check(run_search.per_user(withheld) is None,
          "no resolved acceptance -> no number (never a known-wrong one)")
    plain = {"metrics": {"itl_p99_ms": 20.0}}
    check(run_search.per_user(plain) == 50.0, "drafting off -> step == token -> 1000/20 = 50")
    # worst instance, not the aggregate: the floor promises a rate to EVERY user
    uneven = {"metrics": {"itl_p99_ms": 10.0},
              "metrics_all_instances": [{"itl_p99_ms": 10.0}, {"itl_p99_ms": 50.0}]}
    check(run_search.per_user(uneven) == 20.0,
          f"worst instance sets the rate (20, not 100), got {run_search.per_user(uneven)}")
    # and the search agrees with the report on the same run
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, ".claude", "skills", "config-search", "scripts"))
    import report as _report
    pu_report, _basis = _report._per_user_rate(drafting)
    check(abs(pu_report - run_search.per_user(drafting)) < 0.05,
          f"search {run_search.per_user(drafting)} == report {pu_report} on one run")


def _fin(label, tpg, tps, c_star=128, **extra):
    return dict({"label": label, "config": {}, "gpus": 32, "c_star": c_star, "c_use": 102,
                 "at_c_use": {"tokens_per_sec": tps, "tokens_per_sec_per_gpu": tpg,
                              "tok_per_s_per_user": 90.0, "itl_p99_ms": 11.0,
                              "ttft_p99_ms": 0.0, "preemption_count": 0,
                              "measured_at_concurrency": 102},
                 "curve": []}, **extra)


def test_partially_explored_arm_does_not_win_a_near_tie():
    """A drafting arm swept on ONE topology cannot take the global recommendation on a near-tie.

    The retired drafting backfill swept K only on the no-drafting winner and inserted those rows into the
    global finalist ranking -- and on B200 that changed the overall winner, with the untested
    runner-up 0.2% behind. Drafting changes throughput AND C*, so an untested topology could well
    be better with drafting on: the comparison was never run (namasl review #4)."""
    print("a drafting arm swept on one topology does not win a near-tie")
    scope = {"drafting_scope": {"swept_on": "8x TP4 + EP", "topologies_swept": 1,
                                "note": "winner only"}}
    # 40.2 vs 40.1 tok/s/GPU: inside TIE_TOLERANCE, i.e. not evidence of anything.
    draft = _fin("8x TP4 + EP", 40.2, 1392.3, variant="1 draft token", **scope)
    plain = _fin("4x TP8 + EP", 40.1, 1282.8)
    ranked = run_search.rank_finalists([draft, plain])
    check(ranked[0] is plain,
          f"the fully-explored topology leads a near-tie (got {run_search.winner_name(ranked[0])})")
    check(run_search.partially_explored(draft), "the drafting arm is flagged as partially explored")
    check(not run_search.partially_explored(plain), "a normal finalist is not flagged")

    # A CLEAR win still ranks: 48 vs 40.1 is well outside the tolerance, so the comparison means
    # something even though only one topology was swept.
    big = _fin("8x TP4 + EP", 48.0, 1700.0, variant="1 draft token", **scope)
    ranked2 = run_search.rank_finalists([big, plain])
    check(ranked2[0] is big, "a clear drafting win still takes the top slot")
    # ...and when it does, the reason says what was actually tested.
    reason = run_search.winner_reason(ranked2[0], 50.0, 0.8)
    check("not swept on the other topologies" in reason,
          f"the winner reason states the scope limit (got: {reason[-120:]!r})")
    check("best draft depth for" in reason, "the reason frames it as best-for-this-topology")


def test_operating_point_itself_must_meet_the_floor():
    """The published operating point must be a MEASURED, SERVABLE point.

    c_star proves the shape was servable at SOME load. These curves are not monotonic -- one slow
    instance at a particular load is enough -- so a shape can pass the ramp either side of c_use
    and fail exactly there. The dedicated c_use run was assigned to `at` whenever it merely
    RETURNED metrics, with the fallback reachable only when `at is None`, so a
    successful-but-unservable point became the finalist's operating point: h100's
    `batch tokens 2,048` arm passes c=8 and c=16 at ~104/~103 tok/s/user yet published its c=13
    point at 4.4, and h200's 4x TP1xDP8 + EP published 0.8 (namasl review).

    Required behaviour: prefer the dedicated c_use measurement when servable; else the HIGHEST
    SERVABLE measured point at or below c_use, with measured_at_concurrency exposing where the
    numbers came from; else drop the shape. Never publish an unservable point, never interpolate.
    """
    print("the published operating point is measured, servable, and says where it came from")

    class UnservableAtCUseOnly(FakeRunner):
        """Servable on every ramp rung, catastrophic ONLY at the dedicated c_use load.

        This is the non-monotonic shape the review asks for: C* passes, the c_use point fails.
        """
        def __call__(self, blis, catalog, registry, scenarios_dir, model, hardware,
                     engine_version, workload, num_requests, seed, horizon, config,
                     phase="probe", timeout=120, defaults=None, replay=None):
            rec = super().__call__(blis, catalog, registry, scenarios_dir, model, hardware,
                                   engine_version, workload, num_requests, seed, horizon, config,
                                   phase=phase, timeout=timeout, defaults=defaults, replay=replay)
            if rec.get("returncode") == 0 and phase == "c_use":
                # one instance at 226 ms, as observed in the h100 ledger
                rec["metrics_all_instances"] = [{"itl_p99_ms": rec["metrics"]["itl_p99_ms"]},
                                                {"itl_p99_ms": 226.0}]
            return rec

    r = UnservableAtCUseOnly()
    runs, finalists, (floor_rate, _fi, _h, dropped) = run(scenario(budget=64), r)
    check(bool(finalists), "shapes servable below c_use are still published, not all dropped")
    for f in finalists:
        a = f["at_c_use"]
        pu = a.get("tok_per_s_per_user")
        check(pu is not None and pu >= floor_rate,
              f"{f['label']}: operating point meets the floor (pu={pu}, floor={floor_rate})")
        # the point must be one actually MEASURED, i.e. present on the curve at that load
        mac = a.get("measured_at_concurrency")
        onc = [p for p in f["curve"] if p["concurrency"] == mac]
        check(len(onc) >= 1, f"{f['label']}: operating point c={mac} is a measured curve point")
        check(abs((onc[0].get("tok_per_s_per_user") or -1) - pu) < 0.05,
              f"{f['label']}: published pu matches the measurement at c={mac}")
        check(mac <= f["c_use"], f"{f['label']}: fallback never exceeds c_use ({mac} <= {f['c_use']})")
        if mac != f["c_use"]:
            check("did not meet" in (a.get("operating_point_note") or ""),
                  f"{f['label']}: a substituted point explains itself")
    # and a shape with NOTHING servable at or below c_use is dropped, with the reason
    class NeverServable(FakeRunner):
        def __call__(self, *a, **kw):
            rec = super().__call__(*a, **kw)
            if rec.get("returncode") == 0:
                rec["metrics_all_instances"] = [{"itl_p99_ms": 900.0}]
            return rec
    _runs, fin2, (_fr, _fi2, _h2, dropped2) = run(scenario(budget=64), NeverServable())
    check(not fin2, "a shape unservable at every load yields no finalist")
    check(bool(dropped2), "and it lands in dropped[] with a reason")


def test_recommended_load_is_the_servable_load_everywhere():
    """c_use IS the recommendation, and every consumer must name the servable load.

    The previous fix put the servable measurement into at_c_use but left c_use at the 0.8xC*
    target -- so winner_reason said "usable load c=13", the report said "run at 13 concurrent
    sessions", the UI card showed 13 and the traffic statistics were read from the raw c=13 run,
    all beside c=8's figures, for a load the measurement had just proved unservable
    (namasl review)."""
    print("the recommended load is the measured servable load, and the reason says so")

    class UnservableAtTarget(FakeRunner):
        def __call__(self, blis, catalog, registry, scenarios_dir, model, hardware,
                     engine_version, workload, num_requests, seed, horizon, config,
                     phase="probe", timeout=120, defaults=None, replay=None):
            rec = super().__call__(blis, catalog, registry, scenarios_dir, model, hardware,
                                   engine_version, workload, num_requests, seed, horizon, config,
                                   phase=phase, timeout=timeout, defaults=defaults, replay=replay)
            if rec.get("returncode") == 0 and phase == "c_use":
                rec["metrics_all_instances"] = [{"itl_p99_ms": rec["metrics"]["itl_p99_ms"]},
                                                {"itl_p99_ms": 226.0}]
            return rec

    runs, finalists, (floor_rate, _fi, headroom, _d) = run(scenario(budget=64), UnservableAtTarget())
    backed = [f for f in finalists if f.get("c_use_target") != f["c_use"]]
    check(bool(backed), "the fixture produced at least one backed-off finalist")
    for f in finalists:
        a = f["at_c_use"]
        target = f.get("c_use_target")
        check(target == round(headroom * f["c_star"]),
              f"{f['label']}: c_use_target is the 0.8xC* arithmetic ({target})")
        # c_use is the measured, servable load -- the single number every consumer reads
        check(a["measured_at_concurrency"] == f["c_use"],
              f"{f['label']}: c_use == measured_at ({f['c_use']} vs {a['measured_at_concurrency']})")
        check(a["tok_per_s_per_user"] >= floor_rate,
              f"{f['label']}: the recommended load meets the floor")
        on = [p for p in f["curve"] if p["concurrency"] == f["c_use"]]
        check(len(on) == 1 and abs(on[0]["tok_per_s_per_user"] - a["tok_per_s_per_user"]) < 0.05,
              f"{f['label']}: at_c_use == curve[c_use] still holds")
        check(f["c_use"] <= target, f"{f['label']}: a back-off only lowers the load")
        reason = run_search.winner_reason(f, floor_rate, headroom)
        check(f"usable load c={f['c_use']}" in reason,
              f"{f['label']}: winner_reason names the servable load, not the target")
        if f["c_use"] != target:
            check("backs off" in reason,
                  f"{f['label']}: winner_reason explains the back-off -- got {reason[-130:]!r}")
            check("backed off from the 0.8xC* target" in (a.get("operating_point_note") or ""),
                  f"{f['label']}: the row's note names the target it backed off from")


def test_curve_itl_shares_the_per_user_basis():
    """A row may not pair a worst-instance pu with an aggregate ITL.

    "0.8 tok/s/user at 13.2 ms" is impossible arithmetic (1000/13.2 = 75.8) and is the same
    displayed-vs-judged split the per-user unification closed, reappearing in the curve."""
    print("the curve's ITL is on the same worst-instance basis as its per-user rate")
    rec = {"returncode": 0, "metrics": {"itl_p99_ms": 13.195, "tokens_per_sec": 65.5},
           "metrics_all_instances": [{"itl_p99_ms": 13.19}, {"itl_p99_ms": 1305.63}]}
    itl = run_search._worst_itl(rec)
    pu = run_search.per_user(rec)
    check(abs(itl - 1305.63) < 0.01, f"ITL is the worst instance (got {itl})")
    check(abs(pu - 1000.0 / itl) < 0.1,
          f"pu == 1000/itl on the SAME basis ({pu} vs {1000.0 / itl:.1f})")


def test_tighter_explicit_itl_cap_is_still_enforced():
    """The rate test subsumes an ITL cap only when rate x cap == 1000."""
    print("an explicit ITL cap tighter than the rate implies is still enforced")
    sc = scenario(budget=64)
    # 50 tok/s/user implies a 20 ms cap; ask for 5 ms, which the rate test does NOT cover.
    sc["interactivity_floor"] = {"tok_per_s_per_user": 50.0, "itl_p99_ms_max": 5.0}
    r = FakeRunner()
    runs, finalists, _ = run(sc, r)
    for f in finalists:
        itl = f["at_c_use"].get("itl_p99_ms")
        check(itl is not None and itl <= 5.0,
              f"finalist {f['label']} honors the tighter 5 ms cap (got {itl})")


def test_no_drafting_sweep_without_a_speculator():
    """A model whose graph declares no speculator must not have draft depth swept.

    kimi-k2.5 declares none, so its search issued 108 drafting runs that were ALL refused by
    run_blis, and the dashboard then rendered "drafter: off" beside "draft tokens (K): 1" for each
    (susiejojo UI review)."""
    print("draft depth is not swept on a model with no speculator")
    variants = topologies.knob_variants({"tp": 8, "num_instances": 4},
                                        skip=("num_speculative_tokens",))
    check(not any("num_speculative_tokens" in v for v in variants),
          "skip=() removes the drafting arms from the refinement grid")
    full = topologies.knob_variants({"tp": 8, "num_instances": 4})
    check(any("num_speculative_tokens" in v for v in full),
          "and they are present when the model HAS a speculator")


def test_coverage_is_recorded_and_not_overstated():
    """The sweep is a RANKED SUBSET of the valid layouts, and must be reported as one.

    candidate_topologies ranks best-first and truncates to MAX_CANDIDATES: on 32 GPUs that is 12 of
    19 valid single-node-footprint layouts, with 7 omitted. Every hero said "Every ... layout that
    fills N GPUs", and move_trail said "enumerated 36 topologies" because it summed
    finalists+dropped AFTER knob refinement -- b200 has 36 finalists but only 12 distinct topology
    labels, the other 24 being variants on three shapes (namasl review)."""
    print("topology coverage is recorded, and the trail counts shapes not variants")
    valid = topologies.candidate_topologies(32, MOE, 10 ** 6, 8)
    ranked = topologies.candidate_topologies(32, MOE, run_search.MAX_CANDIDATES, 8)
    check(len(valid) > len(ranked),
          f"the valid space ({len(valid)}) exceeds what is probed ({len(ranked)})")
    check(len(ranked) == run_search.MAX_CANDIDATES,
          f"the sweep is truncated to MAX_CANDIDATES (got {len(ranked)})")
    # ranking is a prefix: truncation must not reorder, or "highest-ranked N" is a false claim
    check([topologies.label(c) for c in ranked]
          == [topologies.label(c) for c in valid[:len(ranked)]],
          "the probed set is the top-ranked prefix of the valid set")

    r = FakeRunner()
    runs, finalists, _ = run(scenario(budget=64), r)
    labels = {f["label"] for f in finalists}
    n_variants = sum(1 for f in finalists if f.get("variant"))
    check(n_variants > 0, "the fixture produced refinement variants")
    check(len(labels) < len(finalists),
          f"finalists ({len(finalists)}) exceed distinct topologies ({len(labels)}), so a count of "
          f"finalists is NOT a count of topologies")


def test_main_writes_topology_coverage_end_to_end():
    """Run main() to completion and assert the WRITTEN scenario.json.

    This is the seam the unit tests never executed. `coverage` was computed inside search() and
    referenced in main(), so every expensive BLIS probe would run and the search would then die
    with `NameError: name 'coverage' is not defined` before writing the report directory -- with CI
    green, because the other tests call candidate_topologies directly and the run() helper claimed
    to tolerate any return arity (namasl review).

    So this test drives the real producer: a stubbed runner, a temp --out, and then it reads the
    file that main() actually wrote."""
    print("main() writes topology_coverage into scenario.json (producer seam)")
    import run_blis, tempfile, shutil
    d = tempfile.mkdtemp()
    sub = os.path.join(d, "t-coverage.yaml")
    with open(sub, "w") as f:
        f.write("model: deepseek-v4-pro\nhardware: h200\ngpu_budget: 64\n"
                "workload:\n  preset: chatbot\nobjective: dual-pareto\n"
                "interactivity_floor:\n  tok_per_s_per_user: 50.0\n"
                "headroom: 0.8\nseed: 42\n")
    out = os.path.join(d, "out")
    # main() checks the binary exists before probing; the RUNS are stubbed, so any file will do.
    fake_blis = os.path.join(d, "blis")
    open(fake_blis, "w").close()
    orig = run_blis.run_one
    run_blis.run_one = FakeRunner()
    try:
        run_search.main(["--submission", sub, "--out", out,
                         "--blis", fake_blis, "--catalog", "/cat", "--registry", "/reg",
                         "--allow-pin-mismatch"])
        dirs = [x for x in os.listdir(out) if os.path.isdir(os.path.join(out, x))]
        check(len(dirs) == 1, f"main() wrote exactly one run dir (got {dirs})")
        sc = json.load(open(os.path.join(out, dirs[0], "scenario.json")))
        cov = sc.get("topology_coverage")
        check(cov is not None, "scenario.json carries topology_coverage")
        if cov:
            check(cov.get("valid_layouts", 0) >= cov.get("probed", 0) > 0,
                  f"coverage counts are sane (got {cov})")
            check(cov.get("max_candidates") == run_search.MAX_CANDIDATES,
                  "coverage records the cap that produced it")
            check(len(cov.get("omitted") or []) == cov["valid_layouts"] - cov["probed"],
                  "the omitted list accounts for every layout not probed")
        # and the move trail counts SHAPES, not finalists+variants
        pa = json.load(open(os.path.join(out, dirs[0], "pareto.json")))
        trail = " ".join((pa.get("winner") or {}).get("move_trail") or [])
        check("candidate topologies" in trail, f"the trail names candidate topologies ({trail[:70]})")
        labels = {f["label"] for f in pa["finalists"]} | {x["label"] for x in (pa.get("dropped") or [])}
        import re as _re
        m = _re.search(r"enumerated (\d+) candidate topologies", trail)
        check(bool(m) and int(m.group(1)) == len(labels),
              f"the trail's count is the DISTINCT topology count ({m.group(1) if m else '?'} "
              f"vs {len(labels)} labels, {len(pa['finalists'])} finalists)")
    finally:
        run_blis.run_one = orig
        shutil.rmtree(d, ignore_errors=True)


def test_pin_mismatch_is_refused():
    """A declared pin that disagrees with the checkout must stop the search.

    The `resolved` block exists so this is checkable rather than trusted, and the first thing it
    caught was real: pins.env said catalog 0.3.0 while every run used 0.4.0 (ba542f0), making the
    "one pinned stack" label false (namasl review)."""
    print("a pin that disagrees with the checkout is refused, not recorded quietly")
    bad = {"catalog": "0.3.0", "registry": "v0.2.0",
           "resolved": {"catalog": {"describe": "0.4.0", "commit": "ba542f0"},
                        "registry": {"describe": "v0.2.0", "commit": "7f6662d"}}}
    m = run_search.pin_mismatches(bad)
    check(len(m) == 1 and "catalog" in m[0],
          f"the catalog mismatch is named, the matching registry is not (got {m})")
    check("0.3.0" in m[0] and "0.4.0" in m[0], "the message states both versions")
    good = {"catalog": "0.4.0", "resolved": {"catalog": {"describe": "0.4.0"}}}
    check(run_search.pin_mismatches(good) == [], "a matching pin is silent")
    # No resolved block (an older run dir) cannot be checked, and must not false-positive.
    check(run_search.pin_mismatches({"catalog": "0.4.0"}) == [],
          "an unresolvable stack is not reported as a mismatch")


def test_censored_c_star_is_flagged():
    """When the ramp ends with a shape still servable, C* is a lower bound, not a ceiling.

    The b200 winner held 115 tok/s/user at the final rung against a floor of 50, so nothing failed
    above it -- yet the report called C* a "measured ceiling", described a "cliff" that was never
    observed, and recommended 80% of an unbracketed number (namasl review)."""
    print("a C* that was never bracketed is flagged as censored")

    class NeverFails(FakeRunner):
        """Servable at every load, so the ramp runs out before capacity does."""
        def __call__(self, *a, **kw):
            rec = super().__call__(*a, **kw)
            if rec.get("returncode") == 0:
                rec["metrics"]["itl_p99_ms"] = 8.0        # ~125 tok/s/user at every rung
                rec["metrics_all_instances"] = []
            return rec

    runs, finalists, _ = run(scenario(budget=64), NeverFails())
    check(bool(finalists), "the fixture produces finalists")
    for f in finalists:
        top = max(p["concurrency"] for p in f["curve"])
        if f["c_star"] == top:
            check(f.get("c_star_censored") is True,
                  f"{f['label']}: C*={f['c_star']} is the top rung, so it must be flagged censored")
        else:
            check(f.get("c_star_censored") is False,
                  f"{f['label']}: a bracketed C* is not flagged")


def test_provenance_split_is_detected():
    """A ledger whose rows span seeds or stacks is not a comparable set."""
    print("a ledger that spans seeds or stacks is flagged, not silently rendered")
    sys.path.insert(0, os.path.join(ROOT, ".claude", "skills", "config-search", "scripts"))
    import report as _report
    split = [{"cmd": ["blis", "replay", "--seed", "0"]},
             {"cmd": ["blis", "replay", "--seed", "42"]}]
    msg = _report.provenance_split(split)
    check("2 seeds" in msg, f"a seed split is reported (got {msg!r})")
    one = [{"cmd": ["blis", "replay", "--seed", "42"]},
           {"cmd": ["blis", "replay", "--seed", "42"]}]
    check(_report.provenance_split(one) == "", "a coherent ledger is not flagged")
    stacks = [{"pinned_stack": {"registry": "v0.1.1"}}, {"pinned_stack": {"registry": "v0.2.0"}}]
    check("distinct resolved stacks" in _report.provenance_split(stacks),
          "a stack split is reported")
    # The stack branch must be reachable from REAL pipeline output, not only from an injected key:
    # run_blis.run_one now stamps every row via row_stack(). Before that, pinned_stack lived only
    # at scenario level, so no run row carried one and this branch guarded an unreachable path --
    # the test passed while the check could never fire (susiejojo review #2).
    import run_blis as _rb
    st = _rb.row_stack(os.path.join(ROOT, ".build", "blis"),
                       os.path.join(ROOT, os.pardir, "blis-catalog"),
                       os.path.join(ROOT, os.pardir, "blis-registry"))
    if st:
        check(isinstance(st, dict) and st, f"row_stack resolves a real stack identity ({st})")
        rows = [{"pinned_stack": st}, {"pinned_stack": {**st, "registry": "deadbeefcafe"}}]
        check("distinct resolved stacks" in _report.provenance_split(rows),
              "and a split between two REAL row stacks is detected")
        check(_report.provenance_split([{"pinned_stack": st}, {"pinned_stack": dict(st)}]) == "",
              "two rows priced by the same stack are not flagged")


def test_pinned_stack_records_resolved_identity():
    """pins.env names an INTENTION; the report must record what actually priced the runs."""
    print("pinned_stack records resolved commits/digest, not just the pin names")
    ps = run_search._pinned_stack(None, None, None)
    check("blis" in ps and "registry" in ps, "the pin names are still recorded")
    # With real checkouts it resolves commits; without them it must not invent any.
    check("resolved" not in ps or ps["resolved"] == {},
          "no resolved block is fabricated when nothing is on disk")
    cat = os.path.join(ROOT, os.pardir, "blis-catalog")
    if os.path.isdir(os.path.join(cat, ".git")):
        ps2 = run_search._pinned_stack(None, cat, None)
        got = ((ps2.get("resolved") or {}).get("catalog") or {}).get("commit")
        check(bool(got) and len(got) == 40, f"the catalog commit is resolved (got {got!r})")


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
    test_replay_workload_takes_the_replay_path()
    test_shape_workload_passes_no_replay()
    test_parallel_probing_matches_sequential_exactly()
    test_c_use_is_measured_not_substituted()
    test_c_use_run_is_labelled_as_its_own_phase()
    test_submission_seed_reaches_every_run()
    test_per_user_is_acceptance_aware_in_the_search()
    test_partially_explored_arm_does_not_win_a_near_tie()
    test_operating_point_itself_must_meet_the_floor()
    test_recommended_load_is_the_servable_load_everywhere()
    test_curve_itl_shares_the_per_user_basis()
    test_tighter_explicit_itl_cap_is_still_enforced()
    test_no_drafting_sweep_without_a_speculator()
    test_coverage_is_recorded_and_not_overstated()
    test_main_writes_topology_coverage_end_to_end()
    test_pin_mismatch_is_refused()
    test_censored_c_star_is_flagged()
    test_provenance_split_is_detected()
    test_pinned_stack_records_resolved_identity()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
