#!/usr/bin/env python3
"""Hermetic tests for candidate topology enumeration (no BLIS, no network).

    python3 -I scripts/test_topologies.py

The rules under test are the ones BLIS itself enforces, measured against the pinned binary:

- a dense model refuses dp>1 ("states dp 2 for a dense model; DP placement is modelled for MoE"),
  and expert parallelism shards nothing without experts;
- an MoE model's expert-parallel width is derived as tp*pcp*dp, so EP with a width of 1 is a
  no-op the schema itself warns about;
- a per-instance footprint must fit one node (tp*dp <= 8) and divide the GPU budget.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import topologies as tp  # noqa: E402

PASS = 0
FAIL = 0

DENSE = {"name": "qwen3-14b", "arch": "dense"}
MOE = {"name": "deepseek-v4-pro", "arch": "MoE", "experts": 384}


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def footprints(cands):
    return {(c["tp"], c.get("dp", 1), bool(c.get("enable_expert_parallel"))) for c in cands}


def test_dense():
    print("dense models")
    cands = tp.candidate_topologies(16, DENSE)
    check(cands, "dense: some candidates")
    # BLIS rejects dp>1 on a dense model outright, and EP shards nothing without experts.
    check(all(c.get("dp", 1) == 1 for c in cands), "dense: never dp>1 (BLIS refuses it)")
    check(all(not c.get("enable_expert_parallel") for c in cands), "dense: never expert-parallel")
    check({c["tp"] for c in cands} == {1, 2, 4, 8}, "dense: tp tiers 1/2/4/8 within budget")
    # every candidate fills the budget
    check(all(c["tp"] * c["num_instances"] == 16 for c in cands), "dense: candidates fill the budget")


def test_moe_explores_ep():
    print("MoE models explore EP")
    cands = tp.candidate_topologies(256, MOE)
    fps = footprints(cands)
    # The headline regression: EP was never enumerated, so the 24%-better arm was unreachable.
    check(any(ep for (_, _, ep) in fps), "MoE: at least one expert-parallel candidate")
    check((8, 1, True) in fps, "MoE: TP8 + EP enumerated (the measured winner)")
    check((8, 1, False) in fps, "MoE: TP8 without EP enumerated (its control)")
    # tp=4 alone dies on memory at width 1; tp=4,dp=2 runs. dp is as missing an axis as EP.
    check((4, 2, True) in fps, "MoE: TP4xDP2 + EP enumerated")
    check(any(c.get("dp", 1) > 1 for c in cands), "MoE: dp is explored, not pinned to 1")


def test_moe_skips_width_one_ep():
    print("MoE skips EP layouts that shard nothing")
    for c in tp.candidate_topologies(256, MOE):
        if c.get("enable_expert_parallel"):
            w = c["tp"] * c.get("dp", 1)       # derived width = tp*pcp*dp, pcp unset = 1
            check(w > 1, f"EP candidate tp={c['tp']} dp={c.get('dp', 1)} has width {w} > 1")


def test_footprint_and_budget():
    print("footprint + budget constraints")
    for budget in (8, 16, 64, 256):
        for c in tp.candidate_topologies(budget, MOE):
            fp = c["tp"] * c.get("dp", 1)
            check(fp <= 8, f"budget {budget}: footprint {fp} fits one node")
            check(c["num_instances"] >= 1, f"budget {budget}: at least one instance")
            check(fp * c["num_instances"] == budget,
                  f"budget {budget}: tp*dp*ni == budget for {c}")
    # a budget smaller than the widest footprint still yields something
    small = tp.candidate_topologies(2, MOE)
    check(small, "budget 2: still enumerates a candidate")
    check(all(c["tp"] * c.get("dp", 1) <= 2 for c in small), "budget 2: footprints fit the budget")


def test_stable_order():
    print("stable order (determinism)")
    a = tp.candidate_topologies(256, MOE)
    b = tp.candidate_topologies(256, MOE)
    check(a == b, "same inputs -> identical candidate list")
    # EP arms must survive any truncation: the old MAX_CANDIDATES sorted by -tp and would have
    # cut every EP variant off the end of the list.
    head = footprints(a[:4])
    check(any(ep for (_, _, ep) in head), "an EP candidate survives in the first 4")


def test_multinode_footprints_for_oversized_models():
    print("multi-node footprints")
    # Some MoE models cannot fit in ONE node at any tp: kimi-k3 reports 332.3 GiB of per-rank
    # fixed occupancy at tp=8 on a 141 GiB H200 and only fits from tp=32 (4 nodes) up, where BLIS
    # runs it fine. A one-node-only grid makes such a model unreachable rather than slow, so the
    # caller can ask for wider footprints.
    big = tp.candidate_topologies(64, MOE, max_candidates=40, max_footprint=32)
    fps = {c["tp"] * c.get("dp", 1) for c in big}
    check(any(f > 8 for f in fps), f"multi-node footprints offered when asked (got {sorted(fps)})")
    check(32 in fps, "a 32-GPU (4-node) footprint is reachable")
    # Every footprint past one node must fill whole nodes -- run_blis refuses a straddling one.
    for c in big:
        f = c["tp"] * c.get("dp", 1)
        check(f <= 8 or f % 8 == 0, f"footprint {f} fills whole nodes")
    # and the budget still bounds it
    for c in big:
        check(c["tp"] * c.get("dp", 1) * c["num_instances"] == 64, f"{c} fills the 64-GPU budget")


def test_single_node_default_is_unchanged():
    print("single-node default")
    # The default must stay one node: widening it for every model would spend runs on shapes a
    # fitting model has no use for, and multi-node pricing carries a fabric term.
    for c in tp.candidate_topologies(256, MOE):
        check(c["tp"] * c.get("dp", 1) <= 8, f"default grid stays within one node: {c}")


def test_knob_variants_cover_the_levers_that_bind():
    print("knob variants")
    base = {"tp": 8, "dp": 1, "enable_expert_parallel": True, "num_instances": 8}
    vs = tp.knob_variants(base)
    check(vs, "some knob variants are produced")
    # every variant keeps the topology it refines — a knob sweep must not change the shape
    for v in vs:
        for k in ("tp", "dp", "enable_expert_parallel", "num_instances"):
            check(v.get(k) == base.get(k), f"variant preserves {k}: {v}")
    keys = {k for v in vs for k in v if k not in base}
    # Routing is still swept, but as an llm-d EndpointPickerConfig FILE: v0.12.0 removed
    # --routing-policy, so an arm is a path (see routing/README.md) and the old policy names
    # (round-robin / least-loaded / always-busiest) no longer exist upstream at all.
    check("router_config" in keys, f"router_config is swept (got {sorted(keys)})")
    check("routing_policy" not in keys,
          "the flag v0.12.0 removed is NOT swept (the runner rejects it)")
    arms = {v.get("router_config") for v in vs if v.get("router_config")}
    for a in arms:
        check(str(a).startswith("routing/") and str(a).endswith(".yaml"),
              f"{a!r} is a routing-arm EPP config path")
        check(os.path.exists(os.path.join(ROOT, a)),
              f"{a!r} exists -- an arm naming a missing file fails every refinement run")
    # the batch/KV knobs bind DOWNWARD (max_num_seqs 16 costs 64% throughput), so the grid must
    # probe below the default, not only above it
    mns = {v.get("max_num_seqs") for v in vs if v.get("max_num_seqs")}
    check(any(m < 256 for m in mns) if mns else True, f"max_num_seqs probes below default ({mns})")


def test_knob_variants_are_bounded_and_deterministic():
    print("knob variant bounds")
    base = {"tp": 8, "dp": 1, "num_instances": 8}
    a, b = tp.knob_variants(base), tp.knob_variants(base)
    check(a == b, "same input -> identical variants")
    check(len(a) <= 12, f"variant count stays small ({len(a)}) — this multiplies the run budget")
    # one knob at a time: a coordinate sweep, not a cross product (which would be 5000+ runs)
    for v in a:
        extra = [k for k in v if k not in base]
        check(len(extra) <= 1, f"variant changes at most one knob: {extra}")


def test_unknown_arch_is_conservative():
    print("unknown arch")
    # No arch (a catalog miss) must not invent dp/EP for a model that may be dense: BLIS would
    # reject the run, and a silently-dropped candidate is worse than a narrow one.
    cands = tp.candidate_topologies(16, {"name": "mystery"})
    check(all(c.get("dp", 1) == 1 and not c.get("enable_expert_parallel") for c in cands),
          "unknown arch: falls back to the dense (safe) grid")


if __name__ == "__main__":
    test_dense()
    test_moe_explores_ep()
    test_moe_skips_width_one_ep()
    test_footprint_and_budget()
    test_stable_order()
    test_multinode_footprints_for_oversized_models()
    test_single_node_default_is_unchanged()
    test_knob_variants_cover_the_levers_that_bind()
    test_knob_variants_are_bounded_and_deterministic()
    test_unknown_arch_is_conservative()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
