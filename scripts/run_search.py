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
import collections
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_blis  # noqa: E402
import topologies  # noqa: E402
import build_catalog  # noqa: E402

MAX_CANDIDATES = 12
# How many of the best topologies get a knob refinement pass. Each costs len(knob_variants) x the
# ramp, so this directly sets the run budget: 3 shapes x 5 variants x ~12 loads ~= 180 extra runs.
REFINE_TOP = 3


def model_entry(model, catalog_root):
    """The catalog row for `model`, or None. Supplies `arch`, which decides whether the search
    explores MoE-only axes (dp / expert parallelism). A catalog miss falls back to the dense grid
    rather than guessing from the name."""
    try:
        cat = build_catalog.build(os.path.join(catalog_root, os.pardir))
    except Exception:
        return None
    return next((m for m in cat.get("models", []) if m.get("name") == model), None)


def _replay_of(scenario):
    """The scenario's replay descriptor, or None for a generated-shape workload."""
    wl = scenario.get("workload") or {}
    return wl.get("replay") if isinstance(wl, dict) else None


def _git_identity(path):
    """(describe, commit) for a checkout, or (None, None) when it is not a git tree.

    A pin NAME is an intention; the commit is what actually priced the runs. Recording only the
    name let two halves of one report be priced by different registry contents under the same
    label (namasl review #3)."""
    import subprocess
    def _q(*args):
        try:
            out = subprocess.run(args, cwd=path, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return (out.stdout or "").strip() or None if out.returncode == 0 else None
    if not (path and os.path.isdir(path)):
        return None, None
    desc = _q("git", "describe", "--tags", "--always", "--dirty")
    commit = _q("git", "rev-parse", "HEAD")
    return desc, commit


def _file_digest(path):
    """sha256 of a file, or None. The simulator binary has no `version` subcommand, so its digest
    is the only thing that identifies which build priced a run."""
    if not (path and os.path.isfile(path)):
        return None
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _pinned_stack(blis=None, catalog=None, registry=None):
    """What actually priced these runs, recorded into scenario.json.

    pins.env gives the intended REFS; this also resolves each dependency to the commit that was
    checked out and the simulator to its binary digest, so a reader can verify that every run in a
    ledger was priced by one cost model. Without that, a report whose baseline ran before a
    registry bump and whose drafting arms ran after carried a single `registry: v0.1.1` label and
    nothing contradicted it (namasl review #3).
    """
    pins = build_catalog.read_pins()
    out = {"blis": pins.get("BLIS_REF", "unpinned"),
           "kernel": pins.get("KERNEL_REF", "unpinned"),
           "catalog": pins.get("CATALOG_REF", "unpinned"),
           "registry": pins.get("REGISTRY_REF", "unpinned")}
    resolved = {}
    for name, path in (("catalog", catalog), ("registry", registry)):
        desc, commit = _git_identity(path)
        if desc or commit:
            resolved[name] = {"describe": desc, "commit": commit}
    digest = _file_digest(blis)
    if digest:
        resolved["blis"] = {"sha256": digest}
    if resolved:
        out["resolved"] = resolved
    return out


def _corpus_traffic_note(runs):
    """A short description of the traffic a replay actually served, measured from the runs.

    A replay cannot advertise a mean ISL/OSL up front, but it CAN report what it served, which is
    what a reader wants from that slot. Taken from the largest completed run so it describes the
    corpus rather than a tiny probe.
    """
    best = None
    for r in runs:
        m = r.get("metrics") or {}
        c = m.get("completed_requests") or 0
        if r.get("returncode") == 0 and c and m.get("total_input_tokens"):
            if best is None or c > (best.get("completed_requests") or 0):
                best = m
    if not best:
        return ""
    c = best["completed_requests"]
    return (f"~{round(best['total_input_tokens'] / c):,} in / "
            f"{round((best.get('total_output_tokens') or 0) / c):,} out per request")


def _workload_label(scenario):
    """How to name this scenario's workload in progress output and the gallery.

    A shape preset is named by its preset; a replay by its harness scenario. A replay has no
    `preset` at all, so reading only that printed "None" for the agentic runs.
    """
    wl = scenario.get("workload") or {}
    if not isinstance(wl, dict):
        return str(wl)
    rep = wl.get("replay") or {}
    return wl.get("preset") or rep.get("scenario") or "workload"


def concurrency_ramp(gpu_budget, replay=None):
    """Probe ladder for a budget, in stable order.

    The old ladder was the fixed [1..32] whatever the budget. On 256 GPUs that never reached the
    interactivity limit, so the search ranked shapes on an arbitrary mid-curve point. It also
    saturated: with --num-requests 80, c=128 and c=256 returned byte-identical throughput because
    the request pool ran dry (measured). The ladder now grows with the budget, and `requests_for`
    keeps every point supplied.

    A REPLAY's load axis is session-tree LANES, not in-flight requests: one lane drives a whole
    agentic trajectory whose sub-requests fan out from it, so a lane is worth many concurrent
    requests and the ladder has to be far shorter. Measured on deepseek-v4.1-flash / h100 /
    agentx-mvp at one TP8xDP4 instance: 16 lanes holds ITL p99 at 9.2 ms, 24 lanes is already at
    453 ms, and throughput PEAKS near 48 lanes and then falls. Per instance the knee sits between
    16 and 32 lanes, and it scales with the fleet (4 instances hold 64 lanes at 10.5 ms), so the
    ladder is sized per instance-equivalent rather than per GPU.
    """
    if replay:
        # ~8 GPUs per instance-equivalent; probe around the measured per-instance knee and a
        # couple of doublings past it, so C* is bracketed rather than merely approached.
        inst = max(1, int(gpu_budget) // topologies.GPUS_PER_NODE)
        ramp, c = [], 4
        top = max(32, 32 * inst)
        while c <= top:
            ramp.append(c)
            c *= 2
        return ramp
    ramp = [1, 2, 4, 8, 16, 24, 32]
    c = 64
    top = max(32, gpu_budget * 2)      # ~2 sessions per GPU is past any interactivity floor
    while c <= top:
        ramp.append(c)
        c *= 2
    return ramp


# Two configs within this relative distance on the ranking metric are a TIE, and the simpler one
# wins. qwen3-14b produced "32x TP8 · seq64" beating plain "32x TP8" by 0.3 tok/s out of 39,801
# (0.0008%): noise, but it promoted a knob-tweaked config to the headline and implied that
# max_num_seqs=64 was the reason. 0.5% is well inside the simulator's own modelling error.
TIE_TOLERANCE = 0.005


def partially_explored(f):
    """True when a finalist's winning knob was explored on only PART of the search space.

    A drafting arm added by add_drafting_arm.py is swept on the no-drafting winner alone. Drafting
    changes throughput AND C*, so such a row establishes "the best draft depth for that topology",
    not "the best topology+drafting combination" -- promoting it to the global recommendation
    claims an optimum the experiment did not test (namasl review #4)."""
    return bool((f.get("drafting_scope") or {}).get("topologies_swept", 0) == 1)


def rank_finalists(finalists):
    """Best first: most tok/s/GPU, then most tok/s — but a near-tie prefers the SIMPLER config.

    Simpler = fewer engine/routing knobs set away from their defaults. A recommendation should
    only carry a knob when that knob earned its place.

    A PARTIALLY EXPLORED arm (see partially_explored) cannot take the top slot on a near-tie: its
    knob was never tried on the shapes it is being ranked against, so a sub-tolerance win over an
    untested topology is not evidence. It still ranks on a clear win, where the margin exceeds the
    tie tolerance and the comparison means something.
    """
    knobs = [k for k, _ in topologies.KNOB_TIERS]
    nknobs = lambda f: sum(1 for k in knobs if f["config"].get(k) is not None)
    best = max((f["at_c_use"]["tokens_per_sec_per_gpu"] or 0) for f in finalists) if finalists else 0

    def key(f):
        pg = f["at_c_use"]["tokens_per_sec_per_gpu"] or 0
        # Bucket by relative distance from the best, so everything inside the tolerance compares
        # equal on efficiency and the knob count decides.
        bucket = 0 if best <= 0 else int((best - pg) / (best * TIE_TOLERANCE))
        # Inside the top bucket, a fully-explored config outranks a partially-explored one.
        scope = 1 if (bucket == 0 and partially_explored(f)) else 0
        return (bucket, scope, nknobs(f), -(f["at_c_use"]["tokens_per_sec"] or 0))

    return sorted(finalists, key=key)


def winner_name(f):
    """Identifying name for a finalist: topology, plus its knob delta when it has one.

    `pareto.json` names the winner by string, and several finalists legitimately share a topology
    once the search refines knobs — so the bare label would not say WHICH `64x TP4xDP2 + EP` won.
    The knob delta is parenthesised rather than glued on with a separator, so the topology still
    reads as one unit."""
    v = f.get("variant")
    return f"{f['label']} ({v})" if v else f["label"]


def winner_reason(f, floor_rate, headroom):
    """One sentence saying what won and why, naming the knobs that made the difference."""
    a = f["at_c_use"]
    v = f.get("variant")
    how = (f"with {v}" if v else "at default engine and routing settings")
    # A partially-explored knob must be framed for what it is: the best setting FOR THIS TOPOLOGY,
    # not a tested global optimum (namasl review #4).
    scope = ""
    if partially_explored(f):
        scope = (f" This is the best draft depth for {f['label']} — draft depth was not swept on "
                 f"the other topologies, so it is not established as the best overall "
                 f"topology+drafting combination.")
    return (f"{f['label']} {how} gives the most tok/s/GPU "
            f"({a['tokens_per_sec_per_gpu']:g} tok/s/GPU, {a['tokens_per_sec']:g} tok/s fleet) "
            f"while holding the {floor_rate:g} tok/s/user floor; "
            f"usable load c={f['c_use']} (C*={f['c_star']}, headroom {headroom:g}).{scope}")


def footprint_tiers(gpu_budget):
    """Per-instance footprint caps to try, narrowest first.

    One node is where a model that fits belongs — a multi-node instance pays a fabric term on
    every collective. But the largest MoE models do not fit in one node at ANY tp, so when a tier
    yields nothing servable the search widens by a node-doubling rather than declaring the model
    unrunnable. Bounded by the budget: a footprint cannot exceed the GPUs available."""
    tiers, cap = [], topologies.GPUS_PER_NODE
    while cap <= gpu_budget:
        tiers.append(cap)
        cap *= 2
    return tiers or [topologies.GPUS_PER_NODE]


def requests_for(concurrency, base=80):
    """Requests to offer at one probe point.

    A point measures the SHAPE only while requests outnumber sessions; otherwise it measures the
    pool draining. Keep a healthy multiple of the session count."""
    return max(base, concurrency * 4)


def scenario_seed(scenario):
    """The ONE seed a scenario runs under. Every path -- the search ramp, the c_use probe, the
    drafting pass, and the seed recorded in scenario.json -- must read the seed through here, so a
    run's recorded seed is always the seed BLIS was given (namasl review #1)."""
    return int((scenario or {}).get("seed", 0) or 0)


def per_user(rec):
    """Per-user tok/s for a recorded run, through the ONE acceptance-aware, worst-instance
    definition in run_blis.per_user_rate -- the same one report.py's feasibility test and displayed
    number use, so the search can no longer reject a configuration the report shows as servable
    (namasl review #2)."""
    pu, _basis, _itl = run_blis.per_user_rate(rec)
    return round(pu, 1) if pu is not None else None


def search(scenario, blis, catalog, registry, scenarios_dir, num_requests=80, seed=0,
           model_entry=None, run_timeout=120, max_parallel=1):
    """Probe every candidate topology over a concurrency ramp; return runs, finalists, dropped.

    `model_entry` is the catalog row (for `arch`); when omitted the search is dense-only, which is
    the safe grid. Candidate shapes come from scripts/topologies.py — the same enumeration the
    agentic path uses, so this automated sweep explores the MoE expert-parallel axis the skill
    advertises instead of silently varying tp alone."""
    model = scenario["model"]
    hardware = scenario["hardware"]
    budget = int(scenario["gpu_budget"])
    wl = scenario.get("workload") or {}
    workload = wl.get("preset") if isinstance(wl, dict) else wl
    # A REPLAY workload is a different BLIS subcommand, not a --workload value: its traffic is a
    # recorded corpus, so there is no ISL/OSL to generate and no request count to offer. The
    # submission states it under `workload.replay:` and run_blis switches onto `blis replay`.
    replay = wl.get("replay") if isinstance(wl, dict) else None
    # Engine knobs every run of this scenario carries (precision, context window, ...). A replay
    # NEEDS max_model_len raised: the AgentX corpus caps requests at 256k, and every trace whose
    # peak prompt+output exceeds the deployment's window is dropped -- at the 32768 default that
    # is the entire corpus ("no trace fits max_context_length 32768").
    base_engine = dict(scenario.get("engine") or {})
    # Acceptance selectors for a drafting candidate. BLIS REFUSES a drafting scenario whose
    # acceptance does not resolve ("the workload states no thinking mode, which selects the
    # profile"), so these must accompany every num_speculative_tokens > 0 run. They describe the
    # TRAFFIC, so they come from the workload: the AgentX replay declares category `coding` and
    # thinking `on`, and a submission may override either.
    accept = dict(scenario.get("acceptance") or {})
    if replay:
        accept.setdefault("speculative_category", replay.get("category") or "coding")
        accept.setdefault("speculative_thinking", str(replay.get("thinking") or "on"))
    floor = scenario.get("interactivity_floor") or {}
    floor_itl = float(floor.get("itl_p99_ms_max", 20.0))
    floor_rate = float(floor.get("tok_per_s_per_user", 50.0))
    headroom = float(scenario.get("headroom", 0.8))
    engine_version = scenario.get("engine_version", "0.29.0")

    ramp = concurrency_ramp(budget, replay)
    runs, finalists, dropped = [], [], []

    def probe_batch(topos):
        """Probe several candidates CONCURRENTLY, then fold their results in input order.

        Candidates are independent of one another (each climbs its own ramp and the ramp is
        fixed, not adaptive), so they parallelise cleanly. What must NOT vary is the order in
        which results are folded in: `runs` ordering, `c_star` selection and the winner all have
        to come from a stable ordering of the batch rather than from whichever worker finished
        first, or the recommendation stops being reproducible. So each candidate's own probe
        stays sequential (it needs its early break), workers run candidate-wide, and the results
        are collected by INPUT index.

        A replay is memory-hungry in a way a generated-workload run is not: it parses a ~569 MB
        corpus per process and peaks around 4.2 GiB RSS, where sweep_pool's default assumption is
        1 GiB. Hence the explicit per-worker footprint below.
        """
        topos = list(topos)
        if not topos:
            return
        if len(topos) == 1 or max_parallel == 1:
            for t in topos:
                _probe_one(t, runs.append, finalists.append, dropped.append)
            return
        import concurrent.futures
        results = [None] * len(topos)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futs = {}
            for i, t in enumerate(topos):
                # Each candidate collects into its OWN lists; nothing is shared until the fold.
                local = {"runs": [], "fin": [], "drop": []}
                results[i] = local
                futs[ex.submit(_probe_one, t, local["runs"].append, local["fin"].append,
                               local["drop"].append)] = i
            for f in concurrent.futures.as_completed(futs):
                f.result()          # surface any worker exception rather than losing the shape
        for local in results:       # fold in INPUT order -> deterministic regardless of timing
            runs.extend(local["runs"])
            finalists.extend(local["fin"])
            dropped.extend(local["drop"])

    def _probe_one(topo, add_run, add_finalist, add_dropped):
        """Probe one candidate up the ramp; append its runs and either a finalist or a dropped row.

        `label` is provisional — the final labels are assigned once, after every candidate is in,
        so a suffix only appears on knobs that actually VARY across the set (see relabel())."""
        label = topologies.label(topo)
        gpus = topo["tp"] * topo.get("dp", 1) * topo["num_instances"]
        curve, c_star, first_error = [], None, None
        for c in ramp:
            cfg = {**base_engine, **topo, "concurrency": c}
            # Only a DRAFTING run carries the acceptance selectors; passing them with drafting off
            # would record selectors that chose nothing.
            if cfg.get("num_speculative_tokens"):
                cfg = {**accept, **cfg}
            # A replay is bounded by its sending window, not a request count.
            nreq = None if replay else requests_for(c, num_requests)
            rec = run_blis.run_one(blis, catalog, registry, scenarios_dir, model, hardware,
                                   engine_version, workload, nreq, seed,
                                   None, cfg, phase="probe", timeout=run_timeout, replay=replay)
            add_run(rec)
            m = rec.get("metrics") or {}
            # rec["error"] is set on rc=0 runs too: run_blis._completion_issue flags a run that is
            # incomplete / overloaded / still queued. Such a run has a tokens_per_sec but is NOT a
            # usable capacity point — report.py:_feasible already rejects it — so excluding only
            # on returncode/missing-tps let an overloaded run set c_star and the winner's numbers
            # while the report marked that same run infeasible. Treat any error as a failed probe.
            if rec.get("returncode") != 0 or not m.get("tokens_per_sec") or rec.get("error"):
                if first_error is None:
                    first_error = rec.get("error") or f"rc={rec.get('returncode')}"
                # A shape BLIS refuses outright (bad layout / memory) fails at EVERY load, so stop
                # re-probing it; a run that merely missed the floor or overloaded keeps climbing.
                if rec.get("returncode") not in (0, None):
                    break
                continue
            pu = per_user(rec)
            # Servability is the per-TOKEN rate against the floor, on the worst-instance,
            # acceptance-scaled basis (run_blis.per_user_rate). The old second clause compared the
            # RAW p99 ITL to the floor's cap, which under drafting is a VERIFICATION-STEP latency
            # and so rejected arms whose per-token rate is comfortably above the floor. The rate
            # test already subsumes it: AL x 1000/itl >= floor_rate is exactly itl/AL <= floor_itl
            # (namasl review #2).
            servable = pu is not None and pu >= floor_rate
            pt = {"concurrency": c, "tokens_per_sec": round(m["tokens_per_sec"], 1),
                  "itl_p99_ms": m.get("itl_p99_ms"), "tok_per_s_per_user": pu,
                  "preemption_count": m.get("preemption_count", 0)}
            curve.append(pt)
            if servable:
                c_star = c
        if c_star is None:
            # Record WHY, so pareto.json shows the shapes that were tried and failed rather than
            # leaving a single survivor looking like the whole search space.
            add_dropped({"label": label, "config": topo, "gpus": gpus,
                         "reason": (first_error or "no probe met the interactivity floor")})
            return
        c_use = max(1, int(round(headroom * c_star)))
        # MEASURE c_use, do not substitute the nearest lower rung. The ramp doubles, so
        # c_use = 0.8 x c_star almost never lands on a rung: with c*=64 the nearest lower probe is
        # 32, and reporting that run's throughput as "the throughput at c=51" understated the
        # winner by 46% (423.5 vs 786.9 tok/s measured here). The same arithmetic understated
        # every doubling-ramp finalist by ~37%, the committed kimi-k2.5 report included, where
        # "34,307.6 tok/s at c=819" is really the c=512 measurement. One extra run per finalist
        # buys a number that is actually the number the report claims.
        at = next((p for p in curve if p["concurrency"] == c_use), None)
        if at is None:
            cfg = {**base_engine, **topo, "concurrency": c_use}
            if cfg.get("num_speculative_tokens"):
                cfg = {**accept, **cfg}
            nreq = None if replay else requests_for(c_use, num_requests)
            rec = run_blis.run_one(blis, catalog, registry, scenarios_dir, model, hardware,
                                   engine_version, workload, nreq, seed, None, cfg,
                                   phase="c_use", timeout=run_timeout, replay=replay)
            add_run(rec)
            m = rec.get("metrics") or {}
            if rec.get("returncode") == 0 and m.get("tokens_per_sec") and not rec.get("error"):
                pu = per_user(rec)
                at = {"concurrency": c_use, "tokens_per_sec": round(m["tokens_per_sec"], 1),
                      "itl_p99_ms": m.get("itl_p99_ms"), "tok_per_s_per_user": pu,
                      "preemption_count": m.get("preemption_count", 0)}
                # Keep the curve ordered by load, so the figures draw it correctly.
                curve.append(at)
                curve.sort(key=lambda p: p["concurrency"])
        if at is None:
            # The dedicated c_use run failed, so fall back to the best probe at or below it and
            # say so, rather than dropping a shape that IS servable at c_star.
            at = min((p for p in curve if p["concurrency"] <= c_use),
                     key=lambda p: abs(p["concurrency"] - c_use), default=None)
        if at is None:
            add_dropped({"label": label, "config": topo, "gpus": gpus,
                         "reason": f"no probe at or below the usable load c={c_use}"})
            return
        tps = at["tokens_per_sec"]
        add_finalist({
            "label": label,
            "config": topo, "gpus": gpus, "c_star": c_star, "c_use": c_use,
            "at_c_use": {"tokens_per_sec": tps, "tokens_per_sec_per_gpu": round(tps / gpus, 1),
                         "tok_per_s_per_user": at["tok_per_s_per_user"], "itl_p99_ms": at["itl_p99_ms"],
                         "ttft_p99_ms": 0.0, "preemption_count": at.get("preemption_count", 0),
                         # The load these numbers were actually MEASURED at. Normally c_use; it
                         # differs only when the dedicated c_use run failed and the nearest lower
                         # probe stood in, and then the discrepancy must be visible rather than
                         # presented as the figure at c_use.
                         "measured_at_concurrency": at["concurrency"]},
            "curve": curve,
        })

    # Single-node footprints first — what a model that fits should use, with no fabric term to pay.
    # If NOTHING there is servable, the model may simply not fit in one node (kimi-k3 needs 332.3
    # GiB per rank at tp=8 on a 141 GiB H200 and only fits from tp=32 up), so widen the per-instance
    # footprint a node at a time rather than reporting "no servable configuration" for a model BLIS
    # can actually run. Escalation only ever ADDS shapes, so a model that fits is unaffected.
    for cap in footprint_tiers(budget):
        seen = {tuple(sorted(c.items())) for c in (f["config"] for f in finalists)} | \
               {tuple(sorted(c.items())) for c in (d["config"] for d in dropped)}
        probe_batch(topo for topo in topologies.candidate_topologies(
            budget, model_entry, MAX_CANDIDATES, cap)
            if tuple(sorted(topo.items())) not in seen)
        if finalists:
            break

    # Knob refinement, coordinate-wise on the BEST topologies only. Routing and the batch/KV
    # limits are real levers — on kimi-k2.5 routing swung throughput 2.3x and max_num_seqs=64 cost
    # 29% — but crossing them with every shape and load would be thousands of near-identical runs.
    # Refining the top few shapes finds the same binding knob for ~1/9th of the budget.
    # The refinement batch is built from the ranking BEFORE any of it runs, so the set of
    # variants explored does not depend on worker timing.
    probe_batch(variant for base in rank_finalists(finalists)[:REFINE_TOP]
                for variant in topologies.knob_variants(base["config"]))

    relabel(finalists, dropped)
    return runs, finalists, (floor_rate, floor_itl, headroom, dropped)


def relabel(finalists, dropped):
    """Keep `label` pure topology and describe knob deltas in a separate `variant` field.

    The label is what a reader scans — GPUs x TP x EP x instances, the things that dominate an
    LLM serving decision. Engine and routing knobs do NOT belong in it: jamming them in produced
    `64x TP4xDP2 + EP · ll · b8k · seq256 · mem0.9`, with invented abbreviations that explain
    nothing. They go in `variant` (a spelled-out phrase) and `knobs` (the raw values), and the
    report surfaces them as filters and panel fields.

    `variant` is scoped to the topology group and measured against that group's baseline (its most
    common value per knob), so an unrefined shape and the all-defaults row both get "". Mutates in
    place."""
    entries = list(finalists) + list(dropped)
    if not entries:
        return ()
    knobs = [k for k, _ in topologies.KNOB_TIERS]
    groups = collections.defaultdict(list)
    for e in entries:
        c = e["config"]
        groups[(c.get("tp"), c.get("dp", 1), bool(c.get("enable_expert_parallel")),
                c.get("num_instances"))].append(e)
    seen = set()
    for rows in groups.values():
        varying = tuple(k for k in knobs
                        if len({json.dumps(r["config"].get(k)) for r in rows}) > 1)
        seen.update(varying)
        base = {k: collections.Counter(json.dumps(r["config"].get(k))
                                       for r in rows).most_common(1)[0][0] for k in varying}
        for e in rows:
            e["label"] = topologies.label(e["config"])          # topology only
            e["knobs"] = {k: e["config"].get(k) for k in varying}
            e["variant"] = ", ".join(
                topologies.knob_phrase(k, e["config"][k]) for k in varying
                if e["config"].get(k) is not None
                and json.dumps(e["config"].get(k)) != base[k])
    return tuple(k for k in knobs if k in seen)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deterministic mini config-search -> reports/<run_id>/")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--blis", default=os.path.join(ROOT, ".build", "blis"))
    ap.add_argument("--catalog", default=os.path.join(ROOT, os.pardir, "blis-catalog"))
    ap.add_argument("--registry", default=os.path.join(ROOT, os.pardir, "blis-registry"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports"))
    ap.add_argument("--num-requests", type=int, default=80)
    # A replay parses a ~569 MB corpus and simulates a 900 s sending window, so a run takes far
    # longer than a generated-workload probe. The default 120 s would time out every one.
    ap.add_argument("--run-timeout", type=int, default=120,
                    help="per-BLIS-run timeout in seconds (raise it for a replay workload)")
    # The corpus is machine-specific and ~569 MB, so the submission states its DIGEST (which
    # pins the bytes) while the path is supplied here. The digest is checked by BLIS, so a
    # wrong file fails loudly rather than replaying different traffic under the same name.
    ap.add_argument("--corpus", help="replay corpus file (required by a replay submission)")
    ap.add_argument("--corpus-cache", help="dir for parsed-corpus snapshots (a large speedup)")
    # Candidates probe independently, so they parallelise; results are still folded in input
    # order, so the recommendation does not depend on worker timing. Default 1 keeps the old
    # sequential behaviour for anyone who does not ask for width.
    ap.add_argument("--max-parallel", type=int, default=1,
                    help="probe this many candidate topologies at once (default 1). A replay "
                         "peaks near 4.2 GiB RSS per run -- it parses the whole corpus -- so "
                         "size this against RAM, not just cores")
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

    # A replay submission needs the corpus path from the command line; fail before any run
    # rather than once per probe.
    rep = _replay_of(scenario)
    if rep is not None:
        if not args.corpus:
            raise SystemExit(
                f"{os.path.basename(sub)} is a replay submission (workload.replay) and needs "
                f"--corpus: the corpus is not vendored by blis-catalog. Fetch traces.jsonl for "
                f"{rep.get('scenario')} and pass its path; the submission's corpus_sha256 holds "
                f"it to the right bytes.")
        if not os.path.isfile(args.corpus):
            raise SystemExit(f"--corpus not found at {args.corpus}")
        rep = dict(rep, corpus=os.path.abspath(args.corpus))
        if args.corpus_cache:
            rep["corpus_cache"] = os.path.abspath(args.corpus_cache)
        scenario = dict(scenario, workload=dict(scenario["workload"], replay=rep))

    entry = model_entry(scenario["model"], os.path.abspath(args.catalog))
    arch = (entry or {}).get("arch", "unknown")
    print(f"searching {scenario.get('model')} ({arch}) on {scenario.get('gpu_budget')}x "
          f"{scenario.get('hardware')} ({_workload_label(scenario)})…", flush=True)
    # The submission's seed MUST reach every run. Omitting it left `search` on its default seed=0
    # while scenario.json recorded the submission's 42, so a later pass that read the recorded seed
    # (add_drafting_arm) replayed DIFFERENT traffic than the baseline it was compared against --
    # AgentX uses the seed to choose each lane's trajectory start, so the "matched-load" tables
    # were not matched workloads (namasl review #1).
    runs, finalists, (floor_rate, floor_itl, headroom, dropped) = search(
        scenario, blis, args.catalog, args.registry, scen_dir, num_requests=args.num_requests,
        seed=scenario_seed(scenario),
        model_entry=entry, run_timeout=args.run_timeout, max_parallel=max(1, args.max_parallel))
    if not finalists:
        detail = "; ".join(f"{d['label']}: {d['reason'][:120]}" for d in dropped[:4])
        raise SystemExit("no servable configuration found for this scenario (nothing met the "
                         f"floor). Tried {len(dropped)} shape(s): {detail}")

    # winner: dual-pareto proxy — most tok/s/GPU, tie-break total tok/s
    finalists[:] = rank_finalists(finalists)
    winner = finalists[0]
    # Same acceptance-aware basis as servability above, so the headline count cannot disagree with
    # the rows the report marks servable (namasl review #2).
    n_servable = sum(1 for r in runs
                     if r.get("returncode") == 0 and (per_user(r) or 0) >= floor_rate)

    scenario_out = {"run_id": run_id, "model": scenario["model"], "hardware": scenario["hardware"],
                    "gpu_budget": scenario["gpu_budget"], "workload": scenario.get("workload"),
                    "objective": "dual-pareto",
                    "interactivity_floor": {"tok_per_s_per_user": floor_rate, "itl_p99_ms_max": floor_itl},
                    "headroom": headroom, "seed": int(scenario.get("seed", 0) or 0),
                    # Which simulator priced these runs. pins.env is the single source of truth
                    # (build_catalog.read_pins); without this the report footer printed "?" for
                    # every component of the stack.
                    "pinned_stack": _pinned_stack(blis, args.catalog, args.registry)}
    # The move trail must say what was actually explored: the previous wording ("single-node
    # topologies") described a tp-only sweep and gave no hint that EP was never tried.
    ep_runs = sum(1 for r in runs if ((r.get("deployment") or {}).get("parallel") or {})
                  .get("enable_expert_parallel"))
    trail = [f"enumerated {len(finalists) + len(dropped)} topologies within budget"
             + (f" ({arch}: tp x dp footprints x expert-parallel arms)" if arch == "MoE"
                else f" ({arch}: tp tiers)"),
             (f"probed a session-lane ramp per topology with real BLIS replay runs"
              f" (ladder to {max(concurrency_ramp(int(scenario['gpu_budget']), _replay_of(scenario)))} lanes)"
              if _replay_of(scenario) else
              "probed a concurrency ramp per topology with real BLIS runs"
              f" (ladder to c={max(concurrency_ramp(int(scenario['gpu_budget'])))})"),
             f"ranked finalists on usable capacity (headroom {headroom})"]
    if arch == "MoE":
        trail.insert(1, f"compared expert-parallel against its non-EP control ({ep_runs} EP runs)")
    if dropped:
        trail.append(f"dropped {len(dropped)} shape(s) BLIS refused or that missed the floor")
    pareto = {"objective": "dual-pareto",
              "interactivity_floor": {"tok_per_s_per_user": floor_rate, "itl_p99_ms_max": floor_itl},
              "headroom": headroom, "finalists": finalists, "dropped": dropped,
              "winner": {"chosen": winner_name(winner), "budget_max": winner_name(winner),
                         "efficiency_per_gpu": winner_name(winner),
                         "label": winner["label"], "variant": winner.get("variant", ""),
                         "reason": winner_reason(winner, floor_rate, headroom),
                         "move_trail": trail}}
    # A replay has no ISL/OSL by construction -- its per-request lengths live in the corpus -- so
    # the chip that would carry a shape's "256/256" describes the traffic instead of rendering
    # blank. A shape workload keeps the empty string the gallery already expects.
    rep_out = _replay_of(scenario)
    isl_osl = ""
    if rep_out:
        isl_osl = f"replay · {_corpus_traffic_note(runs)}" if _corpus_traffic_note(runs) else "replay"
    gallery = {"synthetic": False, "isl_osl": isl_osl,
               "generated": __import__("datetime").date.today().isoformat(),
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
