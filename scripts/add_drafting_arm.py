#!/usr/bin/env python3
"""Sweep speculative-decoding depth on a finished run directory's winning topology.

    python3 -I scripts/add_drafting_arm.py reports/<run-id> --corpus traces.jsonl [--k 1,3,5]

Why this exists as a separate pass rather than only inside run_search: a full re-search is ~2h per
scenario, and the question "how should I set up drafting for this deployment?" is answered by
sweeping K on the shape the search already chose, which is ~20 runs. The pass APPENDS to
runs.jsonl and adds finalists to pareto.json, so the existing topology evidence is kept and the
explorer gains the drafting facets.

It sweeps K at the winner's own C* ladder points plus its c_use, so each drafting arm is
comparable to the baseline at the same loads -- the comparison that matters, since drafting can
change C* itself.

The acceptance length is NOT guessed: it resolves from blis-registry's measured profiles, keyed by
(model, drafter, K, thinking, category), and the depth is skipped when no profile matches rather
than priced at an unknown acceptance.
"""
import argparse, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, ".claude", "skills", "config-search", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_blis            # noqa: E402
import topologies          # noqa: E402
import run_search          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Add a speculative-decoding arm to a finished run dir")
    ap.add_argument("run_dir")
    ap.add_argument("--blis", default=os.path.join(ROOT, ".build", "blis"))
    ap.add_argument("--catalog", default=os.path.join(ROOT, os.pardir, "blis-catalog"))
    ap.add_argument("--registry", default=os.path.join(ROOT, os.pardir, "blis-registry"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus-cache")
    ap.add_argument("--k", default="1,3,5", help="draft depths to sweep")
    ap.add_argument("--run-timeout", type=int, default=1800)
    ap.add_argument("--max-parallel", type=int, default=3)
    args = ap.parse_args(argv)

    D = os.path.abspath(args.run_dir)
    sc = json.load(open(os.path.join(D, "scenario.json")))
    pa = json.load(open(os.path.join(D, "pareto.json")))
    runs = [json.loads(l) for l in open(os.path.join(D, "runs.jsonl")) if l.strip()]

    # This pass is a BACKFILL for run dirs whose search predates native drafting. run_search has
    # swept num_speculative_tokens as a refinement knob since 921c6a7 (on REFINE_TOP topologies,
    # which is the broader experiment namasl asked for), so on a ledger that already holds drafting
    # runs this pass would:
    #   * append DUPLICATE finalists for configs the search already measured, and
    #   * measure them WORSE, because it reuses the baseline's concurrency ladder rather than the
    #     drafting arm's own c_use rung (K=5 landed at c=16 where the search measured c=26).
    # Refuse instead of corrupting the ledger; a re-run search needs no backfill.
    already = sorted({run_blis.draft_depth(r) for r in runs if run_blis.draft_depth(r)})
    if already:
        raise SystemExit(
            f"{D}: the search already swept draft depth (K={already} present in runs.jsonl), and "
            f"on more topologies than this pass covers. This backfill is only for run dirs whose "
            f"search predates native drafting -- running it here would append duplicate finalists "
            f"measured on the wrong ladder. Nothing to do.")

    model, hardware = sc["model"], sc["hardware"]
    chosen = (pa.get("winner") or {}).get("chosen", "")
    win = next((f for f in pa["finalists"] if f["label"] == chosen and not f.get("variant")),
               None) or next((f for f in pa["finalists"] if f["label"] == chosen), None)
    if not win:
        raise SystemExit(f"{D}: cannot find the winning finalist {chosen!r}")

    rep = run_search._replay_of(sc)
    if rep is None:
        raise SystemExit(f"{D}: not a replay run; this pass is written for the AgentX replay")


    # The drafting arms are compared against the baseline runs in THIS ledger at matched loads, so
    # they must replay the same seeded traffic. AgentX picks each lane's trajectory start from the
    # seed; a mismatch silently compares two different workloads and the "matched-load" table is
    # meaningless. Refuse rather than produce it (namasl review #1).
    seed = run_search.scenario_seed(sc)
    ledger_seeds = set()
    for r in runs:
        cmd = r.get("cmd") or []
        if "--seed" in cmd:
            try:
                ledger_seeds.add(int(cmd[cmd.index("--seed") + 1]))
            except (IndexError, ValueError):
                pass
    stale = ledger_seeds - {seed}
    if stale:
        raise SystemExit(
            f"{D}: the committed runs were priced at seed(s) {sorted(stale)} but scenario.json "
            f"says {seed}. A drafting arm added at {seed} would be compared, load for load, "
            f"against a baseline that replayed DIFFERENT traffic. Re-run the search so the whole "
            f"ledger shares one seed, then add the drafting arm.")
    rep = dict(rep, corpus=os.path.abspath(args.corpus))
    if args.corpus_cache:
        rep["corpus_cache"] = os.path.abspath(args.corpus_cache)

    drafter = run_blis.graph_speculator(args.catalog, model)
    if not drafter:
        raise SystemExit(f"{model}: declares no speculator in its catalog graph; nothing to sweep")

    # Acceptance selectors come from the workload, as in run_search.
    accept = {"speculative_category": rep.get("category") or "coding",
              "speculative_thinking": str(rep.get("thinking") or "on")}
    topo = {k: v for k, v in win["config"].items()
            if k in ("tp", "dp", "num_instances", "enable_expert_parallel")}
    # The engine knobs this deployment runs with. scenario.json does NOT persist the submission's
    # `engine:` block, so reading it from there silently produced an empty dict and every drafting
    # run failed on `cache_dtype ""` -- the precision this model requires. Take it from the
    # WINNER'S OWN recorded deployment instead, which is authoritative for what actually ran, and
    # drop the keys run_blis derives per run (block_size is model+part dependent; speculative is
    # what this pass is setting).
    base_engine = dict(sc.get("engine") or {})
    for r in runs:
        d = r.get("deployment") or {}
        par = d.get("parallel") or {}
        if (r.get("returncode") == 0 and par.get("tp") == topo.get("tp")
                and par.get("dp", 1) == topo.get("dp", 1)
                and d.get("num_instances") == topo.get("num_instances")):
            eng = dict(d.get("engine") or {})
            eng.pop("speculative", None)
            eng.pop("block_size", None)
            base_engine = {**eng, **base_engine}
            break
    if not base_engine:
        raise SystemExit(f"{D}: cannot recover the engine knobs from the ledger; "
                         f"re-run the search rather than guessing precision")

    # Compare at the loads the baseline already measured, so every arm is like-for-like.
    loads = sorted({p["concurrency"] for p in win.get("curve") or []} | {win["c_use"]})

    ks = []
    for tok in args.k.split(","):
        k = int(tok)
        al, src = run_blis.acceptance_length(args.registry, model, drafter, k,
                                             accept["speculative_thinking"],
                                             accept["speculative_category"])
        if al is None:
            print(f"  K={k}: no acceptance profile for ({model}, {drafter}, K={k}, "
                  f"thinking={accept['speculative_thinking']}, {accept['speculative_category']}) "
                  f"-- skipped rather than priced at an unknown acceptance", flush=True)
            continue
        ks.append((k, al, src))
    if not ks:
        raise SystemExit("no draft depth had a measured acceptance profile; nothing to run")

    print(f"{os.path.basename(D)}: sweeping {drafter} K={[k for k, _, _ in ks]} on "
          f"{win['label']} over loads {loads}", flush=True)

    import concurrent.futures
    jobs = [({**base_engine, **topo, **accept, "num_speculative_tokens": k, "concurrency": c}, k, c)
            for k, _, _ in ks for c in loads]
    out = [None] * len(jobs)

    def one(i, cfg):
        out[i] = run_blis.run_one(args.blis, args.catalog, args.registry,
                                  os.path.join("/tmp", "cs-draft-scen"), model, hardware,
                                  sc.get("engine_version", "0.29.0"), None, None,
                                  run_search.scenario_seed(sc), None, cfg,
                                  phase="draft", timeout=args.run_timeout, replay=rep)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as ex:
        list(ex.map(lambda t: one(t[0], t[1][0]), list(enumerate(jobs))))

    # Fold in INPUT order so the ledger stays deterministic (INV-6).
    added, curves = 0, {}
    for rec, (cfg, k, c) in zip(out, jobs):
        if rec is None:
            continue
        runs.append(rec)
        added += 1
        m = rec.get("metrics") or {}
        if rec.get("returncode") != 0 or not m.get("tokens_per_sec") or rec.get("error"):
            continue
        itl = m.get("itl_p99_ms")
        # ONE definition for the per-token rate: acceptance-scaled, worst-instance
        # (run_blis.per_user_rate) -- the same one the search scores on and the report judges
        # feasibility with, so this pass cannot publish a rate the report contradicts
        # (namasl review #2).
        pu = run_search.per_user(rec)
        curves.setdefault(k, []).append(
            {"concurrency": c, "tokens_per_sec": round(m["tokens_per_sec"], 1),
             "itl_p99_ms": itl, "tok_per_s_per_user": pu,
             "preemption_count": m.get("preemption_count", 0)})

    floor = pa.get("interactivity_floor") or {}
    floor_rate = float(floor.get("tok_per_s_per_user", 50.0))
    headroom = float(pa.get("headroom", 0.8))
    gpus = win["gpus"]

    for k, al, src in ks:
        curve = sorted(curves.get(k) or [], key=lambda p: p["concurrency"])
        if not curve:
            continue
        servable = [p for p in curve if (p["tok_per_s_per_user"] or 0) >= floor_rate]
        if not servable:
            continue
        c_star = max(p["concurrency"] for p in servable)
        c_use = max(1, int(round(headroom * c_star)))
        at = next((p for p in curve if p["concurrency"] == c_use), None) or \
            min((p for p in curve if p["concurrency"] <= c_use),
                key=lambda p: abs(p["concurrency"] - c_use), default=None)
        if at is None:
            continue
        tps = at["tokens_per_sec"]
        pa["finalists"].append({
            "label": win["label"],
            "variant": topologies.knob_phrase("num_speculative_tokens", k),
            "config": {**topo, **accept, "num_speculative_tokens": k},
            # This arm was swept ONLY on the topology that won with drafting off. Drafting changes
            # both throughput and C*, so it may well have changed WHICH topology is best -- an
            # untested shape could beat this one with drafting on. The scope is recorded on the row
            # so the ranking can refuse to call it a global optimum (namasl review #4).
            "drafting_scope": {"swept_on": win["label"],
                               "topologies_swept": 1,
                               "note": "draft depth explored on the no-drafting winner only; "
                                       "other topologies were not re-tested with drafting on"},
            "gpus": gpus, "c_star": c_star, "c_use": c_use,
            "at_c_use": {"tokens_per_sec": tps, "tokens_per_sec_per_gpu": round(tps / gpus, 1),
                         "tok_per_s_per_user": at["tok_per_s_per_user"],
                         "itl_p99_ms": at["itl_p99_ms"], "ttft_p99_ms": 0.0,
                         "preemption_count": at.get("preemption_count", 0),
                         "measured_at_concurrency": at["concurrency"]},
            "curve": curve,
            "acceptance": {"acceptance_length": al, "source": src},
        })

    pa["finalists"] = run_search.rank_finalists(pa["finalists"])

    # gallery.json is a SIDECAR of display counts, not derived at render time, so appending runs
    # without refreshing it leaves the card quoting the pre-drafting totals (189 where the ledger
    # now holds 210). Recount from the ledger, and refresh the winner blurb, which may have
    # changed: on b200 the drafting arm took the recommendation.
    gpath = os.path.join(D, "gallery.json")
    if os.path.isfile(gpath):
        g = json.load(open(gpath))
        g["n_runs"] = len(runs)
        # Acceptance-aware, worst-instance basis -- the same test the search and the report use.
        # Counting raw aggregate ITL against the cap miscounted every drafting run, whose ITL is a
        # verification-step latency (namasl review #2).
        g["n_servable"] = sum(1 for r in runs
                              if r.get("returncode") == 0
                              and (run_search.per_user(r) or 0) >= floor_rate)
        top = pa["finalists"][0]
        pa["winner"] = {**(pa.get("winner") or {}),
                        "chosen": run_search.winner_name(top),
                        "label": top["label"], "variant": top.get("variant", ""),
                        "reason": run_search.winner_reason(top, floor_rate, headroom)}
        g["reason"] = pa["winner"]["reason"]
        with open(gpath, "w") as f:
            json.dump(g, f, indent=2, sort_keys=True)
    with open(os.path.join(D, "runs.jsonl"), "w") as f:
        for r in runs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(os.path.join(D, "pareto.json"), "w") as f:
        json.dump(pa, f, indent=2, sort_keys=True)
    print(f"  +{added} runs, finalists now {len(pa['finalists'])}; winner stays "
          f"{run_search.winner_name(pa['finalists'][0])}", flush=True)


if __name__ == "__main__":
    main()
