#!/usr/bin/env python3
"""Backfill the DERIVED fields a finished run dir predates, without re-measuring.

    python3 -I scripts/backfill_derived_fields.py reports/<run-id> [...]

Two fields were added to the producer after these reports were searched:

  * `scenario.topology_coverage` -- how much of the valid layout space the sweep covered
  * `finalist.c_star_censored`   -- whether C* was the highest load TESTED (a lower bound) rather
                                    than an observed ceiling

NEITHER depends on a BLIS measurement. topology_coverage is a function of (gpu_budget, model,
footprint caps attempted) through candidate_topologies; c_star_censored is `c_star == max(curve)`,
and the curve is already in pareto.json. So a re-search would reproduce byte-identical values for
them -- at ~50 h of CPU for the two remaining AgentX reports, which is not a sensible way to buy
a value that is already determined by the file.

This pass therefore recomputes exactly those two fields from the artifacts and leaves every
measurement untouched. It REFUSES to change anything else: no finalist metric, no ledger row, no
winner. The narrative is then re-rendered so its prose matches (the honest-coverage and
bracketed-capacity wording), which is the whole point of having the fields.

What it cannot do: a field that depends on runs the search never performed. The footprint caps are
INFERRED from the shapes actually present, which is sound because escalation only ever adds wider
footprints -- if the widest finalist fits one node, no wider tier was attempted.
"""
import json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, ".claude", "skills", "config-search", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import topologies          # noqa: E402
import run_search          # noqa: E402


def footprint_caps_used(pareto):
    """The footprint tiers the search must have attempted, from the shapes it produced.

    Escalation only ever ADDS wider footprints (run_search widens a node at a time and stops at the
    first tier with a servable shape), so the widest per-instance footprint present bounds it."""
    widest = 0
    for f in (pareto.get("finalists") or []) + (pareto.get("dropped") or []):
        cfg = f.get("config") or {}
        widest = max(widest, int(cfg.get("tp", 1)) * int(cfg.get("dp", 1) or 1))
    return widest or topologies.GPUS_PER_NODE


def backfill(run_dir, catalog=None):
    sc_path = os.path.join(run_dir, "scenario.json")
    pa_path = os.path.join(run_dir, "pareto.json")
    sc = json.load(open(sc_path))
    pa = json.load(open(pa_path))
    changed = []

    budget = int(sc["gpu_budget"])
    # The catalog must be the SAME one the search used: `arch` decides whether the candidate grid
    # is the MoE tp x dp x EP space or the narrow dense one, so reading a different catalog gives a
    # different valid-layout count (19 vs 4 for deepseek-v4-pro). Validation against a real search
    # caught exactly that: the search had been stubbed with no catalog while this read the live one.
    entry = run_search.model_entry(
        sc["model"], os.path.abspath(catalog or os.path.join(ROOT, os.pardir, "blis-catalog")))

    if sc.get("topology_coverage") is None:
        cap = footprint_caps_used(pa)
        tiers = [c for c in run_search.footprint_tiers(budget) if c <= cap]
        probed = set()
        for c in tiers:
            probed |= {topologies.label(x) for x in topologies.candidate_topologies(
                budget, entry, run_search.MAX_CANDIDATES, c)}
        every = [topologies.label(x) for x in topologies.candidate_topologies(
            budget, entry, 10 ** 6, cap)]
        sc["topology_coverage"] = {
            "valid_layouts": len(every),
            "probed": len(probed),
            "omitted": [l for l in every if l not in probed],
            "max_candidates": run_search.MAX_CANDIDATES,
            "footprint_caps_attempted": tiers,
            # Say plainly that this was derived after the fact, so a reader never mistakes it for
            # something the search recorded at the time.
            "derived_post_hoc": True,
        }
        changed.append(f"topology_coverage ({len(probed)} of {len(every)})")

    # The DECLARED pin label, when the resolved block already proves what ran. These reports were
    # searched while pins.env still said catalog 0.3.0; the runs used 0.4.0 (ba542f0), which the
    # resolved block recorded faithfully, and pins.env has since been corrected to 0.4.0. Bringing
    # the label into line with the commit the artifact ITSELF records is a relabel, not a claim --
    # and the `resolved` commit stays beside it so the correction is checkable.
    ps = sc.get("pinned_stack") or {}
    res = ps.get("resolved") or {}
    for name in ("catalog", "registry"):
        got = (res.get(name) or {}).get("describe")
        if got and ps.get(name) and ps[name] != got:
            ps[f"{name}_declared_at_search_time"] = ps[name]
            ps[name] = got
            changed.append(f"{name} label {ps[f'{name}_declared_at_search_time']} -> {got} "
                           f"(matching the recorded commit)")

    n_cen = 0
    for f in pa.get("finalists") or []:
        if f.get("c_star_censored") is None:
            top = max((p["concurrency"] for p in f.get("curve") or []), default=None)
            f["c_star_censored"] = (top is not None and f.get("c_star") == top)
            n_cen += 1
    if n_cen:
        changed.append(f"c_star_censored on {n_cen} finalist(s)")

    if changed:
        sc["pinned_stack"] = ps or sc.get("pinned_stack")
        json.dump(sc, open(sc_path, "w"), indent=2, sort_keys=True)
        json.dump(pa, open(pa_path, "w"), indent=2, sort_keys=True)
    return changed


def main(argv=None):
    args = list(argv if argv is not None else sys.argv[1:])
    catalog = None
    if "--catalog" in args:
        i = args.index("--catalog")
        catalog = args[i + 1]
        del args[i:i + 2]
    if not args or args[0] in ("-h", "--help"):
        raise SystemExit(__doc__.strip().splitlines()[0] + "\n\n"
                         "usage: backfill_derived_fields.py [--catalog DIR] <run-dir> [...]")
    for d in args:
        ch = backfill(d, catalog)
        print(f"{os.path.basename(d)}: " + (", ".join(ch) if ch else "already complete"))


if __name__ == "__main__":
    main()
