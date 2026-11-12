#!/usr/bin/env python3
"""The Pareto objective: the non-dominated front of throughput per GPU against interactivity.

The axes are the metric registry's, not this repository's (design DEV-8):

  x  interactivity_of_mean_tpot  1000 / mean(TPOT in ms), tokens/s per user
                                 (InferenceX mean_intvty; not the mean of per-request 1/TPOT, which is
                                 the registry's interactivity_mean and reads 2 to 27 percent higher)
  y  total_tokens_per_second_per_gpu            (input + output tokens)/s over ALL the GPUs, or
     output_tokens_per_second_per_decode_gpu    output tokens/s over the decode GPUs

Both are higher-is-better, so a point is dominated when another is at least as good on both axes and
better on one. Only runs that held the scenario's SLOs are points: the SLO gate is a separate
constraint, not an axis.

NO TIE RULE. Two points are compared exactly. A preference for P/D "when it does no worse" needs to know
when two deployments are not distinguished, and that is a statement about run-to-run noise: paired
replications under common random numbers, an interval on the difference, three outcomes and control of
the false-winner rate over all the comparisons (design DEV-11). One run per arm gives no such interval,
and a fixed relative tolerance copied from another repository would stand in for one (X7). So no
tolerance displaces anything here: the front is the exact non-dominated set, and the report says it
carries no interval. The P/D preference arrives with the replications (M3).
"""

RANKINGS = ("scalar", "pareto")
# throughput form -> (row field, registry metric)
THROUGHPUT_FORMS = {
    "total": ("total_tokens_per_sec_per_gpu", "total_tokens_per_second_per_gpu"),
    "decode": ("output_tokens_per_sec_per_decode_gpu", "output_tokens_per_second_per_decode_gpu"),
}
X_METRIC = "interactivity_of_mean_tpot"
DEFAULT_OBJECTIVE = {"ranking": "scalar", "throughput": "total"}


def objective_of(spec_objective):
    """The spec's `objective:` block with defaults filled in."""
    out = dict(DEFAULT_OBJECTIVE)
    out.update({k: v for k, v in (spec_objective or {}).items() if v is not None})
    return out


def interactivity_of(metrics):
    """(tokens/s per user, basis) for a run's flattened measured-window metrics, or (None, "").

    Only the Result's own registered `interactivity_of_mean_tpot` counts (DEV-8: a definition is read
    from the registry, never rebuilt from another field). A Result that does not state it, or states a
    figure that is not a positive number, leaves the run unrated: it is counted, not placed."""
    v = (metrics or {}).get("interactivity_of_mean_tpot")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return float(v), "result"
    return None, ""


def _dominates(a, b):
    return (a["x"] >= b["x"] and a["y"] >= b["y"]) and (a["x"] > b["x"] or a["y"] > b["y"])


def non_dominated(points):
    """The points no other point dominates, in input order. Equal points do not dominate each other."""
    return [p for p in points if not any(_dominates(q, p) for q in points if q is not p)]


def _point(row, y_field):
    x, y = row.get("interactivity"), row.get(y_field)
    if not row.get("servable") or x is None or y is None:
        return None
    return {"x": float(x), "y": float(y), "row": row}


def _member(p, status="front"):
    r = p["row"]
    return {"label": r["label"], "variant": r.get("variant", ""), "sweep": r.get("sweep"),
            "concurrency": r.get("concurrency"), "disaggregated": bool(r.get("disaggregated")),
            "interactivity": round(p["x"], 3), "throughput_per_gpu": round(p["y"], 3),
            "status": status, "config": r.get("config") or {}}


def front(rows, throughput="total"):
    """The exact front of `rows` (ledger rows, see run_experiment.run_row).

    Returns {"members": [...], "points": n, "unrated": n}: members sorted by interactivity then
    throughput; `unrated` the servable rows lacking either figure (they cannot be placed, and an older
    Result does not state every figure). No member is dominated by any point."""
    y_field = THROUGHPUT_FORMS[throughput][0]
    servable = [r for r in rows if r.get("servable")]
    pts = [p for p in (_point(r, y_field) for r in servable) if p]
    members = [_member(p) for p in non_dominated(pts)]
    members.sort(key=lambda m: (m["interactivity"], m["throughput_per_gpu"], m["label"],
                                m["variant"], m["concurrency"] or 0))
    return {"members": members, "points": len(pts), "unrated": len(servable) - len(pts)}


def best_on_front(rows, throughput="total"):
    """The ledger row that heads the front: the member with the greatest throughput per GPU, then the
    greatest interactivity, then the lowest concurrency. None when no row can be placed.

    A front has no single best; this is the one summary row the report cards need, and it is the
    throughput end of the front, so choosing it never hides a member. The full front is emitted
    beside it."""
    y_field = THROUGHPUT_FORMS[throughput][0]
    f = front(rows, throughput)
    if not f["members"]:
        return None
    top = max(f["members"], key=lambda m: (m["throughput_per_gpu"], m["interactivity"],
                                           -(m["concurrency"] or 0)))
    for r in rows:
        if (r.get("servable") and r["label"] == top["label"] and r.get("variant", "") == top["variant"]
                and r.get("sweep") == top["sweep"] and r.get("concurrency") == top["concurrency"]
                and r.get(y_field) is not None):
            return r
    return None


def describe(objective):
    """One line saying which ranking and objective a report used."""
    form, metric = THROUGHPUT_FORMS[objective["throughput"]][1], X_METRIC
    if objective["ranking"] == "pareto":
        return (f"Pareto front of {form} against {metric} among runs that held the scenario's SLOs, "
                f"compared exactly on one run per arm (no interval)")
    return "max output tok/s per accelerator subject to the scenario's SLOs"
