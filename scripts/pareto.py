#!/usr/bin/env python3
"""The Pareto objective: the non-dominated front of throughput per GPU against interactivity.

The axes are the metric registry's, not this repository's (design DEV-8):

  x  interactivity_mean          the mean over requests of each one's tok/s, 1000 / its TPOT in ms
                                 (formula user_tps). Not interactivity_of_mean_tpot, 1000 / the mean
                                 TPOT (InferenceX mean_intvty), the harmonic mean, which is lower
  y  total_tokens_per_second_per_gpu            (input + output tokens)/s over ALL the GPUs, or
     output_tokens_per_second_per_decode_gpu    output tokens/s over the decode GPUs

A spec that states its per-user statistic (scenario.slo: a tpot_<stat> cap or an interactivity_<stat>
floor, see slo.py) replaces x with each row's tok_per_s_per_user on that statistic (1000 / the worst
instance's tpot_<stat>_ms, or with interactivity_mean stated the worst instance's interactivity_mean),
so the ranking and the SLO gate are one figure; pareto.json then names it (`x`, `x_basis`).

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
# The registry's mean of each request's tok/s (formula user_tps), run_blis.MEAN_RATE.
MEAN_RATE = "interactivity_mean"
X_METRIC = MEAN_RATE
X_FIELD = "interactivity"            # the row field holding X_METRIC
PER_USER_X = "tok_per_s_per_user"       # the figure pareto.json's `x` names when a spec states one
# The row field the front reads it from: the rate to 3 decimals, as `interactivity`. The displayed
# tok_per_s_per_user is rounded to 0.1, and ties from that rounding dropped real front members.
PER_USER_FIELD = "per_user_rate"
DEFAULT_OBJECTIVE = {"ranking": "scalar", "throughput": "total"}


def objective_of(spec_objective):
    """The spec's `objective:` block with defaults filled in."""
    out = dict(DEFAULT_OBJECTIVE)
    out.update({k: v for k, v in (spec_objective or {}).items() if v is not None})
    return out


def interactivity_of(rec):
    """(tokens/s per user, basis) for a run record, or (None, "").

    Only the Result's own registered X_METRIC, as the record keeps it (`measured.metrics`), counts
    (DEV-8: a definition is read from the registry, never rebuilt from another field). A Result that
    does not state it, or states a figure that is not a positive number, leaves the run unrated: it is
    counted, not placed."""
    entry = (((rec or {}).get("measured") or {}).get("metrics") or {}).get(X_METRIC)
    v = entry.get("value") if isinstance(entry, dict) else None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return float(v), "result"
    return None, ""


def _dominates(a, b):
    return (a["x"] >= b["x"] and a["y"] >= b["y"]) and (a["x"] > b["x"] or a["y"] > b["y"])


def non_dominated(points):
    """The points no other point dominates, in input order. Equal points do not dominate each other."""
    return [p for p in points if not any(_dominates(q, p) for q in points if q is not p)]


def _point(row, y_field, x_field=X_FIELD):
    x, y = row.get(x_field), row.get(y_field)
    if not row.get("servable") or x is None or y is None:
        return None
    return {"x": float(x), "y": float(y), "row": row}


def _member(p, status="front"):
    r = p["row"]
    return {"label": r["label"], "variant": r.get("variant", ""), "sweep": r.get("sweep"),
            "concurrency": r.get("concurrency"), "disaggregated": bool(r.get("disaggregated")),
            "interactivity": round(p["x"], 3), "throughput_per_gpu": round(p["y"], 3),
            "status": status, "config": r.get("config") or {}}


def front(rows, throughput="total", x_field=X_FIELD):
    """The exact front of `rows` (ledger rows, see run_experiment.run_row), x read from `x_field`.
    A member's `interactivity` is its x, whichever figure the report's `x` names.

    Returns {"members": [...], "points": n, "unrated": n}: members sorted by interactivity then
    throughput; `unrated` the servable rows lacking either figure (they cannot be placed, and an older
    Result does not state every figure). No member is dominated by any point."""
    y_field = THROUGHPUT_FORMS[throughput][0]
    servable = [r for r in rows if r.get("servable")]
    pts = [p for p in (_point(r, y_field, x_field) for r in servable) if p]
    members = [_member(p) for p in non_dominated(pts)]
    members.sort(key=lambda m: (m["interactivity"], m["throughput_per_gpu"], m["label"],
                                m["variant"], m["concurrency"] or 0))
    return {"members": members, "points": len(pts), "unrated": len(servable) - len(pts)}


def best_on_front(rows, throughput="total", x_field=X_FIELD):
    """The ledger row that heads the front: the member with the greatest throughput per GPU, then the
    greatest interactivity, then the lowest concurrency. None when no row can be placed.

    A front has no single best; this is the one summary row the report cards need, and it is the
    throughput end of the front, so choosing it never hides a member. The full front is emitted
    beside it."""
    y_field = THROUGHPUT_FORMS[throughput][0]
    f = front(rows, throughput, x_field)
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


def describe(objective, per_user=None):
    """One line saying which ranking and objective a report used; `per_user` is the stated per-user
    basis (run_experiment.per_user_basis), which the front's x is then on."""
    form, metric = THROUGHPUT_FORMS[objective["throughput"]][1], X_METRIC
    if per_user and per_user["of"] == MEAN_RATE:
        metric = f"tok/s/user (the worst instance's {MEAN_RATE}, the mean of each request's tok/s)"
    elif per_user:
        metric = f"tok/s/user (1000 / the worst instance's {per_user['of']})"
    if objective["ranking"] == "pareto":
        return (f"Pareto front of {form} against {metric} among runs that held the scenario's SLOs, "
                f"compared exactly on one run per arm (no interval)")
    return "max output tok/s per accelerator subject to the scenario's SLOs"
