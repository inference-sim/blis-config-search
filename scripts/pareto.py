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

TIE RULE. A deployment is judged against an aggregated one at the same GPU count, and the team prefers
P/D when it does no worse. Two measurements are not distinguished below the run-to-run noise, so
"does no worse" is "within `tie_tolerance` of it on both axes". For an aggregated point A on the exact
front, any P/D point P with P.x >= (1 - tol) * A.x and P.y >= (1 - tol) * A.y ties it, and then P
displaces A. The P/D points that displace A and are not on the exact front are reported as
`tie_preferred`; A is listed under `displaced`. The default tolerance is the accuracy dashboard's
documented noise floor (1.6 percent) until repeat noise is measured. Nothing else uses the tolerance:
two aggregated points, or two P/D points, are compared exactly.
"""

# The accuracy dashboard's fallback noise floor (docs/limits.md there). A spec states its own.
DEFAULT_TIE_TOLERANCE = 0.016
RANKINGS = ("scalar", "pareto")
# throughput form -> (row field, registry metric)
THROUGHPUT_FORMS = {
    "total": ("total_tokens_per_sec_per_gpu", "total_tokens_per_second_per_gpu"),
    "decode": ("output_tokens_per_sec_per_decode_gpu", "output_tokens_per_second_per_decode_gpu"),
}
X_METRIC = "interactivity_of_mean_tpot"
DEFAULT_OBJECTIVE = {"ranking": "scalar", "throughput": "total", "tie_tolerance": DEFAULT_TIE_TOLERANCE}


def objective_of(spec_objective):
    """The spec's `objective:` block with defaults filled in."""
    out = dict(DEFAULT_OBJECTIVE)
    out.update({k: v for k, v in (spec_objective or {}).items() if v is not None})
    return out


def interactivity_of(metrics):
    """(tokens/s per user, basis) for a run's flattened measured-window metrics, or (None, "").

    The Result's own interactivity_of_mean_tpot when it states one; otherwise the registry's formula
    applied to the Result's tpot_mean_ms, which is the same quantity (1000 over the mean TPOT of the
    requests with at least two output tokens)."""
    m = metrics or {}
    v = m.get("interactivity_of_mean_tpot")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return float(v), "result"
    t = m.get("tpot_mean_ms")
    if isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0:
        return 1000.0 / t, "1000 / tpot_mean_ms"
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


def _member(p, status, displaces=()):
    r = p["row"]
    out = {"label": r["label"], "variant": r.get("variant", ""), "sweep": r.get("sweep"),
           "concurrency": r.get("concurrency"), "disaggregated": bool(r.get("disaggregated")),
           "interactivity": round(p["x"], 3), "throughput_per_gpu": round(p["y"], 3),
           "status": status, "config": r.get("config") or {}}
    if displaces:
        out["displaces"] = list(displaces)
    return out


def _name(p):
    r = p["row"]
    return f"{r['label']} ({r['variant']}) c={r.get('concurrency')}" if r.get("variant") \
        else f"{r['label']} c={r.get('concurrency')}"


def front(rows, throughput="total", tie_tolerance=DEFAULT_TIE_TOLERANCE):
    """The front of `rows` (ledger rows, see run_experiment.run_row) with the tie rule applied.

    Returns {"members": [...], "displaced": [...], "points": n, "unrated": n}: members sorted by
    interactivity then throughput; `displaced` the aggregated points the tie rule removed, each with
    the P/D points that displaced it; `unrated` the servable rows lacking either figure (they cannot be
    placed, and an older Result does not state every figure)."""
    y_field = THROUGHPUT_FORMS[throughput][0]
    servable = [r for r in rows if r.get("servable")]
    pts = [p for p in (_point(r, y_field) for r in servable) if p]
    exact = non_dominated(pts)
    pd_points = [p for p in pts if p["row"].get("disaggregated")]
    tol = float(tie_tolerance)
    displaced, promoted_for = [], {}
    for a in exact:
        if a["row"].get("disaggregated"):
            continue
        tied = [p for p in pd_points
                if p["x"] >= (1 - tol) * a["x"] and p["y"] >= (1 - tol) * a["y"]]
        if tied:
            displaced.append((a, tied))
            for p in non_dominated(tied):
                promoted_for.setdefault(id(p), (p, []))[1].append(_name(a))
    gone = {id(a) for a, _ in displaced}
    members = [_member(p, "front") for p in exact if id(p) not in gone]
    on_exact = {id(p) for p in exact}
    members += [_member(p, "tie_preferred", names) for p, names in promoted_for.values()
                if id(p) not in on_exact]
    for m_p in exact:
        # A P/D point already on the exact front that ties an aggregated one displaces it too.
        if m_p["row"].get("disaggregated"):
            names = [_name(a) for a, tied in displaced if any(t is m_p for t in tied)]
            if names:
                for m in members:
                    if m["status"] == "front" and m["label"] == m_p["row"]["label"] \
                            and m["variant"] == m_p["row"].get("variant", "") \
                            and m["concurrency"] == m_p["row"].get("concurrency") \
                            and m["sweep"] == m_p["row"].get("sweep"):
                        m["displaces"] = names
    members.sort(key=lambda m: (m["interactivity"], m["throughput_per_gpu"], m["label"],
                                m["variant"], m["concurrency"] or 0))
    return {"members": members,
            "displaced": [{"point": _member(a, "displaced"),
                           "by": [_name(p) for p in non_dominated(tied)]} for a, tied in displaced],
            "points": len(pts), "unrated": len(servable) - len(pts)}


def best_on_front(rows, throughput="total", tie_tolerance=DEFAULT_TIE_TOLERANCE):
    """The ledger row that heads the front: the member with the greatest throughput per GPU, then the
    greatest interactivity, then the lowest concurrency. None when no row can be placed.

    A front has no single best; this is the one summary row the report cards need, and it is the
    throughput end of the front, so choosing it never hides a member. The full front is emitted
    beside it."""
    y_field = THROUGHPUT_FORMS[throughput][0]
    f = front(rows, throughput, tie_tolerance)
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
    tol = f"{objective['tie_tolerance'] * 100:g}"
    if objective["ranking"] == "pareto":
        return (f"Pareto front of {form} against {metric} among runs that held the scenario's SLOs; "
                f"a P/D point within {tol}% of an aggregated one on both axes displaces it")
    return "max output tok/s per accelerator subject to the scenario's SLOs"
