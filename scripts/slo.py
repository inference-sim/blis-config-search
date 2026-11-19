#!/usr/bin/env python3
"""A spec's `scenario.slo` keys: what each one gates, and the per-user statistic they state.

  <metric>_<stat>: 25ms    a cap on the worst instance's `<metric>_<stat>_ms` (tpot_p90, ttft_p95,
                           e2e_mean, ...): any statistic the metric registry defines for a metric
                           whose per-instance figures the record carries
  interactivity_<stat>: R  a floor in tok/s/user: the worst instance's 1000 / tpot_<stat>_ms >= R.
                           A spec key, defined by its tpot figure: interactivity_mean is 1000 / the
                           mean TPOT (the registry's interactivity_of_mean_tpot), not the registry's
                           interactivity_mean, and a report names the figure (`of: tpot_mean_ms`)
  throughput_per_gpu: N    a floor on output tok/s per accelerator
  itl_p99: 20ms            kept as the alias of tpot_p99 (#126: the slot it reads holds TPOT)

The statistics are the registry's (`blis metrics --json`), checked when a spec is validated, and the
Result's (`measured.metrics`), checked on each run: a statistic the Result does not report is refused,
never computed here. The worst-instance figures are rebuilt at run_blis.INSTANCE_STATS, so a statistic
outside it is refused too: one the simulator adds needs that one constant extended.

The per-user rate (the interactivity floor, each row's tok/s/user and the Pareto axis) is on the
statistic the tpot caps and interactivity floors state. A spec that states none keeps the p99 rate and
the interactivity_of_mean_tpot axis it had before, so a committed spec and its report do not change.
"""
import re

import run_blis

KEY_RE = re.compile(r"^(?P<metric>[a-z][a-z0-9_]*?)_(?P<stat>mean|p[0-9]{1,2})$")
THROUGHPUT = "throughput_per_gpu"
ALIASES = {"itl_p99": "tpot_p99"}
LEGACY = ("itl_p99", "ttft_p99", "e2e_p99")
FLOOR_METRIC = "interactivity"
PER_USER_METRIC = "tpot"
DEFAULT_STAT = "p99"
# The metrics a cap can be on: those whose per-instance figures the record rebuilds (worst instance).
JUDGED = tuple(stem for stem, _col in run_blis._ROW_LATENCY)


def _stat_order(stat):
    return (-1, 0) if stat == "mean" else (0, int(stat[1:]))


def figure(metric, stat):
    """The Result's metric id for a latency at a statistic: tpot, p90 -> tpot_p90_ms."""
    return f"{metric}_{stat}_ms"


def parse(key):
    """{kind, key, metric, stat, figure} for one SLO key, or None when it has no meaning here.
    kind is "throughput", "cap" or "floor"; a floor's figure is the TPOT it is 1000 over."""
    if key == THROUGHPUT:
        return {"kind": "throughput", "key": key}
    m = KEY_RE.match(str(key))
    if not m:
        return None
    metric, stat = m.group("metric"), m.group("stat")
    if metric == FLOOR_METRIC:
        return {"kind": "floor", "key": key, "metric": PER_USER_METRIC, "stat": stat,
                "figure": figure(PER_USER_METRIC, stat)}
    target = ALIASES.get(key)
    if target:
        metric, stat = KEY_RE.match(target).group("metric", "stat")
    return {"kind": "cap", "key": key, "metric": metric, "stat": stat, "figure": figure(metric, stat)}


def _registry_stats(registry, metric, unit="ms"):
    """The statistics the registry defines `metric` at, as `<metric>_<stat>_ms` ids, in order."""
    out = [r.get("statistic") for rid, r in registry.items()
           if r.get("unit") == unit and rid == figure(metric, r.get("statistic"))]
    return sorted((s for s in out if s), key=_stat_order)


def key_errors(slo, registry, parse_duration_ms):
    """Why each key of a spec's `scenario.slo` cannot be judged, as validator messages ([] when all
    can). `registry` is blis_cli.metric_registry's {id: record}."""
    errs = []
    for key, val in slo.items():
        g = parse(key)
        where = f"scenario.slo.{key}"
        if g is None:
            errs.append(f"{where}: unknown SLO (a cap is <metric>_<stat> on {', '.join(JUDGED)}, "
                        f"e.g. tpot_p90 or ttft_p95; a floor is interactivity_<stat> in tok/s/user; "
                        f"or {THROUGHPUT}; itl_p99 is the alias of tpot_p99)")
            continue
        if g["kind"] == "throughput":
            if isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0:
                errs.append(f"{where}: {val!r} must be a positive tok/s per accelerator, e.g. 500")
            continue
        if g["kind"] == "floor":
            if g["figure"] not in registry:
                errs.append(f"{where}: the metric registry has no {g['figure']}, which this floor is "
                            f"1000 over; {PER_USER_METRIC} is recorded at "
                            f"{', '.join(_registry_stats(registry, PER_USER_METRIC)) or 'no statistic'}")
            elif g["stat"] not in run_blis.INSTANCE_STATS:
                errs.append(f"{where}: the worst instance is judged at {', '.join(run_blis.INSTANCE_STATS)}, "
                            f"not {g['stat']}")
            elif isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0:
                errs.append(f"{where}: {val!r} must be a positive tok/s per user, e.g. 50")
            continue
        if g["metric"] not in JUDGED:
            hint = "".join(f"; {k} is kept as the alias of {v}" for k, v in ALIASES.items()
                           if k.split("_")[0] == g["metric"])
            errs.append(f"{where}: {g['metric']} is not judged on the worst instance (the record "
                        f"carries per-instance {', '.join(JUDGED)}){hint}")
            continue
        if g["figure"] not in registry:
            errs.append(f"{where}: the metric registry has no {g['figure']}; {g['metric']} is "
                        f"recorded at {', '.join(_registry_stats(registry, g['metric'])) or 'no statistic'}")
            continue
        if g["stat"] not in run_blis.INSTANCE_STATS:
            errs.append(f"{where}: the worst instance is judged at {', '.join(run_blis.INSTANCE_STATS)} "
                        f"(the statistics the per-instance figures are rebuilt at), not {g['stat']}")
            continue
        ms = parse_duration_ms(val)
        if ms is None:
            errs.append(f"{where}: {val!r} needs an explicit unit "
                        f"(e.g. 20ms, 5s) -- a bare number is ambiguous")
        elif ms <= 0:
            errs.append(f"{where}: {val!r} must be a positive duration, e.g. 20ms")
    stats = sorted({g["stat"] for g in map(parse, slo) if g and g.get("metric") == PER_USER_METRIC},
                   key=_stat_order)
    if len(stats) > 1:
        errs.append(f"scenario.slo: the per-user rate is stated at {', '.join(stats)}; state it at one "
                    f"statistic, so the floor, each row's tok/s/user and the Pareto axis are one figure")
    return errs


def per_user(slo):
    """(stat, stated) of the per-user rate: the statistic the tpot caps and interactivity floors
    state, or the p99 default when none states one. The legacy itl_p99 is the default, not a
    statement: its spec keeps the report it had."""
    stated = sorted({g["stat"] for g in map(parse, slo or {})
                     if g and g.get("metric") == PER_USER_METRIC and g["key"] not in LEGACY},
                    key=_stat_order)
    return (stated[0], True) if stated else (DEFAULT_STAT, False)


def gates(slo_ms):
    """The SLO's gates in the order a run is judged: throughput, the per-user gates, then the other
    caps by metric (ttft, e2e) and statistic. Each is parse()'s dict plus `limit`."""
    out = []
    for key, limit in (slo_ms or {}).items():
        g = parse(key)
        if g is not None:
            out.append({**g, "limit": limit})

    def order(g):
        if g["kind"] == "throughput":
            return (0,)
        if g.get("metric") == PER_USER_METRIC:
            return (1, g["kind"] == "floor", g["key"])
        return (2, JUDGED.index(g["metric"]) if g["metric"] in JUDGED else len(JUDGED),
                _stat_order(g["stat"]), g["key"])
    return sorted(out, key=order)


def floor_rate(slo_ms):
    """The per-user rate the per-user gates require (tok/s/user), or None: the tightest of each
    tpot cap's 1000 / cap and each interactivity floor."""
    rates = []
    for g in gates(slo_ms):
        if g.get("metric") != PER_USER_METRIC or not g["limit"]:
            continue
        rates.append(1000.0 / g["limit"] if g["kind"] == "cap" else float(g["limit"]))
    return max(rates) if rates else None


def worst(run, g):
    """(worst, missing) of a gate's figure on a run. A legacy key reads the slot it always read
    (worst_metric over the flat record); every other key reads the Result's own figure."""
    if g["key"] in LEGACY:
        return run_blis.worst_metric(run, "itl_p99_ms" if g["key"] == "itl_p99" else f"{g['key']}_ms")
    return run_blis.worst_figure(run, g["figure"])


def unreported(run, slo_ms):
    """(key, message) for the first gate whose statistic this run's Result does not report although
    it reports the metric at others (tpot_p95 on the pinned release), else None. Such a key is refused:
    the statistic is the simulator's to add, never computed here."""
    measured = ((run.get("measured") or {}).get("metrics")) or {}
    if not isinstance(measured, dict) or not measured:
        if run.get("returncode") == 0 and run.get("metrics"):
            # A record written before the measured window: its legacy slots are not the registry's
            # figures, so only the legacy keys, which read those slots, can be judged on it.
            for g in gates(slo_ms):
                if g["kind"] != "throughput" and g["key"] not in LEGACY:
                    return g["key"], (f"scenario.slo.{g['key']}: this record has no measured window "
                                      f"(it predates the Result's measured.metrics), so {g['figure']} "
                                      f"cannot be read from it; only {', '.join(LEGACY)} can be judged "
                                      f"on such a ledger. Run the sweep again")
        return None
    for g in gates(slo_ms):
        if g["kind"] == "throughput" or g["figure"] in measured:
            continue
        stats = sorted((m.group("stat") for m in map(KEY_RE.match, (k[:-3] for k in measured
                                                                    if k.endswith("_ms"))) if m
                        and m.group("metric") == g["metric"]), key=_stat_order)
        if stats:
            return g["key"], (f"scenario.slo.{g['key']}: the simulator's Result does not report "
                              f"{g['figure']}; it reports {g['metric']} at {', '.join(stats)}")
    return None


def figures(slo_ms):
    """The Result figures the latency gates are judged on, in judging order (DEV-8)."""
    out = []
    for g in gates(slo_ms):
        if g["kind"] != "throughput" and g["figure"] not in out:
            out.append(g["figure"])
    return out
