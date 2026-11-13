#!/usr/bin/env python3
# report.py -- shared helpers for reading a run directory and judging a run's feasibility.
#
# This module USED to render a standalone HTML report (run-dir -> index.html). That report was
# retired once the React SPA under web/ became the only report surface (issue #69); the HTML
# rendering, its template (report/template.html) and vendored libs (report/vendor/) are gone.
#
# What remains is the data layer the rest of the pipeline shares:
#   * run-dir loading   -- load_run_dir / is_run_dir / _read_json / _read_jsonl (build_gallery.py)
#   * feasibility       -- _feasible / _per_user_rate and the floor/SLO helpers (test_experiment.py)
#   * provenance        -- provenance_split, a ledger-coherence check (build_gallery.py, blis_pin.py)
#
# Inputs (all optional-tolerant; report/schema.md documents report.json):
#   scenario.json  resolved Scenario (model/hw/budget/workload/floor/headroom/pinned_stack)
#   pareto.json    finalists / dropped / winner (the analysis artifact)
#   runs.jsonl     one row per blis run (the raw ledger)
#   report.json    the narrative the SPA renders.
#
# It NEVER runs BLIS and makes no subprocess / network calls.
import decimal, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_blis  # noqa: E402  -- the ONE per-user / feasibility definition (namasl review #2)

# report/ still holds the canonical lexicon.json (read by lexicon.py, mirrored to the SPA) and
# schema.md (documents report.json); REPORT_DIR is kept so those readers resolve them.
REPORT_DIR = os.path.normpath(os.path.join(HERE, "..", "report"))


# --- defensive loading -------------------------------------------------------

class MalformedInput(ValueError):
    """An input file EXISTS but failed to parse. Distinct from a missing optional file."""


def _read_json(path, default):
    """Parse a JSON input. A MISSING optional file quietly returns `default`. A file that EXISTS but
    fails to parse raises MalformedInput -- we must NOT silently substitute {} for a truncated
    scenario.json, because that drops the floor/SLOs and would mark unconstrained runs servable
    (review F1). The caller decides whether to abort or surface it."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return default
    except ValueError as e:
        raise MalformedInput(f"{path}: invalid JSON ({e})") from e


def _read_jsonl(path):
    """Return (rows, skipped_line_numbers). A malformed row is skipped but NOT swallowed silently:
    its 1-based line number is returned so the caller can warn and surface a count in the report."""
    rows, skipped = [], []
    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        skipped.append(lineno)
    except OSError:
        pass
    return rows, skipped


def is_run_dir(path):
    """A run dir is identified by the engine's raw ledger runs.jsonl or the analysis pareto.json."""
    return (os.path.isfile(os.path.join(path, "runs.jsonl"))
            or os.path.isfile(os.path.join(path, "pareto.json")))


def load_run_dir(path):
    """Returns (scenario, pareto, runs, report, n_malformed). n_malformed counts non-empty
    runs.jsonl lines that failed to parse -- surfaced in the report rather than hidden (QA G3)."""
    runs, skipped = _read_jsonl(os.path.join(path, "runs.jsonl"))
    if skipped:
        lines = ", ".join(str(n) for n in skipped[:10]) + ("…" if len(skipped) > 10 else "")
        print(f"warning: {os.path.join(path, 'runs.jsonl')}: skipped {len(skipped)} malformed "
              f"line(s): {lines}", file=sys.stderr)
    return (_read_json(os.path.join(path, "scenario.json"), {}),
            _read_json(os.path.join(path, "pareto.json"), {}),
            runs,
            _read_json(os.path.join(path, "report.json"), {}),
            len(skipped))


# --- numeric helpers ---------------------------------------------------------

def _as_num(v):
    """Coerce a metric to float, tolerating a numeric string (e.g. "100"); None/garbage -> None.
    Keeps feasibility comparisons from throwing on a non-numeric metric value (jgchn review)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fnum(v, digits):
    """Format a metric the way the dashboard's `fnum` (web/src/lib.ts) does, so one number is not
    printed `143.9` on one surface and `144` on the other (issue #13 review).

    `fnum` is `toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: 0})`:
      * at most `digits` decimals, with trailing zeros STRIPPED (18.0 -> "18", 6.72 -> "6.72");
      * thousands separators;
      * and ties rounded HALF AWAY FROM ZERO. That last part is why this cannot just be an
        f-string: Python's format() rounds half to EVEN (banker's), so f"{0.5:.0f}" is "0" where
        JS gives "1" -- a residual mismatch on exactly the .5 cases.
    `digits` comes from the metric's lexicon entry, which is where the dashboard reads it too.
    """
    n = _as_num(v)
    if n is None:
        return "–"
    q = decimal.Decimal(1).scaleb(-digits)        # 1, 0.1, 0.01, ...
    d = decimal.Decimal(repr(float(n))).quantize(q, rounding=decimal.ROUND_HALF_UP)
    # strip trailing zeros (minimumFractionDigits: 0) without going to exponent form
    txt = f"{d:,f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return txt or "0"


# --- feasibility: the ONE definition both the sweeper and the SPA data judge on ----------------

def _metric_any(run, key):
    """The WORST (max) value of `key` across the cluster aggregate AND every per-instance metric.
    A latency/count SLO can be met in the aggregate yet violated on one instance; taking the max over
    both (not just trusting the aggregate when present) is what keeps feasibility in parity with the
    search, which drops a run if any instance breaches (QA G4). Returns (worst, missing_everywhere)."""
    vals = []
    agg = _as_num((run.get("metrics") or {}).get(key))   # coerce; tolerate string/garbage (jgchn)
    if agg is not None:
        vals.append(agg)
    for inst in (run.get("metrics_all_instances") or []):
        if isinstance(inst, dict):
            iv = _as_num(inst.get(key))
            if iv is not None:
                vals.append(iv)
    if vals:
        return max(vals), False
    return None, True   # metric entirely absent (or non-numeric) in aggregate and every instance


def _floor_rate(floor):
    """The per-user token rate the interactivity floor requires. The mirror of
    _floor_itl_p99_max: honor the explicit rate, else derive it from an itl_p99_ms_max alias, so a
    scenario that states only the cap is still enforced on the per-token basis."""
    floor = floor or {}
    rate = _as_num(floor.get("tok_per_s_per_user"))
    if rate:
        return rate
    cap = _as_num(floor.get("itl_p99_ms_max"))
    if cap:
        return 1000.0 / cap
    return 0.0


def _floor_itl_p99_max(floor):
    """The p99 ITL cap the interactivity floor implies. Honor the explicit itl_p99_ms_max alias if
    present; otherwise DERIVE it from the documented tok_per_s_per_user floor (p99 cap = 1000 / rate)
    -- a scenario that specifies only the rate (as the checked-in YAML does) must still be enforced,
    not treated as no floor at all (review F2)."""
    floor = floor or {}
    cap = _as_num(floor.get("itl_p99_ms_max"))
    if cap is not None:
        return cap
    rate = _as_num(floor.get("tok_per_s_per_user"))
    if rate:
        return 1000.0 / rate
    return None


_SPEC_DURATION = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(us|ms|s)$")


def _spec_slo_caps(slos, floor_set):
    """Experiment-spec SLO keys (`itl_p99: '20ms'`) as the `<metric>_ms` caps checked here.
    A spec's itl_p99 is already enforced, acceptance-aware, as the interactivity floor."""
    out = {}
    for k, cap in slos.items():
        mt = _SPEC_DURATION.match(str(cap).strip()) if isinstance(cap, str) else None
        if mt and k in ("itl_p99", "ttft_p99", "e2e_p99"):
            if k == "itl_p99" and floor_set:
                continue
            scale = {"us": 0.001, "ms": 1.0, "s": 1000.0}[mt.group(2)]
            k, cap = f"{k}_ms", float(mt.group(1)) * scale
        out[k] = cap
    return out


def _feasible(run, floor, slos):
    """(ok, viol[]) -- did this run meet completion + interactivity floor + any SLOs?
    A declared SLO whose metric is absent from BOTH the aggregate and every instance is treated as a
    violation (cannot prove the SLO was met), not silently passed (QA G4)."""
    viol = []
    if run.get("returncode", 0) != 0 or run.get("error"):
        viol.append(f"run failed: {run.get('error') or 'nonzero exit'}")
    m = run.get("metrics") or {}
    inj, comp = _as_num(m.get("injected_requests")), _as_num(m.get("completed_requests"))
    if inj and comp is not None and comp < inj:   # _as_num tolerates string/None metrics (jgchn review)
        viol.append(f"incomplete: {comp:g}/{inj:g} completed")
    if m.get("timed_out_requests"):
        viol.append(f"timed_out_requests = {m['timed_out_requests']}")
    if m.get("dropped_unservable"):
        viol.append(f"dropped_unservable = {m['dropped_unservable']}")
    # The interactivity floor is a per-TOKEN promise, and the rate tested is 1000 over the worst p99 time
    # per output token (run_blis.per_user_rate), the one definition the display and the sweeper's
    # `servable` also use. TPOT is per token and already holds what speculation gains, so no acceptance
    # factor enters; this replaces the engine's per-step interval, which needed one.
    itl_max = _floor_itl_p99_max(floor)
    if itl_max is not None:
        rate, basis, itl = run_blis.per_user_rate(run)
        floor_rate = _floor_rate(floor)
        if rate is None:
            viol.append("itl_p99_ms missing (cannot verify interactivity floor)")
        elif rate < floor_rate:
            viol.append(f"tok_per_s_per_user = {rate:.1f} < {floor_rate:g} (tpot p99 {itl:.1f} ms)")
    slos = dict(slos or {})
    # Experiment specs write throughput_per_gpu as a floor, not a cap.
    tput = _as_num(slos.pop("throughput_per_gpu", None))
    if tput:
        tps = _as_num(m.get("tokens_per_sec"))
        gpus = _as_num((run.get("deployment") or {}).get("gpus"))
        if tps is None or not gpus:
            viol.append(f"throughput_per_gpu missing (cannot verify SLO >= {tput:g})")
        elif tps / gpus < tput:
            viol.append(f"throughput_per_gpu = {tps / gpus:.1f} < {tput:g}")
    for k, cap in _spec_slo_caps(slos, itl_max is not None).items():
        if cap is None:
            continue
        val, missing = _metric_any(run, k)
        if missing:
            viol.append(f"{k} missing (cannot verify SLO <= {cap})")
        elif val > cap:
            viol.append(f"{k} = {val:.1f} > {cap}")
    return (len(viol) == 0, viol)


def _per_user_rate(run, mtp=0):
    """Per-user token rate for display, through the ONE definition in run_blis.per_user_rate --
    the same one _feasible judges on, so a row can no longer be displayed above the floor while
    being marked infeasible (namasl review #2).

    The rate is 1000 over the worst-instance p99 time per output token, drafting or not (no acceptance
    factor: TPOT is per token). `mtp` is accepted for call-site compatibility but the depth is read from the run itself.

    There is deliberately NO short-circuit on a `metrics.tok_per_s_per_user` field. An earlier
    version preferred it and returned it raw -- an aggregate, neither acceptance-scaled nor
    worst-instance -- while _feasible always judged through run_blis.per_user_rate. Nothing writes
    that field today, so the two agreed by luck; the day the engine emits it, the displayed rate
    and the feasibility verdict would diverge again, which is the exact split this unification
    closed. One definition means one definition (susiejojo review #1).

    Returns (pu_or_None, basis); basis is always "" (a direct rate), kept so callers need no change."""
    pu, basis, _itl = run_blis.per_user_rate(run)
    return (round(pu, 2) if pu is not None else None), basis


# --- provenance --------------------------------------------------------------

def provenance_split(runs):
    """Describe a ledger whose rows were NOT all priced by one cost model, else "".

    Two halves of a report can be priced differently and still carry one `pinned_stack` label: the
    committed DeepSeek reports had baseline rows recorded before the registry was bumped to v0.2.0
    and drafting rows recorded after, under a single `registry: v0.1.1`. The recorded command
    contains only a mutable local path, so nothing in the artifacts contradicted the label
    (namasl review #3).

    What we CAN check from a ledger: the seed every run was issued with, and the resolved stack
    each row recorded. A split in either means the rows are not comparable, and the report must say
    so rather than presenting a matched-load table across the boundary.
    """
    seeds, stacks, bases = set(), set(), set()
    for r in runs:
        # A record written before the basis was recorded was ranked on the whole run; a new one on
        # the measured window. Figures on the two are different populations (DEV-8).
        # Only a run that produced metrics has a basis; an error row (refused config, failed or
        # invalid Result) has none to disagree about.
        if r.get("metrics"):
            bases.add(r.get("basis") or "unrecorded")
        cmd = r.get("cmd") or []
        if "--seed" in cmd:
            try:
                seeds.add(int(cmd[cmd.index("--seed") + 1]))
            except (IndexError, ValueError):
                pass
        st = r.get("pinned_stack") or r.get("stack")
        if isinstance(st, dict):
            stacks.add(json.dumps(st, sort_keys=True))
    issues = []
    if len(seeds) > 1:
        issues.append(f"runs span {len(seeds)} seeds ({sorted(seeds)}): a matched-load comparison "
                      f"across them compares different replayed traffic")
    if len(stacks) > 1:
        issues.append(f"runs span {len(stacks)} distinct resolved stacks: they were not priced by "
                      f"one cost model")
    if len(bases) > 1:
        issues.append(f"runs mix bases ({sorted(bases)}): a whole-run figure and a measured-window "
                      f"figure are different populations and are never compared (DEV-8)")
    return "; ".join(issues)
