#!/usr/bin/env python3
# report.py -- the report half of config-search: a PURE run-dir -> index.html transform.
#
#   python3 -I scripts/report.py <run-dir>            # -> <run-dir>/index.html
#   python3 -I scripts/report.py <run-dir> --out p    # -> p
#   python3 -I scripts/report.py <parent-of-run-dirs> # -> each child's index.html + a card index
#
# It reads the run directory the search engine writes and renders a magazine-style analytical report
# modeled on the CurveBender "glm53flash" format: a numbered-finding narrative, pre-rendered inline-SVG
# figures (figures.py), a client-side crossfilter explorer over every run, and a glossary.
#
# Inputs (all optional-tolerant -- three of four are brain-authored "convention, loose on purpose"):
#   scenario.json  resolved Scenario (model/hw/budget/workload/floor/headroom/pinned_stack)
#   pareto.json    finalists / dropped / winner (the analysis artifact)
#   runs.jsonl     one row per blis run (the raw ledger) -> RECORDS + figures
#   report.json    the brain-authored NARRATIVE (standfirst, hero, sections, glossary, operating pts).
#                  Absent -> the report still renders header + explorer + an auto operating table.
#
# It NEVER runs BLIS and makes no subprocess / network calls at GENERATION time. The rendered page is
# fully self-contained / offline: d3 + crossfilter are inlined via _vendor_js(), so the explorer, the
# article, and the baked figures all render with no network access.
#
# Determinism (INV-6): a pure function of the run-directory CONTENTS. Identical data renders
# byte-identical HTML regardless of where the directory lives or what it is named -- the footer
# identifies the run by the in-data run_id, never the path/basename (and says "unidentified" if none).
import argparse, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figures  # noqa: E402
import lexicon  # noqa: E402

REPORT_DIR = os.path.normpath(os.path.join(HERE, "..", "report"))
TEMPLATE = os.path.join(REPORT_DIR, "template.html")
INDEX_TEMPLATE = os.path.join(REPORT_DIR, "index-template.html")
VENDOR_DIR = os.path.join(REPORT_DIR, "vendor")
# The explorer's JS deps, inlined so the report is self-contained / offline (no CDN at view time).
VENDOR_LIBS = ("d3.min.js", "crossfilter.min.js")
RECORDS_PH = "/*__RECORDS__*/[]"
FACTS_PH = "/*__FACTS__*/{}"
# The shared vocabulary, baked into the page so the explorer's facet/metric/axis names come from
# the same file the React dashboard reads (issue #13). Inlined, not fetched: the report must stay
# self-contained and offline.
LEXICON_PH = "/*__LEXICON__*/{}"
CARDS_PH = "/*__CONFIG_SEARCH_CARDS__*/[]"


def _vendor_js():
    """Inline <script> blocks for the vendored explorer libs (report/vendor/*.min.js). Each lib's
    bytes are wrapped verbatim in its own <script>...</script>; a minified lib never contains a
    literal '</script>' so no escaping is needed, but we assert that to be safe."""
    out = []
    for name in VENDOR_LIBS:
        path = os.path.join(VENDOR_DIR, name)
        with open(path, encoding="utf-8") as f:
            js = f.read()
        if "</script" in js.lower():
            raise RuntimeError(f"vendored {name} contains a literal </script>; cannot inline safely")
        out.append(f"<!-- vendored: {name} -->\n<script>\n{js}\n</script>")
    return "\n".join(out)


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


# --- esc + small helpers -----------------------------------------------------

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _json_for_script(obj):
    """JSON-serialize for embedding inside an inline <script> element. json.dumps alone is NOT safe
    there: a string containing "</script>" would close the element early, which breaks the page and,
    with attacker-influenced data (a model/route name like "</script><script>…"), executes arbitrary
    code in whoever opens the shareable report (review F5). Escaping <, >, & to \\uXXXX keeps the data
    valid JSON to JSON.parse while making those characters inert to the HTML parser. ensure_ascii is
    default-on, so U+2028/U+2029 (which also break inline scripts) are already escaped."""
    return (json.dumps(obj, sort_keys=True)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def _get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _num(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.1f}"
    return str(v)


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


# --- RECORDS: one flat row per blis run -------------------------------------

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
    itl_max = _floor_itl_p99_max(floor)
    if itl_max is not None:
        itl, missing = _metric_any(run, "itl_p99_ms")
        if missing:
            viol.append("itl_p99_ms missing (cannot verify interactivity floor)")
        elif itl > itl_max:
            viol.append(f"itl_p99_ms = {itl:.1f} > {itl_max}")
    for k, cap in (slos or {}).items():
        if cap is None:
            continue
        val, missing = _metric_any(run, k)
        if missing:
            viol.append(f"{k} missing (cannot verify SLO <= {cap})")
        elif val > cap:
            viol.append(f"{k} = {val:.1f} > {cap}")
    return (len(viol) == 0, viol)


def _per_user_rate(run, mtp):
    """Per-user token rate (pu), on the SAME p99 basis as pareto.json's tok_per_s_per_user and the
    interactivity floor -- so one page never shows two incompatible per-user numbers for a point
    (review F3). Prefer the engine's authoritative per-token field (tok_per_s_per_user). Else fall
    back to 1000/itl_p99_ms, and ONLY when one decode step == one token (mtp == 0): under speculative
    decoding BLIS logs ITL per decode STEP, so 1000/ITL is a step rate that undercounts the per-token
    rate -- we return pu=None (flagged) rather than publish a wrong number (QA G1).
    Returns (pu_or_None, is_mtp_without_authoritative_field)."""
    m = run.get("metrics") or {}
    authoritative = _as_num(m.get("tok_per_s_per_user"))
    if authoritative is not None:
        return round(authoritative, 2), False
    itl_p99 = _as_num(m.get("itl_p99_ms"))
    if not itl_p99:
        return None, False
    if mtp:
        return None, True   # step rate would undercount; do not fabricate a per-token number
    return round(1000.0 / itl_p99, 2), False


GPUS_PER_NODE = 8


def _pool_nodes(dep):
    """Nodes the run's pool spans = ceil(total GPUs / 8); 1 when unknown.

    Mirrors run_blis.pool_nodes. Reported because a multi-node pool prices its collectives over a
    fabric, which is a different deployment from the same tp on one node.
    """
    total = dep.get("gpus") or 0
    return max(1, -(-int(total) // GPUS_PER_NODE)) if total else 1


def _blis_cmd(run):
    """The `blis run` invocation for this row, shell-quoted, or None.

    Taken from the recorded `cmd` (what actually ran) rather than rebuilt, so the panel cannot
    drift from the run. Long absolute paths are left intact: the point is reproducibility.
    """
    cmd = run.get("cmd")
    if not cmd:
        return None
    import shlex
    return " ".join(shlex.quote(str(a)) for a in cmd)


def build_records(runs, scenario):
    """Map our per-run rows onto the reference's flat record shape (short keys the explorer uses)."""
    floor = scenario.get("interactivity_floor") or {}
    slos = scenario.get("slos") or {}
    out = []
    for i, r in enumerate(runs):
        m = r.get("metrics") or {}
        dep = r.get("deployment") or {}
        par = dep.get("parallel") or {}
        eng = dep.get("engine") or {}
        cfg = r.get("config") or {}
        itl_mean = m.get("itl_mean_ms")
        ok, viol = _feasible(r, floor, slos)
        gpus = dep.get("gpus") or (par.get("tp", 1) * (dep.get("num_instances", 1) or 1))
        tp = par.get("tp", cfg.get("tp"))
        ni = dep.get("num_instances", cfg.get("num_instances", 1))
        mtp = cfg.get("num_speculative_tokens", 0) or 0
        pu, pu_is_step_estimate = _per_user_rate(r, mtp)
        out.append({
            "id": r.get("config_hash") or f"run-{i}",
            "src": r.get("phase", "run"),
            "conc": r.get("concurrency", cfg.get("concurrency")),
            "tp": tp,
            "dp": par.get("dp", 1),
            "ni": ni,
            "gpus": gpus,
            "series": _series_label(tp, ni, par),
            "route": cfg.get("routing_policy", "round-robin"),
            "scorers": cfg.get("routing_scorers"),
            # NB: `sched` is the scheduling-DELAY metric; the scheduler policy is `schedp`.
            "schedp": cfg.get("scheduler"),
            "admit": cfg.get("admission_policy"),
            "pd": "colocated" if not (cfg.get("prefill_instances") or cfg.get("decode_instances")) else "disagg",
            "mtp": mtp,
            "mnbt": eng.get("max_num_batched_tokens"),
            "mns": eng.get("max_num_seqs"),
            # all-to-all backend: the engine field when the search swept it, else the EP state.
            "a2a": eng.get("all2all_backend") or ("ep" if par.get("enable_expert_parallel") else "n/a"),
            # Expert-parallel width is DERIVED as tp x pcp x dp (pcp unset = 1), mirroring
            # blis-schemas Parallelism.ExpertParallelWidth(). A width of 1 shards no experts --
            # and is exactly what makes a big MoE model refuse to fit ("expert-parallel width 1").
            "epw": (par.get("tp", 1) * par.get("dp", 1)) if par.get("enable_expert_parallel") else None,
            # metrics (reference short keys)
            "pu": pu,                              # per-user tok/s (see _per_user_rate)
            "pu_is_step_estimate": pu_is_step_estimate,   # True = 1000/itl fallback under MTP (undercount)
            "cl": round(m["tokens_per_sec"], 1) if m.get("tokens_per_sec") is not None else None,
            # Deployment detail for the click-to-inspect panel: what BLIS actually consumed.
            "pp": par.get("pp", 1),
            "nodes": _pool_nodes(dep),
            "fabric": "ib-400g" if _pool_nodes(dep) > 1 else None,
            "bs": eng.get("block_size"),
            "mml": eng.get("max_model_len"),
            "gmu": eng.get("gpu_memory_utilization"),
            "cg": eng.get("cudagraph_mode"),
            "quant": eng.get("quantization"),
            "cdt": eng.get("cache_dtype"),
            "scen": dep.get("scenario"),
            "cmd": _blis_cmd(r),
            # Output per GPU: the dual objective's efficiency half, and a brush axis in the
            # explorer. Null (never 0) when throughput is unknown, so a failed run is not plotted
            # as a perfectly efficient one.
            "tpg": (round(m["tokens_per_sec"] / gpus, 2)
                    if m.get("tokens_per_sec") is not None and gpus else None),
            # Every latency view must share ONE statistic, or switching the explorer's axis
            # silently swaps a mean for a tail: itl/ttft/e2e/sched are all p99 (the basis the
            # interactivity floor and `pu` use). The mean is kept separately under itl_mean.
            "itl": round(m["itl_p99_ms"], 3) if m.get("itl_p99_ms") else None,
            "itl_mean": round(itl_mean, 3) if itl_mean else None,
            "ttft": m.get("ttft_p99_ms"),
            "sched": m.get("scheduling_delay_p99_ms"),
            "e2e": m.get("e2e_p99_ms"),
            "hit": m.get("prefix_hit_rate", m.get("hit_rate")),
            "to": float(m.get("timed_out_requests", 0) or 0),
            "drop": float(m.get("dropped_unservable", 0) or 0),
            "p95": m.get("e2e_p95_ms"),
            "ok": ok,
            "status": "complete" if ok else (r.get("phase") or "infeasible"),
            "viol": viol,
        })
    _disambiguate_series(out)
    return out


# Engine/routing knobs that can distinguish two runs of the SAME topology. Order fixes the order
# of a variant description so it is deterministic. Keys are RECORDS fields (see build_records).
DISAMBIGUATING_KNOBS = ("route", "scorers", "mnbt", "mns", "bs", "gmu", "schedp", "admit", "a2a")

# Readable names for each knob, used in the `variant` description. These are spelled out on
# purpose: an invented shorthand ("b2k" for max_num_batched_tokens=2048) tells a reader nothing.
KNOB_NAMES = {"route": "routing", "scorers": "scorers", "mnbt": "batch tokens",
              "mns": "max seqs", "bs": "block size", "gmu": "gpu mem util",
              "schedp": "scheduler", "admit": "admission", "a2a": "MoE dispatch"}


def _disambiguate_series(records):
    """Fill each record's `variant` — the knob delta from its topology group's baseline.

    The `series` label stays PURE TOPOLOGY (`64x TP4xDP2 + EP`): GPUs x TP x EP x instances is
    what matters most in LLM inference and what a reader scans. Engine and routing knobs are not
    squeezed into it — doing that produced `64x TP4xDP2 + EP · ll · b8k · seq256 · mem0.9` with
    invented abbreviations, which is unreadable. They live in `variant` (a spelled-out phrase, for
    the hover card and table), in `knobs` (for the facets), and in the click panel in full.

    Scoped to the topology group: a row is described only by the knobs that differ among the runs
    sharing its own topology, measured against that group's most common value (the unrefined
    runs), so the baseline row has an EMPTY variant and a shape the search never refined keeps a
    clean name.
    """
    for r in records:
        r.setdefault("variant", "")
        r.setdefault("knobs", {})
    if not records:
        return
    import collections
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r.get("tp"), r.get("dp"), r.get("epw"), r.get("ni"))].append(r)
    for rows in groups.values():
        # knobs that actually differ WITHIN this topology group
        varying = [k for k in DISAMBIGUATING_KNOBS
                   if len({json.dumps(r.get(k), sort_keys=True) for r in rows}) > 1]
        for r in rows:
            r["knobs"] = {k: r.get(k) for k in varying}
        if not varying:
            continue
        # the group's baseline value per knob = its most common one (the unrefined runs)
        base = {k: collections.Counter(json.dumps(r.get(k), sort_keys=True)
                                       for r in rows).most_common(1)[0][0] for k in varying}
        for r in rows:
            bits = [_knob_phrase(k, r.get(k)) for k in varying
                    if r.get(k) is not None
                    and json.dumps(r.get(k), sort_keys=True) != base[k]]
            r["variant"] = ", ".join(b for b in bits if b)


def _knob_phrase(key, value):
    """Spelled-out description of one knob setting, e.g. "batch tokens 2,048"."""
    name = KNOB_NAMES.get(key, key)
    if key == "route":
        return f"{value} routing"           # "least-loaded routing"
    if key in ("mnbt", "mns", "bs"):
        return f"{name} {value:,}"          # thousands separator: 2,048 not b2k
    return f"{name} {value}"


def _series_label(tp, ni, par):
    """e.g. "32x TP4xDP2 + EP". dp is shown only when > 1: a label that hid it made TP1 and
    TP1xDP8 — different deployments on different GPU counts — read as the same series."""
    ep = " + EP" if par.get("enable_expert_parallel") else ""
    dp = par.get("dp", 1) or 1
    shape = f"TP{tp}" + (f"xDP{dp}" if dp > 1 else "")
    return f"{ni}x {shape}{ep}" if tp and ni else (f"{shape}{ep}" if tp else "config")


# --- FACTS: precomputed stats the brain's prose references ------------------

def build_facts(records, scenario, pareto):
    servable = [r for r in records if r.get("ok")]
    concs = sorted({r["conc"] for r in records if r.get("conc") is not None})
    cls = [r["cl"] for r in records if r.get("cl") is not None]
    peak = max((r for r in records if r.get("cl") is not None), key=lambda r: r["cl"], default=None)
    pus = [r["pu"] for r in records if r.get("pu") is not None]
    winner = (pareto.get("winner") or {})
    return {
        "n_points": len(records),
        "n_admissible": len(servable),
        "n_infeasible": len(records) - len(servable),
        "n_configs": len({r["series"] for r in records}),
        "conc_list": ", ".join(str(c) for c in concs),
        "conc_min": concs[0] if concs else None,
        "conc_max": concs[-1] if concs else None,
        "peak_cluster": peak["cl"] if peak else None,
        "peak_cluster_conc": peak["conc"] if peak else None,
        "peak_cluster_pu": peak["pu"] if peak else None,
        "baseline_pu": max(pus) if pus else None,
        "max_servable_conc": max((r["conc"] for r in servable if r.get("conc") is not None),
                                  default=None),
        "winner": winner.get("chosen"),
        "model": scenario.get("model"),
        "gpu_budget": scenario.get("gpu_budget"),
        "hardware": scenario.get("hardware"),
    }


# --- figure rendering --------------------------------------------------------

def render_figure(name, records, scenario):
    """Dispatch a figure name to figures.py. Two degrade paths, both -> a captioned placeholder:
      * unknown figure name (not in the registry);
      * a registered figure whose keying knob was NOT varied across the runs (QA G7/F3) -- e.g. a
        routing figure on a search that never swept routing. The claim in the docs that such figures
        placeholder is now actually enforced here."""
    spec = figures.FIGURES.get(name)
    if not spec:
        return figures.placeholder(f"figure '{name}' is not available for this run")
    for knob in spec.get("keys", ()):
        if not figures.varies(records, knob):
            return figures.placeholder(f"'{knob}' was not varied, so '{name}' has nothing to compare")
    builder = spec["builder"]
    if builder is figures.fig_operating_curve:
        floor_pu = (scenario.get("interactivity_floor") or {}).get("tok_per_s_per_user")
        return builder(records, floor_pu=floor_pu)
    return builder(records)


# --- narrative body ----------------------------------------------------------

def _prose(paras):
    return '<div class="prose">' + "".join(f"<p>{p}</p>" for p in paras) + "</div>"


def _header(report, scenario, facts):
    title = report.get("title") or f"Serving {facts.get('model') or 'this model'} on " \
        f"{facts.get('gpu_budget') or '?'}× {facts.get('hardware') or 'GPUs'}"
    standfirst = report.get("standfirst") or (
        f"A configuration search over {facts.get('n_configs')} serving shapes and "
        f"{facts.get('n_points')} simulated runs, and what they say about how to run this model.")
    meta = report.get("meta") or [
        f"BLIS discrete-event simulation", f"{facts.get('n_points')} runs"]
    meta_html = "".join(f"<span>{m}</span>" for m in meta)
    return (f"<header><h1>{esc(title)}</h1>"
            f'<p class="standfirst">{standfirst}</p>'
            f'<div class="meta">{meta_html}</div></header>')


def _hero(report):
    cells = report.get("hero")
    if not cells:
        return ""
    out = []
    for i, c in enumerate(cells[:3]):
        cls = "pcell run" if (c.get("run") or i == len(cells[:3]) - 1) else "pcell"
        out.append(f'<div class="{cls}"><h4>{esc(c.get("h",""))}</h4>'
                   f'<p>{c.get("html", c.get("p",""))}</p></div>')
    return f'<section class="hero"><div class="panel">{"".join(out)}</div></section>'


def _section_measurements(report):
    intro = report.get("explore_prose") or [
        "Everything that follows comes from these runs: each is one configuration at one load.",
        "Click any value to filter — every count, the chart and the table update together."]
    return (f'<section class="finding" id="explore"><h2><span class="num">1</span>'
            f'The measurements</h2>{_prose(intro)}<div id="xf"></div></section>')


def _caption(section, fignum):
    """The figcaption for one section's figure: "Figure N." then the figure's SHARED caption.

    The caption comes from report/lexicon.json, which the React dashboard reads too, so one plot
    cannot be described in two vocabularies across the two surfaces (issue #13, item 7b). A
    section may still APPEND its own run-specific sentence via `caption_note` -- the shared
    sentence says what the axes mean, the note says what this run's numbers do. A section whose
    figure has no lexicon entry falls back to its authored `caption` verbatim.
    """
    name = section.get("figure")
    shared = lexicon.FIGURES.get(name, {}).get("caption")
    if not shared:
        return section.get("caption", "")
    note = section.get("caption_note") or ""
    lead = f"<b>Figure {fignum}. {esc(shared)}</b>"
    return f"{lead} {note}".strip()


def _sections(report, records, scenario):
    out = []
    fignum = 0
    for s in report.get("sections", []):
        num = s.get("num", "")
        title = esc(s.get("title", ""))
        fig = ""
        if s.get("figure"):
            fignum += 1
            svg = render_figure(s["figure"], records, scenario)
            fig = (f'<figure><div class="figwrap">{svg}</div>'
                   f'<figcaption>{_caption(s, fignum)}</figcaption></figure>')
        prose = _prose(s.get("prose", [])) if s.get("prose") else ""
        out.append(f'<section class="finding"><h2><span class="num">{num}</span>{title}</h2>'
                   f'{fig}{prose}</section>')
    return "".join(out)


# Flags the "What to run" table prints. This is the list a reader COPIES, so an omission is a
# wrong deployment: it previously showed only tp + num-instances, leaving out expert parallelism,
# data parallelism and routing — exactly the knobs that distinguish two rows whose topology
# matches. `epw`/`dp` render as the flags vLLM actually takes.
#
# The list and the skip table come from report/lexicon.json, which the React dashboard reads too
# (web/src/lib.ts `FLAGS`/`FLAG_SKIP`), so the two surfaces cannot print different deployments for
# one run. A key absent from _FLAG_SKIP still skips None -- see lexicon.FLAG_SKIP.
_FLAG_LABELS = lexicon.FLAG_LABELS
_FLAG_SKIP = lexicon.FLAG_SKIP


def _obj_rank(r):
    """Sort key ranking one record on the SHARED objective (lexicon.objective): tok/s per GPU,
    then fleet tok/s, then tok/s per user, then run id. The tiebreaks make the pick deterministic
    rather than ledger-order dependent -- two runs at one load with equal efficiency would
    otherwise resolve differently on the two surfaces (and between two renders)."""
    def num(key):
        v = _as_num(r.get(key))
        return v if v is not None else float("-inf")
    primary = num(lexicon.OBJECTIVE["key"])
    ties = [num(k) if k != "id" else 0.0 for k in lexicon.OBJECTIVE["tiebreak"]]
    return (primary, *ties)


def operating_points(records):
    """One operating point per load: the best SERVABLE run at that concurrency on the shared
    objective (tok/s per GPU -- see lexicon.objective), concurrency-ascending.

    This is the report half of the parity fix for issue #13. It previously ranked by per-user
    rate while the dashboard's `operatingPoints()` ranked by tok/s per GPU, so at high load the
    two "What to run" tables recommended DIFFERENT configurations for one scenario -- and the
    report's own recommendation card, which quotes the search winner, ranks per-GPU too. Both
    surfaces now rank on the one objective the search itself optimises.

    Rows with conc=None are skipped: a null load is not a meaningful operating point, and mixing
    None with ints breaks sorted() (jgchn review). build_facts/figures already treat conc=None as
    not-a-value, so this matches. A run with no `tpg` cannot be ranked on the objective, so it is
    skipped rather than ranked as infinitely inefficient.
    """
    by_conc = {}
    for r in records:
        if not r.get("ok") or r.get("conc") is None or _as_num(r.get("tpg")) is None:
            continue
        cur = by_conc.get(r["conc"])
        # strict >: the first record at a load wins a tie, and the id tiebreak below decides
        # between genuinely equal rows, so the pick does not depend on ledger order
        if cur is None or _obj_rank(r) > _obj_rank(cur) or (
                _obj_rank(r) == _obj_rank(cur) and str(r.get("id")) < str(cur.get("id"))):
            by_conc[r["conc"]] = r
    return [by_conc[c] for c in sorted(by_conc)]


def _operating_row(r):
    """One derived operating point, in the authored `operating_points` shape (so both paths feed
    the same renderer). Carries every column lexicon.operating_columns declares."""
    return {"conc": r.get("conc"), "series": r.get("series"), "variant": r.get("variant"),
            "flags": {k: r.get(k) for k, _ in _FLAG_LABELS},
            "pu": r.get("pu"), "cl": r.get("cl"), "tpg": r.get("tpg"),
            "itl": r.get("itl"), "ttft": r.get("ttft"), "e2e": r.get("e2e")}


def _operating_table(report, records):
    """The "What to run" table. DERIVED from the records whenever there are any, so it cannot
    disagree with the dashboard's table over the same run set (issue #13): a hand-authored
    `operating_points` list carried its own metrics and its own choice of rows, which is exactly
    the drift the issue is about. The authored list is still honoured when the run dir has no
    usable ledger -- a brain-written table beats no table at all."""
    derived = operating_points(records)
    pts = [_operating_row(r) for r in derived] if derived else (report.get("operating_points") or [])
    if not pts:
        return ""
    cols = lexicon.OPERATING_COLUMNS
    rows = []
    for p in pts:
        flags = p.get("flags", {})
        fl = "".join(
            # a boolean flag prints bare (--enable-expert-parallel, not "... 8")
            f'<span class="fl"><i>{esc(lbl)}</i>'
            + ("" if k in lexicon.BOOL_FLAGS else f' {esc(flags[k])}') + "</span>"
            for k, lbl in _FLAG_LABELS
            if flags.get(k) is not None and flags.get(k) not in _FLAG_SKIP.get(k, (None,)))
        # The configuration cell names the topology and its knob delta above the flags, matching
        # the dashboard's ConfigCell -- a flag list alone does not say WHICH config a row is.
        name = (f'<div class="op-name">{esc(p["series"])}</div>' if p.get("series") else "")
        var = (f'<div class="op-var">{esc(p["variant"])}</div>' if p.get("variant") else "")
        cells = [f'<td class="n hero">{esc(p.get("conc"))}</td>', f"<td>{name}{var}{fl}</td>"]
        cells += [f'<td class="n">{_num(p.get(c))}</td>' for c in cols[2:]]
        rows.append(f'<tr>{"".join(cells)}</tr>')
    heads = "".join(
        f'<th{"" if i < 2 else " class=\"n\""}>{esc(lexicon.label(c))}</th>'
        for i, c in enumerate(cols))
    return (f'<table class="short"><thead><tr>{heads}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _what_to_run(report, records):
    prose = report.get("recommendation_prose")
    if not prose and not records:
        return ""
    num = report.get("recommendation_num", 9)
    # The objective the rows are ranked on is STATED, not implied: the issue asks for the two
    # surfaces to rank on the same objective *and say which*. The dashboard prints the same
    # sentence (lexicon.objective.blurb) over its own copy of this table.
    # ONE prose block: two adjacent .prose divs lost the inter-paragraph spacing, so the shared
    # blurb ran straight on from the brain's last sentence.
    body = _prose(list(prose or []) + [lexicon.OBJECTIVE["blurb"]])
    return (f'<section class="finding"><h2><span class="num">{num}</span>What to run</h2>'
            f'{body}{_operating_table(report, records)}</section>')


def _glossary(report):
    entries = report.get("glossary")
    if not entries:
        return ""
    num = report.get("glossary_num", 10)
    items = []
    for g in entries:
        gid = esc(g.get("id", ""))
        term = esc(g.get("term", ""))
        definition = g.get("def", "")
        long = g.get("long", "")
        items.append(f'<div class="gl-entry" id="{gid}"><dt>{term}</dt>'
                     f'<dd><b>{definition}</b> {long}</dd></div>')
    return (f'<section class="finding" id="glossary"><h2><span class="num">{num}</span>Terms</h2>'
            f'<div class="prose"><p>Each term is linked from its first use above. Hover a link '
            f'for the short definition.</p></div><dl class="gl-list">{"".join(items)}</dl></section>')


def _footer(scenario, pareto, records, n_malformed=0):
    # run_path intentionally NOT a parameter: the footer must not depend on the directory path/name
    # (QA G2). The run is identified by the in-data run_id only.
    ps = scenario.get("pinned_stack") or {}
    stack = ""
    if ps:
        stack = (f" · BLIS {esc(ps.get('blis','?'))}, kernel {esc(ps.get('kernel','?'))}, "
                 f"catalog {esc(ps.get('catalog','?'))}, registry {esc(ps.get('registry','?'))}")
    # Identify the run ONLY by the id carried IN the data (scenario/pareto), never by the directory
    # path or name, so the HTML is a pure function of the run-dir CONTENTS: identical data renders
    # byte-identical no matter where the directory lives or what it is called (QA G2). If the data
    # carries no run_id we say so plainly rather than falling back to the (location-dependent)
    # basename -- a missing id must not make the output depend on the directory name.
    run_id = (scenario.get("run_id") or pareto.get("run_id")
              or "unidentified (no run_id in scenario.json/pareto.json)")
    # Surface skipped malformed ledger rows instead of hiding them (QA G3).
    malformed = ""
    if n_malformed:
        malformed = (f' <b>Note:</b> {n_malformed} malformed line(s) in <code>runs.jsonl</code> were '
                     f'skipped and are not reflected in the figures or counts.')
    return (f'<div class="foot"><b>These are simulator predictions</b> from the '
            f'blis-latency-kernel backend — spot-check against the real llm-d measurements in '
            f'curvebender-tools/reports before acting. Run <code>{esc(run_id)}</code>{stack}. '
            f'Raw per-run ledger, scenario and analysis sit beside this report in the run directory '
            f'({len(records)} runs).{malformed}</div>')


def render_report(scenario, pareto, runs, report, run_path=None, n_malformed=0):
    # run_path is accepted for call-site compatibility but intentionally UNUSED in the output: the
    # report is a pure function of the run-dir contents, independent of where it lives (QA G2).
    _ = run_path
    records = build_records(runs, scenario)
    facts = build_facts(records, scenario, pareto)
    body = (_header(report, scenario, facts) + _hero(report) + _section_measurements(report)
            + _sections(report, records, scenario) + _what_to_run(report, records)
            + _glossary(report) + _footer(scenario, pareto, records, n_malformed))
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    title = report.get("title") or facts.get("model") or "config-search report"
    # Single-pass substitution: each placeholder is replaced exactly once, so a literal "__BODY__"
    # or "__TITLE__" that legitimately appears in brain-authored prose (now part of `body`) is NOT
    # re-scanned and mistaken for an unconsumed marker (QA G8). We verify the TEMPLATE had each slot
    # before substituting, instead of scanning the finished HTML for the markers afterwards.
    replacements = {
        "__TITLE__": esc(title),
        "__BODY__": body,
        "__VENDOR_JS__": _vendor_js(),      # inline d3 + crossfilter (self-contained / offline)
        RECORDS_PH: _json_for_script(records),
        FACTS_PH: _json_for_script(facts),
        LEXICON_PH: _json_for_script(lexicon.LEX),
    }
    for ph in replacements:
        if ph not in html:
            raise RuntimeError(f"template placeholder {ph!r} is missing from {TEMPLATE}")
    # Single-pass over the TEMPLATE with a FUNCTION replacement: inserted text (vendor JS, prose) is
    # never re-scanned, and backslashes in the minified libs are not treated as regex escapes.
    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    return pattern.sub(lambda mo: replacements[mo.group(0)], html)


# --- multi-run card index ----------------------------------------------------

def _card_for(scenario, pareto, report, href):
    finalists = pareto.get("finalists") or []
    winner = (pareto.get("winner") or {}).get("chosen") or (
        finalists[0].get("label") if finalists else "?")
    wl = scenario.get("workload") or {}
    return {"href": href, "model": scenario.get("model", "?"),
            "hw": f"{scenario.get('gpu_budget','?')}x {scenario.get('hardware','?')}",
            "workload": wl.get("preset") or wl.get("name") or "workload",
            "winner": winner, "summary": report.get("standfirst", "")[:120],
            "run_id": scenario.get("run_id") or os.path.basename(os.path.dirname(href))}


def render_card_index(cards):
    with open(INDEX_TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    return html.replace(CARDS_PH, _json_for_script(cards))   # script-safe (review F5)


# --- entrypoints -------------------------------------------------------------

def write_one(run_dir, out=None):
    scenario, pareto, runs, report, n_malformed = load_run_dir(run_dir)
    if not runs and not pareto:
        raise SystemExit(f"{run_dir}: no runs.jsonl or pareto.json — not a config-search run directory")
    out = out or os.path.join(run_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_report(scenario, pareto, runs, report, os.path.abspath(run_dir), n_malformed))
    return out


def write_tree(parent):
    children = sorted(d for d in os.listdir(parent) if is_run_dir(os.path.join(parent, d)))
    if not children:
        raise SystemExit(f"{parent}: no run directories (no child has runs.jsonl/pareto.json)")
    cards = []
    for name in children:
        rd = os.path.join(parent, name)
        write_one(rd)
        scenario, pareto, _, report, _ = load_run_dir(rd)
        cards.append(_card_for(scenario, pareto, report, f"{name}/index.html"))
    index = os.path.join(parent, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(render_card_index(cards))
    return index, children


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="config-search HTML report (run-dir -> index.html); never runs BLIS")
    ap.add_argument("path", help="a run directory, or a parent of several run directories")
    ap.add_argument("--out", help="output HTML path (single run dir only; default <run-dir>/index.html)")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.path):
        raise SystemExit(f"{args.path}: not a directory")
    try:
        if is_run_dir(args.path):
            print(f"wrote {write_one(args.path, args.out)}")
        else:
            if args.out:
                raise SystemExit("--out is only valid for a single run directory, not a parent of many")
            index, children = write_tree(args.path)
            print(f"wrote {index} + {len(children)} report(s): {', '.join(children)}")
    except MalformedInput as e:
        # A present-but-unparseable scenario/pareto/report must abort, not silently unconstrain the
        # recommendation (review F1).
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
