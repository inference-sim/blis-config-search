#!/usr/bin/env python3
"""Build the static config-search gallery site from committed run directories.

    python3 -I scripts/build_gallery.py                 # reports/ -> site/
    python3 -I scripts/build_gallery.py --reports R --out O
    python3 -I scripts/build_gallery.py --no-mock       # exclude synthetic run dirs (production)

Pure transform: reads each run dir's scenario.json / pareto.json / gallery.json, renders
each report with report.py (which never runs BLIS), and emits:

    <out>/<run_id>/index.html   one self-contained report per scenario (+ its data files)
    <out>/data.js               window.SCENARIOS = [...]  (the gallery dataset)
    <out>/index.html            the gallery shell (from gallery/index.template.html)

Deterministic (INV-6): run dirs are processed in sorted run_id order and the dataset is
serialized stably, so identical inputs produce identical output. Never runs BLIS or the
LLM search; it only reads JSON and renders/copies HTML.

Safety: every run dir is validated against a strict whitelist (known keys, enum fields,
run_id pattern, numeric sanity) before it contributes a row. An invalid dir aborts the
build loudly rather than emitting a half-trusted site.
"""
import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SKILL = os.path.join(ROOT, ".claude", "skills", "config-search")
TEMPLATE = os.path.join(ROOT, "gallery", "index.template.html")

# report.py lives in the skill; import it so a report is a function call, not a subprocess.
sys.path.insert(0, os.path.join(SKILL, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for build_catalog
import report as report_py  # noqa: E402

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# Hardware is validated by FORM, not against an allowlist.
#
# This was a hardcoded set, which drifted both ways: it listed `mi300x`, which blis-catalog does
# not carry, and omitted `b200`/`b300`/`l40s`, which it does -- so a legitimate B200 report failed
# the build with "unknown hardware 'b200'". Reading the set from the catalog instead looked like
# the fix, but it made a pure, offline transform depend on a SIBLING CLONE: CI has no
# ../blis-catalog, so the lookup silently fell back to the stale set and failed there while
# passing locally. That is worse than the original bug, because it only shows up in CI.
#
# The authority on what hardware a run used is the RUN, which BLIS already refused to produce for
# a chip it cannot price (NS-6) -- re-deciding that here adds nothing. What this build does need is
# that the value is a safe, well-formed token, since it reaches file names and the page.
HARDWARE_RE = re.compile(r"^[a-z][a-z0-9.-]{1,31}$")
# Report-data files copied verbatim next to the rendered index.html (so a report stays
# regenerable from the published site). Nothing executable; no arbitrary contributor files.
REPORT_DATA_FILES = ("scenario.json", "pareto.json", "runs.jsonl", "report.json")


class Invalid(Exception):
    """A run directory failed validation; the build aborts."""


def _req(d, key, where):
    if key not in d:
        raise Invalid(f"{where}: missing required key '{key}'")
    return d[key]


def _num(v, where, lo=0.0, hi=1e9):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise Invalid(f"{where}: expected a number, got {v!r}")
    if not (lo <= float(v) <= hi):
        raise Invalid(f"{where}: {v} out of range [{lo}, {hi}]")
    return float(v)


def _workload_name(scenario):
    wl = scenario.get("workload") or {}
    if isinstance(wl, str):
        return wl
    # A REPLAY workload has no `preset` -- its traffic is a recorded corpus, named by the harness
    # scenario it replays. Reading only `preset` labelled every replay row the literal word
    # "workload", which tells a reader nothing and makes two different replays indistinguishable.
    rep = wl.get("replay") or {}
    return wl.get("preset") or rep.get("scenario") or wl.get("name") or "workload"


def validate_and_project(run_dir):
    """Validate one run dir and project its gallery row. Raises Invalid on any problem."""
    name = os.path.basename(run_dir.rstrip("/"))
    if not RUN_ID_RE.match(name):
        raise Invalid(f"{name}: directory name is not a safe run_id (^[a-z0-9][a-z0-9-]{{0,63}}$)")

    scenario = report_py._read_json(os.path.join(run_dir, "scenario.json"), {})
    pareto = report_py._read_json(os.path.join(run_dir, "pareto.json"), {})
    gallery = report_py._read_json(os.path.join(run_dir, "gallery.json"), {})
    if not pareto:
        raise Invalid(f"{name}: pareto.json missing or empty")

    run_id = scenario.get("run_id") or name
    if not RUN_ID_RE.match(run_id):
        raise Invalid(f"{name}: scenario run_id {run_id!r} is not safe")

    model = str(_req(scenario, "model", name))
    hardware = str(_req(scenario, "hardware", name)).lower()
    if not HARDWARE_RE.match(hardware):
        raise Invalid(f"{name}: hardware {hardware!r} is not a safe catalog key "
                      f"(expected lowercase letters, digits, dots and dashes)")
    gpu_budget = int(_num(_req(scenario, "gpu_budget", name), name, 1, 100000))
    workload = _workload_name(scenario)

    finalists_in = pareto.get("finalists") or []
    if not finalists_in:
        raise Invalid(f"{name}: pareto.json has no finalists")
    chosen = (pareto.get("winner") or {}).get("chosen")

    # The winner is named with its knob delta ("4x TP2 (least-loaded routing)") because several
    # finalists share a topology once the search refines engine/routing knobs. Match on the
    # (label, variant) PAIR — matching the bare label silently fell through to the last finalist
    # and showed the wrong winner on the site.
    chosen_label = (pareto.get("winner") or {}).get("label")
    chosen_variant = (pareto.get("winner") or {}).get("variant")

    finalists, winner_row = [], None
    for f in finalists_in:
        label = str(_req(f, "label", f"{name} finalist"))
        variant = str(f.get("variant") or "")
        at = f.get("at_c_use") or {}
        row = {
            "label": label,
            "variant": variant,
            # what the site shows: topology, plus the knob delta when there is one
            "name": f"{label} ({variant})" if variant else label,
            "tp": int(f.get("config", {}).get("tp", 0) or 0),
            "ni": int(f.get("config", {}).get("num_instances", 0) or 0),
            "gpus": int(_num(f.get("gpus", 0), f"{name}:{label} gpus", 0, 100000)),
            "c_use": int(_num(f.get("c_use", 0), f"{name}:{label} c_use", 0, 1e7)),
            "tps": _num(at.get("tokens_per_sec", 0), f"{name}:{label} tps", 0, 1e9),
            "tps_per_gpu": _num(at.get("tokens_per_sec_per_gpu", 0), f"{name}:{label} tps/gpu", 0, 1e9),
            "per_user": _num(at.get("tok_per_s_per_user", 0), f"{name}:{label} per_user", 0, 1e6),
            "itl_p99": _num(at.get("itl_p99_ms", 0), f"{name}:{label} itl", 0, 1e6),
            # The load the numbers above were MEASURED at. It equals c_use for a well-formed
            # finalist; it differs only when the dedicated c_use run failed and a lower probe
            # stood in, and the card pairs these figures with c_use, so a silent substitution
            # would read as the throughput at a load never run. Absent in a run dir written
            # before the field existed -> fall back to c_use rather than inventing a mismatch.
            "measured_at": int(_num(at.get("measured_at_concurrency", f.get("c_use", 0)) or 0,
                                    f"{name}:{label} measured_at", 0, 1e7)),
        }
        finalists.append(row)
        # prefer the explicit (label, variant) pair; fall back to the legacy `chosen` string so a
        # report written before the variant field still resolves its winner
        if chosen_label is not None:
            if label == chosen_label and variant == (chosen_variant or ""):
                winner_row = row
        elif chosen in (row["name"], label):
            winner_row = row
    # Resolve the winner. If pareto.json named one (by label/variant or legacy `chosen`) it MUST
    # match a finalist — silently falling back to finalists[-1] showed the worst-ranked config as
    # the recommendation. Only when no winner is named at all (older reports) do we fall back.
    named = chosen_label is not None or chosen is not None
    if winner_row is not None:
        winner = winner_row
    elif named:
        want = chosen_label if chosen_label is not None else chosen
        raise Invalid(f"{name}: winner {want!r} does not match any finalist "
                      f"({[r['name'] for r in finalists]})")
    else:
        winner = finalists[-1]

    floor = scenario.get("interactivity_floor") or {}
    floor_str = ""
    if floor.get("tok_per_s_per_user") is not None:
        floor_str = f"{floor['tok_per_s_per_user']:g} tok/s/user"
        if floor.get("itl_p99_ms_max") is not None:
            floor_str += f" (ITL p99 ≤ {floor['itl_p99_ms_max']:g} ms)"

    # gallery.json sidecar: display fields not carried by scenario/pareto. A MOCK marker
    # file OR gallery.synthetic flags a synthetic row; both are honored.
    synthetic = bool(gallery.get("synthetic")) or os.path.exists(os.path.join(run_dir, "MOCK"))

    row = {
        "run_id": run_id, "synthetic": synthetic,
        "model": model, "hardware": hardware.upper(), "gpu_budget": gpu_budget,
        "workload": workload, "isl_osl": str(gallery.get("isl_osl", "")),
        "floor": floor_str, "headroom": scenario.get("headroom"),
        "n_runs": int(gallery.get("n_runs", 0) or 0),
        "n_servable": int(gallery.get("n_servable", 0) or 0),
        "generated": str(gallery.get("generated", "")),
        "winner": winner, "finalists": finalists,
        "reason": str(gallery.get("reason") or (pareto.get("winner") or {}).get("reason", "")),
        # A run-dir-authored caveat about the DATA itself (not the recommendation), shown on the
        # card and banners the report. Used for the reports whose workload label does not match
        # the traffic they were priced on -- see _CAVEAT_BANNER.
        "caveat": str(gallery.get("caveat", "")),
    }
    return run_id, row


# A fixed banner/back-link injected into each rendered report. report.py stays a pure,
# gallery-agnostic transform (and lives in the read-only skill); these are gallery concerns,
# so the build adds them after the fact, right inside the report's .wrap container.
# The report is served by the SPA at data/reports/<run_id>/index.html, so the gallery shell
# (the SPA's own index.html) is THREE levels up — through <run_id>, reports/, and data/. The
# old ../index.html resolved to data/reports/index.html and 404'd on Pages.
_BACK_LINK = ('<div style="font:600 13px/1.4 system-ui,sans-serif;margin:16px 0 0">'
              '<a href="../../../index.html#/reports" style="color:#1f6e5c;text-decoration:none">← back to the gallery</a></div>')
_MOCK_BANNER = ('<div role="note" style="font:600 13px/1.5 system-ui,sans-serif;margin:14px 0 0;'
                'padding:10px 14px;border:1px solid #d8b984;background:#f6e8ca;color:#6b4508;border-radius:6px">'
                'MOCK scenario — illustrative placeholder data for the gallery demo, not a BLIS measurement. '
                'Do not deploy from these numbers.</div>')


def _caveat_banner(text):
    """A banner for a data caveat the run dir declares in gallery.json's `caveat`.

    Distinct from the MOCK banner: these ARE real BLIS measurements, but something about what
    they measured does not match what the report says it measured, and a reader deciding a
    deployment from them has to know that.
    """
    return ('<div role="note" style="font:600 13px/1.5 system-ui,sans-serif;margin:14px 0 0;'
            'padding:10px 14px;border:1px solid #d8b984;background:#f6e8ca;color:#6b4508;'
            'border-radius:6px">' + report_py.esc(text) + '</div>')


def _decorate_report(path, synthetic, caveat=""):
    """Inject a back-to-gallery link (all reports), a MOCK banner (synthetic only) and a data
    caveat banner (when the run dir declares one) just inside the report's .wrap container,
    without touching report.py."""
    try:
        with open(path, encoding="utf-8") as f:
            html = f.read()
    except OSError:
        return
    anchor = '<div class="wrap">'
    i = html.find(anchor)
    if i == -1:
        return
    inject = _BACK_LINK + (_MOCK_BANNER if synthetic else "") + (_caveat_banner(caveat) if caveat else "")
    at = i + len(anchor)
    html = html[:at] + inject + html[at:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def build(reports_dir, out_dir, include_mock=True):
    reports_dir = os.path.abspath(reports_dir)
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(reports_dir):
        raise SystemExit(f"{reports_dir}: not a directory")

    run_dirs = sorted(
        os.path.join(reports_dir, d) for d in os.listdir(reports_dir)
        if report_py.is_run_dir(os.path.join(reports_dir, d))
    )
    if not run_dirs:
        raise SystemExit(f"{reports_dir}: no run directories (none has pareto.json/runs.jsonl)")

    os.makedirs(out_dir, exist_ok=True)
    scenarios, built = [], []
    for rd in run_dirs:
        run_id, row = validate_and_project(rd)
        if row["synthetic"] and not include_mock:
            continue
        # render the report fresh from committed JSON (ignores any committed index.html,
        # guaranteeing it is report.py output — the safe default from the spec, Q3)
        dest = os.path.join(out_dir, run_id)
        os.makedirs(dest, exist_ok=True)
        scenario, pareto, runs, report, _ = report_py.load_run_dir(rd)
        report_py.write_one(rd, out=os.path.join(dest, "index.html"))
        _decorate_report(os.path.join(dest, "index.html"), row["synthetic"], row.get("caveat", ""))
        for fn in REPORT_DATA_FILES:
            src = os.path.join(rd, fn)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(dest, fn))
        # records.json: the flat RECORDS the in-app scenario-detail explorer cross-filters
        # (same projection report.py bakes into each report). Empty list if no runs.jsonl.
        records = report_py.build_records(runs, scenario) if runs else []
        with open(os.path.join(dest, "records.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, sort_keys=True)
        row["n_records"] = len(records)
        scenarios.append(row)
        built.append(run_id)

    scenarios.sort(key=lambda r: r["run_id"])  # stable output
    data_js = "/* generated by build_gallery.py — do not edit */\nwindow.SCENARIOS = " \
        + json.dumps(scenarios, indent=2, sort_keys=True) + ";\n"
    with open(os.path.join(out_dir, "data.js"), "w", encoding="utf-8") as f:
        f.write(data_js)

    # Legacy standalone shell: the webpage is now the Vite+React SPA in web/, which consumes
    # this build's data.js/catalog.js/reports via web/scripts/stage-data.mjs. The old vanilla
    # shell is written only when its template is still present, so build_gallery stays a pure
    # data producer either way.
    if os.path.isfile(TEMPLATE):
        with open(TEMPLATE, encoding="utf-8") as f:
            shell = f.read()
        with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(shell)
        app_src = os.path.join(ROOT, "gallery", "app")
        if os.path.isdir(app_src):
            for fn in sorted(os.listdir(app_src)):
                if fn.endswith((".js", ".css")):
                    shutil.copyfile(os.path.join(app_src, fn), os.path.join(out_dir, fn))

    return built, scenarios


def _write_mode(out_dir, mode):
    """Bake the site mode in as a static constant. 'pages' is read-only; 'local' enables
    the live submission form. The Pages build must never emit 'local'."""
    with open(os.path.join(out_dir, "mode.js"), "w", encoding="utf-8") as f:
        f.write(f'/* generated by build_gallery.py */\nwindow.MODE = {json.dumps(mode)};\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the static config-search gallery site.")
    ap.add_argument("--reports", default=os.path.join(ROOT, "reports"), help="parent of run dirs")
    ap.add_argument("--out", default=os.path.join(ROOT, "site"), help="output site directory")
    ap.add_argument("--no-mock", action="store_true", help="exclude synthetic run dirs (production)")
    ap.add_argument("--mode", choices=["pages", "local"], default="pages",
                    help="pages = read-only (default, for GitHub Pages); local = live submission form")
    ap.add_argument("--siblings", default=os.path.join(ROOT, os.pardir),
                    help="dir with blis-catalog/blis-registry/inference-sim for the catalog tabs")
    args = ap.parse_args(argv)
    try:
        built, scenarios = build(args.reports, args.out, include_mock=not args.no_mock)
    except Invalid as e:
        raise SystemExit(f"validation error: {e}")

    # catalog.js (the models / hardware / workloads spec-sheets) — degrades gracefully if siblings absent
    import build_catalog
    cat = build_catalog.build(os.path.abspath(args.siblings))
    with open(os.path.join(args.out, "catalog.js"), "w", encoding="utf-8") as f:
        f.write("/* generated by build_gallery.py (via build_catalog) */\nwindow.CATALOG = "
                + json.dumps(cat, indent=2, sort_keys=True) + ";\n")
    _write_mode(args.out, args.mode)

    n_mock = sum(1 for s in scenarios if s["synthetic"])
    print(f"built {len(built)} report(s) -> {os.path.relpath(args.out, ROOT)}/  "
          f"({len(scenarios) - n_mock} real, {n_mock} mock) · mode={args.mode}")
    print(f"  catalog: {len(cat['models'])} models, {len(cat['hardware'])} GPUs, "
          f"{len(cat['workloads'])} workloads")
    print("  " + ", ".join(built))


if __name__ == "__main__":
    main()
