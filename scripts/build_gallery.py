#!/usr/bin/env python3
"""Build the site data from the committed run directories in reports/.

    python3 -I scripts/build_gallery.py                 # reports/ -> site/
    python3 -I scripts/build_gallery.py --reports R --out O

A pure transform (it never runs BLIS) that writes:

    <out>/scenarios.json        one row per report, with its run records inlined
    <out>/catalog.json          the model / GPU / workload spec-sheets (build_catalog.py)
    <out>/<run_id>/index.html   the standalone report (report.py), plus its data files

Run dirs are read in sorted order and serialized stably, so identical inputs give identical
output. Every run dir is validated before it contributes a row, and an invalid one aborts the
build rather than publishing a half-trusted site.
"""
import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_catalog  # noqa: E402
import report as report_py  # noqa: E402

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# Validated by form, not an allowlist: BLIS already refused any chip it cannot price, and the
# value only has to be safe in file names and on the page.
HARDWARE_RE = re.compile(r"^[a-z][a-z0-9.-]{1,31}$")
# Copied next to each rendered report; runs.jsonl (~20 MB each) stays in reports/, unpublished.
REPORT_DATA_FILES = ("scenario.json", "pareto.json", "report.json")
# The report is served at data/reports/<run_id>/index.html, three levels below the SPA.
BACK_LINK = ('<div style="font:600 13px/1.4 system-ui,sans-serif;margin:16px 0 0">'
             '<a href="../../../index.html#/reports" style="color:#1f6e5c;text-decoration:none">'
             '← back to the reports</a></div>')


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
    """A preset name, or the harness scenario a replay workload replays."""
    wl = scenario.get("workload") or {}
    if isinstance(wl, str):
        return wl
    return wl.get("preset") or (wl.get("replay") or {}).get("scenario") or "workload"


def _finalist(f, where):
    label = str(_req(f, "label", f"{where} finalist"))
    variant = str(f.get("variant") or "")
    at = f.get("at_c_use") or {}
    cfg = f.get("config") or {}
    w = f"{where}:{label}"
    return {
        "label": label,
        "variant": variant,
        "name": f"{label} ({variant})" if variant else label,
        "tp": int(cfg.get("tp", 0) or 0),
        "ni": int(cfg.get("num_instances", 0) or 0),
        "gpus": int(_num(f.get("gpus", 0), f"{w} gpus", 0, 100000)),
        "c_use": int(_num(f.get("c_use", 0), f"{w} c_use", 0, 1e7)),
        "tps": _num(at.get("tokens_per_sec", 0), f"{w} tps"),
        "tps_per_gpu": _num(at.get("tokens_per_sec_per_gpu", 0), f"{w} tps/gpu"),
        "per_user": _num(at.get("tok_per_s_per_user", 0), f"{w} per_user", 0, 1e6),
        "itl_p99": _num(at.get("itl_p99_ms", 0), f"{w} itl", 0, 1e6),
    }


def validate_and_project(run_dir):
    """Validate one run dir and return (run_id, its row for scenarios.json)."""
    name = os.path.basename(run_dir.rstrip("/"))
    if not RUN_ID_RE.match(name):
        raise Invalid(f"{name}: directory name is not a safe run_id (^[a-z0-9][a-z0-9-]{{0,63}}$)")
    scenario, pareto, runs, _report, _ = report_py.load_run_dir(run_dir)
    totals = report_py._read_json(os.path.join(run_dir, "experiment.json"), {}).get("totals") or {}
    if not pareto:
        raise Invalid(f"{name}: pareto.json missing or empty")

    run_id = scenario.get("run_id") or name
    if not RUN_ID_RE.match(run_id):
        raise Invalid(f"{name}: scenario run_id {run_id!r} is not safe")
    hardware = str(_req(scenario, "hardware", name)).lower()
    if not HARDWARE_RE.match(hardware):
        raise Invalid(f"{name}: hardware {hardware!r} is not a safe catalog key")

    finalists = [_finalist(f, name) for f in pareto.get("finalists") or []]
    if not finalists:
        raise Invalid(f"{name}: pareto.json has no finalists")
    # Match the winner on (label, variant): finalists share a topology once knobs are swept.
    win = pareto.get("winner") or {}
    winner = next((r for r in finalists if r["label"] == win.get("label")
                   and r["variant"] == (win.get("variant") or "")), None)
    if winner is None:
        raise Invalid(f"{name}: winner {win.get('label')!r} does not match any finalist")

    floor = scenario.get("interactivity_floor") or {}
    floor_str = ""
    if floor.get("tok_per_s_per_user") is not None:
        floor_str = f"{floor['tok_per_s_per_user']:g} tok/s/user"
        if floor.get("itl_p99_ms_max") is not None:
            floor_str += f" (ITL p99 ≤ {floor['itl_p99_ms_max']:g} ms)"

    # A ledger mixing seeds or stacks is not one comparison; say so on the card.
    split = report_py.provenance_split(runs)
    return run_id, {
        "run_id": run_id,
        "model": str(_req(scenario, "model", name)),
        "hardware": hardware.upper(),
        "gpu_budget": int(_num(_req(scenario, "gpu_budget", name), name, 1, 100000)),
        "workload": _workload_name(scenario),
        "floor": floor_str,
        "headroom": scenario.get("headroom"),
        "n_runs": int(totals.get("runs", 0) or 0),
        "n_servable": int(totals.get("servable", 0) or 0),
        "winner": winner,
        "finalists": finalists,
        "reason": str(win.get("reason", "")),
        "caveat": f"Provenance: {split}." if split else "",
        "records": report_py.build_records(runs, scenario) if runs else [],
    }


def _write_report(run_dir, dest):
    """Render the standalone report into dest/index.html with a link back to the site."""
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "index.html")
    report_py.write_one(run_dir, out=path)
    with open(path, encoding="utf-8") as f:
        html = f.read()
    anchor = '<div class="wrap">'
    html = html.replace(anchor, anchor + BACK_LINK, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    for fn in REPORT_DATA_FILES:
        if os.path.exists(os.path.join(run_dir, fn)):
            shutil.copyfile(os.path.join(run_dir, fn), os.path.join(dest, fn))


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, separators=(",", ":"))


def _within(path, parent):
    return os.path.commonpath([path, parent]) == parent


def build(reports_dir, out_dir):
    """Validate and render every run dir under reports_dir into out_dir; return the rows."""
    reports_dir, out_dir = os.path.realpath(reports_dir), os.path.realpath(out_dir)
    if not os.path.isdir(reports_dir):
        raise SystemExit(f"{reports_dir}: not a directory")
    run_dirs = sorted(os.path.join(reports_dir, d) for d in os.listdir(reports_dir)
                      if report_py.is_run_dir(os.path.join(reports_dir, d)))
    if not run_dirs:
        raise SystemExit(f"{reports_dir}: no run directories (none has pareto.json/runs.jsonl)")

    # out_dir is wiped, so it must not hold the repo, reports/, or be one of the run dirs.
    if (_within(os.path.realpath(ROOT), out_dir) or _within(reports_dir, out_dir)
            or out_dir in run_dirs):
        raise SystemExit(f"{out_dir}: refusing to wipe the repo, reports/, or a run dir")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    rows = []
    for rd in run_dirs:
        run_id, row = validate_and_project(rd)
        _write_report(rd, os.path.join(out_dir, run_id))
        rows.append(row)
    rows.sort(key=lambda r: r["run_id"])
    _dump(os.path.join(out_dir, "scenarios.json"), rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the site data from reports/.")
    ap.add_argument("--reports", default=os.path.join(ROOT, "reports"), help="parent of run dirs")
    ap.add_argument("--out", default=os.path.join(ROOT, "site"), help="output directory")
    ap.add_argument("--siblings", default=os.path.join(ROOT, os.pardir),
                    help="dir with blis-catalog/blis-registry/inference-sim for the catalog tabs")
    args = ap.parse_args(argv)
    try:
        rows = build(args.reports, args.out)
    except Invalid as e:
        raise SystemExit(f"validation error: {e}")
    cat = build_catalog.build(os.path.abspath(args.siblings))
    _dump(os.path.join(args.out, "catalog.json"), cat)
    print(f"built {len(rows)} report(s) -> {os.path.relpath(args.out, ROOT)}/")
    print(f"  catalog: {len(cat['models'])} models, {len(cat['hardware'])} GPUs, "
          f"{len(cat['workloads'])} workloads")


if __name__ == "__main__":
    main()
