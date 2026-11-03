#!/usr/bin/env python3
"""Rewrite topology labels in already-committed reports without re-sweeping.

Labels are baked into report JSON at sweep time (run_experiment.label_of), so a fix to the
labeller only reaches FUTURE reports. This rewrites the committed ones in place: it recomputes
every label from the config the report already carries, so e.g. a dropped pp (#89) reappears
as `4x TP8xPP2 + EP`. No BLIS, no LLM -- only the label strings change, never a measurement.

    python3 -I scripts/relabel_reports.py                 # rewrite every report that changes
    python3 -I scripts/relabel_reports.py --check         # exit 1 if any label is stale (CI)
    python3 -I scripts/relabel_reports.py reports/<id>    # just these

experiment.json rows carry the full config (pp included), so their labels are recomputed
directly. pareto.json finalists and report.json prose do NOT carry pp, so they are remapped by
the old->new pairs those recomputations produce -- verified 1:1 per report, and applied
longest-old-first so `1x TP4` never clobbers inside `1x TP4 + EP`.
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiment as E  # noqa: E402


def _relabel_from_config(cfg):
    """The label this config should carry now. None if it has no topology (not a finalist row)."""
    if not cfg or not (cfg.get("tp") or cfg.get("prefill_instances")):
        return None
    return E.label_of(cfg)


def _old_to_new(experiment):
    """Map each stale label to its corrected one, from rows whose inline config carries pp.

    Raises if a stale label would map to two different corrected labels (it never does in the
    committed set, but a silent pick would corrupt pareto/report where the config lacks pp)."""
    mapping = {}

    def visit(row):
        cfg = row.get("config")
        new = _relabel_from_config(cfg)
        if not new:
            return
        old = row.get("label")
        if old and old != new:
            if mapping.get(old, new) != new:
                raise SystemExit(f"ambiguous relabel: {old!r} -> {mapping[old]!r} and {new!r}")
            mapping[old] = new

    for sw in experiment.get("sweeps", []):
        for r in sw.get("rows", []):
            visit(r)
        if sw.get("best"):
            visit(sw["best"])
    if experiment.get("best_overall"):
        visit(experiment["best_overall"])
    return mapping


def _recompute_struct_labels(obj):
    """Rewrite every `label` sitting beside a `config` (experiment.json rows/best). Count changes."""
    n = 0
    if isinstance(obj, dict):
        new = _relabel_from_config(obj.get("config")) if "label" in obj else None
        if new and obj["label"] != new:
            obj["label"] = new
            n += 1
        for v in obj.values():
            n += _recompute_struct_labels(v)
    elif isinstance(obj, list):
        for v in obj:
            n += _recompute_struct_labels(v)
    return n


def _apply_map_struct(obj, mapping):
    """Rewrite `label` fields by exact old->new lookup (pareto.json finalists/winner)."""
    n = 0
    if isinstance(obj, dict):
        if isinstance(obj.get("label"), str) and obj["label"] in mapping:
            obj["label"] = mapping[obj["label"]]
            n += 1
        for v in obj.values():
            n += _apply_map_struct(v, mapping)
    elif isinstance(obj, list):
        for v in obj:
            n += _apply_map_struct(v, mapping)
    return n


def _apply_map_text(text, mapping):
    """Replace stale labels in prose, longest old-label first so a shorter one is not a substring
    of a longer one it would corrupt (`1x TP4` inside `1x TP4 + EP`)."""
    for old in sorted(mapping, key=len, reverse=True):
        text = text.replace(old, mapping[old])
    return text


def relabel_report(report_dir, check=False):
    """Relabel one report dir in place. Returns the number of label changes (0 if already current)."""
    ej = os.path.join(report_dir, "experiment.json")
    if not os.path.isfile(ej):
        return 0
    experiment = json.load(open(ej))
    mapping = _old_to_new(experiment)
    if not mapping:
        return 0
    if check:
        return len(mapping)

    # experiment.json: recompute from config directly (authoritative, carries pp).
    changed = _recompute_struct_labels(experiment)
    _write_json(ej, experiment)

    # pareto.json: structured label fields, remapped (no pp in its config).
    pj = os.path.join(report_dir, "pareto.json")
    if os.path.isfile(pj):
        pareto = json.load(open(pj))
        if _apply_map_struct(pareto, mapping):
            _write_json(pj, pareto)

    # report.json: labels live inside prose HTML, so substitute in the raw text.
    rj = os.path.join(report_dir, "report.json")
    if os.path.isfile(rj):
        raw = open(rj).read()
        new = _apply_map_text(raw, mapping)
        if new != raw:
            open(rj, "w").write(new)

    return changed


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", help="report dirs (default: reports/*)")
    ap.add_argument("--check", action="store_true",
                    help="report stale labels and exit 1 without writing")
    a = ap.parse_args(argv)
    dirs = a.dirs or sorted(d for d in glob.glob("reports/*") if os.path.isdir(d))

    stale, changed = [], 0
    for d in dirs:
        n = relabel_report(d, check=a.check)
        if n and a.check:
            stale.append((os.path.basename(d), n))
        elif n:
            changed += n
            print(f"{os.path.basename(d)}: {n} label(s) rewritten")

    if a.check:
        for name, n in stale:
            print(f"::warning::{name}: {n} stale topology label(s); run relabel_reports.py")
        return 1 if stale else 0
    print(f"done: {changed} label(s) rewritten" if changed else "all labels already current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
