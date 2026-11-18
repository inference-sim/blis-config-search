#!/usr/bin/env python3
"""The knob table: every setting the pinned simulator reads, and how a spec can vary it (CFG-6).

The rows are generated, never written here (CFG-3): an engine row for each entry of an engine's
field list, and a document row for each property of the scenario and deployment documents (`blis
schema scenario|deployment`). What a spec can do with a row is decided in two places only:

  * an engine row is sweepable when the field list marks it `modelled` and the validator's own
    vocabulary (settings.py) accepts the name; any other status is the reason it is not;
  * a document row is classified in knobs/coverage.yaml, with a sweep key the spec format accepts
    or a reason (and the open issue that would lift it).

A document property in neither is an error, and so is a classification of a property that no
longer exists. A new simulator capability therefore cannot arrive unseen.

  knob_table.py --blis BLIS --fields vllm=FILE [--fields sglang=FILE] [--check | --write]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_cli  # noqa: E402
import settings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COVERAGE = os.path.join(ROOT, "knobs", "coverage.yaml")
TABLE_MD = os.path.join(ROOT, "knobs", "knob-table.md")
SPEC_SCHEMA = os.path.join(ROOT, "experiments", "schema.json")
MAX_DEPTH = 5


def document_paths(schema):
    """Every property path of a document's JSON Schema, an array's element marked `[]`.

    An object property with no properties of its own (a free-form map) is a path; one with
    properties is expanded.
    """
    defs = schema["$defs"]
    out = []

    def ref(node):
        r = node.get("$ref") or (node.get("items") or {}).get("$ref") or ""
        return r.rsplit("/", 1)[-1] or None

    def walk(name, prefix, depth, stack):
        for key, node in sorted((defs.get(name, {}).get("properties") or {}).items()):
            path = prefix + key + ("[]" if node.get("type") == "array" else "")
            target = ref(node)
            if (target in defs and target not in stack and depth < MAX_DEPTH
                    and defs[target].get("properties")):
                walk(target, path + ".", depth + 1, stack + (target,))
            else:
                out.append(path)

    top = schema["$ref"].rsplit("/", 1)[-1]
    walk(top, "", 0, (top,))
    return out


def engine_rows(kind, field_list, deployment_schema):
    """One row per field-list entry: sweepable (a spec name) or the reason it is not."""
    vocab = settings.vocabulary_from(field_list, deployment_schema, renames=settings.renames_for(kind))
    by_entry = {id(e): n for n, e in vocab.status.items()}
    sweep_name = {(e["source"], e["name"]): by_entry.get(id(e)) for e in field_list["fields"]}
    rows = []
    for e in field_list["fields"]:
        name, status = e["name"], e["status"]
        spec = sweep_name.get((e["source"], name))
        row = {"id": f"engine:{kind}:{e['source']}:{name}", "kind": "engine", "engine": kind,
               "setting": name, "source": e["source"], "status": status}
        if spec and status == "modelled" and settings.inert_reason(vocab, spec) is None:
            row["sweep"] = f"engine.{spec}"
        else:
            row["sweep"] = None
            if e["source"] == "env":
                row["reason"] = "a launch environment variable; a spec does not set the environment"
            elif status == "modelled":
                row["reason"] = "modelled, but the spec vocabulary does not name it"
            else:
                row["reason"] = f"the engine's field list marks it {status}"
            issue = (e.get("evidence") or {}).get("issue")
            if issue:
                row["issue"] = issue
        rows.append(row)
    return rows


def load_coverage(path=COVERAGE):
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or not isinstance(doc.get("knobs"), dict):
        raise SystemExit(f"{path}: expected a `knobs:` mapping")
    return doc


def _matches(pattern, path):
    """A coverage key is an exact property path, or a prefix ending in `.*`."""
    if pattern.endswith(".*"):
        return path.startswith(pattern[:-1])
    return pattern == path


def classify(document, path, coverage):
    """(entry, key) for the coverage entry of a `document:path`, preferring an exact key."""
    full = f"{document}:{path}"
    hit = None
    for key, entry in coverage["knobs"].items():
        if key == full:
            return entry, key
        if key.endswith(".*") and _matches(key, full) and (hit is None or len(key) > len(hit[1])):
            hit = (entry, key)
    return hit if hit else (None, None)


def document_rows(scenario_schema, deployment_schema, coverage):
    rows, problems, used = [], [], set()
    for document, schema in (("scenario", scenario_schema), ("deployment", deployment_schema)):
        for path in document_paths(schema):
            entry, key = classify(document, path, coverage)
            row = {"id": f"{document}:{path}", "kind": "document", "document": document, "path": path}
            if entry is None:
                problems.append(f"{document}:{path} is a property of the {document} document that "
                                f"knobs/coverage.yaml neither maps to a sweep key nor gives a reason for")
                row["sweep"], row["reason"] = None, "unclassified"
            else:
                used.add(key)
                row["class"] = key
                sweep, reason = entry.get("sweep"), entry.get("reason")
                if bool(sweep) == bool(reason):
                    problems.append(f"{key}: give exactly one of `sweep` and `reason`")
                row["sweep"] = sweep
                if reason:
                    row["reason"] = reason
                    if entry.get("issue"):
                        row["issue"] = entry["issue"]
            rows.append(row)
    for key in coverage["knobs"]:
        if key not in used:
            problems.append(f"{key} classifies no property of the scenario or deployment document "
                            f"(it was removed or renamed upstream)")
    return rows, problems


def check_sweep_keys(coverage, spec_schema, parallel_keys):
    """Every sweep key a coverage entry names is a key the spec format has (and, for `parallel`, a
    property of the deployment document's Parallelism)."""
    props = spec_schema["$defs"]["sweep"]["properties"]
    scenario_props = spec_schema["$defs"]["scenario"]["properties"]
    problems = []
    for key, entry in coverage["knobs"].items():
        for target in re.split(r",\s*", entry.get("sweep") or ""):
            sm = re.match(r"scenario\.([\w]+)", target)
            if sm:
                if sm.group(1) not in scenario_props:
                    problems.append(f"{key}: the spec format has no scenario key {sm.group(1)!r}")
                continue
            m = re.match(r"sweeps\[\]\.([\w-]+)(?:\.([\w]+))?", target)
            if not m:
                if target:
                    problems.append(f"{key}: sweep key {target!r} is not a sweeps[] path")
                continue
            head, inner = m.groups()
            if head not in props:
                problems.append(f"{key}: the spec format has no sweep key {head!r}")
            elif head == "parallel" and inner and inner not in parallel_keys:
                problems.append(f"{key}: parallel.{inner} is no property of the deployment document's Parallelism")
    return problems


def build(scenario_schema, deployment_schema, field_lists, coverage, spec_schema=None):
    rows, problems = document_rows(scenario_schema, deployment_schema, coverage)
    engines = {}
    for kind, fl in sorted(field_lists.items()):
        engines[kind] = fl["engine"]["version"]
        rows += engine_rows(kind, fl, deployment_schema)
    table = {"kind": "KnobTable", "engines": engines, "knobs": rows}
    if spec_schema is not None:
        parallel = deployment_schema["$defs"]["deployment.Parallelism"]["properties"]
        problems += check_sweep_keys(coverage, spec_schema, parallel)
    problems += check_capabilities(rows, coverage, engines)
    return table, problems


def pending_capabilities(table_engines, coverage):
    """Capabilities that need an engine kind the build does not report yet (until the pin moves)."""
    return [n for n, c in (coverage.get("capabilities") or {}).items()
            if c.get("requires_engine") and c["requires_engine"] not in table_engines]


def check_capabilities(rows, coverage, engines=()):
    """Each named capability resolves to rows that exist and are sweepable or carry an open issue."""
    ids = {r["id"]: r for r in rows}
    problems = []
    for name, cap in (coverage.get("capabilities") or {}).items():
        if cap.get("requires_engine") and cap["requires_engine"] not in engines:
            continue
        for rid in cap.get("rows") or []:
            row = next((ids[i] for i in ids if _matches(rid, i) or i == rid), None)
            if row is None:
                problems.append(f"capability {name}: no knob row {rid}")
            elif not row.get("sweep") and not (row.get("issue") or row.get("reason")):
                problems.append(f"capability {name}: {rid} is neither sweepable nor explained")
        if cap.get("rows") and not cap.get("note"):
            problems.append(f"capability {name}: say in `note` what the sweep varies")
    return problems


def render_md(table, coverage):
    rows = table["knobs"]
    out = ["# The knob table", "",
           "Generated by `scripts/knob_table.py` from the pinned simulator (CFG-3, CFG-6); do not edit.",
           "Engines: " + ", ".join(f"{k} {v}" for k, v in table["engines"].items()) + ".", ""]
    out += ["## Document knobs", "", "One row for each coverage entry; `knob_table.py --json FILE` writes every property.", "",
            "| coverage entry | properties | sweep key | reason / issue |", "|---|---|---|---|"]
    groups = {}
    for r in rows:
        if r["kind"] == "document":
            groups.setdefault(r.get("class", r["id"]), []).append(r)
    for key, rs in sorted(groups.items()):
        r = rs[0]
        why = ""
        if r.get("reason"):
            why = r["reason"] + (f" ({r['issue']})" if r.get("issue") else "")
        out.append(f"| `{key}` | {len(rs)} | {('`' + r['sweep'] + '`') if r.get('sweep') else ''} | {why} |")
    for kind in table["engines"]:
        mine = [r for r in rows if r["kind"] == "engine" and r["engine"] == kind]
        out += ["", f"## Engine settings: {kind} {table['engines'][kind]}", ""]
        counts = {}
        for r in mine:
            key = (r["source"], r["status"], bool(r.get("sweep")))
            counts[key] = counts.get(key, 0) + 1
        out += ["| source | status | sweepable | settings |", "|---|---|---|---|"]
        for (src, status, sw), n in sorted(counts.items()):
            out.append(f"| {src} | {status} | {'yes' if sw else 'no'} | {n} |")
        sweepable = sorted(r["sweep"] for r in mine if r.get("sweep"))
        out += ["", "Sweepable: " + (", ".join(f"`{s}`" for s in sweepable) or "none") + ".",
                "", "The other settings, with their status and issue, are in the full table (`knob_table.py --json FILE`)."]
    pending = pending_capabilities(table["engines"], coverage)
    caps = coverage.get("capabilities") or {}
    if caps:
        out += ["", "## M3.1 capabilities", "", "| capability | what a sweep varies | knobs |", "|---|---|---|"]
        for name, cap in caps.items():
            wait = " Pending the pinned build." if name in pending else ""
            out.append(f"| {name} | {cap.get('note', '')}{wait} | " + ", ".join(f"`{x}`" for x in cap.get("rows", [])) + " |")
    return "\n".join(out) + "\n"


def render_json(table):
    """One knob per line: a diff of the table reads as the knobs that moved."""
    rows = ",\n".join("  " + json.dumps(r, sort_keys=True) for r in table["knobs"])
    return ('{"engines": ' + json.dumps(table["engines"], sort_keys=True) + ', "kind": "KnobTable",\n "knobs": [\n'
            + rows + "\n ]}\n")


def parse_fields_args(items):
    out = {}
    for item in items or []:
        kind, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--fields wants kind=FILE, got {item!r}")
        out[kind] = path
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blis", default=os.environ.get("BLIS"))
    ap.add_argument("--fields", action="append", help="kind=FILE, once per engine kind")
    ap.add_argument("--coverage", default=COVERAGE)
    ap.add_argument("--check", action="store_true", help="fail if the committed table differs")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--json", help="also write the full table, every setting and its reason, to this file")
    args = ap.parse_args(argv)
    paths = parse_fields_args(args.fields)
    if not paths:
        raise SystemExit("give at least one --fields kind=FILE")
    scenario = blis_cli.schema(args.blis, "scenario")
    deployment = blis_cli.schema(args.blis, "deployment")
    lists = {kind: blis_cli.fields(args.blis, kind, path) for kind, path in paths.items()}
    coverage = load_coverage(args.coverage)
    with open(SPEC_SCHEMA, encoding="utf-8") as f:
        spec_schema = json.load(f)
    table, problems = build(scenario, deployment, lists, coverage, spec_schema)
    for p in problems:
        print(f"knob table: {p}", file=sys.stderr)
    if problems:
        return 1
    md = render_md(table, coverage)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(render_json(table))
    if args.write:
        with open(TABLE_MD, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {TABLE_MD}: {len(table['knobs'])} knobs")
        return 0
    if args.check:
        have = open(TABLE_MD, encoding="utf-8").read() if os.path.exists(TABLE_MD) else None
        if have != md:
            print(f"knob table: {os.path.relpath(TABLE_MD, ROOT)} differs from the generated one; run "
                  f"`scripts/knob_table.py --write`", file=sys.stderr)
            return 1
        print("knob table: matches")
        return 0
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
