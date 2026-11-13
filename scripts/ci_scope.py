#!/usr/bin/env python3
"""Decide which CI jobs a change needs, from the paths it touches.

A required check must always REPORT, so the workflows run on every pull request and each job asks
this script whether its inputs changed; a job whose inputs did not change finishes in seconds
instead of repeating work whose result cannot differ.

  python3 -I scripts/ci_scope.py --base <sha-or-ref> [--pins-moved true|false] [--reports] [--github-output]
  python3 -I scripts/ci_scope.py --files a b c      # explicit change list (the harness uses this)

Without a change list (a push, a manual dispatch with no base, a base that cannot be diffed) every
job runs: the script never guesses that nothing changed.

What each job reads (the single place this is written down):

  python       the hermetic suite (scripts/test_*.py): every script, the pins, the specs and
               reports and routing/workload files the tests load, the README and SETUP they check.
  experiments  validates and dry-runs every spec with the pinned simulator: the specs, routing/ and
               workloads/, pins.env (which simulator), the scripts the validator and runner import.
  web          typecheck, test and build of web/, which stages reports/ through build_gallery.py.
  pages        the same build, published as the PR preview.
  smoke        re-runs a few rows of one report on the pinned simulator. It reads that report's own
               spec and ledger (experiments/<id>.yaml, reports/<id>/), the routing/ and workloads/
               files that spec names, the pin VALUES, and the scripts blis_pin.py imports. A data
               file therefore selects only the reports that use it; only a pin move or a changed
               runner script selects every report.

The scripts each job reads are found by following `import` statements, so a new import cannot go
unlisted. A path this table does not know selects every job.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
JOBS = ("python", "experiments", "web", "pages")

# Roots whose import closure (plus the root itself) is what a job reads from scripts/.
# fetch_fields.py and clone_pinned.sh are run as commands, not imported, so they are named.
CLOSURE_ROOTS = {
    "experiments": ("validate_experiment.py", "run_experiment.py", "fetch_fields.py", "knob_profile.py"),
    "web": ("build_gallery.py",),
    "smoke": ("blis_pin.py", "fetch_fields.py", "clone_pinned.sh"),
}

# Paths no job reads. Everything else must be claimed by a rule below or it selects every job.
INERT = ("docs/", ".claude/", "LICENSE", "CLAUDE.md", "AGENTS.md", ".gitignore")

PYTHON_DIRS = ("scripts/", "experiments/", "routing/", "workloads/", "reports/", "report/", "examples/",
               "deploy/")
PYTHON_FILES = ("pins.env", "requirements.in", "requirements.txt", "README.md", "SETUP.md")
SHARED_CI = (".github/actions/",)


def _imports(path):
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names.add(n.module.split(".")[0])
    return names


def closure(roots, root=ROOT):
    """The scripts/ files `roots` read: themselves and every scripts/ module they import, transitively."""
    sdir = os.path.join(root, "scripts")
    seen, todo = set(), list(roots)
    while todo:
        fn = todo.pop()
        if fn in seen:
            continue
        seen.add(fn)
        if fn.endswith(".py"):
            for mod in _imports(os.path.join(sdir, fn)):
                if os.path.isfile(os.path.join(sdir, mod + ".py")):
                    todo.append(mod + ".py")
    return {"scripts/" + f for f in seen}


def report_ids(root=ROOT):
    d = os.path.join(root, "reports")
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, "scenario.json")))


def _spec_text(rid, root):
    try:
        with open(os.path.join(root, "experiments", rid + ".yaml"), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def scope(files, pins_moved=False, root=ROOT, workflow_file=None):
    """{job: bool, 'smoke_reports': [ids], 'smoke_all': bool} for a change list. files=None means unknown."""
    ids = report_ids(root)
    if files is None:
        return _result({j: True for j in JOBS}, ids, True)
    files = [f.strip() for f in files if f.strip()]
    clo = {k: closure(v, root) for k, v in CLOSURE_ROOTS.items()}
    on = {j: False for j in JOBS}
    smoke_ids, smoke_all = set(), bool(pins_moved)
    everything = False
    own = {"python": ".github/workflows/checks.yml", "experiments": ".github/workflows/checks.yml",
           "web": ".github/workflows/checks.yml", "pages": ".github/workflows/pages.yml"}
    for f in files:
        if f.startswith(INERT):
            continue
        if f in own.values() or f.startswith(SHARED_CI):
            for j, w in own.items():
                on[j] = on[j] or f == w or f.startswith(SHARED_CI)
            if f.startswith(SHARED_CI) or f == ".github/workflows/pin-compat.yml":
                smoke_all = True
            continue
        if f == ".github/workflows/pin-compat.yml":
            smoke_all = True
            continue
        if f.startswith(".github/workflows/"):
            continue          # the other workflows are not read by these jobs
        known = False
        if f.startswith(PYTHON_DIRS) or f in PYTHON_FILES:
            on["python"], known = True, True
        if f.startswith(("experiments/", "routing/", "workloads/")) or f == "pins.env" or f in clo["experiments"]:
            on["experiments"], known = True, True
        if f.startswith(("web/", "reports/", "report/", "experiments/", "routing/", "workloads/")) \
                or f in ("pins.env", "requirements.txt") or f in clo["web"]:
            on["web"], on["pages"], known = True, True, True
        if f in ("requirements.txt", "requirements.in"):
            on["experiments"] = True
        # smoke
        if f in clo["smoke"] or f in ("requirements.txt",):
            smoke_all, known = True, True
        m = re.match(r"(?:experiments/(.+)\.yaml|reports/([^/]+)/.*)$", f)
        if m:
            rid = m.group(1) or m.group(2)
            if rid in ids:
                smoke_ids.add(rid)
            known = True
        elif f.startswith(("routing/", "workloads/")):
            s = _stem(f)
            smoke_ids.update(r for r in ids if s in _spec_text(r, root))
            known = True
        if f.startswith(("scripts/test_", "scripts/testdata/")):
            known = True      # tests are read by the python job only
        if not known:
            everything = True
    if everything:
        return _result({j: True for j in JOBS}, ids, True)
    return _result(on, ids, smoke_all, smoke_ids)


def _result(on, ids, smoke_all, smoke_ids=()):
    sel = list(ids) if smoke_all else [i for i in ids if i in set(smoke_ids)]
    out = dict(on)
    out["smoke_all"] = bool(smoke_all)
    out["smoke_reports"] = sel
    out["smoke"] = bool(sel)
    return out


def changed_files(base, root=ROOT):
    if not base:
        return None
    for ref in (base, "origin/" + base):
        r = subprocess.run(["git", "-C", root, "diff", "--name-only", ref, "HEAD"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.splitlines()
    return None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="", help="diff HEAD against this commit or branch")
    ap.add_argument("--files", nargs="*", default=None, help="explicit change list instead of a diff")
    ap.add_argument("--pins-moved", default="false", choices=("true", "false"))
    ap.add_argument("--github-output", action="store_true")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)
    files = a.files if a.files is not None else changed_files(a.base, a.root)
    res = scope(files, a.pins_moved == "true", a.root)
    res["changed"] = None if files is None else len(files)
    print(json.dumps(res, sort_keys=True))
    if a.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            for k in JOBS + ("smoke", "smoke_all"):
                f.write(f"{k}={str(res[k]).lower()}\n")
            f.write("smoke_reports=" + json.dumps(res["smoke_reports"]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
