#!/usr/bin/env python3
"""Hermetic tests for scripts/ci_scope.py: which CI jobs a change list selects."""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import ci_scope as S  # noqa: E402

FAIL = 0


def check(ok, msg):
    global FAIL
    if not ok:
        FAIL += 1
        print("  FAIL:", msg)


IDS = S.report_ids()
ONE = IDS[0]


def jobs(files, moved=False):
    r = S.scope(files, moved)
    return {j for j in S.JOBS if r[j]}, r


def test_unknown_change_list_runs_everything():
    print("test_unknown_change_list_runs_everything")
    on, r = jobs(None)
    check(on == set(S.JOBS) and r["smoke_reports"] == IDS, "no change list must select every job and report")
    on, r = jobs(["some/new/top-level/file.bin"])
    check(on == set(S.JOBS) and r["smoke_reports"] == IDS, "a path no rule knows must select everything")


def test_docs_only_runs_nothing():
    print("test_docs_only_runs_nothing")
    on, r = jobs(["docs/objective-refusals-router-files.md", "CLAUDE.md"])
    check(not on and not r["smoke"], "docs-only change selects no job")


def test_web_only():
    print("test_web_only")
    on, r = jobs(["web/src/App.tsx"])
    check(on == {"web", "pages"} and not r["smoke"], f"web/ selects web and pages only, got {on}")


def test_spec_selects_its_own_smoke():
    print("test_spec_selects_its_own_smoke")
    on, r = jobs([f"experiments/{ONE}.yaml"])
    check(r["smoke_reports"] == [ONE], "a spec selects only its own report's smoke")
    check({"python", "experiments"} <= on, "a spec selects the python and experiments jobs")
    on, r = jobs([f"reports/{ONE}/ledger.jsonl"])
    check(r["smoke_reports"] == [ONE] and "experiments" not in on, "a report's data selects its smoke, not spec validation")


def test_pin_move_and_runner_scripts_select_all_smoke():
    print("test_pin_move_and_runner_scripts_select_all_smoke")
    on, r = jobs(["pins.env"], moved=True)
    check(r["smoke_reports"] == IDS, "a moved pin selects every report")
    on, r = jobs(["pins.env"], moved=False)
    check(not r["smoke"] and "experiments" in on, "a comment-only pins.env edit selects no smoke")
    for f in ("scripts/blis_pin.py", "scripts/run_blis.py", "scripts/settings.py", "scripts/fetch_fields.py"):
        check(S.scope([f])["smoke_reports"] == IDS, f"{f} is read by the smoke and selects every report")
    check(not S.scope(["scripts/test_build.py"])["smoke"], "a test file is not read by the smoke")
    check(not S.scope(["scripts/build_gallery.py"])["smoke"], "the site builder is not read by the smoke")


def test_closure_follows_imports():
    print("test_closure_follows_imports")
    c = S.closure(S.CLOSURE_ROOTS["smoke"])
    for f in ("scripts/blis_pin.py", "scripts/run_blis.py", "scripts/settings.py", "scripts/blis_cli.py",
              "scripts/fetch_corpus.py"):
        check(f in c, f"{f} missing from the smoke closure")
    check("scripts/build_gallery.py" not in c, "build_gallery.py is not imported by the smoke")


def test_shared_data_selects_only_its_users():
    print("test_shared_data_selects_only_its_users")
    for fn in sorted(os.listdir(os.path.join(ROOT, "routing"))):
        if not fn.endswith(".yaml"):
            continue
        users = [i for i in IDS if os.path.splitext(fn)[0] in S._spec_text(i, ROOT)]
        check(S.scope(["routing/" + fn])["smoke_reports"] == users, f"routing/{fn} selects exactly the specs naming it")


def test_every_test_file_is_run_by_the_python_job():
    print("test_every_test_file_is_run_by_the_python_job")
    with open(os.path.join(ROOT, ".github", "workflows", "checks.yml")) as f:
        wf = f.read()
    tests = sorted(n for n in os.listdir(os.path.join(ROOT, "scripts")) if re.fullmatch(r"test_.*\.py", n))
    for t in tests:
        check("scripts/" + t in wf, f"{t} is run by no checks.yml step")


if __name__ == "__main__":
    test_unknown_change_list_runs_everything()
    test_docs_only_runs_nothing()
    test_web_only()
    test_spec_selects_its_own_smoke()
    test_pin_move_and_runner_scripts_select_all_smoke()
    test_closure_follows_imports()
    test_shared_data_selects_only_its_users()
    test_every_test_file_is_run_by_the_python_job()
    print("failed" if FAIL else "ok")
    sys.exit(1 if FAIL else 0)
