#!/usr/bin/env python3
"""Hermetic tests for the backend scripts (no network, no port binding, no BLIS).

    python3 -I scripts/test_backend.py

Covers serve_local (scenario validation, yaml shaping, path-traversal guard, submission
naming) and run_submission (validation, unsafe-path refusal, no-runner loud failure). These
are the non-build backend pieces; the build/catalog are covered by test_build.py.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = [name]  # argparse-at-import guard
    spec.loader.exec_module(m)
    return m


def test_serve_local():
    print("test_serve_local")
    m = _load("serve_local", os.path.join(ROOT, "scripts", "serve_local.py"))

    # valid scenario
    s, e = m.validate_scenario({"model": "qwen3-14b", "hardware": "h100",
                                "workload": "chatbot", "gpu_budget": 8})
    check(e == [], "valid scenario has no errors")
    check(s["workload"]["preset"] == "chatbot", "workload nested as preset")
    check(s["objective"] == "dual-pareto", "objective defaulted")

    # every invalid field caught. The MODEL check is conditional by design: validate_scenario
    # only rejects an unknown model when a staged catalog exists (`if models and ...` in
    # serve_local), so it degrades gracefully where web/dist/data/catalog.json was never built —
    # which is exactly CI, and why asserting a flat 4 errors failed there while passing locally.
    s2, e2 = m.validate_scenario({"model": "nope", "hardware": "x",
                                  "workload": "agentic", "gpu_budget": 0})
    has_catalog = bool(m._catalog_models())
    want = 4 if has_catalog else 3
    check(len(e2) == want,
          f"all {want} invalid fields reported (got {len(e2)}; "
          f"catalog {'staged' if has_catalog else 'absent'}): {e2}")
    for field in ("hardware", "workload", "gpu_budget"):
        check(any(field in err for err in e2), f"{field} error reported")
    if has_catalog:
        check(any("unknown model" in err for err in e2), "unknown model reported")

    # yaml shaping round-trips to the same scenario
    import yaml as _yaml
    y = m.scenario_yaml(s)
    check(_yaml.safe_load(y)["model"] == "qwen3-14b", "scenario_yaml parses back")

    # catalog models load from the built catalog.js (if present)
    models = m._catalog_models()
    check(isinstance(models, set), "catalog models is a set")
    if models:
        check("qwen3-14b" in models, "known model present in catalog set")

    # path-traversal guard on the static handler. Mirror the handler's real logic:
    # strip querystring, lstrip leading "/", realpath-join under site_root, then require
    # containment. A genuine `..` climb must land OUTSIDE and be rejected; an absolute path
    # or an encoded-but-not-decoded name must be CONTAINED (treated as a nonexistent file
    # under the root), which is equally safe.
    site_root = os.path.realpath(m.SITE)
    def contained(path):
        rel = path.split("?", 1)[0].lstrip("/") or "index.html"
        full = os.path.realpath(os.path.join(site_root, rel))
        return full == site_root or full.startswith(site_root + os.sep)
    check(not contained("/../../etc/passwd"), "real ../ climb escapes root -> rejected")
    check(not contained("/../" + os.path.basename(site_root) + "-sibling/x"), "sibling climb rejected")
    check(contained("/etc/passwd"), "absolute path is re-rooted under site (safe, not an escape)")
    check(contained("/%2e%2e/secret"), "undecoded %2e stays a literal name under root (safe)")
    check(contained("/reports/x.html"), "normal path stays contained")


def test_run_submission():
    print("test_run_submission")
    path = os.path.join(ROOT, "scripts", "run_submission.py")
    m = _load("run_submission", path)

    errs, parsed = m.validate({"model": "qwen3-14b", "hardware": "h100",
                               "workload": {"preset": "chatbot"}, "gpu_budget": 8})
    check(errs == [], "valid submission validates")
    check(parsed == ("qwen3-14b", "h100", "chatbot", 8), "fields parsed")

    errs2, _ = m.validate({"model": "", "hardware": "zzz",
                           "workload": {"preset": "agentic"}, "gpu_budget": 99999999})
    check(len(errs2) >= 3, "invalid submission reports multiple errors")

    # a bare-number interactivity_floor (what the old Submit form emitted) must be rejected here
    # with a clear message, not crash the pipeline with AttributeError downstream.
    errs3, _ = m.validate({"model": "qwen3-14b", "hardware": "h100",
                           "workload": {"preset": "chatbot"}, "gpu_budget": 8,
                           "interactivity_floor": 50})
    check(any("mapping" in e for e in errs3), "bare-number interactivity_floor is rejected")
    # the nested shape still validates cleanly
    errs4, _ = m.validate({"model": "qwen3-14b", "hardware": "h100",
                           "workload": {"preset": "chatbot"}, "gpu_budget": 8,
                           "interactivity_floor": {"tok_per_s_per_user": 50}})
    check(errs4 == [], "nested interactivity_floor validates")

    # end-to-end CLI behavior: no runner -> loud non-zero exit, no fabricated report
    tmp = tempfile.mkdtemp()
    sub = os.path.join(ROOT, "submissions", "_test_backend_tmp.yaml")
    os.makedirs(os.path.dirname(sub), exist_ok=True)
    with open(sub, "w") as f:
        f.write("model: qwen3-14b\nhardware: h100\ngpu_budget: 8\nworkload:\n  preset: chatbot\n")
    try:
        r = subprocess.run([sys.executable, "-I", path, "--submission",
                            os.path.relpath(sub, ROOT), "--out", tmp],
                           cwd=ROOT, capture_output=True, text=True)
        check(r.returncode != 0, "no-runner run exits non-zero (loud failure)")
        check("runner" in (r.stdout + r.stderr).lower(), "no-runner message mentions runner")
        check(not os.listdir(tmp), "no report fabricated when runner absent")

        # unsafe submission path refused
        r2 = subprocess.run([sys.executable, "-I", path, "--submission", "../etc/passwd"],
                           cwd=ROOT, capture_output=True, text=True)
        check(r2.returncode != 0, "unsafe submission path refused")
    finally:
        os.remove(sub)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_workflows_sane():
    print("test_workflows_sane")
    import yaml as _yaml
    gr = os.path.join(ROOT, ".github", "workflows", "generate-report.yml")
    txt = open(gr).read()
    doc = _yaml.safe_load(txt)
    check(doc is not None, "generate-report.yml parses")
    # the gate: it must refuse committing to a protected branch and be manually dispatched
    check("workflow_dispatch" in txt, "generate-report is manually dispatched (not auto on PR)")
    check("report-generation" in txt, "uses the protected environment")
    check("pr_branch must be" in txt or "not '$branch'" in txt, "refuses main as pr_branch")

    pg = os.path.join(ROOT, ".github", "workflows", "pages.yml")
    ptxt = open(pg).read()
    check(_yaml.safe_load(ptxt) is not None, "pages.yml parses")
    check("pages: write" in ptxt and "refs/heads/main" in ptxt, "pages publish gated to main")


if __name__ == "__main__":
    test_serve_local()
    test_run_submission()
    test_workflows_sane()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
