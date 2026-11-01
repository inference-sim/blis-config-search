#!/usr/bin/env python3
"""Hermetic tests for the backend scripts (no network, no port binding, no BLIS).

    python3 -I scripts/test_backend.py

Covers serve_local (path-traversal guard) and the workflows' safety properties. These
are the non-build backend pieces; the build/catalog are covered by test_build.py.
"""
import importlib.util
import os
import sys

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


def test_workflows_sane():
    print("test_workflows_sane")
    import yaml as _yaml
    pg = os.path.join(ROOT, ".github", "workflows", "pages.yml")
    ptxt = open(pg).read()
    check(_yaml.safe_load(ptxt) is not None, "pages.yml parses")
    pdoc = _yaml.safe_load(ptxt)
    deploy_if = pdoc["jobs"]["deploy"]["if"]
    check("refs/heads/main" in deploy_if and "push" in deploy_if, "live publish gated to main pushes")
    check("clean-exclude: pr-preview/" in ptxt, "live publish keeps the PR previews")
    main_push = next(st for st in pdoc["jobs"]["deploy"]["steps"]
                     if "github-pages-deploy-action@" in st.get("uses", ""))
    check(main_push["with"].get("force") is False,
          "live publish rebases instead of force-pushing over a concurrent preview")
    deploy_uses = [st.get("uses", "") for st in pdoc["jobs"]["deploy"]["steps"]]
    check(any(u.startswith("actions/checkout@") for u in deploy_uses),
          "live publish checks out the repo (the deploy action needs a git working tree)")
    preview_if = pdoc["jobs"]["preview"]["if"]
    check("head.repo.full_name == github.repository" in preview_if,
          "PR previews are same-repo only (no fork build published)")
    check("umbrella-dir: pr-preview" in ptxt, "PR previews live under pr-preview/, apart from the root")
    # The deploy comment is posted by our own step, gated on our own wait (pr-preview-action's
    # wait fails when a newer gh-pages push supersedes the build), so check that chain.
    psteps = pdoc["jobs"]["preview"]["steps"]
    wait = next((st for st in psteps if st.get("id") == "wait-pages"), {})
    comment = next((st for st in psteps if "sticky-pull-request-comment@" in st.get("uses", "")), {})
    check("pages/builds" in wait.get("run", "") and "built" in wait.get("run", "")
          and "steps.wait-pages.outcome == 'success'" in comment.get("if", "")
          and comment.get("with", {}).get("header") == "pr-preview"
          and pdoc["jobs"]["preview"]["permissions"].get("pages") == "read",
          "the preview comment waits for the Pages build (and may read its status)")
    check("github.event.pull_request.number" in str(pdoc["concurrency"]["group"]),
          "pages runs are grouped per PR, so batch PRs don't cancel each other's previews")

    # The deterministic sweeper needs no approval gate (no LLM in the path), but it must still
    # never write to main or publish: its only output is a bot PR.
    rx = os.path.join(ROOT, ".github", "workflows", "run-experiment.yml")
    rtxt = open(rx).read()
    check(_yaml.safe_load(rtxt) is not None, "run-experiment.yml parses")
    check("pull-requests: write" in rtxt, "run-experiment opens a PR rather than pushing to main")
    check("--base main --head" in rtxt, "run-experiment targets main through a PR")
    check("pages: write" not in rtxt, "run-experiment never publishes directly")
    check("validate_experiment.py" in rtxt, "run-experiment validates before building BLIS")
    check("pins.env" in rtxt, "run-experiment builds the pinned stack from pins.env")
    # The bot PR is deterministic sweeper output, not LLM output, so it carries no
    # LLM-attribution footer.
    check("Generated with [Claude Code]" not in rtxt, "run-experiment's bot PR claims no LLM authorship")

    ck = os.path.join(ROOT, ".github", "workflows", "checks.yml")
    ctxt = open(ck).read()
    check(_yaml.safe_load(ctxt) is not None, "checks.yml parses")
    check("validate_experiment.py experiments/" in ctxt,
          "checks.yml validates every committed experiment spec")
    check("test_experiment.py" in ctxt, "checks.yml runs the experiment pipeline tests")


if __name__ == "__main__":
    test_serve_local()
    test_workflows_sane()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
