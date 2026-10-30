#!/usr/bin/env python3
"""Hermetic tests for the gallery/catalog build (no BLIS, no network).

    python3 -I scripts/test_build.py

Covers: run-dir validation (valid / malicious / out-of-range), the gallery row projection,
build determinism, --no-mock filtering, and the catalog ingester shape + graceful degradation
when a sibling repo is absent.
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_gallery as bg
import build_catalog as bc

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def _mk_run_dir(d, run_id, synthetic=True, workload="chatbot", hardware="h100"):
    os.makedirs(d, exist_ok=True)
    scenario = {"run_id": run_id, "model": "qwen3-14b", "hardware": hardware,
                "gpu_budget": 8, "workload": {"preset": workload},
                "interactivity_floor": {"tok_per_s_per_user": 50, "itl_p99_ms_max": 20}, "headroom": 0.8}
    pareto = {"finalists": [{"label": "4x TP2", "config": {"tp": 2, "num_instances": 4},
              "gpus": 8, "c_use": 24, "at_c_use": {"tokens_per_sec": 1842.0,
              "tokens_per_sec_per_gpu": 230.3, "tok_per_s_per_user": 68.0, "itl_p99_ms": 14.7}}],
              "winner": {"chosen": "4x TP2", "reason": "test"}}
    json.dump(scenario, open(os.path.join(d, "scenario.json"), "w"))
    json.dump(pareto, open(os.path.join(d, "pareto.json"), "w"))
    json.dump({"synthetic": synthetic, "isl_osl": "ShareGPT", "generated": "2026-10-20",
               "n_runs": 10, "n_servable": 7}, open(os.path.join(d, "gallery.json"), "w"))
    open(os.path.join(d, "runs.jsonl"), "w").close()
    if synthetic:
        open(os.path.join(d, "MOCK"), "w").close()


def test_winner_with_a_knob_variant():
    print("test_winner_with_a_knob_variant")
    # pareto.json names the winner with its knob delta ("64x TP4xDP2 + EP (least-loaded
    # routing)") because several finalists share a topology once the search refines knobs. The
    # gallery matched `label == chosen` and, finding no match, silently fell back to the LAST
    # finalist — showing the wrong winner on the site.
    tmp = tempfile.mkdtemp()
    try:
        rd = os.path.join(tmp, "qwen3-14b-h100-chatbot-0002")
        _mk_run_dir(rd, "qwen3-14b-h100-chatbot-0002")
        at = {"tokens_per_sec": 2500.0, "tokens_per_sec_per_gpu": 312.5,
              "tok_per_s_per_user": 80.0, "itl_p99_ms": 12.0}
        pareto = {"finalists": [
            {"label": "4x TP2", "variant": "least-loaded routing",
             "config": {"tp": 2, "num_instances": 4}, "gpus": 8, "c_use": 32, "at_c_use": at},
            {"label": "4x TP2", "variant": "", "config": {"tp": 2, "num_instances": 4},
             "gpus": 8, "c_use": 24, "at_c_use": {"tokens_per_sec": 1842.0,
             "tokens_per_sec_per_gpu": 230.3, "tok_per_s_per_user": 68.0, "itl_p99_ms": 14.7}}],
            "winner": {"chosen": "4x TP2 (least-loaded routing)", "label": "4x TP2",
                       "variant": "least-loaded routing", "reason": "test"}}
        json.dump(pareto, open(os.path.join(rd, "pareto.json"), "w"))
        _, row = bg.validate_and_project(rd)
        check(row["winner"]["tps"] == 2500.0,
              f"the knob-variant winner is matched, not the last finalist "
              f"(got tps={row['winner']['tps']})")
        check(row["winner"].get("variant") == "least-loaded routing",
              f"the winner row carries its knob delta (got {row['winner'].get('variant')!r})")
        # every finalist row must carry its variant, or the site cannot tell them apart either
        check(all("variant" in f for f in row["finalists"]),
              "every finalist row carries a variant field")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_winner_mismatch_raises_not_falls_back():
    print("test_winner_mismatch_raises_not_falls_back")
    # When pareto.json names a winner that matches NO finalist, the old code silently showed the
    # worst-ranked finalist (finalists[-1]) as the recommendation. A named-but-unmatched winner
    # means the report is inconsistent and must raise, not pick the last row.
    tmp = tempfile.mkdtemp()
    try:
        rd = os.path.join(tmp, "qwen3-14b-h100-chatbot-0003")
        _mk_run_dir(rd, "qwen3-14b-h100-chatbot-0003")
        pareto = {"finalists": [{"label": "4x TP2", "config": {"tp": 2, "num_instances": 4},
                  "gpus": 8, "c_use": 24, "at_c_use": {"tokens_per_sec": 1842.0,
                  "tokens_per_sec_per_gpu": 230.3, "tok_per_s_per_user": 68.0, "itl_p99_ms": 14.7}}],
                  "winner": {"chosen": "8x TP1 (does not exist)", "reason": "test"}}
        json.dump(pareto, open(os.path.join(rd, "pareto.json"), "w"))
        try:
            bg.validate_and_project(rd)
            check(False, "a named winner matching no finalist should raise")
        except bg.Invalid:
            check(True, "winner that matches no finalist raises Invalid")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_back_link_resolves_on_pages():
    print("test_report_back_link_resolves_on_pages")
    # The report is served at data/reports/<run_id>/index.html; the gallery shell is the SPA root.
    # The back-link must climb three levels (../../../) or it 404s on Pages.
    tmp = tempfile.mkdtemp()
    try:
        rd = os.path.join(tmp, "qwen3-14b-h100-chatbot-0004")
        _mk_run_dir(rd, "qwen3-14b-h100-chatbot-0004", synthetic=False)
        out = os.path.join(tmp, "site")
        bg.build(os.path.dirname(rd), out, include_mock=True)
        html = open(os.path.join(out, "qwen3-14b-h100-chatbot-0004", "index.html")).read()
        check("../../../index.html#/reports" in html,
              "back-link climbs data/reports/<id> to the SPA root")
        check("\"../index.html#/reports\"" not in html,
              "the old one-level back-link is gone")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validation():
    print("test_validation")
    tmp = tempfile.mkdtemp()
    try:
        rd = os.path.join(tmp, "qwen3-14b-h100-chatbot-0001")
        _mk_run_dir(rd, "qwen3-14b-h100-chatbot-0001")
        run_id, row = bg.validate_and_project(rd)
        check(run_id == "qwen3-14b-h100-chatbot-0001", "valid run_id projected")
        check(row["synthetic"] is True, "MOCK marker -> synthetic")
        check(row["winner"]["tps"] == 1842.0, "winner tps projected")
        check(row["hardware"] == "H100", "hardware uppercased")

        # bad run_id (directory name)
        bad = os.path.join(tmp, "Bad_Name!!")
        _mk_run_dir(bad, "x")
        try:
            bg.validate_and_project(bad)
            check(False, "bad dir name should raise")
        except bg.Invalid:
            check(True, "bad dir name raises Invalid")

        # Hardware is validated by FORM, not against an allowlist. An allowlist drifted from the
        # catalog in both directions and, when it was made to READ the catalog, turned this pure
        # offline transform into something that depended on a sibling clone -- so it passed
        # locally and failed in CI, where no ../blis-catalog exists. BLIS already refuses a chip
        # it cannot price (NS-6); what this build needs is that the value is a safe token.
        for hw in ("b200", "b300", "l40s", "gb200-nvl72", "a100-80"):
            ok_dir = os.path.join(tmp, f"qwen3-14b-{hw}-chatbot-0003")
            _mk_run_dir(ok_dir, f"qwen3-14b-{hw}-chatbot-0003", hardware=hw)
            try:
                _, row = bg.validate_and_project(ok_dir)
                check(row["hardware"] == hw.upper(), f"real catalog part {hw!r} is accepted")
            except bg.Invalid as e:
                check(False, f"{hw!r} should be accepted, got {e}")

        # ... while an unsafe value still raises, since hardware reaches file names and the page.
        for hw in ("../etc", "Z Z", "a" * 64, "9bad", ""):
            badhw = os.path.join(tmp, "qwen3-14b-bad-chatbot-0004")
            _mk_run_dir(badhw, "qwen3-14b-bad-chatbot-0004", hardware=hw)
            try:
                bg.validate_and_project(badhw)
                check(False, f"unsafe hardware {hw!r} should raise")
            except bg.Invalid:
                check(True, f"unsafe hardware {hw!r} raises Invalid")
    finally:
        shutil.rmtree(tmp)


def test_build_determinism_and_nomock():
    print("test_build_determinism_and_nomock")
    tmp = tempfile.mkdtemp()
    try:
        reports = os.path.join(tmp, "reports")
        _mk_run_dir(os.path.join(reports, "qwen3-14b-h100-chatbot-0001"), "qwen3-14b-h100-chatbot-0001", synthetic=False)
        _mk_run_dir(os.path.join(reports, "qwen3-14b-h200-summarization-0002"), "qwen3-14b-h200-summarization-0002",
                    synthetic=True, workload="summarization", hardware="h200")
        o1 = os.path.join(tmp, "s1")
        o2 = os.path.join(tmp, "s2")
        bg.build(reports, o1, include_mock=True)
        bg.build(reports, o2, include_mock=True)
        d1 = open(os.path.join(o1, "data.js")).read()
        d2 = open(os.path.join(o2, "data.js")).read()
        check(d1 == d2, "data.js byte-identical across builds (determinism)")

        oprod = os.path.join(tmp, "prod")
        _, scen = bg.build(reports, oprod, include_mock=False)
        check(len(scen) == 1 and all(not s["synthetic"] for s in scen), "--no-mock keeps only real rows")

        # records.json emitted per scenario
        check(os.path.isfile(os.path.join(o1, "qwen3-14b-h100-chatbot-0001", "records.json")),
              "records.json emitted per scenario")
    finally:
        shutil.rmtree(tmp)


def test_catalog_workload_kinds():
    """The two arms of the catalog's workloads/ namespace, on a fixture.

    test_catalog() below reads the real sibling repos, which are ABSENT in CI, so its
    workload assertions never run there. These do: a hand-built catalog with one shape
    and one replay pins the projection contract hermetically.
    """
    tmp = tempfile.mkdtemp()
    try:
        wdir = os.path.join(tmp, "blis-catalog", "workloads")
        os.makedirs(wdir)
        with open(os.path.join(wdir, "sharegpt.yaml"), "w", encoding="utf-8") as f:
            f.write("prefix_tokens: 0\n"
                    "prompt: {tokens: 208, tokens_stdev: 231, tokens_min: 4, tokens_max: 1013}\n"
                    "output: {tokens: 185, tokens_stdev: 190, tokens_min: 4, tokens_max: 979}\n")
        # the replay arm, shaped as blis-catalog's agentx-mvp.yaml is
        with open(os.path.join(wdir, "agentx-mvp.yaml"), "w", encoding="utf-8") as f:
            f.write('_comment: "provenance prose the loader discards"\n'
                    "replay:\n"
                    "  scenario: inferencex-agentx-mvp\n"
                    "  harness_version: 0.12.0\n"
                    "  corpus:\n"
                    "    loader: semianalysis_cc_traces_weka_062126_256k\n"
                    "    source:\n"
                    "      huggingface: semianalysisai/cc-traces-weka-062126-256k\n"
                    "      revision: 8fecd2fc56694469f758f0afbbb6335ad3043740\n"
                    "  category: coding\n"
                    '  thinking: "on"\n')
        wl = {w["name"]: w for w in bc.build(tmp)["workloads"]}
        check(set(wl) == {"sharegpt", "agentx-mvp"}, "both workload arms are projected")

        sg = wl["sharegpt"]
        check(sg["kind"] == "shape", "a prompt/output file projects as a shape")
        check((sg["isl_mean"], sg["osl_mean"]) == (208, 185), "shape ISL/OSL means projected")
        check(sg["scenario"] is None and sg["corpus"] is None, "a shape carries no replay provenance")

        ax = wl["agentx-mvp"]
        check(ax["kind"] == "replay", "a top-level replay: file projects as a replay")
        check(ax["scenario"] == "inferencex-agentx-mvp", "replay scenario projected")
        check(ax["corpus"] == "semianalysisai/cc-traces-weka-062126-256k", "replay corpus projected")
        check(ax["corpus_revision"] == "8fecd2fc", "corpus revision shortened for display")
        check(ax["harness_version"] == "0.12.0", "harness version projected as a string")
        check(ax["thinking"] == "on", "thinking projected as text, never a bool")
        # the point of the fix: absent distributions must stay absent, not read as 0
        check(ax["isl_mean"] is None and ax["osl_mean"] is None and ax["prefix_tokens"] is None,
              "a replay states no ISL/OSL distribution and no scalar prefix")
    finally:
        shutil.rmtree(tmp)


def test_pins_are_the_single_source_of_the_sim_version():
    """pins.env is the authority for "which simulator priced these reports".

    Three surfaces have to agree, or the site publishes a version that no run was made with:
      * pins.env            -- BLIS_REF / BLIS_SHA (read by build_catalog + the workflow)
      * SETUP.md            -- what a human clones by hand
      * generate-report.yml -- what CI clones; it must source pins.env, not hard-code a ref

    The regression this guards: `versions["inference-sim"]` used to be a `git describe` of an
    ambient ../inference-sim checkout. Nothing is ingested from that repo, so the pill published
    whatever branch a developer left lying around (v0.7.10-40-g220aa1dd, off the pinned tag's
    ancestry entirely), and read "absent" in CI, which clones the pin to ./upstream instead.
    """
    print("test_pins_are_the_single_source_of_the_sim_version")
    pins = bc.read_pins()
    ref, sha = pins.get("BLIS_REF"), pins.get("BLIS_SHA")
    check(bool(ref) and bool(sha), "pins.env must define BLIS_REF and BLIS_SHA")
    check(len(sha or "") == 40, "BLIS_SHA should be a full 40-char commit sha")

    # The pill names what PRICED the committed reports, and is never a description of some
    # sibling checkout. While no report predates the pin those are the same string; a pin bump
    # makes them diverge, and then publishing BLIS_REF would claim the new simulator priced runs
    # it never saw. So the reports are asked first, via their recorded pinned_stack.blis.
    cat = bc.build("/nonexistent-siblings-xyz")
    pill = cat["versions"]["inference-sim"]
    check("absent" not in pill,
          "the sim version must not depend on a sibling checkout being present")
    recorded = []
    rdir = os.path.join(ROOT, "reports")
    if os.path.isdir(rdir):
        for name in sorted(os.listdir(rdir)):
            p = os.path.join(rdir, name, "scenario.json")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                got = ((json.load(f).get("pinned_stack") or {}).get("blis"))
            if got and got not in recorded:
                recorded.append(got)
    expect = ref if not recorded else (
        ref if recorded == [ref]
        else ", ".join(([ref] if ref in recorded else []) + [v for v in recorded if v != ref]))
    check(pill == expect,
          f"the pill should name what priced the committed reports ({expect!r}), got {pill!r}")
    for v in recorded:
        check(v in pill, f"a report priced by {v} must be represented in the pill, got {pill!r}")

    # ...and the three branches of that rule, on synthetic report dirs rather than the real
    # gallery, so the test keeps meaning after the next pin bump.
    tmp = tempfile.mkdtemp()
    try:
        def _report(name, blis):
            d = os.path.join(tmp, name)
            os.makedirs(d)
            with open(os.path.join(d, "scenario.json"), "w") as f:
                json.dump({"pinned_stack": {"blis": blis}}, f)

        pin = {"BLIS_REF": "v9.9.9"}
        empty = os.path.join(tmp, "no-reports-here")
        os.makedirs(empty)
        check(bc._priced_by(pin, empty) == "v9.9.9",
              "with no committed report to go on, the pill falls back to the pin")
        check("unpinned" in bc._priced_by({}, empty),
              "with neither a pin nor a report, the pill says so instead of guessing")
        _report("a", "v9.9.9")
        _report("b", "v9.9.9")
        check(bc._priced_by(pin, tmp) == "v9.9.9",
              "reports that all agree with the pin publish as that one version")
        _report("c", "v9.9.8")
        check(bc._priced_by(pin, tmp) == "v9.9.9, v9.9.8",
              "a gallery spanning two simulators must name both, pin first, not read as one")
        check(bc._priced_by({"BLIS_REF": "v9.9.7"}, tmp) == "v9.9.9, v9.9.8",
              "a pin nothing was priced by is not published as having priced anything")
    finally:
        shutil.rmtree(tmp)

    # SETUP.md documents the same pin a human would clone
    with open(os.path.join(ROOT, ".claude", "skills", "config-search", "SETUP.md")) as f:
        setup = f.read()
    check(f"--branch {ref} " in setup, f"SETUP.md should clone {ref}")
    check(sha in setup, f"SETUP.md should verify the pinned sha {sha[:7]}")

    # CI must read pins.env rather than carry its own copy of the refs
    with open(os.path.join(ROOT, ".github", "workflows", "generate-report.yml")) as f:
        wf = f.read()
    check(". ./pins.env" in wf, "generate-report.yml should source pins.env")
    for key in ("BLIS_REF", "CATALOG_REF", "REGISTRY_REF"):
        check(f'"${key}"' in wf, f"the workflow should take {key} from pins.env")
    check(f"--branch {ref} " not in wf,
          "generate-report.yml hard-codes the sim ref; source it from pins.env instead")

    # The workflow that sweeps merged experiment specs clones the same three pins.
    with open(os.path.join(ROOT, ".github", "workflows", "run-experiment.yml")) as f:
        exp_wf = f.read()
    check(". ./pins.env" in exp_wf, "run-experiment.yml should source pins.env")
    # It must CHECK the sha, by reading it from pins.env -- a literal sha here would be a second
    # copy of the pin, which is the drift pins.env exists to prevent.
    check("$BLIS_SHA" in exp_wf, "run-experiment.yml should verify BLIS_SHA from pins.env")
    check(sha not in exp_wf, "run-experiment.yml hard-codes the sha; read it from pins.env")

    # ...and the SWEEPER must check it too, not just CI. The gap this closes (#23): only the
    # workflows verified BLIS_SHA, so `grep BLIS_SHA scripts/run_experiment.py` came back empty
    # and a contributor sweeping locally against an off-pin binary got real-looking numbers with
    # nothing in the output to say so. CI still refuses outright (it verifies on clone, above);
    # the runner WARNS, because testing against an unreleased simulator is a legitimate thing to
    # do locally and is how a pin bump gets evaluated at all.
    import run_search
    for fn in ("blis_pin_warning", "_blis_source_commit"):
        check(hasattr(run_search, fn), f"run_search should define {fn} (the runner-side pin check)")
    # The check compares a COMMIT to BLIS_SHA. It must not compare the binary digest: the same
    # tag builds to different bytes per toolchain, so a digest pin would fire on honest builds
    # (v0.12.0 is recorded with two different digests across the committed reports alone).
    with open(os.path.join(ROOT, "pins.env")) as f:
        pins_txt = f.read()
    check("BLIS_DIGEST" not in pins_txt and "BLIS_SHA256" not in pins_txt,
          "pins.env must not pin a binary DIGEST: it is build-dependent, so it would be wrong "
          "for every contributor but the one who wrote it (see test_search's pin tests)")
    on_pin = {"blis": ref, "resolved": {"blis": {"sha256": "ab" * 32, "commit": sha}}}
    check(run_search.blis_pin_warning(on_pin, pins) == "",
          "a binary built from the pinned commit must not warn")
    off = {"blis": ref, "resolved": {"blis": {"sha256": "ab" * 32, "commit": "c" * 40}}}
    check(bool(run_search.blis_pin_warning(off, pins)),
          "a binary built from another commit must warn")
    # Both runners must apply it; the sweeper is the one that was missing it.
    with open(os.path.join(ROOT, "scripts", "run_experiment.py")) as f:
        exp_py = f.read()
    check("_blis_pin_warning" in exp_py,
          "run_experiment.py should warn when an off-pin simulator prices a sweep (#23)")

    # The catalog must be cloned BEFORE the validate step, or the existence check on
    # scenario.model/hardware/workload silently no-ops and a typo'd identifier still reaches the
    # Go build before failing every run (susiejojo review on #22). Ordering is the whole point,
    # so assert the positions, not just the presence.
    clone_at = exp_wf.find("blis-catalog.git")
    validate_at = exp_wf.find("validate_experiment.py")
    check(clone_at != -1 and validate_at != -1 and clone_at < validate_at,
          "run-experiment.yml must clone blis-catalog BEFORE validating, so the "
          "model/hardware/workload existence check can run")
    check('validate_experiment.py "$spec" --catalog' in exp_wf,
          "run-experiment.yml should validate WITH --catalog, or the existence check no-ops")

    # The schema/validator agreement tests SKIP when jsonschema is absent, so the job that runs
    # them has to install it -- otherwise the drift guard quietly becomes a no-op and the
    # published schema can fall behind with CI still green (susiejojo review on #22).
    # Match the pip INVOCATION, not the word: a comment mentioning jsonschema would satisfy a
    # bare substring check while the package went uninstalled.
    with open(os.path.join(ROOT, ".github", "workflows", "checks.yml")) as f:
        checks_wf = f.read()
    installs = [ln.strip() for ln in checks_wf.splitlines() if "pip install" in ln]
    # The `python` job is the one that runs test_experiment.py, so that is the install line that
    # has to carry it -- and the fork-safe `experiments` job must NOT need a new dependency.
    py_job = checks_wf.split("experiments:")[0]
    py_installs = [ln.strip() for ln in py_job.splitlines() if "pip install" in ln]
    check(any("jsonschema" in ln for ln in py_installs),
          "checks.yml's `python` job must `pip install jsonschema`, or "
          "SchemaAgreesWithValidator skips in CI and the schema/validator drift guard never runs")
    check(all("pyyaml" in ln for ln in installs),
          "every checks.yml job still needs pyyaml")

    # GUIDANCE must not restate a superseded version. A doc that tells a contributor to clone a
    # ref the stack no longer uses is worse than one saying nothing: SETUP.md was correct at
    # v0.12.0 while the skill's README and SKILL.md still read "build v0.10.0 ... catalog 0.2.1",
    # and nothing failed, because only SETUP.md was checked (qa-review on #22, F2/F3).
    #
    # Historical statements are legitimate and stay: a dated measurement, a "removed in vX" note,
    # a recorded pinned_stack. So this reads only the *instructional* surfaces and asks that any
    # version they name is one pins.env actually pins.
    pinned_refs = {pins.get(k) for k in ("BLIS_REF", "CATALOG_REF", "REGISTRY_REF", "KERNEL_REF")}
    superseded = {"v0.10.0", "0.2.1", "v0.1.1"} - pinned_refs
    for rel in (os.path.join(".claude", "skills", "config-search", "README.md"),
                os.path.join("experiments", "README.md"),
                os.path.join(".claude", "skills", "blis-experiment", "SKILL.md")):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            text = f.read()
        for stale in sorted(superseded):
            check(stale not in text,
                  f"{rel} names superseded ref {stale}; cite pins.env instead of a version")


def test_lexicon_mirror_is_in_sync():
    """The shared report vocabulary (issue #13) lives in the skill at
    .claude/skills/config-search/report/lexicon.json; web/src/lexicon.json is a byte-identical
    mirror, because Vite cannot import out of the read-only skill dir. A mirror that drifts is
    worse than no sharing at all -- it looks shared and is not -- so the build fails on it."""
    print("test_lexicon_mirror_is_in_sync")
    canon = os.path.join(ROOT, ".claude", "skills", "config-search", "report", "lexicon.json")
    mirror = os.path.join(ROOT, "web", "src", "lexicon.json")
    check(os.path.exists(canon), "skill lexicon.json is missing")
    if not os.path.exists(mirror):
        check(False, "web/src/lexicon.json is missing (copy the skill's lexicon.json over it)")
        return
    with open(canon, "rb") as a, open(mirror, "rb") as b:
        ca, cb = a.read(), b.read()
    check(ca == cb, "web/src/lexicon.json has drifted from the skill's report/lexicon.json; "
                    "copy the canonical file over it")
    lex = json.loads(ca)
    # the dashboard and the report both key off these, so a rename here must be deliberate
    check(lex["objective"]["key"] == "tpg", "the shared objective is no longer tok/s per GPU")
    for key in lex["operating_columns"]:
        src = lex["metrics"].get(key) or lex["facets"].get(key, {})
        check(bool(src.get("label")), f"operating column {key} has no label")
        check(bool(src.get("help")), f"operating column {key} has no help text")


def test_catalog():
    print("test_catalog")
    cat = bc.build(os.path.join(ROOT, os.pardir))
    # against the real sibling repos if present; else graceful empty
    if cat["models"]:
        names = {m["name"] for m in cat["models"]}
        check("qwen3-14b" in names, "qwen3-14b in catalog")
        q = next(m for m in cat["models"] if m["name"] == "qwen3-14b")
        check(q["arch"] == "dense" and q["layers"] == 40, "qwen3-14b dense/40 layers")
        glm = next((m for m in cat["models"] if m["name"] == "glm-5.3-flash"), None)
        if glm:
            check(glm["arch"] == "MoE" and glm["experts"] == 288, "glm-5.3-flash MoE/288 experts")
        hw = {h["name"] for h in cat["hardware"]}
        check("h100" in hw and "h200" in hw, "h100/h200 in hardware")
        wl = {w["name"] for w in cat["workloads"]}
        # The four shape presets are the floor, not the ceiling: the catalog's workloads/
        # namespace also carries trace-driven entries (agentx-mvp, and more over time), so
        # asserting an exact set made this test fail the moment upstream added one. Assert
        # the presets are present and every entry is projected, which is the contract.
        check(wl >= {"chatbot", "summarization", "contentgen", "multidoc"},
              "the four shape presets are all present")
        check(all(w["kind"] in ("shape", "replay") for w in cat["workloads"]),
              "every workload is projected as a shape or a replay")
        shapes = [w for w in cat["workloads"] if w["kind"] == "shape"]
        check(all(w["isl_mean"] and w["osl_mean"] for w in shapes),
              "every shape states prompt and output means")
        for w in cat["workloads"]:
            if w["kind"] != "replay":
                continue
            # a replay's provenance is the only thing worth showing about it, so it must
            # project; its distributions must stay absent rather than read as zeros
            check(w["scenario"] and w["corpus"], f"replay {w['name']} projects scenario + corpus")
            check(w["isl_mean"] is None and w["prefix_tokens"] is None,
                  f"replay {w['name']} states no ISL/OSL distribution")
        sg = next((w for w in cat["workloads"] if w["name"] == "sharegpt"), None)
        if sg:  # present once blis-catalog#(sharegpt) lands; skipped against an older checkout
            # 223/200, the MEASURED pair the catalog states, with its provenance in the file:
            # ShareGPT_V3_unfiltered_cleaned_split.json under Qwen/Qwen2.5-14B, vLLM's three-rule
            # filter (min_len 4, max_prompt_len 1024 on the prompt ALONE, max_total_len 2048), over
            # the 82,744 qualifying pairs. The old 208/185 here predates blis-catalog#44 landing
            # the measurement; reading the 1024 as a COMBINED cap is the specific error that review
            # caught, and its signature is osl_max ~979 instead of the correct 1917.
            check(sg["kind"] == "shape" and sg["isl_mean"] == 223 and sg["osl_mean"] == 200,
                  f"sharegpt preset projects its measured ISL/OSL "
                  f"(got {sg.get('isl_mean')}/{sg.get('osl_mean')}, expected 223/200)")
    else:
        print("  (sibling repos absent — catalog empty; that is the graceful path)")

    # graceful degradation: nonexistent siblings
    empty = bc.build("/nonexistent-siblings-xyz")
    check(empty["models"] == [] and empty["versions"]["blis-catalog"] == "absent",
          "absent siblings -> empty catalog + 'absent' version")


if __name__ == "__main__":
    test_winner_with_a_knob_variant()
    test_winner_mismatch_raises_not_falls_back()
    test_report_back_link_resolves_on_pages()
    test_validation()
    test_build_determinism_and_nomock()
    test_catalog_workload_kinds()
    test_pins_are_the_single_source_of_the_sim_version()
    test_lexicon_mirror_is_in_sync()
    test_catalog()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
