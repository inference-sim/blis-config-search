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

        # unknown hardware
        badhw = os.path.join(tmp, "qwen3-14b-zzz-chatbot-0002")
        _mk_run_dir(badhw, "qwen3-14b-zzz-chatbot-0002", hardware="zzz")
        try:
            bg.validate_and_project(badhw)
            check(False, "unknown hardware should raise")
        except bg.Invalid:
            check(True, "unknown hardware raises Invalid")
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
        check(wl == {"chatbot", "summarization", "contentgen", "multidoc"}, "exactly the 4 real workloads")
    else:
        print("  (sibling repos absent — catalog empty; that is the graceful path)")

    # graceful degradation: nonexistent siblings
    empty = bc.build("/nonexistent-siblings-xyz")
    check(empty["models"] == [] and empty["versions"]["blis-catalog"] == "absent",
          "absent siblings -> empty catalog + 'absent' version")


if __name__ == "__main__":
    test_winner_with_a_knob_variant()
    test_validation()
    test_build_determinism_and_nomock()
    test_catalog()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
