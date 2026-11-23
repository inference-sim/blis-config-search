#!/usr/bin/env python3
"""Hermetic tests for the pin-bump automation (scripts/blis_pin.py + its three workflows).

    python3 -I scripts/test_blis_pin.py

No network, no BLIS: the upstream lookups are the only non-hermetic calls in blis_pin.py, and
every function below takes their results as arguments instead.
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import blis_pin as B  # noqa: E402

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


# TEMPORARY: the inference-sim commands, from scripts/testdata/blis-m1 (inference-sim#2077).
FIXTURE_BLIS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "blis-m1",
                            "blis")

GO_MOD = """module github.com/inference-sim/inference-sim

require (
\tgithub.com/inference-sim/blis-substrate v0.11.0
\tgithub.com/inference-sim/blis-vllm v0.32.0-blis.2
)

replace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router v1.0.0-blis.2
"""

# M3.2 on: the substrate and engines are directories of inference-sim; only the router fork is a module.
GO_MOD_M32 = """module github.com/inference-sim/inference-sim

replace github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router v1.0.0-blis.5.0.20261121014725-cb8d93fcb0fa
"""


def test_version_ordering():
    print("test_version_ordering")
    check(B.newer("v0.15.0", "v0.14.0"), "a later minor is newer")
    check(B.newer("v0.10.0", "v0.9.9"), "versions compare numerically, not as strings")
    check(not B.newer("v0.14.0", "v0.14.0"), "the same tag is not newer")
    check(not B.newer("v0.13.0", "v0.14.0"), "an older tag is never a bump")
    check(not B.newer("v0.16.0-rc1", "v0.14.0"), "a prerelease tag never becomes the pin")
    check(not B.newer("v0.16.0", None), "with no current pin the bot cannot order, so it waits")


def test_go_mod_versions():
    print("test_go_mod_versions")
    mods = B.go_mod_versions(GO_MOD)
    check(mods.get("blis-substrate") == "v0.11.0", "the substrate version is read")
    check(mods.get("blis-vllm") == "v0.32.0-blis.2", "the engine version is read")
    check(mods.get("llm-d-router") == "v1.0.0-blis.2", "the router fork is read from its replace")


def test_sync_pins_reads_replace():
    print("test_sync_pins_reads_replace")
    import sync_pins
    got = sync_pins.derive_from_text(GO_MOD)
    check(got == {"SUBSTRATE_REF": "v0.11.0", "VLLM_REF": "v0.32.0-blis.2",
                  "ROUTER_REF": "v1.0.0-blis.2"}, f"derived pins: {got}")
    block = sync_pins.derive_from_text("replace (\n\tgithub.com/llm-d/llm-d-router v0.0.0 => "
                                       "github.com/inference-sim/llm-d-router v1.0.0-blis.3\n)\n")
    check(block == {"ROUTER_REF": "v1.0.0-blis.3"}, f"a replace block entry is read: {block}")
    dropped = sync_pins.derive_from_text(GO_MOD.split("replace ")[0])
    check("ROUTER_REF" not in dropped, f"a dropped replace derives no ROUTER_REF: {dropped}")
    check("ROUTER_REF=" not in sync_pins.render(dropped), "so the regenerated block clears it")


def test_smoke_skips_dropped_sweeps():
    print("test_smoke_skips_dropped_sweeps")
    runs = [{"sweep": "a", "config": {}}, {"sweep": "gone", "config": {}}, {"sweep": "b", "config": {}}]
    spec = {"sweeps": [{"name": "a"}, {"name": "b"}]}
    check([r["sweep"] for r in B.live_rows(runs, spec)] == ["a", "b"],
          "a sweep the spec dropped is not re-run, so its refusal is not reported as a break")
    check(B.live_rows(runs, {}) == runs, "with no sweep list, every row is still checked")


def test_smoke_fetches_the_corpus_a_workload_file_names():
    print("test_smoke_fetches_the_corpus_a_workload_file_names")
    import fetch_corpus
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "workloads"))
        os.makedirs(os.path.join(d, "experiments"))
        with open(os.path.join(d, "workloads", "w.yaml"), "w") as f:
            f.write("corpus:\n  source: {huggingface: org/ds, revision: rev1, file: t.jsonl}\n"
                    "  sha256: abc\n")
        with open(os.path.join(d, "experiments", "e.yaml"), "w") as f:
            f.write("scenario:\n  workload_spec: workloads/w.yaml\n")
        asked = []
        real = fetch_corpus.fetch
        fetch_corpus.fetch = lambda dest, url, sha, token=None: asked.append((url, sha))
        try:
            B.ensure_corpus({"source": "experiments/e.yaml"}, os.path.join(d, "no-catalog"),
                            os.path.join(d, "out"), root=d)
        finally:
            fetch_corpus.fetch = real
    check(asked == [("https://huggingface.co/datasets/org/ds/resolve/rev1/t.jsonl", "abc")],
          f"a spec that names its workload by file is smoked on that file's corpus: {asked}")


def test_smoke_skips_rows_of_a_shape_the_spec_dropped():
    print("test_smoke_skips_rows_of_a_shape_the_spec_dropped")
    spec = {"sweeps": [{"name": "a", "num-instances": [1], "parallel": {"tp": [8], "dp": [1]},
                        "engine": {"max_num_seqs": [8, 16], "speculative": {"method": ["x"]}}}]}
    keep = {"sweep": "a", "config": {"tp": 8, "dp": 1, "num_instances": 1, "max_num_seqs": 8, "concurrency": 4}}
    gone = {"sweep": "a", "config": {"tp": 2, "dp": 4, "num_instances": 1, "max_num_seqs": 8, "concurrency": 4}}
    knob = {"sweep": "a", "config": {"tp": 8, "dp": 1, "num_instances": 1, "max_num_seqs": 99}}
    check(B.live_rows([keep, gone, knob], spec) == [keep],
          "a row whose deployment or knob value the spec no longer lists is not re-run")
    spec = {"sweeps": [{"name": "a", "engine": {"engram_config": {"cpu_offload": [True]}}}]}
    on = {"sweep": "a", "config": {"engram_config": {"cpu_offload": True}}}
    off = {"sweep": "a", "config": {"engram_config": {"cpu_offload": False}}}
    check(B.live_rows([on, off], spec) == [on], "the same for a knob given as a block of lists")


def test_rewrite_pins_keeps_comments():
    print("test_rewrite_pins_keeps_comments")
    text = "# BLIS_REF=v0.0.1 is a comment\nBLIS_REF=v0.14.0\nBLIS_SHA=aaa\n\nSUBSTRATE_REF=v0.13.0\n"
    out = B.rewrite_pins(text, {"BLIS_REF": "v0.15.0", "BLIS_SHA": "bbb"})
    check("BLIS_REF=v0.15.0\n" in out and "BLIS_SHA=bbb\n" in out, "assignments move")
    check("# BLIS_REF=v0.0.1 is a comment" in out, "a comment naming the key is left alone")
    check("SUBSTRATE_REF=v0.13.0" in out, "keys not named in the update are untouched")
    try:
        B.rewrite_pins(text, {"NEW_PIN": "x"})
        check(False, "inventing a pin that pins.env does not assign must be refused")
    except ValueError:
        check(True, "")


def test_setup_md_sources_the_pins():
    print("test_setup_md_sources_the_pins")
    with open(os.path.join(B.ROOT, "SETUP.md")) as f:
        setup = f.read()
    pins = B.read_pins()
    check(". ./pins.env" in setup and 'fetch -q --depth 1 origin "$BLIS_REF"' in setup
          and '"$(git rev-parse HEAD)" = "$BLIS_SHA"' in setup,
          "the clone recipe sources pins.env and uses $BLIS_REF and $BLIS_SHA")
    check(pins["BLIS_REF"] not in setup and pins["BLIS_SHA"] not in setup,
          "SETUP.md copies neither the pinned release nor its commit (X23): a bump has nothing to rewrite there")
    check(not hasattr(B, "rewrite_setup"), "bump edits pins.env only")


def test_bump_end_to_end_on_copies():
    print("test_bump_end_to_end_on_copies")
    tmp = tempfile.mkdtemp()
    try:
        pins = os.path.join(tmp, "pins.env")
        shutil.copy(B.PINS, pins)
        cur = B.read_pins(pins)
        # The copy of pins.env may pin a commit (the M3.2 stack head): a bump moves it to the first
        # release tag, and from then on to later ones.
        setup_before = open(os.path.join(B.ROOT, "SETUP.md"), "rb").read()
        moved = B.bump("v99.0.0", sha="e" * 40, go_mod=GO_MOD_M32, release_name="v99.0.0: big", pins_path=pins)
        after = B.read_pins(pins)
        check(after["BLIS_REF"] == "v99.0.0" and after["BLIS_SHA"] == "e" * 40, "pin moved")
        check(after["ROUTER_REF"] == "v1.0.0-blis.5.0.20261121014725-cb8d93fcb0fa",
              "the router pin follows the release's go.mod")
        check("SUBSTRATE_REF" not in after and "VLLM_REF" not in after,
              "the substrate and the engines are in-tree: no pin of their own")
        check("CATALOG_REF" not in after and "REGISTRY_REF" not in after,
              "the catalog and registry are embedded in the release: no pin of their own")
        check(set(moved) >= {"BLIS_REF", "BLIS_SHA"}, f"bump reports what moved: {moved}")
        check(open(os.path.join(B.ROOT, "SETUP.md"), "rb").read() == setup_before,
              "a bump leaves SETUP.md alone: it sources pins.env and holds no version")
        try:
            B.bump(after["BLIS_REF"], sha="e" * 40, go_mod=GO_MOD_M32, pins_path=pins)
            check(False, "bumping to a tag that is not newer must refuse")
        except SystemExit:
            check(True, "")
    finally:
        shutil.rmtree(tmp)

def test_changed_ignores_comments():
    print("test_changed_ignores_comments")
    a = {"BLIS_REF": "v1.0.0", "BLIS_SHA": "a", "SUBSTRATE_REF": "v0.12.0"}
    check(B.changed(a, dict(a)) == {}, "identical values -> nothing to regenerate")
    check(B.changed(a, dict(a, SUBSTRATE_REF="v0.13.0")) == {"SUBSTRATE_REF": ("v0.12.0", "v0.13.0")},
          "a substrate move regenerates: its embedded data prices the runs")
    check(B.changed(a, dict(a, CATALOG_REF="0.5.0")) == {},
          "a catalog ref is not a stack pin: the data is the release's embedded data")
    check(B.changed(a, dict(a, SOMETHING_ELSE="x")) == {},
          "a key outside the stack pins does not queue a campaign")


def _fake_repo():
    root = tempfile.mkdtemp()
    for d in ("reports", "experiments"):
        os.makedirs(os.path.join(root, d))

    def rdir(rid, model, files):
        d = os.path.join(root, "reports", rid)
        os.makedirs(d)
        with open(os.path.join(d, "scenario.json"), "w") as f:
            json.dump({"model": model, "hardware": "h100", "pinned_stack": {"blis": "v1.0.0"}}, f)
        for name in files:
            with open(os.path.join(d, name), "w") as f:
                f.write("{}")

    def write(rel, text):
        with open(os.path.join(root, rel), "w") as f:
            f.write(text)

    # an experiment whose spec is in experiments/
    rdir("exp-a", "m1", ["experiment.json"])
    write("experiments/exp-a.yaml", "name: exp-a\nscenario: {model: m1}\n")
    # an experiment whose spec file was RENAMED but still names the report
    rdir("exp-b", "m1", ["experiment.json"])
    write("experiments/renamed.yaml", "name: exp-b\nscenario: {model: m1, replay: {corpus: x}}\n")
    # an experiment whose spec is gone: falls back to the run dir's own spec.yaml
    rdir("exp-c", "m2", ["experiment.json", "spec.yaml"])
    write("reports/exp-c/spec.yaml", "name: exp-c\nscenario: {model: m2}\n")
    # nothing to regenerate from: no experiment.json (e.g. a retired search report)
    rdir("orphan", "m3", ["report.json"])
    for i in range(5):
        rdir(f"many-{i}", "m4", ["experiment.json"])
        write(f"experiments/many-{i}.yaml", f"name: many-{i}\nscenario: {{model: m4}}\n")
    return root


def test_same_stack():
    print("test_same_stack")
    with tempfile.TemporaryDirectory() as t:
        def mk(name, blis="v0.15.0", resolved=None):
            d = os.path.join(t, name)
            os.makedirs(d)
            with open(os.path.join(d, "scenario.json"), "w") as f:
                json.dump({"pinned_stack": {"blis": blis, "engine_version": "0.32.0",
                                            "resolved": resolved or {}}}, f)
            return d
        a = mk("a", resolved={"blis": {"commit": "c1", "describe": "v0.15.0", "sha256": "aa"}})
        b, c = mk("b", resolved={"blis": {"sha256": "aa"}}), mk("c", blis="v0.16.0")
        rebuilt, unhashed = mk("rebuilt", resolved={"blis": {"sha256": "bb"}}), mk("unhashed")
        check(B.same_stack(a, b), "build provenance beyond the binary hash does not make a different stack")
        check(not B.same_stack(a, rebuilt), "a same-tag rebuild with a different binary is a different stack")
        check(B.same_stack(a, unhashed), "labels decide when one side records no binary hash")
        check(not B.same_stack(a, c), "a different blis ref is a different stack")
        check(not B.same_stack(a, os.path.join(t, "missing")), "a missing report matches nothing")


def test_classify_and_batch():
    print("test_classify_and_batch")
    root = _fake_repo()
    try:
        c = {i: B.classify(i, root) for i in B.report_ids(root)}
        check(c["exp-a"]["kind"] == "experiment" and c["exp-a"]["source"] == "experiments/exp-a.yaml",
              "an experiment dir regenerates from its spec")
        check(c["exp-b"]["source"] == "experiments/renamed.yaml" and c["exp-b"]["replay"],
              "a renamed spec is found by its `name`, and its replay flag is read")
        check(c["exp-c"]["source"] == os.path.join("reports", "exp-c", "spec.yaml"),
              "a removed spec falls back to the run dir's own spec.yaml")
        check(c["orphan"]["kind"] == "orphan", "no source -> orphan, never a guess")

        p = B.plan(root, max_batch=4)
        names = [b["name"] for b in p["batches"]]
        check([o["id"] for o in p["orphans"]] == ["orphan"], "orphans are listed, not batched")
        check(all(len(b["ids"]) <= 4 for b in p["batches"]), "no batch exceeds max_batch")
        check("m4-1" in names and "m4-2" in names, f"a large model splits into numbered batches: {names}")
        m1 = next(b for b in p["batches"] if b["model"] == "m1")
        check(sorted(m1["ids"]) == ["exp-a", "exp-b"], "one model's reports review together")
        every = sorted(i for b in p["batches"] for i in b["ids"])
        check(every == sorted(r["id"] for r in p["reports"]),
              "every runnable report lands in exactly one batch")
        check(len(every) == len(set(every)), "no report is regenerated twice")

        only = B.plan(root, only=["exp-c"])
        check([r["id"] for r in only["reports"]] == ["exp-c"], "--only restricts the plan")
        try:
            B.plan(root, only=["nope"])
            check(False, "--only with an unknown id must refuse, not silently do nothing")
        except SystemExit:
            check(True, "")
    finally:
        shutil.rmtree(root)


def test_plan_covers_the_real_gallery():
    """Every committed report must be regenerable, or a pin bump leaves it stale by design."""
    print("test_plan_covers_the_real_gallery")
    p = B.plan()
    check(not p["orphans"],
          "every committed report needs an experiments/ spec to regenerate from; "
          f"orphans: {[o['id'] for o in p['orphans']]}")
    check(len(p["reports"]) == len(B.report_ids()), "the plan covers every committed report")


def test_summary_table():
    print("test_summary_table")
    old, new = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        def put(root, rid, blis, chosen, tpg, runs, servable):
            d = os.path.join(root, rid)
            os.makedirs(d)
            json.dump({"pinned_stack": {"blis": blis}}, open(os.path.join(d, "scenario.json"), "w"))
            json.dump({"winner": {"chosen": chosen}}, open(os.path.join(d, "pareto.json"), "w"))
            json.dump({"totals": {"runs": runs, "servable": servable},
                       "best_overall": {"tokens_per_sec_per_gpu": tpg}},
                      open(os.path.join(d, "experiment.json"), "w"))
        put(old, "r1", "v1.0.0", "4x TP8", 20.0, 100, 50)
        put(new, "r1", "v2.0.0", "4x TP8", 22.5, 100, 50)
        put(old, "r2", "v1.0.0", "8x TP4", 10.0, 80, 40)
        put(new, "r2", "v1.0.0", "8x TP4", 10.0, 80, 40)  # stale copy: the job failed
        md = B.summary(old, new, ["r1", "r2"], failed=["r2"])
        check("v1.0.0 → **v2.0.0**" in md, "the stack move is shown")
        check("20 → **22.5**" in md, "a moved number is shown old → new")
        check("| 4x TP8 |" in md, "an unchanged winner is shown once, not as a move")
        check("**failed**" in md and "`r2`" in md.split("Not regenerated")[1],
              "a failed report is called out, never read from its stale directory")
    finally:
        shutil.rmtree(old)
        shutil.rmtree(new)


def _rec(cfg, tps, rc=0, error=None):
    return {"config": cfg, "returncode": rc, "error": error,
            "metrics": {"tokens_per_sec": tps, "itl_p99_ms": 5.0, "ttft_p99_ms": 50.0,
                        "completed_requests": 80}}


def test_smoke_pairs():
    print("test_smoke_pairs")
    base = {"tp": 8, "num_instances": 2, "concurrency": 4}
    runs = [
        _rec(dict(base), 100.0),
        _rec(dict(base, max_num_seqs=64), 80.0),                     # moved: a real lever
        _rec(dict(base, gpu_memory_utilization=0.7), 100.0),         # did not move
        _rec(dict(base, concurrency=1), 30.0),
        _rec(dict(base, concurrency=1, max_num_seqs=64), 25.0),      # cheaper moved pair
        _rec(dict(base, tp=4, num_instances=4, max_num_seqs=64), 70.0),  # different SHAPE
        _rec(dict(base, max_num_seqs=64, router_config="/x/queue.yaml"), 60.0),  # two knobs
        _rec(dict(base, num_speculative_tokens=1, speculative_category="coding"), 120.0),
        _rec(dict(base, max_num_batched_tokens=2048), 0, rc=1, error="boom"),  # failed before
    ]
    pairs = B.smoke_pairs(runs)
    knobs = [p["knob"] for p in pairs]
    check(len(knobs) == len(set(knobs)), f"one pair per knob: {knobs}")
    check("max_num_seqs" in knobs and "gpu_memory_utilization" in knobs, f"knobs found: {knobs}")
    check("max_num_seqs,router_config" not in knobs,
          "a pair differing in two knobs cannot attribute an effect, so it is never chosen")
    check("num_speculative_tokens" in knobs,
          "acceptance selectors ride with the drafting knob instead of counting as knobs")
    check("max_num_batched_tokens" not in knobs, "a row that already failed is not a smoke target")
    seqs = next(p for p in pairs if p["knob"] == "max_num_seqs")
    check(seqs["moved"] and seqs["base"]["config"]["concurrency"] == 1,
          "the cheapest pair where the knob MOVED is preferred")
    check(seqs["base"]["config"].get("tp") == seqs["row"]["config"].get("tp"),
          "a pair never spans two shapes")
    check(not next(p for p in pairs if p["knob"] == "gpu_memory_utilization")["moved"],
          "a knob that never moved is still re-run, but cannot be judged inert")
    check(len(B.smoke_pairs(runs, max_pairs=2)) == 2, "max_pairs caps the run count")
    check([(p["knob"], p["base"]["config"]) for p in B.smoke_pairs(list(reversed(runs)))]
          == [(p["knob"], p["base"]["config"]) for p in pairs],
          "the choice does not depend on ledger order")


def test_smoke_verdicts():
    """LOUD and SILENT are judged from the new records; run_one is stubbed, the rest is real."""
    print("test_smoke_verdicts")
    import run_blis
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "reports", "r"))
        os.makedirs(os.path.join(root, "experiments"))
        open(os.path.join(root, "reports", "r", "experiment.json"), "w").close()
        with open(os.path.join(root, "experiments", "r.yaml"), "w") as f:
            f.write("name: r\nscenario: {model: m, hardware: h100, gpus: 16, workload: chatbot}\n")
        base = {"tp": 8, "num_instances": 2, "concurrency": 1}
        old = [_rec(dict(base), 100.0), _rec(dict(base, max_num_seqs=64), 80.0),
               _rec(dict(base, router_config="/elsewhere/routing/queue.yaml"), 90.0)]
        os.makedirs(os.path.join(root, "routing"))
        open(os.path.join(root, "routing", "queue.yaml"), "w").close()
        with open(os.path.join(root, "reports", "r", "runs.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in old) + "\n")
        with open(os.path.join(root, "reports", "r", "scenario.json"), "w") as f:
            json.dump({"model": "m"}, f)
        real, seen = run_blis.run_one, []

        def fake(behaviour):
            def run_one(blis, cat, reg, sd, model, hw, ev, wl, nreq, seed, hz, cfg, **kw):
                seen.append(cfg)
                return behaviour(cfg)
            return run_one
        try:
            run_blis.run_one = fake(lambda c: _rec(c, 81.0 if c.get("max_num_seqs") else
                                                   91.0 if c.get("router_config") else 101.0))
            r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
            check(not r["failures"], f"numbers moved but knobs still act: no failure {r['failures']}")
            check(any(c.get("router_config") == os.path.join(root, "routing", "queue.yaml")
                      for c in seen), "a ledger's absolute router path is mapped to this checkout")
            run_blis.run_one = fake(lambda c: _rec(c, 100.0))
            r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
            check(any(f.startswith("SILENT") and "max_num_seqs" in f for f in r["failures"]),
                  f"a knob that moved before and is identical now is SILENT: {r['failures']}")
            run_blis.run_one = fake(lambda c: _rec(c, 99.0, error="overloaded: still_queued=3"))
            r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
            check(not r["failures"] and len(r["warnings"]) == 3,
                  f"a row that runs but tips into overload is a warning, not LOUD: {r}")
            check("**Warnings**" in B.smoke_markdown([r]), "the comment lists the warnings")
            run_blis.run_one = fake(lambda c: _rec(c, 0, error="unsupported config: x"))
            r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
            check(sum(f.startswith("LOUD") for f in r["failures"]) == 3,
                  "a config the new stack refuses is LOUD")
            fit = ('kernelmodel: x does not fit 79.2 GiB devices at tp=8: one rank\'s weights alone are 80.8 GiB. '
                   'No KV budget exists')
            run_blis.run_one = fake(lambda c: _rec(c, 0, rc=1, error=fit))
            r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
            check(not r["failures"] and len(r["warnings"]) == 3 and "no longer fits" in r["warnings"][0],
                  f"a row the new release says does not fit is a warning naming the refusal, not LOUD: {r}")
            def unreleased(c):
                rec = _rec(c, 81.0 if c.get("max_num_seqs") else 91.0 if c.get("router_config") else 101.0)
                rec["released_build"] = False
                return rec
            run_blis.run_one = fake(unreleased)
            r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root, pin_ref="0fb85a76192eb1cb3f55aadc42d3f24876385c0c")
            check(not r["failures"] and sum("is a commit" in w for w in r["warnings"]) == 1,
                  f"an unreleased binary at a commit pin is one warning per report, not a failure per row: {r}")
            r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root, pin_ref="v0.19.0")
            check(sum(f.startswith("LOUD") and "not a tagged release" in f for f in r["failures"]) == 3,
                  f"an unreleased binary at a release-tag pin is LOUD: {r['failures']}")
            # A known kernel issue is strict: refused with its signature is a loud note, not a pass;
            # the case running again, or refused for another reason, fails the job.
            occ = "kernelmodel: the kernel reports 210.4 GiB of per-rank fixed occupancy on a 178.4 GiB device"
            saved_known = B.KNOWN_KERNEL_ISSUES
            B.KNOWN_KERNEL_ISSUES = ({"report": "r", "config": {"tp": 8, "num_instances": 2}, "signature":
                                      "per-rank fixed occupancy", "reason": "an over-count"},)
            try:
                run_blis.run_one = fake(lambda c: _rec(c, 0, rc=1, error=occ))
                r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
                md = B.smoke_markdown([r])
                check(not r["failures"] and len(r["known_issues"]) == 3 and "Known kernel issues" in md
                      and "known kernel issue" in md and "an over-count" in md,
                      f"a refused known case is printed as a known kernel issue, not a pass: {r['failures']}")
                run_blis.run_one = fake(lambda c: _rec(c, 81.0 if c.get("max_num_seqs") else
                                                       91.0 if c.get("router_config") else 101.0))
                r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
                check(sum("KNOWN_KERNEL_ISSUES" in f for f in r["failures"]) == 3 and not r["known_issues"],
                      f"a known case that now runs fails the job: {r['failures']}")
                run_blis.run_one = fake(lambda c: _rec(c, 0, rc=1, error="field x not found"))
                r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
                check(sum("KNOWN_KERNEL_ISSUES" in f for f in r["failures"]) == 3,
                      "a known case refused for another reason fails the job")
                B.KNOWN_KERNEL_ISSUES = ({"report": "other", "config": {"tp": 8}, "signature": "per-rank fixed occupancy",
                                          "reason": "x"},)
                run_blis.run_one = fake(lambda c: _rec(c, 0, rc=1, error=occ))
                r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
                check(sum(f.startswith("LOUD") for f in r["failures"]) == 3 and not r["known_issues"],
                      "an entry for another report excuses nothing")
            finally:
                B.KNOWN_KERNEL_ISSUES = saved_known
            check(B._known_issue("deepseek-v4-1-flash-b200-agentx-4node",
                                 {"tp": 4, "dp": 1, "num_instances": 8, "enable_expert_parallel": True,
                                  "concurrency": 64}) is not None
                  and B._known_issue("deepseek-v4-1-flash-b200-agentx-4node", {"tp": 8, "dp": 1, "num_instances": 4,
                                                                               "enable_expert_parallel": True}) is None
                  and B._known_issue("deepseek-v4-1-flash-b200-agentx-4node",
                                     {"tp": 4, "dp": 1, "num_instances": 8, "enable_expert_parallel": True,
                                      "max_num_batched_tokens": 2048}) is None,
                  "the committed entry matches the tp4 ep4 8-instance default case at any concurrency, not tp8 "
                  "and not the same shape with max_num_batched_tokens set (that one fits)")
            # A known non-binding knob is strict too: identical numbers are a note, the knob moving
            # or the entry matching no picked pair fails the job, and other knobs stay SILENT.
            saved_nb = B.KNOWN_NON_BINDING_KNOBS
            B.KNOWN_NON_BINDING_KNOBS = ({"report": "r", "knob": "max_num_seqs", "config": {"tp": 8, "num_instances": 2},
                                          "reason": "the cap cannot bind"},)
            try:
                run_blis.run_one = fake(lambda c: _rec(c, 91.0 if c.get("router_config") else 100.0))
                r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
                md = B.smoke_markdown([r])
                check(not r["failures"] and len(r["known_non_binding"]) == 1 and "Known non-binding knobs" in md
                      and "the cap cannot bind" in md,
                      f"a listed knob that measures identically is a note, not SILENT: {r['failures']}")
                run_blis.run_one = fake(lambda c: _rec(c, 81.0 if c.get("max_num_seqs") else
                                                       91.0 if c.get("router_config") else 101.0))
                r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
                check(any("KNOWN_NON_BINDING_KNOBS" in f and "now moves" in f for f in r["failures"])
                      and not r["known_non_binding"],
                      f"a listed knob that now moves fails the job: {r['failures']}")
                B.KNOWN_NON_BINDING_KNOBS = ({"report": "r", "knob": "max_num_seqs", "config": {"tp": 4},
                                              "reason": "x"},)
                run_blis.run_one = fake(lambda c: _rec(c, 91.0 if c.get("router_config") else 100.0))
                r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
                check(any(f.startswith("SILENT") for f in r["failures"])
                      and any("no longer picks a pair" in f for f in r["failures"]),
                      f"an entry matching no picked pair excuses nothing and is itself a failure: {r['failures']}")
                B.KNOWN_NON_BINDING_KNOBS = ({"report": "r", "knob": "router_config", "config": {"tp": 8},
                                              "reason": "x"},)
                r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
                check(any(f.startswith("SILENT") and "max_num_seqs" in f for f in r["failures"]),
                      "an entry for one knob does not excuse another knob")
            finally:
                B.KNOWN_NON_BINDING_KNOBS = saved_nb
            check(B._non_binding("muse-spark-open-816b-a42b-gb200-agentx-rack", "max_num_seqs",
                                 {"tp": 8, "num_instances": 8, "concurrency": 128}) is not None
                  and B._non_binding("muse-spark-open-816b-a42b-gb200-agentx-rack", "max_num_seqs",
                                     {"tp": 8, "num_instances": 8, "concurrency": 512}) is None,
                  "the committed entry covers the 128-lane pair and not the 512-lane pair, where the cap binds")
            run_blis.run_one = fake(lambda c: _rec(c, 0, rc=1, error="field x not found"))
            r = B.smoke("r", FIXTURE_BLIS, "cat", "reg", root=root)
            check(sum(f.startswith("LOUD") for f in r["failures"]) == 3,
                  "every row that ran before and fails now is LOUD")
            check("field x not found" in r["failures"][0], "the engine's own reason is quoted")
            md = B.smoke_markdown([r])
            check("**FAIL**" in md and "fix these on this PR" in md, "the comment says what to do")
        finally:
            run_blis.run_one = real
    finally:
        shutil.rmtree(root)


def test_regen_refuses_an_unreleased_pin():
    print("test_regen_refuses_an_unreleased_pin")
    commit = "0fb85a76192eb1cb3f55aadc42d3f24876385c0c"
    for ref in (commit, "main", "v0.18", "v0.18.2-rc1", ""):
        try:
            B.require_release({"BLIS_REF": ref})
            check(False, f"BLIS_REF={ref!r} must be refused")
        except SystemExit as e:
            check("not a release tag" in str(e) and repr(ref) in str(e), f"a clear refusal for {ref!r}: {e}")
    B.require_release({"BLIS_REF": "v0.18.2"})
    check(True, "a release tag passes")
    # regen itself refuses before it touches anything (the guard is its first step, so no BLIS is run).
    try:
        B.regen("any-report", "blis", "cat", "reg", tempfile.mkdtemp(), pins={"BLIS_REF": commit})
        check(False, "regen on a commit pin must refuse")
    except SystemExit as e:
        check("not a release tag" in str(e), f"regen refuses on a commit pin: {e}")
    # The campaign refuses at its plan step, so no report job starts, and the plan step does not hide it.
    wf = open(os.path.join(ROOT, ".github", "workflows", "regen-reports.yml")).read()
    check("blis_pin.py plan --require-release" in wf, "regen-reports.yml plans with --require-release")
    check("|| true" not in wf.split("blis_pin.py plan --require-release")[1].split("PY\n")[0],
          "the plan step does not swallow the refusal")


def test_workflow_wiring():
    print("test_workflow_wiring")
    wf = os.path.join(ROOT, ".github", "workflows")
    bump = open(os.path.join(wf, "bump-blis-pin.yml")).read()
    regen = open(os.path.join(wf, "regen-reports.yml")).read()
    sha = B.read_pins()["BLIS_SHA"]
    for name, text in (("bump-blis-pin.yml", bump), ("regen-reports.yml", regen)):
        check(sha not in text, f"{name} must read the sha from pins.env, not carry a copy")
        check(". ./pins.env" in text, f"{name} must source pins.env")
        check("go-version-file: upstream/go.mod" in text,
              f"{name} must build with the toolchain the release's go.mod names")
    check("$BLIS_SHA" in regen, "regen-reports.yml must refuse a moved tag")
    check("scripts/blis_pin.py bump" in bump, "the bump PR is written by blis_pin.py")
    check("bump/blis-pin-" in bump and 'gh pr list --head "$branch" --state all' in bump,
          "the daily poll must not open a second PR for a release it already proposed")
    check("exists=$exists" in bump and "steps.target.outputs.exists" in bump,
          "a pushed branch with no PR (a failed create) is resumed, not skipped forever")
    check("github.event.before" in regen and "fetch-depth: 0" in regen,
          "pins are compared across the whole push: a rebase-merged pin PR ends in its fixes")
    check("../blis-substrate" not in regen and "--catalog inference-sim/substrate/catalog" in regen
          and "--data-override" not in regen,
          "regen checks out inference-sim's substrate/ inside the checkout, so ledger cmds stay relative, and "
          "prices with the simulator's embedded data")
    check('paths: ["pins.env"]' in regen and "branches: [main]" in regen,
          "regeneration triggers on a pins.env change reaching main")
    check("scripts/blis_pin.py changed" in regen,
          "a comment-only pins.env edit must not queue a campaign")
    check("fail-fast: false" in regen, "one failed report must not cancel the others")
    check("blis_pin.py same-stack" in regen and '"$SWEPT..HEAD" -- "reports/$id"' in regen,
          "a report re-swept on main during the campaign keeps main's copy")
    check('[ ! -d "reports/$id" ]' in regen and 'stale="$stale $id"' in regen,
          "a report removed, or re-swept on the old stack, on main mid-campaign is not swapped in")
    check('git show "$SWEPT:pins.env"' in regen and "steps.stack.outputs.current == 'true'" in regen,
          "a campaign whose pins moved on main mid-sweep opens no PR")
    check("always()" in regen, "batch PRs open even when some report jobs failed")
    check("scripts/test_blis_pin.py" in open(os.path.join(wf, "checks.yml")).read(),
          "checks.yml must run these tests")
    ck = open(os.path.join(wf, "checks.yml")).read()
    check("FETCH_HEAD^{commit}" in ck and 'rev-parse FETCH_HEAD)' not in ck,
          "an annotated tag fetches as the tag object: checks.yml compares the commit it points at")
    compat = open(os.path.join(wf, "pin-compat.yml")).read()
    check("pull_request:" in compat and "paths:" not in compat.split("\njobs:")[0].replace(
              "without a `paths:` filter", ""),
          "the gate triggers on EVERY PR: a path-filtered required check stays pending forever")
    check("blis_pin.py changed" in compat and "needs.changed.outputs.run == 'true'" in compat,
          "the heavy jobs run only when a pin's VALUE moved, or a spec, ledger or smoke script changed")
    gate = compat.split("\n  gate:", 1)[-1]
    check("if: always()" in gate and "needs: [changed, build, smoke]" in gate,
          "`gate` always reports, so it is the one check that can be marked required")
    for name, text in (("pin-compat.yml", compat), ("regen-reports.yml", regen)):
        check("validate_experiment.py experiments/ --blis upstream/blis" in text,
              f"{name} validates every spec against the new binary's commands")
        check("blis_pin.py drift" not in text and "blis-schemas" not in text,
              f"{name} compares no copied settings table against sibling checkouts")
    action = open(os.path.join(ROOT, ".github", "actions", "pinned-blis", "action.yml")).read()
    check(sha not in compat and ". ./pins.env" in compat and "pinned-blis" in compat
          and ". ./pins.env" in action and "no longer resolves to $BLIS_SHA" in action
          and "git ls-remote" in action,
          "pin-compat.yml reads the pin from pins.env and refuses a moved tag")
    for step in ("blis_pin.py smoke",):
        check(step in compat, f"the gate runs `{step}`")
    check("fail-fast: false" in compat, "one report's smoke failure must not hide the others'")
    check("<!-- pin-compat -->" in compat and "comment \"$PR\" --edit-last" not in compat,
          "the gate comment is found by its marker, not as the token user's latest comment")
    check("head.repo.full_name == github.repository" in compat,
          "a fork's pin move does not run fork code on the regen runner")
    check("needs.changed.outputs.reports != '[]'" in compat and 'REPORTS" = "[]"' in gate,
          "no committed reports means no smoke matrix, and the gate still passes")
    check("--draft" in bump, "the bot's pin PR opens as a draft: it is where fixes get pushed")
    check("test_experiment.py" not in bump,
          "the bump workflow must not require the suite to pass: a release that needs code "
          "changes still needs its PR (pin-compat says what to change)")
    for name, text, kind in (("bump-blis-pin.yml", bump, "blis-pin"),
                             ("regen-reports.yml", regen, "report-regen")):
        check("--label auto-update" in text and f"--label {kind}" in text,
              f"{name}'s PRs carry `auto-update` + `{kind}`, so a bot PR is told apart at a glance")
        check("gh label create auto-update --force" in text and f"gh label create {kind} --force" in text,
              f"{name} creates its labels first: `gh pr create --label` fails on a missing label")
        check("issues: write" in text, f"{name} needs issues: write to create labels")
    check("gh label create full-smoke --force" in bump and "--label full-smoke" in bump,
          "the pin bot's PR carries `full-smoke`: pin-compat smokes only a core pair on a PR without it, "
          "and a release bump (with every fix pushed to it) must smoke every report")
    checks_yml = open(os.path.join(wf, "checks.yml")).read()
    check("workflow_dispatch:" in checks_yml.split("\njobs:")[0],
          "checks.yml is dispatchable: a GITHUB_TOKEN PR event starts no workflows")
    check("inputs.base" in compat and "inputs.pr" in compat,
          "a dispatched pin-compat still compares against the base and comments on the PR")
    for name, text in (("bump-blis-pin.yml", bump), ("regen-reports.yml", regen)):
        check("actions: write" in text and "gh workflow run checks.yml" in text
              and "gh workflow run pin-compat.yml" in text
              and ("-f base=main" in text or '-f base="$BASE_BRANCH"' in text),
              f"{name} dispatches CI on its own PR, so the required gate reports without a PAT")
    check("--stale-only" in regen, "a pin move reaching main regenerates only the stale reports")
    import yaml
    jobs = yaml.safe_load(regen)["jobs"]
    swept = [j for j, d in jobs.items() if any(
        (st.get("with") or {}).get("ref") == "${{ needs.plan.outputs.sha }}" for st in d.get("steps") or [])]
    check(sorted(swept) == ["build", "merge", "regen"] and "github.event.inputs.base || github.ref" in regen,
          "every regen job sweeps the base's pins, not the dispatch ref's")
    # The REGEN_RUNS_ON runner (the ARC image, deploy/openshift/arc/) ships no gh CLI: a job there
    # that calls gh fails with exit 127 (#183, the merge job's shard download).
    import re
    big = [j for j, d in jobs.items() if "vars.REGEN_RUNS_ON" in str(d.get("runs-on"))]
    check(sorted(big) == ["build", "merge", "regen"], "the heavy jobs run on REGEN_RUNS_ON")
    for j in big:
        for st in jobs[j].get("steps") or []:
            check(not re.search(r"(^|[\s;|&(])gh\s", st.get("run") or ""),
                  f"regen-reports.yml job {j} step {st.get('name') or st.get('uses')!r} calls gh, "
                  "which the REGEN_RUNS_ON runner image does not have")
    msteps = jobs["merge"]["steps"]
    dl = next((st for st in msteps if st.get("name") == "Download every shard's ledger"), {})
    check(dl.get("uses", "").startswith("actions/download-artifact@")
          and dl.get("with", {}).get("pattern") == "regen-${{ matrix.report.id }}--shard-*"
          and dl.get("with", {}).get("path") == "${{ runner.temp }}/shards",
          "merge downloads its shards by pattern, one regen-<id>--shard-<i>/ directory each")
    shard_dir = '"$RUNNER_TEMP/shards/regen-$ID--shard-$i/runs.jsonl"'
    check(sum(shard_dir in (st.get("run") or "") for st in msteps) == 2,
          "merge checks, then assembles, every shard's ledger from the directory the pattern made")


PINS_NEW = {"BLIS_REF": "v2.0.0", "BLIS_SHA": "b" * 40, "ROUTER_REF": "r2", "SUBSTRATE_REF": "s2"}


def _write_stack(root, rid, blis, router, substrate, commit=None):
    path = os.path.join(root, "reports", rid, "scenario.json")
    with open(path) as f:
        scen = json.load(f)
    scen["pinned_stack"] = {"blis": blis, "router": router, "substrate": substrate,
                            "engine_version": "0.32.0", "data": "embedded"}
    if commit:
        scen["pinned_stack"]["resolved"] = {"blis": {"commit": commit, "sha256": "aa"}}
    with open(path, "w") as f:
        json.dump(scen, f)


def test_stale_reports_and_the_post_merge_plan():
    """A pin move regenerates what the new pins did not price, and nothing once all are current.

    This is the post-merge guarantee: a pin PR regenerates its reports on its own branch, so the
    push of that merge to main plans no report and regen-reports opens no PR."""
    print("test_stale_reports_and_the_post_merge_plan")
    root = _fake_repo()
    try:
        with open(os.path.join(root, "pins.env"), "w") as f:
            f.write("# pins\n" + "".join(f"{k}={v}\n" for k, v in PINS_NEW.items()))
        every = B.report_ids(root)
        check(B.stale_ids(root) == every, "before regeneration every report carries the old stack")
        check(sorted(r["id"] for r in B.plan(root, stale_only=True)["reports"])
              == sorted(r["id"] for r in B.plan(root)["reports"]),
              "a fresh pin move plans every runnable report")

        # Regenerate two reports onto the new stack: only the rest stay stale.
        _write_stack(root, "exp-a", "v2.0.0", "r2", "s2", commit="b" * 40)
        _write_stack(root, "exp-b", "v2.0.0", "r2", "s2")
        check("exp-a" not in B.stale_ids(root) and "exp-b" not in B.stale_ids(root),
              "a report on the pinned labels (and commit, where recorded) is current")
        planned = [r["id"] for r in B.plan(root, stale_only=True)["reports"]]
        check("exp-a" not in planned and "exp-c" in planned,
              f"the stale-only plan skips the current reports and keeps the rest: {planned}")

        # Each label, and a recorded commit, is part of the stack.
        for rid, args, why in (
                ("exp-c", ("v2.0.0", "r1", "s2"), "an old router ref"),
                ("many-0", ("v2.0.0", "r2", "s1"), "an old substrate ref"),
                ("many-1", ("v1.0.0", "r2", "s2"), "an old blis ref")):
            _write_stack(root, rid, *args)
            check(rid in B.stale_ids(root), f"{why} is stale")
        _write_stack(root, "many-2", "v2.0.0", "r2", "s2", commit="c" * 40)
        check("many-2" in B.stale_ids(root), "a binary built from another commit is stale, whatever its tag")

        # The whole gallery regenerated: the merge's push plans nothing.
        for rid in every:
            _write_stack(root, rid, "v2.0.0", "r2", "s2", commit="b" * 40)
        check(B.stale_ids(root) == [], "after regeneration no report is stale")
        check(B.plan(root, stale_only=True)["reports"] == [] and B.plan(root, stale_only=True)["batches"] == [],
              "after the pin PR merges, main's regen-reports plans no report and opens no PR")
        check(len(B.plan(root)["reports"]) > 0, "a forced (rehearsal) plan still covers every report")
        rc = B.main(["stale", "--root", root])
        check(rc == 0, "`blis_pin.py stale` exits 0 when every report is current")
        _write_stack(root, "exp-a", "v1.0.0", "r2", "s2")
        check(B.main(["stale", "--root", root]) == 1, "`blis_pin.py stale` exits 1 while one is stale")
    finally:
        shutil.rmtree(root)

def test_shards_fit_a_long_report_into_jobs():
    print("test_shards_fit_a_long_report_into_jobs")
    root = _fake_repo()
    try:
        with open(os.path.join(root, "reports", "exp-a", "runs.jsonl"), "w") as f:
            f.write("".join(json.dumps({"wall_s": 3600}) + "\n" for _ in range(10)))   # 10 h serial
        check(B.shards_for("exp-a", 4, 6 * 3600, growth=1.0, root=root) == 1, "10 h at 4 fits one 6 h job")
        check(B.shards_for("exp-a", 1, 6 * 3600, growth=1.0, root=root) == 2, "10 h at 1 needs two 6 h jobs")
        check(B.shards_for("exp-a", 2, 6 * 3600, growth=3.0, root=root) == 3, "the growth margin adds jobs")
        check(B.shards_for("orphan", 1, 6 * 3600, root=root) == 1, "a report with no ledger gets one job")
    finally:
        shutil.rmtree(root)


def _stale_pair_repo(marker_text):
    """A fake repo pinned to v2.0.0 with exp-a priced by v1.0.0 (carrying `marker_text`) and exp-c stale."""
    root = _fake_repo()
    with open(os.path.join(root, "pins.env"), "w") as f:
        f.write("".join(f"{k}={v}\n" for k, v in PINS_NEW.items()))
    for rid in B.report_ids(root):
        _write_stack(root, rid, "v2.0.0", "r2", "s2", commit="b" * 40)
    _write_stack(root, "exp-a", "v1.0.0", "r2", "s2")
    _write_stack(root, "exp-c", "v1.0.0", "r2", "s2")
    if marker_text is not None:
        with open(os.path.join(root, "reports", "exp-a", B.PENDING), "w") as f:
            f.write(marker_text)
    return root


def test_a_pending_regeneration_is_held_only_until_its_release_is_pinned():
    """A report waiting on a release newer than the pin is held; once the pin reaches that release the
    hold lapses by itself and the plan names it (#139 follow-up: M3.0 reports said "regenerates in M3.1",
    the pin moved to M3.1, and the push run still skipped them)."""
    print("test_a_pending_regeneration_is_held_only_until_its_release_is_pinned")
    root = _stale_pair_repo("until: v3.0.0\nregenerated against v3.0.0\n")
    try:
        p = B.plan(root, stale_only=True)
        check([r["id"] for r in p["reports"]] == ["exp-c"],
              f"a report waiting on an unpinned release is held back: {p['reports']}")
        check([x["id"] for x in p["pending"]] == ["exp-a"] and not p["lapsed"], "the plan names the held-back report")
        check("exp-a" in [r["id"] for r in B.plan(root)["reports"]], "a forced plan still regenerates it")
    finally:
        shutil.rmtree(root)
    for text, what in (("until: v2.0.0\nthe pin has reached it\n", "its release is the pinned one"),
                       ("until: v1.5.0\nthe pin is past it\n", "its release is behind the pin"),
                       ("Regenerates against the M3.0 simulator in M3.1.\n", "it names no release")):
        root = _stale_pair_repo(text)
        try:
            p = B.plan(root, stale_only=True)
            check(sorted(r["id"] for r in p["reports"]) == ["exp-a", "exp-c"],
                  f"a marker whose hold has ended ({what}) does not skip the stale report: {p['reports']}")
            check([x["id"] for x in p["lapsed"]] == ["exp-a"] and not p["pending"],
                  f"the plan names the lapsed marker ({what}): {p['lapsed']}")
        finally:
            shutil.rmtree(root)


def test_committed_pending_markers_hold_something():
    """Every REGENERATION_PENDING marker in the repository must still hold its report: it names a release the
    pin has not reached, and the report is stale. Anything else is a silent skip waiting to happen."""
    print("test_committed_pending_markers_hold_something")
    pins = B.read_pins(os.path.join(B.ROOT, "pins.env"))
    check(B.report_ids(B.ROOT), "the repository commits reports")
    for rid in B.report_ids(B.ROOT):
        if B.marker(rid) is None:
            continue
        scen = os.path.join(B.ROOT, "reports", rid)
        check(B.pending(rid, pins=pins) is not None and not B.is_current(scen, pins),
              f"reports/{rid}/{B.PENDING} holds nothing: delete it (its release is pinned, it names none, "
              f"or the report is current)")


def test_smoke_prices_a_sglang_report_as_sglang():
    """A SGLang report's smoke re-run must be built for the SGLang engine and record its release; the
    process default is vLLM, and a smoke run that kept it would re-run the report as a different engine."""
    print("test_smoke_prices_a_sglang_report_as_sglang")
    import settings
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "experiments"))
        with open(os.path.join(root, "experiments", "s.yaml"), "w") as f:
            f.write("name: s\nscenario: {model: m, engine_kind: sglang, hardware: h200, workload: w, gpus: 8}\n")
        for kind, version, name in (("vllm", "0.33.0", "fields.json"), ("sglang", "0.5.23", "fields.sglang.json")):
            with open(os.path.join(root, name), "w") as f:
                json.dump({"kind": "FieldList", "engine": {"kind": kind, "version": version},
                           "fields": [{"name": "x", "source": "config", "status": "modeled"}]}, f)
        settings.use_engine("vllm")
        ctx = B._run_context({"source": "experiments/s.yaml", "replay": False}, root, os.path.join(root, "no-blis"),
                             None, None, None, fields=os.path.join(root, "fields.json"))
        check(settings.ENGINE_KIND == "sglang", "the smoke run is set up for the engine the spec names")
        check(ctx[2] == "0.5.23", f"it records the SGLang release, not vLLM's: {ctx[2]}")
    finally:
        settings.use_engine("vllm")
        shutil.rmtree(root)


if __name__ == "__main__":
    test_committed_pending_markers_hold_something()
    test_smoke_prices_a_sglang_report_as_sglang()
    test_version_ordering()
    test_go_mod_versions()
    test_sync_pins_reads_replace()
    test_smoke_skips_dropped_sweeps()
    test_smoke_skips_rows_of_a_shape_the_spec_dropped()
    test_smoke_fetches_the_corpus_a_workload_file_names()
    test_rewrite_pins_keeps_comments()
    test_setup_md_sources_the_pins()
    test_bump_end_to_end_on_copies()
    test_changed_ignores_comments()
    test_same_stack()
    test_classify_and_batch()
    test_plan_covers_the_real_gallery()
    test_summary_table()
    test_smoke_pairs()
    test_smoke_verdicts()
    test_workflow_wiring()
    test_regen_refuses_an_unreleased_pin()
    test_stale_reports_and_the_post_merge_plan()
    test_shards_fit_a_long_report_into_jobs()
    test_a_pending_regeneration_is_held_only_until_its_release_is_pinned()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
