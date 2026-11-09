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


def test_rewrite_setup_moves_only_instructions():
    print("test_rewrite_setup_moves_only_instructions")
    with open(B.SETUP) as f:
        setup = f.read()
    pins = B.read_pins()
    old, sha = pins["BLIS_REF"], pins["BLIS_SHA"]
    out = B.rewrite_setup(setup, old, "v9.9.9", "f" * 40, "release v9.9.9")
    check(f"git clone --branch v9.9.9 --depth 1 https://github.com/inference-sim/inference-sim.git"
          in out, "the clone line moves")
    check(f"EXPECT={'f' * 40}" in out and sha not in out, "the EXPECT sha moves")
    check("v9.9.9 no longer resolves to $EXPECT" in out, "the abort message moves")
    # Historical mentions of the old version are statements of fact and must survive the bump.
    hist = setup.count(old) - 4
    check(out.count(old) == hist,
          f"only the 4 instructional lines move; {hist} historical mention(s) of {old} remain")
    try:
        B.rewrite_setup("no clone block here\n", old, "v9.9.9", "f" * 40, "x")
        check(False, "a SETUP.md whose clone block changed shape must fail loudly")
    except ValueError:
        check(True, "")


def test_bump_end_to_end_on_copies():
    print("test_bump_end_to_end_on_copies")
    tmp = tempfile.mkdtemp()
    try:
        pins, setup = os.path.join(tmp, "pins.env"), os.path.join(tmp, "SETUP.md")
        shutil.copy(B.PINS, pins)
        shutil.copy(B.SETUP, setup)
        cur = B.read_pins(pins)
        if not B.semver(cur["BLIS_REF"]):
            # A draft pins a branch commit; the bot orders only release tags, so start the copy
            # from one. rewrite_setup moves the same four lines whatever the old ref is.
            with open(pins) as f:
                text = B.rewrite_pins(f.read(), {"BLIS_REF": "v0.15.0"})
            with open(pins, "w") as f:
                f.write(text)
            with open(setup) as f:
                text = B.rewrite_setup(f.read(), cur["BLIS_REF"], "v0.15.0", cur["BLIS_SHA"], "x")
            with open(setup, "w") as f:
                f.write(text)
            cur = B.read_pins(pins)
        moved = B.bump("v99.0.0", sha="e" * 40, go_mod=GO_MOD, release_name="v99.0.0: big",
                       pins_path=pins, setup_path=setup)
        after = B.read_pins(pins)
        check(after["BLIS_REF"] == "v99.0.0" and after["BLIS_SHA"] == "e" * 40, "pin moved")
        check(after["SUBSTRATE_REF"] == "v0.11.0", "SUBSTRATE_REF follows the release's go.mod")
        check(after["VLLM_REF"] == "v0.32.0-blis.2" and after["ROUTER_REF"] == "v1.0.0-blis.2",
              "every derived pin follows the release's go.mod")
        check("CATALOG_REF" not in after and "REGISTRY_REF" not in after,
              "the catalog and registry are embedded in the release: no pin of their own")
        check(set(moved) >= {"BLIS_REF", "BLIS_SHA"}, f"bump reports what moved: {moved}")
        with open(setup) as f:
            check('"v99.0.0: big"' in f.read(), "the release title is recorded beside the sha")
        try:
            B.bump(cur["BLIS_REF"], sha="e" * 40, go_mod=GO_MOD, pins_path=pins, setup_path=setup)
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
    check("../blis-catalog" not in regen and "--catalog blis-catalog" in regen,
          "regen clones the catalog inside the checkout, so ledger cmds stay relative")
    check('paths: ["pins.env"]' in regen and "branches: [main]" in regen,
          "regeneration triggers on a pins.env change reaching main")
    check("scripts/blis_pin.py changed" in regen,
          "a comment-only pins.env edit must not queue a campaign")
    check("fail-fast: false" in regen, "one failed report must not cancel the others")
    check("blis_pin.py same-stack" in regen and '"$GITHUB_SHA..HEAD" -- "reports/$id"' in regen,
          "a report re-swept on main during the campaign keeps main's copy")
    check('[ ! -d "reports/$id" ]' in regen and 'stale="$stale $id"' in regen,
          "a report removed, or re-swept on the old stack, on main mid-campaign is not swapped in")
    check('git show "$GITHUB_SHA:pins.env"' in regen and "steps.stack.outputs.current == 'true'" in regen,
          "a campaign whose pins moved on main mid-sweep opens no PR")
    check("always()" in regen, "batch PRs open even when some report jobs failed")
    check("scripts/test_blis_pin.py" in open(os.path.join(wf, "checks.yml")).read(),
          "checks.yml must run these tests")
    compat = open(os.path.join(wf, "pin-compat.yml")).read()
    check("pull_request:" in compat and "paths:" not in compat.split("\njobs:")[0].replace(
              "without a `paths:` filter", ""),
          "the gate triggers on EVERY PR: a path-filtered required check stays pending forever")
    check("blis_pin.py changed" in compat and "needs.changed.outputs.moved == 'true'" in compat,
          "the heavy jobs run only when a pin's VALUE moved against the base branch")
    gate = compat.split("\n  gate:", 1)[-1]
    check("if: always()" in gate and "needs: [changed, build, smoke]" in gate,
          "`gate` always reports, so it is the one check that can be marked required")
    for name, text in (("pin-compat.yml", compat), ("regen-reports.yml", regen)):
        check("validate_experiment.py experiments/ --blis upstream/blis" in text,
              f"{name} validates every spec against the new binary's commands")
        check("blis_pin.py drift" not in text and "blis-schemas" not in text,
              f"{name} compares no copied settings table against sibling checkouts")
    check(sha not in compat and ". ./pins.env" in compat and "$BLIS_SHA" in compat,
          "pin-compat.yml reads the pin from pins.env and refuses a moved tag")
    for step in ("blis_pin.py smoke",):
        check(step in compat, f"the gate runs `{step}`")
    check("fail-fast: false" in compat, "one report's smoke failure must not hide the others'")
    check("<!-- pin-compat -->" in compat and "comment \"$PR\" --edit-last" not in compat,
          "the gate comment is found by its marker, not as the token user's latest comment")
    check("head.repo.full_name == github.repository" in compat,
          "a fork's pin move does not run fork code on the regen runner")
    check("needs.build.outputs.reports != '[]'" in compat and 'REPORTS" = "[]"' in gate,
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


if __name__ == "__main__":
    test_version_ordering()
    test_go_mod_versions()
    test_sync_pins_reads_replace()
    test_smoke_skips_dropped_sweeps()
    test_rewrite_pins_keeps_comments()
    test_rewrite_setup_moves_only_instructions()
    test_bump_end_to_end_on_copies()
    test_changed_ignores_comments()
    test_same_stack()
    test_classify_and_batch()
    test_plan_covers_the_real_gallery()
    test_summary_table()
    test_smoke_pairs()
    test_smoke_verdicts()
    test_workflow_wiring()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
