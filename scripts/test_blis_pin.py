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


GO_MOD = """module github.com/inference-sim/inference-sim

require (
\tgithub.com/inference-sim/blis-schemas v0.10.0
\tgithub.com/inference-sim/blis-vllm v0.32.0-blis.1
)

require (
\tgithub.com/inference-sim/blis-latency-kernel v0.5.0 // indirect
)
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
    check(mods.get("blis-latency-kernel") == "v0.5.0",
          "the kernel is read from go.mod even when it is an indirect requirement")
    check(mods.get("blis-schemas") == "v0.10.0", "schemas version is read")


def test_rewrite_pins_keeps_comments():
    print("test_rewrite_pins_keeps_comments")
    text = "# BLIS_REF=v0.0.1 is a comment\nBLIS_REF=v0.14.0\nBLIS_SHA=aaa\n\nCATALOG_REF=0.4.0\n"
    out = B.rewrite_pins(text, {"BLIS_REF": "v0.15.0", "BLIS_SHA": "bbb"})
    check("BLIS_REF=v0.15.0\n" in out and "BLIS_SHA=bbb\n" in out, "assignments move")
    check("# BLIS_REF=v0.0.1 is a comment" in out, "a comment naming the key is left alone")
    check("CATALOG_REF=0.4.0" in out, "keys not named in the update are untouched")
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
        moved = B.bump("v99.0.0", sha="e" * 40, go_mod=GO_MOD, release_name="v99.0.0: big",
                       pins_path=pins, setup_path=setup)
        after = B.read_pins(pins)
        check(after["BLIS_REF"] == "v99.0.0" and after["BLIS_SHA"] == "e" * 40, "pin moved")
        check(after["KERNEL_REF"] == "v0.5.0", "KERNEL_REF follows the release's go.mod")
        check(after["CATALOG_REF"] == cur["CATALOG_REF"]
              and after["REGISTRY_REF"] == cur["REGISTRY_REF"],
              "catalog/registry are NOT guessed -- the PR asks a reviewer")
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
    a = {"BLIS_REF": "v1.0.0", "BLIS_SHA": "a", "CATALOG_REF": "0.4.0"}
    check(B.changed(a, dict(a)) == {}, "identical values -> nothing to regenerate")
    check(B.changed(a, dict(a, CATALOG_REF="0.5.0")) == {"CATALOG_REF": ("0.4.0", "0.5.0")},
          "a catalog-only move still regenerates: the catalog prices runs too")
    check(B.changed(a, dict(a, SOMETHING_ELSE="x")) == {},
          "a key outside the stack pins does not queue a campaign")


def _fake_repo():
    root = tempfile.mkdtemp()
    for d in ("reports", "experiments", "submissions"):
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
    # submissions: a replay one with a narrative, a shape one without
    rdir("sub-a", "m1", ["report.json"])
    write("submissions/sub-a.yaml", "model: m1\nworkload: {replay: {scenario: s}}\n")
    rdir("sub-b", "m3", [])
    write("submissions/sub-b.yaml", "model: m3\nworkload: {preset: chatbot}\n")
    # nothing to regenerate from
    rdir("orphan", "m3", [])
    for i in range(5):
        rdir(f"many-{i}", "m4", [])
        write(f"submissions/many-{i}.yaml", "model: m4\nworkload: {preset: chatbot}\n")
    return root


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
        check(c["sub-a"]["kind"] == "submission" and c["sub-a"]["replay"] and c["sub-a"]["narrative"],
              "a replay submission with a report.json re-derives its narrative")
        check(not c["sub-b"]["replay"] and not c["sub-b"]["narrative"], "a shape submission")
        check(c["orphan"]["kind"] == "orphan", "no source -> orphan, never a guess")

        p = B.plan(root, max_batch=4)
        names = [b["name"] for b in p["batches"]]
        check([o["id"] for o in p["orphans"]] == ["orphan"], "orphans are listed, not batched")
        check(all(len(b["ids"]) <= 4 for b in p["batches"]), "no batch exceeds max_batch")
        check("m4-1" in names and "m4-2" in names, f"a large model splits into numbered batches: {names}")
        m1 = next(b for b in p["batches"] if b["model"] == "m1")
        check(sorted(m1["ids"]) == ["exp-a", "exp-b", "sub-a"],
              "one model's experiment AND submission reports review together")
        every = sorted(i for b in p["batches"] for i in b["ids"])
        check(every == sorted(r["id"] for r in p["reports"]),
              "every runnable report lands in exactly one batch")
        check(len(every) == len(set(every)), "no report is regenerated twice")

        only = B.plan(root, only=["sub-b"])
        check([r["id"] for r in only["reports"]] == ["sub-b"], "--only restricts the plan")
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
          "every committed report needs an experiments/ spec or a submissions/ file to "
          f"regenerate from; orphans: {[o['id'] for o in p['orphans']]}")
    check(len(p["reports"]) == len(B.report_ids()), "the plan covers every committed report")


def test_summary_table():
    print("test_summary_table")
    old, new = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        def put(root, rid, blis, chosen, tpg, runs, servable):
            d = os.path.join(root, rid)
            os.makedirs(d)
            json.dump({"pinned_stack": {"blis": blis}}, open(os.path.join(d, "scenario.json"), "w"))
            json.dump({"winner": {"chosen": chosen},
                       "finalists": [{"at_c_use": {"tokens_per_sec_per_gpu": tpg}}]},
                      open(os.path.join(d, "pareto.json"), "w"))
            json.dump({"n_runs": runs, "n_servable": servable},
                      open(os.path.join(d, "gallery.json"), "w"))
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
        os.makedirs(os.path.join(root, "submissions"))
        with open(os.path.join(root, "submissions", "r.yaml"), "w") as f:
            f.write("model: m\nhardware: h100\ngpu_budget: 16\nworkload: {preset: chatbot}\n")
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
            r = B.smoke("r", "blis", "cat", "reg", root=root)
            check(not r["failures"], f"numbers moved but knobs still act: no failure {r['failures']}")
            check(any(c.get("router_config") == os.path.join(root, "routing", "queue.yaml")
                      for c in seen), "a ledger's absolute router path is mapped to this checkout")
            run_blis.run_one = fake(lambda c: _rec(c, 100.0))
            r = B.smoke("r", "blis", "cat", "reg", root=root)
            check(any(f.startswith("SILENT") and "max_num_seqs" in f for f in r["failures"]),
                  f"a knob that moved before and is identical now is SILENT: {r['failures']}")
            run_blis.run_one = fake(lambda c: _rec(c, 0, rc=1, error="field x not found"))
            r = B.smoke("r", "blis", "cat", "reg", root=root)
            check(sum(f.startswith("LOUD") for f in r["failures"]) == 3,
                  "every row that ran before and fails now is LOUD")
            check("field x not found" in r["failures"][0], "the engine's own reason is quoted")
            md = B.smoke_markdown([r])
            check("**FAIL**" in md and "fix these on this PR" in md, "the comment says what to do")
        finally:
            run_blis.run_one = real
    finally:
        shutil.rmtree(root)


def test_drift():
    print("test_drift")
    import validate_experiment as V
    root = tempfile.mkdtemp()
    try:
        def put(rel, text):
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        eng = sorted(V.ENGINE_KNOBS)
        put("blis-vllm/rules/engineconfig/fieldnames_test.go",
            "want := map[string]string{\n" + "".join(f'\t"{n}": "x",\n' for n in eng) + "}\n")
        put("blis-schemas/spec/deployment/deployment.go",
            "type Parallelism struct {\n" + "".join(f'\tX int `yaml:"{n},omitempty"`\n'
                                                   for n in sorted(V.PARALLEL_KNOBS)) + "}\n")
        pack = "".join(f'\t{f}: set({", ".join(chr(34) + v + chr(34) for v in V.ENGINE_ENUMS[k])}),\n'
                       for f, k in (("CUDAGraphModes", "cudagraph_mode"),
                                    ("CacheDTypes", "cache_dtype")))
        put(f"blis-vllm/rules/{V.RULES_PACK}/pack.go", pack)
        check(B.drift(root) == [], f"tables that match the stack: no drift {B.drift(root)}")
        put("blis-vllm/rules/engineconfig/fieldnames_test.go",
            "want := map[string]string{\n" + "".join(f'\t"{n}": "x",\n' for n in eng[1:])
            + '\t"brand_new_knob": "x",\n}\n')
        d = B.drift(root)
        check(any("brand_new_knob" in x for x in d), "a knob the stack added is reported")
        check(any(eng[0] in x and "no longer defines" in x for x in d),
              "a knob the stack removed is reported")
        put(f"blis-vllm/rules/{V.RULES_PACK}/pack.go", pack.replace('"FULL"', '"FULL", "NEW_MODE"'))
        check(any("NEW_MODE" in x for x in B.drift(root)), "a new enum value is reported")
        put("blis-vllm/rules/v0_99/pack.go", pack)
        check(any("v0_99" in x for x in B.drift(root)),
              "a newer rules pack upstream means the reader is reading a stale vLLM")
        check(len(B.drift(os.path.join(root, "nothing-here"))) >= 3,
              "missing upstream files fail loudly, never read as 'no drift'")
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
    check("bump/blis-pin-" in bump and "ls-remote --exit-code --heads" in bump,
          "the daily poll must not open a second PR for a release it already proposed")
    check('paths: ["pins.env"]' in regen and "branches: [main]" in regen,
          "regeneration triggers on a pins.env change reaching main")
    check("scripts/blis_pin.py changed" in regen,
          "a comment-only pins.env edit must not queue a campaign")
    check("fail-fast: false" in regen, "one failed report must not cancel the others")
    check("always()" in regen, "batch PRs open even when some report jobs failed")
    check("scripts/test_blis_pin.py" in open(os.path.join(wf, "checks.yml")).read(),
          "checks.yml must run these tests")
    compat = open(os.path.join(wf, "pin-compat.yml")).read()
    check('paths: ["pins.env"]' in compat and "pull_request" in compat,
          "the compatibility gate runs on every PR that moves a pin")
    check(sha not in compat and ". ./pins.env" in compat and "$BLIS_SHA" in compat,
          "pin-compat.yml reads the pin from pins.env and refuses a moved tag")
    for step in ("blis_pin.py drift", "blis_pin.py smoke", "blis_pin.py modules"):
        check(step in compat, f"the gate runs `{step}`")
    check("fail-fast: false" in compat, "one report's smoke failure must not hide the others'")
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
    check("blis_pin.py drift" in regen,
          "regeneration re-checks drift, so a pin that skipped the gate cannot start a campaign")


if __name__ == "__main__":
    test_version_ordering()
    test_go_mod_versions()
    test_rewrite_pins_keeps_comments()
    test_rewrite_setup_moves_only_instructions()
    test_bump_end_to_end_on_copies()
    test_changed_ignores_comments()
    test_classify_and_batch()
    test_plan_covers_the_real_gallery()
    test_summary_table()
    test_smoke_pairs()
    test_smoke_verdicts()
    test_drift()
    test_workflow_wiring()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
