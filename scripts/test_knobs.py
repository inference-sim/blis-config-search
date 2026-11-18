#!/usr/bin/env python3
"""Tests of the knob table and its check (CFG-6). Hermetic: synthetic schemas and field lists.

Run: python3 -I scripts/test_knobs.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knob_table as kt  # noqa: E402

SCENARIO = {"$ref": "#/$defs/scenario.Scenario", "$defs": {
    "scenario.Scenario": {"properties": {"model": {"type": "string"},
                                         "cluster": {"$ref": "#/$defs/scenario.Cluster"}}},
    "scenario.Cluster": {"properties": {"fabric": {"type": "string"},
                                        "pools": {"type": "array", "items": {"$ref": "#/$defs/scenario.Pool"}}}},
    "scenario.Pool": {"properties": {"hardware": {"type": "string"}}},
}}
DEPLOYMENT = {"$ref": "#/$defs/deployment.Deployment", "$defs": {
    "deployment.Deployment": {"properties": {"pools": {"type": "array", "items": {"$ref": "#/$defs/deployment.Pool"}}}},
    "deployment.Pool": {"properties": {"parallel": {"$ref": "#/$defs/deployment.Parallelism"}}},
    "deployment.Parallelism": {"properties": {"tp": {"type": "integer"}, "dp": {"type": "integer"}}},
}}
FIELDS = {"kind": "FieldList", "engine": {"kind": "vllm", "version": "0.33.0"}, "fields": [
    {"name": "block_size", "source": "config", "status": "modelled"},
    {"name": "moe_backend", "source": "config", "status": "not_modelled",
     "evidence": {"issue": "inference-sim/blis-vllm#9"}},
    {"name": "VLLM_X", "source": "env", "status": "not_modelled"},
]}
SPEC = {"$defs": {"sweep": {"properties": {"parallel": {}, "engine": {}, "num-instances": {}}},
                  "scenario": {"properties": {"engine_kind": {}, "model": {}}}}}
COVERAGE = {"knobs": {
    "scenario:model": {"reason": "one model per experiment"},
    "scenario:cluster.fabric": {"reason": "a scenario constant"},
    "scenario:cluster.pools[].hardware": {"reason": "no spec key", "issue": "inference-sim/x#1"},
    "deployment:pools[].parallel.tp": {"sweep": "sweeps[].parallel.tp"},
    "deployment:pools[].parallel.dp": {"sweep": "sweeps[].parallel.dp"},
}}


def build(scenario=SCENARIO, deployment=DEPLOYMENT, coverage=COVERAGE, fields=FIELDS):
    return kt.build(scenario, deployment, {"vllm": fields}, coverage, SPEC)


def test_a_fully_classified_schema_has_no_problem():
    table, problems = build()
    assert problems == [], problems
    ids = {r["id"] for r in table["knobs"]}
    assert "scenario:cluster.pools[].hardware" in ids and "deployment:pools[].parallel.tp" in ids


def test_a_field_added_to_the_schema_is_an_error():
    """The known-bad input of CFG-6: the simulator gains a property nobody classified."""
    grown = copy.deepcopy(SCENARIO)
    grown["$defs"]["scenario.Cluster"]["properties"]["uplink"] = {"type": "number"}
    _, problems = build(scenario=grown)
    assert any("scenario:cluster.uplink" in p and "coverage.yaml" in p for p in problems), problems


def test_a_classification_of_a_removed_property_is_an_error():
    shrunk = copy.deepcopy(DEPLOYMENT)
    del shrunk["$defs"]["deployment.Parallelism"]["properties"]["dp"]
    _, problems = build(deployment=shrunk)
    assert any("deployment:pools[].parallel.dp" in p and "no property" in p for p in problems), problems


def test_an_entry_needs_exactly_one_of_a_sweep_key_and_a_reason():
    both = copy.deepcopy(COVERAGE)
    both["knobs"]["scenario:model"] = {"reason": "r", "sweep": "sweeps[].num-instances"}
    none = copy.deepcopy(COVERAGE)
    none["knobs"]["scenario:model"] = {}
    for cov in (both, none):
        _, problems = build(coverage=cov)
        assert any("exactly one of" in p for p in problems), problems


def test_a_prefix_entry_covers_what_lies_under_it_and_an_exact_one_wins():
    cov = copy.deepcopy(COVERAGE)
    del cov["knobs"]["scenario:cluster.pools[].hardware"]
    cov["knobs"]["scenario:cluster.*"] = {"reason": "the cluster is a constant", "covers": [
        "scenario:cluster.fabric", "scenario:cluster.pools[].hardware"]}
    del cov["knobs"]["scenario:cluster.fabric"]
    table, problems = build(coverage=cov)
    assert problems == [], problems
    assert {r["class"] for r in table["knobs"] if r["id"].startswith("scenario:cluster")} == {"scenario:cluster.*"}


def prefix_coverage():
    cov = copy.deepcopy(COVERAGE)
    del cov["knobs"]["scenario:cluster.pools[].hardware"], cov["knobs"]["scenario:cluster.fabric"]
    cov["knobs"]["scenario:cluster.*"] = {"reason": "the cluster is a constant", "covers": [
        "scenario:cluster.fabric", "scenario:cluster.pools[].hardware"]}
    return cov


def test_a_property_added_under_a_prefix_entry_is_an_error_until_reviewed():
    """A wildcard must not absorb a property nobody looked at."""
    grown = copy.deepcopy(SCENARIO)
    grown["$defs"]["scenario.Cluster"]["properties"]["uplink"] = {"type": "number"}
    _, problems = build(scenario=grown, coverage=prefix_coverage())
    assert any("scenario:cluster.uplink" in p and "new under scenario:cluster.*" in p for p in problems), problems
    reviewed = prefix_coverage()
    reviewed["knobs"]["scenario:cluster.*"]["covers"].append("scenario:cluster.uplink")
    _, problems = build(scenario=grown, coverage=reviewed)
    assert problems == [], problems


def test_a_prefix_entry_without_a_reviewed_list_is_an_error():
    cov = prefix_coverage()
    del cov["knobs"]["scenario:cluster.*"]["covers"]
    _, problems = build(coverage=cov)
    assert any("scenario:cluster.*" in p and "covers" in p for p in problems), problems


def test_a_property_nested_deeper_than_the_old_cap_is_still_listed():
    """document_paths once stopped expanding at depth 5, leaving what lay below it unlisted."""
    defs = {"n.Top": {"properties": {"a": {"$ref": "#/$defs/n.L1"}}}}
    for i in range(1, 9):
        defs[f"n.L{i}"] = {"properties": {"x": {"$ref": f"#/$defs/n.L{i + 1}"}}}
    defs["n.L9"] = {"properties": {"leaf": {"type": "string"}}}
    paths = kt.document_paths({"$ref": "#/$defs/n.Top", "$defs": defs})
    assert paths == ["a.x.x.x.x.x.x.x.x.leaf"], paths


def test_a_sweep_key_the_spec_format_lacks_is_an_error():
    cov = copy.deepcopy(COVERAGE)
    cov["knobs"]["deployment:pools[].parallel.tp"] = {"sweep": "sweeps[].warp.tp"}
    _, problems = build(coverage=cov)
    assert any("no sweep key 'warp'" in p for p in problems), problems
    cov["knobs"]["deployment:pools[].parallel.tp"] = {"sweep": "sweeps[].parallel.tpx"}
    _, problems = build(coverage=cov)
    assert any("tpx" in p for p in problems), problems


def test_a_scenario_key_the_spec_format_lacks_is_an_error():
    cov = copy.deepcopy(COVERAGE)
    cov["knobs"]["scenario:model"] = {"sweep": "scenario.engine_kind"}
    _, problems = build(coverage=cov)
    assert problems == [], problems
    cov["knobs"]["scenario:model"] = {"sweep": "scenario.warp"}
    _, problems = build(coverage=cov)
    assert any("no scenario key 'warp'" in p for p in problems), problems


def test_a_classification_that_classifies_nothing_is_an_error_for_any_kind_of_key():
    cov = copy.deepcopy(COVERAGE)
    cov["knobs"]["scenario:cluster.gone"] = {"reason": "a property the schema never had"}
    _, problems = build(coverage=cov)
    assert any("scenario:cluster.gone" in p and "classifies no property" in p for p in problems), problems


def test_a_sweep_key_that_is_no_spec_path_is_an_error():
    cov = copy.deepcopy(COVERAGE)
    cov["knobs"]["scenario:model"] = {"sweep": "engine.block_size"}
    _, problems = build(coverage=cov)
    assert any("not a sweeps[] path" in p for p in problems), problems


def test_a_capability_row_that_is_neither_swept_nor_explained_is_an_error():
    table, _ = build()
    rows = copy.deepcopy(table["knobs"])
    for r in rows:
        if r["id"] == "scenario:model":
            r.pop("reason", None), r.pop("issue", None), r.update(sweep=None)
    cap = {"capabilities": {"c": {"rows": ["scenario:model"], "note": "n"}}}
    problems = kt.check_capabilities(rows, cap, engines=("vllm",))
    assert any("neither sweepable nor explained" in p for p in problems), problems


def test_engine_rows_follow_the_field_list_status():
    table, _ = build()
    rows = {r["setting"]: r for r in table["knobs"] if r["kind"] == "engine"}
    assert rows["block_size"]["sweep"] == "engine.block_size"
    assert rows["moe_backend"]["sweep"] is None and rows["moe_backend"]["issue"] == "inference-sim/blis-vllm#9"
    assert rows["VLLM_X"]["sweep"] is None and "environment" in rows["VLLM_X"]["reason"]


def test_a_capability_must_resolve_to_rows_that_are_swept_or_explained():
    cov = copy.deepcopy(COVERAGE)
    cov["capabilities"] = {"layout": {"rows": ["deployment:pools[].parallel.tp"], "note": "tp"},
                           "moe": {"rows": ["engine:vllm:config:moe_backend"], "note": "kernel"}}
    _, problems = build(coverage=cov)
    assert problems == [], problems
    cov["capabilities"]["ghost"] = {"rows": ["engine:vllm:config:no_such_setting"], "note": "x"}
    cov["capabilities"]["mute"] = {"rows": ["deployment:pools[].parallel.tp"]}
    _, problems = build(coverage=cov)
    assert any("ghost" in p and "no knob row" in p for p in problems), problems
    assert any("mute" in p and "note" in p for p in problems), problems


def test_a_capability_that_needs_an_engine_the_build_lacks_waits_for_the_pin():
    cov = copy.deepcopy(COVERAGE)
    cov["capabilities"] = {"sg": {"requires_engine": "sglang", "rows": ["engine:sglang:config:x"], "note": "n"}}
    table, problems = build(coverage=cov)
    assert problems == [], problems
    assert kt.pending_capabilities(table["engines"], cov) == ["sg"]
    assert "Pending the pinned build" in kt.render_md(table, cov)


def test_the_committed_coverage_file_parses_and_every_entry_has_one_of_the_two():
    cov = kt.load_coverage()
    for key, entry in cov["knobs"].items():
        assert bool(entry.get("sweep")) != bool(entry.get("reason")), key
        assert key.endswith(".*") == bool(entry.get("covers")), key
        if entry.get("issue"):
            assert entry["issue"].count("#") == 1, key


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
