#!/usr/bin/env python3
"""Unit tests for run_blis.py, the BLIS runner. No network / no BLIS binary needed.

    python3 scripts/test_run_blis.py          # or: python3 -m unittest -v  (from this dir)
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_cli
import run_blis
import settings

# TEMPORARY: the inference-sim commands, from scripts/testdata/blis-m1 (inference-sim#2077).
FIXTURE_BLIS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "blis-m1",
                            "blis")
VOCAB = settings.vocabulary(FIXTURE_BLIS)


def classify(config, node_size=8):
    return run_blis.classify_config(config, VOCAB, node_size)


class ParseMetrics(unittest.TestCase):
    def test_picks_cluster_among_instances(self):
        s = ('{"instance_id":"i0","tokens_per_sec":10.0}\n'
             '{"instance_id":"cluster","tokens_per_sec":42.0}\n')
        self.assertEqual(run_blis.parse_metrics(s).get("tokens_per_sec"), 42.0)

    def test_braces_inside_string_values(self):
        s = '{"instance_id":"cluster","tokens_per_sec":7.0,"path":"a{b}c","err":"oops}"}'
        self.assertEqual(run_blis.parse_metrics(s).get("tokens_per_sec"), 7.0)

    def test_nested_envelope(self):
        self.assertEqual(run_blis.parse_metrics(
            'x {"result":{"instance_id":"cluster","tokens_per_sec":5.0}} y').get("tokens_per_sec"), 5.0)

    def test_no_metrics_returns_empty(self):
        self.assertEqual(run_blis.parse_metrics("no json {not valid}"), {})

    def test_all_instances_excludes_cluster(self):
        s = ('{"instance_id":"i0","tokens_per_sec":1.0}\n{"instance_id":"i1","tokens_per_sec":2.0}\n'
             '{"instance_id":"cluster","tokens_per_sec":3.0}')
        self.assertEqual([o["instance_id"] for o in run_blis.parse_all_instances(s)], ["i0", "i1"])


# A schema 2 Result document trimmed to the sections run_blis reads, with the figures of a real
# two-instance run on the pinned simulator (qwen3-14b on H100, 40 requests, tp 1, two instances).
# The measured window and the whole run disagree on purpose: whole-run throughput and step-interval
# p99 differ from the window's, so a test that reads the wrong block gets the wrong number.
def _m(value, samples=40):
    return {"value": value, "samples": samples}


def _row(i, inst, ttft_ms, e2e_ms, tpot_ms, phase="profile", fate="completed"):
    return {"id": f"request_{i}", "fate": fate, "phase": phase, "instance": inst, "arrival_us": 0,
            "ttft_us": ttft_ms * 1000, "e2e_us": e2e_ms * 1000, "tpot_us": tpot_ms * 1000}


RESULT_DOC = json.dumps({
    "kind": "Result", "schema_version": "1",
    "window": {"start_us": 100, "end_us": 4_900_000, "duration_s": 4.8, "requests": 8},
    "requests": {"generated": 8, "completed": 8, "queued_at_horizon": 0,
                 "in_flight_at_horizon": 0, "timed_out": 0, "length_capped": 0},
    "metrics": {"output_tok_s": _m(106.83, 8), "request_throughput": _m(1.67, 8),
                "ttft_mean_ms": _m(300.0, 8), "ttft_p99_ms": _m(340.0, 8),
                "e2e_mean_ms": _m(2000.0, 8), "e2e_p99_ms": _m(2400.0, 8),
                "step_interval_mean_ms": _m(27.8, 400), "step_interval_p99_ms": _m(33.9, 400),
                # The per-request time per output token disagrees with the engine's step interval, so
                # reading the wrong one fails with a number: TPOT passes a 20 ms gate (18 ms), the
                # step interval does not (33.9 ms).
                "tpot_mean_ms": _m(15.0, 8), "tpot_p50_ms": _m(14.5, 8), "tpot_p90_ms": _m(16.5, 8),
                "tpot_p95_ms": _m(17.2, 8), "tpot_p99_ms": _m(18.0, 8),
                "preemptions": _m(1, 400), "interactivity_mean": _m(57.2, 8),
                "interactivity_p50": _m(57.1, 8), "ttft_p50_ms": _m(290.0, 8)},
    "engine": {"kind": "vllm", "version": "0.32.0"},
    "request_rows": [
        _row(0, "instance_0", 100, 1000, 20), _row(1, "instance_0", 200, 1500, 21),
        _row(2, "instance_1", 300, 2000, 22), _row(3, "instance_1", 900, 2600, 25),
        _row(4, "instance_0", 5000, 9000, 40, phase="warmup"),
        _row(5, "instance_1", 7000, 9500, 41, phase="warmup")],
    "reports": {"engine": {"cluster": {
        "preemptions": 2, "kv_allocation_failures": 0, "scheduling_delay_p99_ms": 1.5,
        "total_input_tokens": 32768, "total_output_tokens": 512, "simulated_duration_s": 5.29,
        "step_interval_p99_ms": 111.0,
        "whole_run": {"tokens_per_sec": 96.80435952382842, "responses_per_sec": 1.5125681175598191,
                      "ttft_p99_ms": 349.06543, "ttft_mean_ms": 342.987,
                      "e2e_p99_ms": 2479.9476, "e2e_mean_ms": 2097.138625,
                      "step_interval_mean_ms": 31.0, "step_interval_p90_ms": 16.729,
                      "step_interval_p95_ms": 16.729, "step_interval_p99_ms": 294.031}},
        "instances": {"instance_0": {"instance_id": "instance_0", "tokens_per_sec": 1.0,
                                     "ttft_p99_ms": 6000.0, "itl_p99_ms": 400.0},
                      "instance_1": {"instance_id": "instance_1", "tokens_per_sec": 2.0,
                                     "ttft_p99_ms": 8000.0, "itl_p99_ms": 500.0}}}}})


class MeasuredWindow(unittest.TestCase):
    """Every figure the report ranks and gates on is the measured window (CFG-4, DEV-8).

    The window and the whole run differ in RESULT_DOC on every figure below, so reading the wrong
    block fails each test with a number, not a shrug.
    """

    def test_throughput_is_the_measured_windows_not_the_whole_runs(self):
        m = run_blis.parse_metrics(RESULT_DOC)
        self.assertEqual(m["tokens_per_sec"], 106.83)
        self.assertNotEqual(m["tokens_per_sec"], 96.80435952382842)
        self.assertEqual(m["responses_per_sec"], 1.67)

    def test_latency_is_the_measured_windows(self):
        m = run_blis.parse_metrics(RESULT_DOC)
        self.assertEqual((m["ttft_p99_ms"], m["e2e_p99_ms"]), (340.0, 2400.0))
        self.assertEqual((m["ttft_mean_ms"], m["e2e_mean_ms"]), (300.0, 2000.0))

    def test_the_itl_gate_is_the_registrys_per_request_tpot_not_the_step_interval(self):
        m = run_blis.parse_metrics(RESULT_DOC)
        self.assertEqual((m["itl_p99_ms"], m["itl_mean_ms"]), (18.0, 15.0))
        self.assertEqual((m["tpot_p99_ms"], m["tpot_mean_ms"]), (18.0, 15.0))
        self.assertNotEqual(m["itl_p99_ms"], 33.9)         # the measured window's step interval
        self.assertNotEqual(m["itl_p99_ms"], 294.031)      # the whole run's
        self.assertNotEqual(m["itl_p99_ms"], 111.0)        # the cluster block's own copy

    def test_a_result_whose_tpot_passes_and_step_interval_fails_clears_a_20_ms_gate(self):
        doc = json.loads(RESULT_DOC)
        doc["metrics"]["tpot_p99_ms"] = _m(10.0, 8)
        doc["metrics"]["step_interval_p99_ms"] = _m(100.0, 400)
        rec = {"metrics": run_blis.parse_metrics(json.dumps(doc)), "metrics_all_instances": []}
        self.assertEqual(rec["metrics"]["itl_p99_ms"], 10.0)
        worst, missing = run_blis.worst_itl_p99(rec)
        self.assertEqual((worst, missing), (10.0, False))
        self.assertAlmostEqual(run_blis.per_user_rate(rec)[0], 100.0)

    def test_a_document_without_a_measured_window_yields_nothing(self):
        # Never a fall back to the whole run: the run fails as "no cluster metrics parsed".
        doc = json.loads(RESULT_DOC)
        del doc["metrics"]
        self.assertEqual({}, run_blis.parse_metrics(json.dumps(doc)))
        doc["metrics"] = {}
        self.assertEqual({}, run_blis.parse_metrics(json.dumps(doc)))

    def test_no_code_path_reads_whole_run_for_display_ranking_or_feasibility(self):
        # The record keeps the whole-run aggregate as raw data under its own key; no other line of
        # any module that displays, ranks or gates mentions it outside a comment.
        src_dir = os.path.dirname(os.path.abspath(run_blis.__file__))
        for name in ("run_blis.py", "run_experiment.py", "report.py", "explore_data.py", "build_gallery.py"):
            with open(os.path.join(src_dir, name), encoding="utf-8") as f:
                code = [l.strip() for l in f.read().splitlines() if not l.lstrip().startswith("#")]
            hits = [l for l in code if "whole_run" in l]
            allowed = {'whole = (_engine_report(doc).get("cluster") or {}).get("whole_run")',
                       'rec["whole_run"] = whole'} if name == "run_blis.py" else set()
            self.assertEqual([l for l in hits if l not in allowed], [], name)

    def test_completion_counters_follow_the_requests_block(self):
        m = run_blis.parse_metrics(RESULT_DOC)
        self.assertEqual((m["injected_requests"], m["completed_requests"]), (8, 8))
        self.assertIsNone(run_blis._completion_issue(m))

    def test_an_overloaded_result_is_still_caught(self):
        doc = json.loads(RESULT_DOC)
        doc["requests"].update({"completed": 6, "queued_at_horizon": 2})
        issue = run_blis._completion_issue(run_blis.parse_metrics(json.dumps(doc)))
        self.assertIsNotNone(issue)
        self.assertIn("injected=8", issue)

    def test_run_totals_that_are_counts_survive(self):
        m = run_blis.parse_metrics(RESULT_DOC)
        self.assertEqual(m["preemption_count"], 1)
        self.assertNotIn("total_output_tokens", m)     # whole-run totals nothing reads are not copied

    def test_instances_are_rebuilt_from_profile_rows_not_read_from_the_whole_run_blocks(self):
        inst = run_blis.parse_all_instances(RESULT_DOC)
        self.assertEqual([o["instance_id"] for o in inst], ["instance_0", "instance_1"])
        by = {o["instance_id"]: o for o in inst}
        # instance_1's profile requests: ttft 300 and 900 ms. Its warm-up request (7000 ms) and the
        # whole-run block's 8000 ms are both outside the window.
        self.assertEqual(by["instance_1"]["requests"], 2)
        self.assertAlmostEqual(by["instance_1"]["ttft_p99_ms"], 300 + 0.99 * 600)
        self.assertAlmostEqual(by["instance_0"]["ttft_p99_ms"], 100 + 0.99 * 100)
        # The gate's slot holds each instance's per-request TPOT (instance_0: 20 and 21 ms), not a step quantity.
        self.assertAlmostEqual(by["instance_0"]["tpot_p99_ms"], 20 + 0.99)
        self.assertEqual(by["instance_0"]["itl_p99_ms"], by["instance_0"]["tpot_p99_ms"])

    def test_grace_requests_inside_the_window_count_and_requests_past_it_do_not(self):
        doc = json.loads(RESULT_DOC)
        doc["request_rows"] = [
            _row(0, "instance_0", 100, 1000, 20),
            dict(_row(1, "instance_0", 900, 1500, 21), phase="grace", arrival_us=0),
            dict(_row(2, "instance_0", 9000, 9999, 30), phase="grace", arrival_us=4_000_000),   # ends at 14 s, past 4.9 s
            _row(3, "instance_0", 7000, 9500, 41, phase="warmup")]
        for r in doc["request_rows"][:1]:
            r["arrival_us"] = 0
        by = {o["instance_id"]: o for o in run_blis.measured_instances(doc)}
        self.assertEqual(2, by["instance_0"]["requests"])
        self.assertAlmostEqual(892.0, by["instance_0"]["ttft_p99_ms"])      # 100 and 900 ms, linear p99

    def test_the_worst_instance_ttft_is_judged_on_the_window(self):
        rec = {"metrics": run_blis.parse_metrics(RESULT_DOC),
               "metrics_all_instances": run_blis.parse_all_instances(RESULT_DOC)}
        worst, missing = run_blis.worst_metric(rec, "ttft_p99_ms")
        self.assertFalse(missing)
        self.assertAlmostEqual(worst, 894.0)      # instance_1, profile phase; not 8000 (whole run)

    def test_the_percentile_is_the_results_own_method(self):
        # numpy's `linear`: 112-sample p99 of a real run matched Result.metrics to every digit.
        xs = list(range(1, 101))
        self.assertAlmostEqual(run_blis._percentile(xs, 99), 99.01)
        self.assertEqual(run_blis._percentile([5.0], 99), 5.0)
        self.assertEqual(run_blis._percentile([1.0, 3.0], 50), 2.0)

    def test_an_itl_slo_can_be_judged_at_all(self):
        rec = {"metrics": run_blis.parse_metrics(RESULT_DOC), "metrics_all_instances": []}
        worst, missing = run_blis.worst_itl_p99(rec)
        self.assertFalse(missing, "a single-instance run must still measure ITL")
        self.assertEqual(worst, 18.0)       # TPOT p99, not the step interval's 33.9

    def test_a_pre_v0150_flat_output_still_parses(self):
        # A contributor testing an older binary: the flat MetricsOutput shape is read as written.
        s = '{"instance_id":"cluster","tokens_per_sec":11.0,"itl_p99_ms":5.47}'
        m = run_blis.parse_metrics(s)
        self.assertEqual((m["tokens_per_sec"], m["itl_p99_ms"]), (11.0, 5.47))


class RecordFields(unittest.TestCase):
    """What a run record keeps of the Result (CFG-4, DEV-8): the measured window verbatim, the basis,
    the whole run as raw data, and a configuration key that leaves the load out."""

    def test_measured_metrics_are_copied_verbatim_with_the_result_names(self):
        doc = json.loads(RESULT_DOC)
        got = run_blis.measured_record(doc)
        self.assertEqual(got["metrics"], doc["metrics"])
        for name in ("interactivity_mean", "interactivity_p50", "ttft_p50_ms"):
            self.assertIn(name, got["metrics"])
        self.assertEqual(got["window"], doc["window"])
        self.assertEqual([o["instance_id"] for o in got["instances"]], ["instance_0", "instance_1"])

    def test_metric_defs_are_copied_when_the_document_has_them_and_absent_otherwise(self):
        doc = json.loads(RESULT_DOC)
        self.assertNotIn("metric_defs", run_blis.measured_record(doc))
        doc["metric_defs"] = {"ttft_p99_ms": {"formula": "ttft", "statistic": "p99"}}
        self.assertEqual(run_blis.measured_record(doc)["metric_defs"], doc["metric_defs"])

    def test_a_result_with_no_metrics_block_has_no_measured_field(self):
        doc = json.loads(RESULT_DOC)
        del doc["metrics"]
        self.assertIsNone(run_blis.measured_record(doc))
        doc["metrics"] = {}
        self.assertIsNone(run_blis.measured_record(doc))

    def test_config_key_leaves_the_load_out_and_config_hash_keeps_it(self):
        a = run_blis._record_shell({"tp": 4, "num_instances": 2, "concurrency": 8}, "p")
        b = run_blis._record_shell({"tp": 4, "num_instances": 2, "concurrency": 64}, "p")
        c = run_blis._record_shell({"tp": 8, "num_instances": 2, "concurrency": 8}, "p")
        self.assertEqual(a["config_key"], b["config_key"])
        self.assertNotEqual(a["config_hash"], b["config_hash"])
        self.assertNotEqual(a["config_key"], c["config_key"])
        # config_hash is unchanged by this field: it is still the hash of the whole config.
        self.assertEqual(a["config_hash"], run_blis._config_hash({"tp": 4, "num_instances": 2, "concurrency": 8}))

    def test_a_ledger_gives_one_key_per_configuration_across_its_loads(self):
        recs = [run_blis._record_shell({"tp": tp, "num_instances": n, "concurrency": c}, "p")
                for tp in (4, 8) for n in (1, 2) for c in (4, 8, 16)]
        self.assertEqual(len({r["config_hash"] for r in recs}), 12)
        self.assertEqual(len({r["config_key"] for r in recs}), 4)


class CompletionIssue(unittest.TestCase):
    def test_clean(self):
        self.assertIsNone(run_blis._completion_issue(
            {"injected_requests": 100, "completed_requests": 100, "length_capped_requests": 5}))

    def test_incomplete(self):
        self.assertIn("incomplete", run_blis._completion_issue(
            {"injected_requests": 100, "completed_requests": 90}))

    def test_overloaded(self):
        self.assertIn("overloaded", run_blis._completion_issue(
            {"injected_requests": 100, "completed_requests": 100, "timed_out_requests": 3}))


class ClassifyConfig(unittest.TestCase):
    def test_splits_and_fills_parallel_defaults(self):
        par, eng, cli, _ = classify(
            {"tp": 8, "enable_expert_parallel": True, "block_size": 32, "num_instances": 2, "concurrency": 64})
        self.assertEqual((par["tp"], par["pp"], par["dp"]), (8, 1, 1))
        self.assertTrue(par["enable_expert_parallel"])
        self.assertEqual(eng["block_size"], 32)
        self.assertEqual(cli, {"num_instances": 2, "concurrency": 64})

    def test_precision_not_defaulted(self):
        _, eng, _, _ = classify({"tp": 8})
        self.assertNotIn("quantization", eng)       # empty precision lets the checkpoint govern
        self.assertNotIn("cache_dtype", eng)

    def test_block_size_default_is_model_aware(self):
        # block_size cannot be a single constant. The kernel adapter REQUIRES a positive
        # block_size on the deployment ("deployment states no block_size"), so it must be stated;
        # but a DSA sparse-MLA model runs a 64-token indexer and refuses a stated 16 ("a stated
        # size must be a multiple of 64, and the engine refuses this layout at startup"). Our flat
        # default of 16 therefore made the whole GLM-5.x family unrunnable.
        self.assertEqual(run_blis.default_block_size(None, "qwen3-14b"), 16)
        self.assertEqual(run_blis.DSA_BLOCK_SIZE, 64)
        # a stated value always wins — it stays a searchable knob
        _, eng, _, _ = classify({"tp": 8, "block_size": 64})
        self.assertEqual(eng["block_size"], 64)

    def test_dsa_detection_from_catalog_graph(self):
        # DSA is a property of the model GRAPH (a sparse_mla attention node), not of its name.
        import tempfile
        d = tempfile.mkdtemp()
        for name, body in (("dsa-model", "nodes:\n  - op: Attention\n    kind: sparse_mla\n"),
                           ("plain-model", "nodes:\n  - op: Attention\n    kind: gqa\n")):
            md = os.path.join(d, "models", name)
            os.makedirs(md)
            with open(os.path.join(md, "graph.yaml"), "w") as f:
                f.write(body)
        self.assertTrue(run_blis.is_dsa_model(d, "dsa-model"))
        self.assertFalse(run_blis.is_dsa_model(d, "plain-model"))
        self.assertEqual(run_blis.default_block_size(d, "dsa-model"), 64)
        self.assertEqual(run_blis.default_block_size(d, "plain-model"), 16)
        # an absent graph must not raise -- it falls back to the non-DSA default
        self.assertFalse(run_blis.is_dsa_model(d, "no-such-model"))
        self.assertEqual(run_blis.default_block_size(d, "no-such-model"), 16)

    def test_scenario_always_states_a_block_size(self):
        # the adapter refuses a deployment with no block_size, so the generated YAML must carry one
        par = {"tp": 8, "pp": 1, "dp": 1, "enable_expert_parallel": True}
        eng = dict(run_blis.ENGINE_DEFAULTS, block_size=64)
        self.assertIn("block_size: 64", run_blis.scenario_yaml("glm-5.3", "h200", "0.32.0", par,
                                                               eng, "n", 1))

    def test_precision_passes_through_when_set(self):
        _, eng, _, _ = classify({"tp": 8, "quantization": "fp8", "cache_dtype": "fp8"})
        self.assertEqual(eng["quantization"], "fp8")

    def test_policy_knobs_go_to_cli(self):
        _, _, cli, _ = classify(
            {"tp": 8, "scheduler": "fcfs", "preemption_policy": "fcfs"})
        self.assertEqual(cli, {"scheduler": "fcfs", "preemption_policy": "fcfs"})

    def test_rejects_routing_flags_removed_in_v0_12_0(self):
        """v0.12.0 moved routing into the embedded llm-d router, so the old flags are gone.

        They must fail LOUDLY with the migration path: silently dropping them would run an
        all-defaults router while the ledger row still claimed a routing policy.
        """
        for key in ("routing_policy", "routing_scorers", "admission_policy"):
            with self.assertRaises(run_blis.ConfigError) as cm:
                classify({"tp": 8, key: "least-loaded"})
            self.assertIn("router_config", str(cm.exception))

    def test_router_config_goes_to_the_scenario(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml") as epp:
            _, _, cli, router = classify({"tp": 8, "router_config": epp.name})
            self.assertEqual(router, {"router_config": epp.name})
            self.assertEqual(cli, {})            # it is a SCENARIO key, never a CLI flag

    def test_router_config_must_exist(self):
        with self.assertRaises(run_blis.ConfigError):
            classify({"tp": 8, "router_config": "/nonexistent/epp.yaml"})

    def test_rejects_unknown_key(self):
        with self.assertRaises(run_blis.ConfigError):
            classify({"tp": 8, "totally_made_up": 1})

    def test_pd_keys_need_both_instance_counts(self):
        """P/D is supported now, but a candidate naming per-role layout without the instance
        counts describes no deployment -- BLIS refuses a pool with no counterpart."""
        with self.assertRaises(run_blis.ConfigError):
            classify({"prefill_tp": 4, "decode_tp": 8})

    def test_engine_names_come_from_the_field_list(self):
        """The runner keeps no list of settings (CFG-3): a name is an engine setting exactly
        when the simulator's field list has it. Removing it from the field list refuses it."""
        _par, eng, _cli, _r = classify({"tp": 8, "max_num_seqs": 64})
        self.assertEqual(64, eng["max_num_seqs"])
        doc = blis_cli.fields(FIXTURE_BLIS, "vllm")
        trimmed = dict(doc, fields=[f for f in doc["fields"] if f["name"] != "max_num_seqs"])
        vocab = settings.vocabulary_from(trimmed, blis_cli.schema(FIXTURE_BLIS, "deployment"))
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config({"tp": 8, "max_num_seqs": 64}, vocab)
        self.assertIn("unknown key", str(e.exception))

    def test_layout_names_come_from_the_deployment_schema(self):
        par, _eng, _cli, _r = classify({"tp": 8, "dcp": 2})
        self.assertEqual(2, par["dcp"])
        schema = blis_cli.schema(FIXTURE_BLIS, "deployment")
        props = dict(schema["$defs"]["deployment.Parallelism"]["properties"])
        props.pop("dcp")
        trimmed = {"$defs": {"deployment.Parallelism": {"properties": props}}}
        vocab = settings.vocabulary_from(blis_cli.fields(FIXTURE_BLIS, "vllm"), trimmed)
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "dcp": 2}, vocab)

    def test_accepts_long_prefill(self):
        """`long_prefill_token_threshold` IS a field of the pinned blis-schemas (v0.5.0) Engine
        struct, and a run naming it completes and moves the number (measured 1983.2 -> 1973.9
        tok/s at 256 on chatbot tp8 c=8). It was listed unsupported on the strength of a stale
        note about the kernel, which cost the search a real prefill-admission knob."""
        _par, engine, _cli, _router = classify(
            {"tp": 8, "long_prefill_token_threshold": 4096})
        self.assertEqual(4096, engine["long_prefill_token_threshold"])

    def test_accepts_pp_gt_1(self):
        """pp>1 was refused through kernel v0.4.0 and is PRICED from v0.5.0, so it must pass.

        The refusal is gone rather than narrowed: there is no shape of pp this layer still turns
        away. Verified against the pinned binary on the Muse armB layout -- tp8 x pp2 = 96.80
        tok/s against tp16 x pp1 = 99.47 at the same 16-GPU footprint with EP held on -- so the
        two layouts price apart and pp is a lever rather than a no-op.
        """
        par, _, _, _ = classify({"tp": 8, "pp": 2})
        self.assertEqual((par["tp"], par["pp"]), (8, 2))
        self.assertEqual(run_blis.footprint_gpus(par), 16)

    def test_pp_reaches_the_scenario_yaml(self):
        """A swept pp must land in the pool's `parallel:` block, not be dropped on the floor.

        The inert-knob lesson of the v0.14.0 bump: a knob that decodes but never arrives costs a
        whole campaign and reads as a finding. pp is a GPU-consuming factor, so a dropped pp would
        also mis-size the cluster.
        """
        y = run_blis.scenario_yaml("muse-spark-open-816b-a42b-nvfp4", "h100", "0.32.0",
                                   {"tp": 8, "pp": 2, "dp": 1, "enable_expert_parallel": True},
                                   {"max_model_len": 262144}, "muse-pp2tp8")
        self.assertIn("pp: 2", y)
        self.assertIn("nodes: 2", y)          # tp8 x pp2 = 16 GPUs = 2 whole nodes
        self.assertIn("fabric:", y)           # multi-node, so a fabric is required
        self.assertIn("cost-model-fabric", y)

    def test_coefficients_include_fabric(self):
        """cost-model-fabric prices cross-node links and is what REGISTRY_REF v0.3.0 adds.

        A scenario omitting it runs rc=0 and prices the interconnect differently (48.61 vs 48.38
        tok/s measured on Muse armB), so its absence would be a silent mispricing.
        """
        self.assertIn("cost-model-fabric", run_blis.COEFFICIENTS)

    def test_carries_the_wider_engine_surface(self):
        """Every engine setting the field list names reaches the scenario's engine block."""
        cfg = {"tp": 8, "num_instances": 1, "cudagraph_mode": "FULL",
               "max_cudagraph_capture_size": 256, "enable_prefix_caching": False,
               "scheduling_policy": "priority", "async_scheduling": True,
               "disable_cascade_attn": True,
               "all2all_backend": "deepep_low_latency", "moe_backend": "triton",
               "long_prefill_token_threshold": 2048}
        _par, engine, _cli, _router = classify(cfg)
        for key, want in cfg.items():
            if key in ("tp", "num_instances"):
                continue
            self.assertEqual(want, engine[key], key)

    def test_booleans_and_int_lists_render_as_valid_yaml(self):
        """json.dumps gives YAML-valid `true`/`false` and `[1, 2, 4]`; a Python repr would not."""
        par, engine, _cli, _r = classify(
            {"tp": 8, "enable_prefix_caching": False, "async_scheduling": True,
             "cudagraph_capture_sizes": [1, 2, 4]})
        doc = run_blis.scenario_yaml("qwen3-14b", "h100", "0.32.0", par, engine, "x", 1)
        self.assertIn("enable_prefix_caching: false", doc)
        self.assertIn("async_scheduling: true", doc)
        self.assertIn("cudagraph_capture_sizes: [1, 2, 4]", doc)
        self.assertNotIn("False", doc)
        self.assertNotIn("True", doc)


class FieldListPath(unittest.TestCase):
    """The engine's field list is the file --fields names, until the binary prints it itself
    (CFG-3): no ambient environment variable and no network read decides it."""

    def doc(self):
        return blis_cli.fields(FIXTURE_BLIS, "vllm")

    def write(self, doc):
        import json
        import tempfile
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d)
        path = os.path.join(d, "fields.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        return path

    def test_the_named_file_is_the_field_list(self):
        doc = self.doc()
        trimmed = dict(doc, fields=[f for f in doc["fields"] if f["name"] != "max_num_seqs"])
        vocab = settings.vocabulary(FIXTURE_BLIS, self.write(trimmed))
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config({"tp": 8, "max_num_seqs": 64}, vocab)
        self.assertIn("unknown key", str(e.exception))
        # The binary's own list still has it: the file, not the binary, decided.
        run_blis.classify_config({"tp": 8, "max_num_seqs": 64}, settings.vocabulary(FIXTURE_BLIS))

    def test_the_engine_release_is_the_files(self):
        doc = self.doc()
        other = dict(doc, engine=dict(doc["engine"], version="9.9.9"))
        self.assertEqual("9.9.9", blis_cli.engine_release(FIXTURE_BLIS, "vllm", self.write(other)))

    def test_an_environment_variable_does_not_name_the_file(self):
        doc = self.doc()
        other = dict(doc, engine=dict(doc["engine"], version="9.9.9"))
        old = os.environ.get("BLIS_FIELDS")
        os.environ["BLIS_FIELDS"] = self.write(other)
        blis_cli._CACHE.clear()      # a cached read would hide the loader
        try:
            self.assertEqual(doc["engine"]["version"], blis_cli.engine_release(FIXTURE_BLIS, "vllm"))
        finally:
            if old is None:
                del os.environ["BLIS_FIELDS"]
            else:
                os.environ["BLIS_FIELDS"] = old
            blis_cli._CACHE.clear()

    def test_a_file_that_is_not_a_field_list_is_refused(self):
        with self.assertRaises(blis_cli.BlisCommandError):
            blis_cli.fields(FIXTURE_BLIS, "vllm", self.write({"kind": "Scenario"}))
        with self.assertRaises(blis_cli.BlisCommandError):
            blis_cli.fields(FIXTURE_BLIS, "vllm", "/nonexistent/fields.json")


class Footprint(unittest.TestCase):
    def test_tp_pp_dp(self):
        self.assertEqual(run_blis.footprint_gpus({"tp": 8, "pp": 1, "dp": 2}), 16)
        self.assertEqual(run_blis.footprint_gpus({"tp": 4, "pp": 1, "dp": 1}), 4)

    def test_within_node_dp_allowed(self):
        _, _, _, _ = classify({"tp": 4, "dp": 2})          # footprint 8 == one node, OK

    def test_multinode_single_instance_allowed_on_whole_nodes(self):
        # A multi-node single instance is REAL in BLIS: kimi-k3 needs a 32-GPU footprint to fit its
        # weights (332.3 GiB/rank at tp=8 on a 141 GiB device) and runs at tp=32 across 4 nodes
        # once the scenario names a fabric. The runner used to refuse it, which made the largest
        # models unreachable rather than slow.
        par, _, _, _ = classify({"tp": 32, "enable_expert_parallel": True})
        self.assertEqual(run_blis.footprint_gpus(par), 32)
        par2, _, _, _ = classify({"tp": 8, "dp": 2})
        self.assertEqual(run_blis.footprint_gpus(par2), 16)

    def test_multinode_footprint_must_fill_whole_nodes(self):
        # A footprint that straddles a node boundary (e.g. 12 GPUs = 1.5 nodes) describes no real
        # placement: the pool is allocated in whole nodes, so the remainder would sit idle and the
        # cross-node cost would be mispriced. Refuse it rather than silently round the pool up.
        with self.assertRaises(run_blis.ConfigError):
            classify({"tp": 2, "dp": 6})                # 12 GPUs: 1.5 nodes
        # The node size is the chip's, not a constant: 12 GPUs fill three 4-GPU nodes.
        classify({"tp": 2, "dp": 6}, node_size=4)
        with self.assertRaises(run_blis.ConfigError):
            classify({"tp": 2, "dp": 6}, node_size=8)

    def test_multinode_scenario_names_a_fabric(self):
        # Cross-node cost resolves from the fabric, not the chip; BLIS refuses a multi-node
        # cluster that does not name one.
        par, eng, _, _ = classify({"tp": 32, "enable_expert_parallel": True})
        eng.setdefault("block_size", 64)
        y = run_blis.scenario_yaml("kimi-k3", "h200", "0.32.0", par, eng, "s", 1)
        self.assertIn("fabric:", y)
        self.assertIn("nodes: 4", y)                                    # 32 GPUs / 8 per node


class ScenarioYaml(unittest.TestCase):
    def test_omits_unset_precision_lists_memory_coeff(self):
        par, eng, _, _ = classify({"tp": 8})
        y = run_blis.scenario_yaml("qwen3-14b", "h100", "0.32.0", par, eng, "s")
        self.assertIn("kind: Deployment", y)
        self.assertNotIn("quantization", y)          # BF16 checkpoint governs -- no forced FP8
        self.assertIn("cost-model-memory", y)


class PoolSizing(unittest.TestCase):
    """The pool must hold EVERY instance, and a multi-node cluster must name a fabric. Sizing the
    pool from one instance capped searches at a single node (BLIS refuses extra instances)."""

    def test_pool_nodes_covers_all_instances(self):
        par = {"tp": 4, "pp": 1, "dp": 1}
        self.assertEqual(run_blis.pool_nodes(par, 1), 1)    # 4 GPUs -> 1 node
        self.assertEqual(run_blis.pool_nodes(par, 2), 1)    # 8 GPUs -> 1 node
        self.assertEqual(run_blis.pool_nodes(par, 8), 4)    # 32 GPUs -> 4 nodes
        self.assertEqual(run_blis.pool_nodes({"tp": 8, "pp": 1, "dp": 1}, 16), 16)  # 128 GPUs

    def test_multinode_scenario_names_a_fabric(self):
        par, eng = {"tp": 8, "pp": 1, "dp": 1}, {"block_size": 16}
        many = run_blis.scenario_yaml("m", "h100", "0.32.0", par, eng, "n", num_instances=16)
        self.assertIn("nodes: 16", many)
        self.assertIn("fabric: ", many)                     # required once nodes > 1
        one = run_blis.scenario_yaml("m", "h100", "0.32.0", par, eng, "n", num_instances=1)
        self.assertIn("nodes: 1", one)
        self.assertNotIn("fabric:", one)                    # single node prices no cross-node traffic

    def test_instance_count_changes_the_scenario_identity(self):
        """Two runs differing only in instance count need different files, else the second reuses
        a pool too small for it."""
        import tempfile
        d = tempfile.mkdtemp()
        par, eng = {"tp": 4, "pp": 1, "dp": 1}, {"block_size": 16}
        a = run_blis.write_scenario(FIXTURE_BLIS, d, "m", "h100", "0.32.0", par, eng, 1)
        b = run_blis.write_scenario(FIXTURE_BLIS, d, "m", "h100", "0.32.0", par, eng, 8)
        self.assertNotEqual(a, b)


class FabricSelection(unittest.TestCase):
    """A multi-node scenario is priced over the chosen fabric; omitting one keeps today's IB
    default, so existing specs are byte-identical (issue #71)."""

    def test_chosen_fabric_reaches_the_scenario_yaml(self):
        par, eng = {"tp": 8, "pp": 1, "dp": 1}, {"block_size": 16}
        y = run_blis.scenario_yaml("m", "h100", "0.32.0", par, eng, "n", num_instances=16,
                                   fabric="roce-200g")
        self.assertIn("fabric: roce-200g", y)

    def test_default_fabric_is_ib_400g(self):
        par, eng = {"tp": 8, "pp": 1, "dp": 1}, {"block_size": 16}
        y = run_blis.scenario_yaml("m", "h100", "0.32.0", par, eng, "n", num_instances=16)
        self.assertIn(f"fabric: {run_blis.DEFAULT_FABRIC}", y)
        self.assertEqual(run_blis.DEFAULT_FABRIC, "ib-400g")

    def test_multinode_fabric_changes_the_scenario_identity(self):
        """Two MULTI-NODE runs differing only in fabric must get different scenario files, or the
        second reuses a pool priced on the wrong interconnect."""
        import tempfile
        d = tempfile.mkdtemp()
        par, eng = {"tp": 8, "pp": 1, "dp": 1}, {"block_size": 16}
        a = run_blis.write_scenario(FIXTURE_BLIS, d, "m", "h100", "0.32.0", par, eng, 16, "ib-400g")
        b = run_blis.write_scenario(FIXTURE_BLIS, d, "m", "h100", "0.32.0", par, eng, 16, "roce-200g")
        self.assertNotEqual(a, b)

    def test_cli_exposes_fabric_with_the_ib_default(self):
        import argparse
        p = argparse.ArgumentParser()
        run_blis.add_common_args(p)
        self.assertEqual(p.parse_args(["--scenarios", "s", "--model", "m",
                                       "--hardware", "h100"]).fabric, run_blis.DEFAULT_FABRIC)
        self.assertEqual(p.parse_args(["--scenarios", "s", "--model", "m", "--hardware", "h100",
                                       "--fabric", "roce-200g"]).fabric, "roce-200g")


class BuildCmd(unittest.TestCase):
    def test_kernel_flags_and_cli_knobs(self):
        cmd = run_blis.build_cmd("blis", None, None, "scn", "s.yaml", "chatbot", 100, 0, None, None,
                                 {"num_instances": 2, "concurrency": 64, "scheduler": "fcfs"})
        # v0.10.0 selects the kernel backend from the scenario's `coefficients:` list, NOT a
        # `--latency-model` CLI flag (the flag was removed and is rejected as unknown).
        self.assertNotIn("--latency-model", cmd)
        self.assertNotIn("blis-latency-kernel", cmd)
        for a, b in (("--scenario", "s.yaml"), ("--scenarios", "scn"),
                     ("--num-instances", "2"), ("--concurrency", "64"), ("--scheduler", "fcfs")):
            self.assertEqual(cmd[cmd.index(a) + 1], b)
        for refused in ("--tp", "--model", "--hardware", "--enable-expert-parallel", "--dp"):
            self.assertNotIn(refused, cmd)


class DataEnvironment(unittest.TestCase):
    """The simulator reads BLIS_CATALOG and BLIS_REGISTRY as whole-data replacement paths. The scripts
    name the checkout they read files from SUBSTRATE_CATALOG and SUBSTRATE_REGISTRY, so those never
    collide with the simulator's; and a default run removes the simulator's own two from its child
    environment even when the caller exported them (CFG-1). These run a real child process."""

    def child_env(self, data_override):
        import subprocess
        import sys
        old = {k: os.environ.get(k) for k in run_blis.SIM_DATA_ENV}
        for k in run_blis.SIM_DATA_ENV:
            os.environ[k] = "/some/checkout"
        try:
            env = run_blis.sim_env(data_override)
            out = subprocess.run([sys.executable, "-c",
                                  "import os;print([k for k in ('BLIS_CATALOG','BLIS_REGISTRY') if k in os.environ])"],
                                 capture_output=True, text=True, env=env).stdout
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return out.strip()

    def test_a_default_run_does_not_inherit_the_data_paths(self):
        self.assertEqual("[]", self.child_env(False))

    def test_an_override_keeps_them(self):
        self.assertEqual("['BLIS_CATALOG', 'BLIS_REGISTRY']", self.child_env(True))

    def test_the_data_a_result_names_decides_the_label(self):
        emb = {"source": "embedded"}
        self.assertEqual("embedded", run_blis.data_of({"data": {"catalog": emb, "registry": emb, "released_data": True}}))
        # A development binary prices with embedded data too; released_data is a separate fact.
        self.assertEqual("embedded", run_blis.data_of({"data": {"catalog": emb, "registry": emb, "released_data": False}}))
        self.assertIs(False, run_blis.released_build({"data": {"released_data": False}}))
        self.assertEqual("override", run_blis.data_of({"data": {"catalog": {"source": "path"}, "registry": emb,
                                                                  "released_data": True}}))
        self.assertIsNone(run_blis.data_of({}))


class EmbeddedData(unittest.TestCase):
    """A committed report is priced with the pinned release's embedded catalog and registry (CFG-1):
    the simulator is given neither unless the run is an explicit, experimental override."""

    def cmds(self, catalog, registry):
        run = run_blis.build_cmd("blis", catalog, registry, "scn", "s.yaml", "chatbot", 100, 0, None, None, {})
        replay = run_blis.build_replay_cmd("blis", catalog, registry, "scn", "s.yaml", 0, None, {},
                                           {"scenario": "x", "corpus": "c", "corpus_sha256": "d"})
        return run, replay

    def test_no_data_flag_without_an_override(self):
        for cmd in self.cmds(None, None):
            self.assertNotIn("--catalog", cmd)
            self.assertNotIn("--registry", cmd)

    def test_an_override_names_both_directories(self):
        for cmd in self.cmds("/cat", "/reg"):
            self.assertEqual("/cat", cmd[cmd.index("--catalog") + 1])
            self.assertEqual("/reg", cmd[cmd.index("--registry") + 1])

    def _record(self, **kw):
        seen = []
        real = run_blis.subprocess.run

        class Done:
            returncode, stdout, stderr = 0, "", ""

        def fake(cmd, **k):
            if cmd[1:2] in (["run"], ["replay"]):
                seen.append(cmd)
                return Done()
            return real(cmd, **k)
        run_blis.subprocess.run = fake
        try:
            d = tempfile.mkdtemp()
            cat = os.path.join(d, "catalog")
            os.makedirs(os.path.join(cat, "workloads"))
            with open(os.path.join(cat, "workloads", "chatbot.yaml"), "w") as f:
                f.write("prompt:\n  tokens: 100\n  tokens_stdev: 10\n  tokens_min: 1\n  tokens_max: 200\n"
                        "output:\n  tokens: 50\n  tokens_stdev: 5\n  tokens_min: 1\n  tokens_max: 100\n")
            rec = run_blis.run_one(FIXTURE_BLIS, cat, "/reg", d, "m", "h100", "0.32.0", "chatbot", 10, 0,
                                   None, {"tp": 8, "concurrency": 4}, gpus_per_node=8, **kw)
        finally:
            run_blis.subprocess.run = real
        return rec, seen

    def test_a_default_run_hands_the_simulator_no_data_and_says_embedded(self):
        rec, seen = self._record()
        self.assertEqual("embedded", rec["data"])
        self.assertTrue(seen, "the run command was not built")
        self.assertNotIn("--catalog", seen[0])
        self.assertNotIn("--registry", seen[0])
        self.assertNotIn("catalog", rec["pinned_stack"])

    def test_an_override_run_passes_the_directories_and_is_marked(self):
        rec, seen = self._record(data_override=True)
        self.assertEqual("override", rec["data"])
        self.assertTrue(seen[0][seen[0].index("--catalog") + 1].endswith("catalog"))
        self.assertEqual("/reg", seen[0][seen[0].index("--registry") + 1])

    def test_the_cli_has_the_override_flag_off_by_default(self):
        import argparse
        p = argparse.ArgumentParser()
        run_blis.add_common_args(p)
        base = ["--scenarios", "s", "--model", "m", "--hardware", "h100"]
        self.assertFalse(p.parse_args(base).data_override)
        self.assertTrue(p.parse_args(base + ["--data-override"]).data_override)


class NodeSize(unittest.TestCase):
    """The node size comes from the catalog entry where it fixes one, else the specification (#96)."""

    def test_a_chip_the_catalog_fixes_takes_the_catalogs_size_and_writes_none(self):
        size, writes = run_blis.cluster_node_size(FIXTURE_BLIS, None, "gb200-nvl72")
        self.assertEqual((size, writes), (4, False))
        y = run_blis.scenario_yaml("m", "gb200-nvl72", "0.32.0", dict(run_blis.PARALLEL_DEFAULTS),
                                   {}, "n", num_instances=2, node_size=size, states_node_size=writes)
        self.assertNotIn("gpus_per_node", y)
        self.assertNotIn("gpus_per_rack", y)

    def test_a_chip_with_no_fixed_size_needs_the_specification_to_state_it(self):
        self.assertEqual(run_blis.cluster_node_size(FIXTURE_BLIS, None, "h100", 8), (8, True))
        self.assertEqual(run_blis.cluster_node_size(FIXTURE_BLIS, None, "h200", 4), (4, True))
        with self.assertRaises(run_blis.ConfigError):
            run_blis.cluster_node_size(FIXTURE_BLIS, None, "h100")

    def test_a_stated_size_that_contradicts_the_catalog_is_refused(self):
        with self.assertRaises(run_blis.ConfigError) as cm:
            run_blis.cluster_node_size(FIXTURE_BLIS, None, "gb200-nvl72", 8)
        self.assertIn("omit it", str(cm.exception))

    def test_the_pool_is_sized_in_the_chips_nodes(self):
        par = {"tp": 8, "pp": 1, "dp": 1}
        self.assertEqual(run_blis.pool_nodes(par, 8, node_size=8), 8)
        self.assertEqual(run_blis.pool_nodes(par, 8, node_size=4), 16)
        y = run_blis.scenario_yaml("m", "h200", "0.32.0", dict(run_blis.PARALLEL_DEFAULTS, **par), {}, "n",
                                   num_instances=8, node_size=4, states_node_size=True)
        self.assertIn("gpus_per_node: 4", y)
        self.assertIn("nodes: 16", y)


class RunOneRejects(unittest.TestCase):
    def test_unsupported_config_is_an_error_row_not_a_crash(self):
        rec = run_blis.run_one(FIXTURE_BLIS, "cat", "reg", "/tmp/scn-x", "m", "h100", "0.32.0",
                               None, None, 0, None, {"tp": 8, "prefill_tp": 4})
        self.assertIsNone(rec["returncode"])
        self.assertIn("unsupported config", rec["error"])
        self.assertEqual(rec["metrics"], {})
        # same row SHAPE as a real run so the sweeper and the report never trip on missing keys
        for k in ("wall_s", "deployment", "metrics_all_instances", "stderr_tail", "cmd"):
            self.assertIn(k, rec)
        self.assertEqual(rec["wall_s"], 0.0)


def _write_catalog(root, model, arch=None, sparse_mla=False, hardware=None, cc=None,
                   workloads=None):
    """A minimal on-disk catalog: only the bytes these helpers actually scan."""
    mdir = os.path.join(root, "models", model)
    os.makedirs(mdir, exist_ok=True)
    lines = ["kind: ModelGraph", f"name: {model}", "global:"]
    if arch:
        lines.append(f"  architecture: {arch}")
    if sparse_mla:
        lines.append("  attention_kind: sparse_mla")
    with open(os.path.join(mdir, "graph.yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if hardware:
        hdir = os.path.join(root, "hardware")
        os.makedirs(hdir, exist_ok=True)
        with open(os.path.join(hdir, hardware + ".yaml"), "w") as f:
            f.write('_comment_cc: "ComputeCapability is prose that must NOT be parsed"\n')
            if cc:
                f.write(f'ComputeCapability: "{cc}"\n')
    for name, body in (workloads or {}).items():
        wdir = os.path.join(root, "workloads")
        os.makedirs(wdir, exist_ok=True)
        with open(os.path.join(wdir, name + ".yaml"), "w") as f:
            f.write(body)


class BlockSizeIsModelAndPartAware(unittest.TestCase):
    """block_size cannot be one constant: the kernel's rule depends on BOTH the architecture and
    the part (blis-latency-kernel sparse_mla.go `v4BlockSize`). A flat 64 made every
    DeepSeek-V4.1 Blackwell run fail; a flat 16 made every DSA model fail."""

    def _bs(self, **kw):
        wl = kw.pop("workloads", None)
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m", workloads=wl, **kw)
            run_blis._ARCH_CACHE.clear(); run_blis._CC_CACHE.clear(); run_blis._DSA_CACHE.clear()
            return run_blis.default_block_size(d, "m", kw.get("hardware"))

    def test_v41_on_hopper_is_64(self):
        self.assertEqual(self._bs(arch="DeepseekV41ForCausalLM", hardware="h100", cc="9.0"), 64)

    def test_v41_on_blackwell_is_128(self):
        self.assertEqual(self._bs(arch="DeepseekV41ForCausalLM", hardware="b200", cc="10.0"), 128)
        self.assertEqual(self._bs(arch="DeepseekV41ForCausalLM", hardware="b300", cc="10.3"), 128)

    def test_v4_is_256_on_every_part(self):
        self.assertEqual(self._bs(arch="DeepseekV4ForCausalLM", hardware="h200", cc="9.0"), 256)
        self.assertEqual(self._bs(arch="DeepseekV4ForCausalLM", hardware="b200", cc="10.0"), 256)

    def test_v41_unstated_capability_falls_back_to_hopper(self):
        """The kernel infers the family and records the assumption rather than failing, so the
        runner matches it instead of guessing Blackwell."""
        self.assertEqual(self._bs(arch="DeepseekV41ForCausalLM", hardware="mystery"), 64)

    def test_dsa_non_v4_is_64_and_plain_model_is_16(self):
        self.assertEqual(self._bs(sparse_mla=True, hardware="h100", cc="9.0"), 64)
        self.assertEqual(self._bs(hardware="h100", cc="9.0"), 16)

    def test_comment_prose_is_not_read_as_the_capability(self):
        """hardware/*.yaml carries a _comment_cc line containing the word ComputeCapability; a
        looser scan would read the prose and mis-size the block."""
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m", arch="DeepseekV41ForCausalLM", hardware="b200", cc="10.0")
            run_blis._CC_CACHE.clear()
            self.assertEqual(run_blis.compute_capability_major(d, "b200"), 10)


class WorkloadShapeIsPassedExplicitly(unittest.TestCase):
    """`--workload <preset>` is IGNORED under --concurrency (cmd/root.go takes the concurrency
    branch before the preset loader), so the preset's shape must reach BLIS as explicit flags or
    every run is priced on the 512/512 defaults while claiming the preset."""

    SHAPE = ("prompt:\n  tokens: 10240\n  tokens_stdev: 1200\n  tokens_min: 500\n"
             "  tokens_max: 20480\noutput:\n  tokens: 1536\n  tokens_stdev: 300\n"
             "  tokens_min: 50\n  tokens_max: 4096\n")

    def test_shape_preset_becomes_token_flags(self):
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m", workloads={"multidoc": self.SHAPE})
            run_blis._PRESET_CACHE.clear()
            shape = run_blis.load_workload_shape(d, "multidoc")
        self.assertEqual(shape["--prompt-tokens"], 10240)
        self.assertEqual(shape["--output-tokens"], 1536)
        self.assertEqual(shape["--prompt-tokens-max"], 20480)

    def test_replay_preset_is_refused_not_silently_defaulted(self):
        """A replay has no ISL/OSL by construction. Returning {} would let `blis run` serve the
        CLI defaults under a replay's name -- the exact mislabelling this guards."""
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m", workloads={"agentx-mvp": "replay:\n  scenario: x\n"})
            run_blis._PRESET_CACHE.clear()
            with self.assertRaises(run_blis.ConfigError) as cm:
                run_blis.load_workload_shape(d, "agentx-mvp")
            self.assertIn("REPLAY", str(cm.exception))

    def test_unknown_preset_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m")
            run_blis._PRESET_CACHE.clear()
            with self.assertRaises(run_blis.ConfigError):
                run_blis.load_workload_shape(d, "no-such-preset")

    def test_build_cmd_emits_the_shape_flags(self):
        cmd = run_blis.build_cmd("/b", None, "/reg", "/sc", "s.yaml", "multidoc", 100, 0, None,
                                 None, {}, {"--prompt-tokens": 10240, "--output-tokens": 1536})
        self.assertIn("--prompt-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--prompt-tokens") + 1], "10240")


    def test_a_yaml_result_document_parses_like_the_json_one(self):
        """An infinite grace has no JSON form, so those runs print YAML (run_blis.INF_NEEDS_YAML)."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        doc = json.loads(RESULT_DOC)
        doc["input"] = {"run": {"workload": {"replay": {"grace_period_s": float("inf")}}}}
        s = yaml.safe_dump(doc, sort_keys=False)
        self.assertTrue(s.startswith("kind: Result") and ".inf" in s)
        self.assertEqual(run_blis.parse_metrics(RESULT_DOC), run_blis.parse_metrics(s))
        # A truncated or malformed document fails like a malformed JSON one: no metrics.
        for bad in ("kind: Result\nengine: [\n", "kind: Result\n---\n- 1\n", "kind: Result\n"):
            self.assertEqual({}, run_blis.parse_metrics(bad), bad)


class ReplayCommand(unittest.TestCase):
    REPLAY = {"scenario": "inferencex-agentx-mvp", "corpus": "/c/traces.jsonl",
              "corpus_sha256": "abc123", "max_context_length": "max_model_len",
              "benchmark_duration": "900s", "corpus_cache": "/cache"}

    def test_uses_the_replay_subcommand_with_the_corpus_digest(self):
        cmd = run_blis.build_replay_cmd("/b", "/cat", "/reg", "/sc", "s.yaml", 42, None,
                                        {"concurrency": 16, "num_instances": 2}, self.REPLAY)
        self.assertEqual(cmd[1], "replay")
        self.assertEqual(cmd[cmd.index("--agentic-scenario") + 1], "inferencex-agentx-mvp")
        # The digest is REQUIRED upstream: the harness reads the dataset's default branch, so only
        # the file's bytes tie a run to its corpus.
        self.assertEqual(cmd[cmd.index("--corpus-sha256") + 1], "abc123")
        self.assertEqual(cmd[cmd.index("--concurrency") + 1], "16")

    def test_an_infinite_grace_asks_for_yaml(self):
        for grace, yaml_out in (("inf", True), (".inf", True), ("30s", False)):
            cmd = run_blis.build_replay_cmd("/b", "/cat", "/reg", "/sc", "s.yaml", 42, None, {},
                                            dict(self.REPLAY, grace_period=grace))
            self.assertEqual(cmd[cmd.index("--grace-period") + 1], grace)
            self.assertEqual(yaml_out, "--format" in cmd and cmd[cmd.index("--format") + 1] == "yaml",
                             grace)
        cmd = run_blis.build_replay_cmd("/b", "/cat", "/reg", "/sc", "s.yaml", 42, None, {},
                                        self.REPLAY)
        self.assertNotIn("--format", cmd)

    def test_passes_no_num_requests(self):
        """A replay is bounded by its sending window, not a request count; --num-requests is not
        even a replay flag."""
        cmd = run_blis.build_replay_cmd("/b", "/cat", "/reg", "/sc", "s.yaml", 0, None, {},
                                        self.REPLAY)
        self.assertNotIn("--num-requests", cmd)
        self.assertNotIn("--workload", cmd)

    def test_requires_corpus_and_digest(self):
        class A:
            agentic_scenario = "inferencex-agentx-mvp"
            corpus = None
            corpus_sha256 = None
        with self.assertRaises(ValueError):
            run_blis.replay_from_args(A())

    def test_none_without_an_agentic_scenario(self):
        class A:
            agentic_scenario = None
        self.assertIsNone(run_blis.replay_from_args(A()))


class RouterBlockInScenario(unittest.TestCase):
    def test_router_is_deployment_level_not_pool_level(self):
        """blis-schemas puts Router on Deployment (a sibling of pools:), so a pool-indented block
        would be rejected as an unknown pool field."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as epp:
            y = run_blis.scenario_yaml("m", "h100", "0.32.0", dict(run_blis.PARALLEL_DEFAULTS),
                                       {"block_size": 64}, "n", 1, run_blis.DEFAULT_FABRIC,
                                       {"router_config": epp.name})
        self.assertRegex(y, r"(?m)^router:$")
        self.assertRegex(y, r"(?m)^  endpoint_picker_config: ")

    def test_absent_router_emits_no_block(self):
        y = run_blis.scenario_yaml("m", "h100", "0.32.0", dict(run_blis.PARALLEL_DEFAULTS),
                                   {"block_size": 16}, "n")
        self.assertNotIn("router:", y)

    def test_router_participates_in_the_scenario_hash(self):
        """Two routing arms differ ONLY in the router block: sharing a file would run the second
        arm with the first arm's router."""
        import shutil
        d = tempfile.mkdtemp()
        try:
            with tempfile.NamedTemporaryFile(suffix=".yaml") as a, \
                 tempfile.NamedTemporaryFile(suffix=".yaml") as b:
                args = ("m", "h100", "0.32.0", dict(run_blis.PARALLEL_DEFAULTS),
                        {"block_size": 64}, 1, run_blis.DEFAULT_FABRIC)
                n1 = run_blis.write_scenario(FIXTURE_BLIS, d, *args, {"router_config": a.name})
                n2 = run_blis.write_scenario(FIXTURE_BLIS, d, *args, {"router_config": b.name})
                n0 = run_blis.write_scenario(FIXTURE_BLIS, d, *args, None)
            self.assertEqual(len({n0, n1, n2}), 3)
        finally:
            shutil.rmtree(d)


class SpeculativeDecoding(unittest.TestCase):
    """Drafting is priced by the kernel; see run_blis's speculative comment."""

    def test_depth_becomes_a_nested_engine_block(self):
        _, eng, _, _ = classify({"tp": 4, "num_speculative_tokens": 5})
        self.assertEqual(eng["speculative"], {"num_spec_tokens": 5})

    def test_method_is_carried_when_stated(self):
        _, eng, _, _ = classify(
            {"tp": 4, "num_speculative_tokens": 3, "speculative_method": "dspark"})
        self.assertEqual(eng["speculative"], {"method": "dspark", "num_spec_tokens": 3})

    def test_zero_depth_is_drafting_off(self):
        _, eng, _, _ = classify({"tp": 4, "num_speculative_tokens": 0})
        self.assertNotIn("speculative", eng)

    def test_method_without_depth_is_refused(self):
        """Naming a drafter while drafting 0 tokens would record a drafter that never ran."""
        with self.assertRaises(run_blis.ConfigError):
            classify({"tp": 4, "speculative_method": "dspark"})

    def test_acceptance_selectors_go_to_the_cli(self):
        _, _, cli, _ = classify(
            {"tp": 4, "num_speculative_tokens": 5,
             "speculative_category": "coding", "speculative_thinking": "on"})
        self.assertEqual(cli, {"speculative_category": "coding", "speculative_thinking": "on"})
        cmd = run_blis.build_cmd("/b", None, "/r", "/s", "s.yaml", None, None, 0, None, None, cli)
        self.assertEqual(cmd[cmd.index("--speculative-category") + 1], "coding")
        self.assertEqual(cmd[cmd.index("--speculative-thinking") + 1], "on")

    def test_nested_block_is_rendered_as_yaml_not_json_flow(self):
        """json.dumps on the dict would emit flow style, which the strict loader rejects."""
        y = run_blis.scenario_yaml("m", "b200", "0.32.0", dict(run_blis.PARALLEL_DEFAULTS),
                                   {"block_size": 128,
                                    "speculative": {"method": "dspark", "num_spec_tokens": 5}},
                                   "n")
        self.assertIn("      speculative_config:\n", y)
        self.assertIn('        method: "dspark"', y)
        self.assertIn("        num_speculative_tokens: 5", y)
        self.assertNotIn('{"method"', y)

    def test_the_engine_block_takes_vllms_names_and_the_sweeps_record_keeps_its_own(self):
        """The pin at inference-sim PR 5 refuses the earlier names ("engine key cudagraph_mode is now
        compilation_config.cudagraph_mode"); the move happens where the scenario is rendered."""
        eng = {"cache_dtype": "fp8", "cudagraph_mode": "FULL", "indexer_kv_dtype": "fp8", "block_size": 64,
               "eplb": {"enable_eplb": True, "window_size": 5}, "speculative": {"method": "mtp", "num_spec_tokens": 2}}
        y = run_blis.scenario_yaml("m", "b200", "0.32.0", dict(run_blis.PARALLEL_DEFAULTS), eng, "n")
        for want in ('      kv_cache_dtype: "fp8"', "      compilation_config:\n", '        cudagraph_mode: "FULL"',
                     "      attention_config:\n", "      enable_eplb: true", "      eplb_config:\n",
                     "      speculative_config:\n", "        num_speculative_tokens: 2", "      block_size: 64"):
            self.assertIn(want, y)
        for old in ("cache_dtype:", "  cudagraph_mode:\n", "num_spec_tokens", "\n      speculative:"):
            self.assertNotIn(old if old != "cache_dtype:" else "\n      cache_dtype:", y)
        self.assertEqual(eng["cache_dtype"], "fp8")             # the caller's block is not changed
        self.assertEqual(run_blis.to_vllm_block({"block_size": 8}), {"block_size": 8})

    def test_drafter_is_read_from_the_models_own_graph(self):
        """The right drafter is a property of the MODEL (dspark here, mtp on v4-pro), so it is
        read from graph.yaml rather than hardcoded or demanded from every caller."""
        with tempfile.TemporaryDirectory() as d:
            mdir = os.path.join(d, "models", "m")
            os.makedirs(mdir)
            with open(os.path.join(mdir, "graph.yaml"), "w") as f:
                f.write("kind: ModelGraph\nspeculator:\n  method: dspark\n  num_spec_tokens: 5\n"
                        "stack:\n  pattern: [a]\n")
            run_blis._SPEC_CACHE.clear()
            self.assertEqual(run_blis.graph_speculator(d, "m"), "dspark")

    def test_model_with_no_speculator_reads_empty(self):
        with tempfile.TemporaryDirectory() as d:
            mdir = os.path.join(d, "models", "m")
            os.makedirs(mdir)
            with open(os.path.join(mdir, "graph.yaml"), "w") as f:
                f.write("kind: ModelGraph\nstack:\n  pattern: [a]\n")
            run_blis._SPEC_CACHE.clear()
            self.assertEqual(run_blis.graph_speculator(d, "m"), "")


class AcceptanceLength(unittest.TestCase):
    """The acceptance length is what makes the per-user axis correct under drafting, and BLIS
    never echoes it, so the runner resolves and records it."""

    PROFILES = ("kind: AcceptanceProfileSet\nname: test-al\nprofiles:\n"
                "- model: deepseek-v4.1-flash\n  speculative_method: dspark\n"
                "  num_spec_tokens: 5\n  thinking: 'on'\n  category: coding\n"
                "  acceptance_length: 3.51\n  provenance:\n    hardware: b200\n")

    def _reg(self, d):
        os.makedirs(os.path.join(d, "acceptance"), exist_ok=True)
        with open(os.path.join(d, "acceptance", "p.yaml"), "w") as f:
            f.write(self.PROFILES)
        run_blis._AL_CACHE.clear()

    def test_resolves_the_measured_value_with_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            self._reg(d)
            al, src = run_blis.acceptance_length(
                d, "deepseek-v4.1-flash", "dspark", 5, "on", "coding")
        self.assertEqual(al, 3.51)
        self.assertIn("b200", src)

    def test_unmatched_key_resolves_to_none(self):
        """An unknown acceptance factor must yield None, so the report withholds `pu` rather than
        publishing a rate wrong by an unknown multiple."""
        with tempfile.TemporaryDirectory() as d:
            self._reg(d)
            al, _ = run_blis.acceptance_length(
                d, "deepseek-v4.1-flash", "dspark", 4, "on", "coding")   # K=4 absent
        self.assertIsNone(al)

    def test_registry_without_acceptance_dir_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            run_blis._AL_CACHE.clear()
            self.assertEqual(run_blis.acceptance_length(d, "m", "dspark", 5, "on", "coding"),
                             (None, None))


class GeneratedScenario(unittest.TestCase):
    def test_names_the_engine_by_kind_and_version(self):
        """A generated scenario names its engine as engine: {kind, version} (SIM-1), and the
        version is the release the simulator's field list describes, not a pin of our own."""
        par, eng, _, _ = classify({"tp": 8})
        y = run_blis.scenario_yaml("qwen3-14b", "h100", VOCAB.release, par, eng, "s")
        self.assertIn('engine:\n  kind: vllm\n  version: "0.32.0"\n', y)
        self.assertNotIn("engine_version", y)
        self.assertEqual("0.32.0", blis_cli.engine_release(FIXTURE_BLIS, "vllm"))

    def test_a_refused_scenario_is_never_written(self):
        """write_scenario validates with `blis schema validate scenario` before publishing."""
        par, eng, _, _ = classify({"tp": 8})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(blis_cli.DocumentInvalid):
                run_blis.write_scenario(FIXTURE_BLIS, d, "INVALID-FOR-TEST", "h100", "0.32.0",
                                        par, eng)
            self.assertEqual([], [f for f in os.listdir(d) if not f.startswith(".")])
            name = run_blis.write_scenario(FIXTURE_BLIS, d, "qwen3-14b", "h100", "0.32.0",
                                           par, eng)
            self.assertTrue(os.path.exists(os.path.join(d, name)))

    def test_a_refused_result_is_an_error_row(self):
        self.assertIsNone(run_blis.result_invalid(FIXTURE_BLIS, '{"kind": "Result"}'))
        self.assertIn("refused", run_blis.result_invalid(
            FIXTURE_BLIS, '{"kind": "Result", "name": "INVALID-FOR-TEST"}'))


class FigureForms(unittest.TestCase):
    """Result schema versions and the form of a figure (substrate layer 10: scope, member, aggregate,
    over). The check is keyed on the document's schema_version, so it lights up with the pin bump to a
    simulator that writes version 3: nothing before it is refused, and nothing in version 3 is read
    unchecked. NOT TESTABLE YET against a real binary: no pinned simulator writes version 3."""

    def doc(self, version, **per_figure):
        d = json.loads(RESULT_DOC)
        d["schema_version"] = version
        for name, extra in per_figure.items():
            d["metrics"][name] = {**d["metrics"][name], **extra}
        if int(version) >= run_blis.FIGURE_SCOPE_FROM_SCHEMA:
            for name, m in d["metrics"].items():
                m.setdefault("scope", "cluster")
        return d

    def test_the_versions_read_and_the_one_the_scope_check_starts_at(self):
        self.assertEqual(run_blis.READ_RESULT_SCHEMAS, (1, 2, 3))
        self.assertEqual(run_blis.FIGURE_SCOPE_FROM_SCHEMA, 3)

    def test_version_2_and_earlier_carry_no_scope_and_are_read_as_cluster_figures(self):
        for v in ("1", 2):
            self.assertEqual(run_blis.figure_form_problems(self.doc(v)), [])

    def test_a_version_3_cluster_figure_is_read(self):
        d = self.doc(3)
        self.assertEqual(run_blis.figure_form_problems(d), [])
        self.assertEqual(run_blis.parse_metrics(json.dumps(d))["tokens_per_sec"], 106.83)

    def test_a_version_3_figure_that_is_not_cluster_scope_is_refused_naming_it(self):
        for extra, text in (({"scope": "pool"}, "scope 'pool'"), ({"scope": "instance"}, "scope 'instance'"),
                            ({"aggregate": "max"}, "aggregate 'max'"), ({"over": "pool"}, "over 'pool'"),
                            ({"member": "decode"}, "member 'decode'")):
            d = self.doc(3, ttft_p99_ms=extra)
            got = run_blis.figure_form_problems(d)
            self.assertEqual(len(got), 1, extra)
            self.assertTrue(got[0].startswith("ttft_p99_ms is not a cluster figure") and text in got[0], got)

    def test_a_figure_with_no_scope_is_refused_from_version_3_and_not_before(self):
        d = self.doc(3)
        del d["metrics"]["e2e_p99_ms"]["scope"]
        self.assertEqual(len(run_blis.figure_form_problems(d)), 1)
        d["schema_version"] = 2
        self.assertEqual(run_blis.figure_form_problems(d), [])

    def test_a_version_this_runner_has_not_read_is_refused(self):
        self.assertEqual(run_blis.figure_form_problems(self.doc(4)),
                         ["Result schema_version 4 is not one this runner reads (1, 2, 3)"])

    def test_the_feasibility_aggregate_is_stated_as_max_over_instances(self):
        self.assertEqual(run_blis.FEASIBILITY_AGGREGATE["aggregate"], "max")
        self.assertEqual(run_blis.FEASIBILITY_AGGREGATE["over"], "instance")
        self.assertEqual(sorted(run_blis.FEASIBILITY_AGGREGATE["metrics"]), ["e2e_p99_ms", "itl_p99_ms", "ttft_p99_ms"])


class FailureLine(unittest.TestCase):
    def test_panic_message_beats_its_stack_frames(self):
        err = ("time=x level=info msg=start\npanic: engine build: no pool\n\ngoroutine 1 [running]:\n"
               "main.main()\n\t/upstream/main.go:19 +0x58\n")
        self.assertEqual(run_blis._failure_line(err), "panic: engine build: no pool")

    def test_fatal_log_then_last_line(self):
        self.assertEqual(run_blis._failure_line("a\ntime=x level=fatal msg=\"bad pool\"\nexit 1\n"),
                         'time=x level=fatal msg="bad pool"')
        self.assertEqual(run_blis._failure_line("first\nlast\n"), "last")
        self.assertEqual(run_blis._failure_line(""), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
