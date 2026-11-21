#!/usr/bin/env python3
"""Tests that a spec states the engine's native speculative settings as a nested block and that each
leaf's origin (sweep, scenario, catalog, runner) is recorded recursively and survives a round trip.

Hermetic: the stand-in `blis` of scripts/testdata/blis-m1 supplies the field list.
Run: python3 -I scripts/test_spec_settings.py
"""
import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_blis  # noqa: E402
import run_experiment as E  # noqa: E402
import settings  # noqa: E402

BLIS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "blis-m1", "blis")

SWEEP = {"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
         "engine": {"speculative": {"method": ["dspark"], "num_spec_tokens": [0, 3],
                                    "rejection_sample_method": ["standard", "block"]}}}
SHARED = {"max_num_seqs": 64, "speculative": {"kv_cache_dtype": "fp8",
                                              "enable_adaptive_verification": True}}


def run_config(shared, cand):
    """The config a run gets: the scenario's engine block under the candidate, nested blocks merged."""
    return E.merge_nested(shared, cand)


class NestedNativeSettings(unittest.TestCase):
    def test_native_fields_stay_nested_and_the_control_arm_stays_one(self):
        cands = E._engine_candidates(SWEEP["engine"])
        self.assertEqual([{"num_speculative_tokens": 0}],
                         [c for c in cands if c["num_speculative_tokens"] == 0])
        on = [c for c in cands if c["num_speculative_tokens"] == 3]
        self.assertEqual(["block", "standard"],
                         sorted(c["speculative"]["rejection_sample_method"] for c in on))
        self.assertTrue(all(c["speculative_method"] == "dspark" for c in on))

    def test_a_grid_with_no_native_field_is_the_flat_one_it_was(self):
        cands = E._engine_candidates({"speculative": {"method": ["dspark"], "num_spec_tokens": [1, 3]}})
        self.assertEqual([{"num_speculative_tokens": 1, "speculative_method": "dspark"},
                          {"num_speculative_tokens": 3, "speculative_method": "dspark"}], cands)

    def test_the_fixed_and_swept_fields_of_one_block_merge(self):
        cand = {"num_speculative_tokens": 3, "speculative_method": "dspark",
                "speculative": {"rejection_sample_method": "block"}}
        vocab = settings.vocabulary(BLIS)
        engine = run_blis.classify_config({"tp": 8, **run_config(SHARED, cand)}, vocab)[1]
        self.assertEqual({"method": "dspark", "num_spec_tokens": 3, "kv_cache_dtype": "fp8",
                          "enable_adaptive_verification": True,
                          "rejection_sample_method": "block"}, engine["speculative"])

    def test_drafting_off_records_no_block_and_the_fixed_dict_is_not_mutated(self):
        shared = copy.deepcopy(SHARED)
        vocab = settings.vocabulary(BLIS)
        off = run_blis.classify_config({"tp": 8, **run_config(shared, {"num_speculative_tokens": 0})}, vocab)[1]
        self.assertNotIn("speculative", off)
        on = run_blis.classify_config({"tp": 8, **shared, "num_speculative_tokens": 2}, vocab)[1]
        on["speculative"]["method"] = "mtp"
        self.assertEqual(SHARED, shared)

    def test_the_engines_native_field_list_is_accepted_and_written_in_vllms_names(self):
        vocab = settings.vocabulary(BLIS)
        for name in ("kv_cache_dtype", "draft_sample_method", "rejection_sample_method",
                     "synthetic_acceptance_length", "enable_adaptive_verification"):
            self.assertIn(name, vocab.blocks["speculative"])
        out = run_blis.to_vllm_block({"speculative": {"method": "dspark", "num_spec_tokens": 3,
                                                      "synthetic_acceptance_length": 2.5}})
        self.assertEqual({"speculative_config": {"method": "dspark", "num_speculative_tokens": 3,
                                                 "synthetic_acceptance_length": 2.5}}, out)


class RecursiveProvenance(unittest.TestCase):
    ENGINE = {"max_model_len": 32768, "max_num_seqs": 64, "block_size": 64,
              "speculative": {"method": "dspark", "num_spec_tokens": 3, "kv_cache_dtype": "fp8",
                              "rejection_sample_method": "block"}}

    def origin(self, engine=None):
        swept = E.leaf_paths(SWEEP["engine"]) | {("max_num_seqs",)}
        return E.engine_origin(engine or self.ENGINE, E.leaf_paths(SHARED), swept)

    def test_each_leaf_keeps_its_own_origin_through_the_nesting(self):
        self.assertEqual({"max_model_len": "runner", "max_num_seqs": "sweep", "block_size": "runner",
                          "speculative": {"method": "sweep", "num_spec_tokens": "sweep",
                                          "kv_cache_dtype": "scenario",
                                          "rejection_sample_method": "sweep"}}, self.origin())

    def test_a_method_the_runner_took_from_the_graph_is_the_catalogs(self):
        engine = {"speculative": {"num_spec_tokens": 3, "method": "mtp"}}
        got = E.engine_origin(engine, set(), {("speculative", "num_spec_tokens")})
        self.assertEqual({"speculative": {"num_spec_tokens": "sweep", "method": "catalog"}}, got)

    def test_provenance_round_trips(self):
        origin = self.origin()
        with tempfile.TemporaryDirectory() as d:
            first, second = os.path.join(d, "a.json"), os.path.join(d, "b.json")
            E._dump(first, {"rows": [{"engine_origin": origin}]})
            with open(first, encoding="utf-8") as f:
                parsed = json.load(f)
            E._dump(second, parsed)
            with open(first, "rb") as a, open(second, "rb") as b:
                self.assertEqual(a.read(), b.read())
        self.assertEqual(origin, parsed["rows"][0]["engine_origin"])
        self.assertEqual(E.leaf_paths(self.ENGINE), E.leaf_paths(parsed["rows"][0]["engine_origin"]),
                         "every leaf of the engine has an origin, at the depth it sits")


PD_SWEEP = {"name": "d", "prefill-instances": [1], "decode-instances": [1],
            "engine": {"speculative": {"method": ["dspark"], "num_spec_tokens": [3]}},
            "prefill": {"parallel": {"tp": [4]},
                        "engine": {"max_num_seqs": [32],
                                   "eplb": {"enable_eplb": [True], "window_size": [100, 200]}}},
            "decode": {"parallel": {"tp": [4]}, "engine": {"max_num_seqs": [64, 128]}}}


def pd_classify(cand, shared=None):
    """The parallel dict classify_config returns for a disaggregated candidate over the scenario's engine."""
    return run_blis.classify_config(run_config(shared or {}, cand), settings.vocabulary(BLIS))[0]


class PoolEngineBlocks(unittest.TestCase):
    """A disaggregated sweep's per-pool engine block is expanded and merged like the shared one."""

    def test_a_pool_engine_block_expands_to_one_candidate_per_combination(self):
        kept, dropped = E.expand_sweep(PD_SWEEP, 8)
        self.assertEqual([], dropped)
        eplbs = sorted((c["prefill_eplb"] for c in kept), key=lambda b: b["window_size"])
        self.assertEqual([{"enable_eplb": True, "window_size": 100},
                          {"enable_eplb": True, "window_size": 200}], [eplbs[0], eplbs[-1]])
        self.assertEqual(4, len(kept), "2 prefill windows x 2 decode batch sizes, on the one drafting arm")
        self.assertTrue(all(isinstance(c["prefill_eplb"], dict) for c in kept))

    def test_a_pool_block_merges_field_wise_onto_the_scenarios_block(self):
        shared = {"eplb": {"enable_eplb": True, "num_redundant_experts": 8}}
        cand = {"tp": 4, "prefill_instances": 1, "decode_instances": 1, "prefill_tp": 4, "decode_tp": 4,
                "prefill_eplb": {"window_size": 100}}
        before = copy.deepcopy(shared)
        pd = pd_classify(cand, shared)["pd"]
        self.assertEqual({"enable_eplb": True, "num_redundant_experts": 8, "window_size": 100},
                         pd["prefill"]["engine"]["eplb"])
        self.assertEqual(shared["eplb"], pd["decode"]["engine"]["eplb"], "the other pool keeps the scenario's block")
        self.assertEqual(before, shared)

    def test_drafting_is_one_configuration_per_run_in_every_spelling(self):
        base = {"tp": 4, "prefill_instances": 1, "decode_instances": 1, "prefill_tp": 4, "decode_tp": 4}
        for key, value in (("prefill_speculative", {"kv_cache_dtype": "fp8"}),
                           ("decode_speculative", {"method": "dspark", "num_spec_tokens": 3}),
                           ("prefill_num_speculative_tokens", 3),
                           ("decode_speculative_method", "dspark")):
            with self.subTest(key):
                with self.assertRaises(run_blis.ConfigError) as e:
                    pd_classify({**base, key: value})
                self.assertIn("one configuration per run", str(e.exception))

    def test_the_validator_refuses_a_pool_that_states_drafting(self):
        import test_engine_kind as K
        import validate_experiment as V
        spec = {"name": "t", "scenario": {"model": "qwen3-14b", "hardware": "h100", "workload": "chatbot",
                                          "cluster": {"gpus_per_node": 8}, "gpus": 8, "concurrency": [4]},
                "sweeps": [copy.deepcopy(PD_SWEEP)]}
        spec["sweeps"][0]["engine"] = {"max_num_seqs": [64]}
        self.assertFalse([e for e in V.validate(copy.deepcopy(spec), BLIS, K.ROUTING, None, None)
                          if "speculative" in e])
        spec["sweeps"][0]["decode"]["engine"] = {"speculative": {"kv_cache_dtype": ["fp8"]}}
        errs = V.validate(copy.deepcopy(spec), BLIS, K.ROUTING, None, None)
        self.assertTrue(any("decode.engine.speculative" in e and "one configuration per run" in e
                            for e in errs), errs)


class PoolProvenance(unittest.TestCase):
    ENGINE = {"max_model_len": 32768, "max_num_seqs": 16, "block_size": 64,
              "speculative": {"method": "mtp", "num_spec_tokens": 3}}
    PREFILL = {**ENGINE, "max_num_seqs": 32, "eplb": {"enable_eplb": True, "window_size": 100}}
    DECODE = {**ENGINE, "max_num_seqs": 64}
    REC = {"deployment": {"engine": ENGINE, "parallel": {"pd": {"prefill": {"engine": PREFILL},
                                                                  "decode": {"engine": DECODE}}}}}
    SWEEP = {"engine": {"speculative": {"num_spec_tokens": [3]}},
             "prefill": {"engine": {"max_num_seqs": [32], "eplb": {"window_size": [100]}}},
             "decode": {"engine": {"max_num_seqs": [64]}}}
    FIXED = {("max_num_seqs",), ("eplb", "enable_eplb")}

    def origins(self, rec=None, sweep=None):
        sweep = sweep or self.SWEEP
        return E.row_origins(rec or self.REC, self.FIXED, E.leaf_paths(sweep.get("engine")), sweep)

    def test_each_pool_records_the_origin_of_its_own_leaves(self):
        got = self.origins()["pool_engine_origin"]
        self.assertEqual({"max_model_len": "runner", "max_num_seqs": "sweep", "block_size": "runner",
                          "speculative": {"method": "catalog", "num_spec_tokens": "sweep"},
                          "eplb": {"enable_eplb": "scenario", "window_size": "sweep"}}, got["prefill"])
        self.assertEqual("sweep", got["decode"]["max_num_seqs"])
        self.assertNotIn("eplb", got["decode"])

    def test_a_pool_is_not_credited_with_the_other_pools_sweep(self):
        sweep = {**self.SWEEP, "decode": {}}
        got = self.origins(sweep=sweep)["pool_engine_origin"]
        self.assertEqual("sweep", got["prefill"]["max_num_seqs"])
        self.assertEqual("scenario", got["decode"]["max_num_seqs"],
                         "the decode pool's batch size was fixed by the scenario, whatever the prefill pool swept")

    def test_the_shared_origin_is_unchanged_and_a_colocated_row_has_no_pool_tree(self):
        self.assertEqual(E.engine_origin(self.ENGINE, self.FIXED, E.leaf_paths(self.SWEEP["engine"])),
                         self.origins()["engine_origin"])
        colocated = {"deployment": {"engine": self.ENGINE, "parallel": {"tp": 8}}}
        self.assertEqual({"engine_origin"}, set(E.row_origins(colocated, self.FIXED, set(), {})))
        self.assertEqual({}, E.row_origins({"deployment": None}, self.FIXED, set(), {}))

    def test_pool_origins_round_trip_at_every_depth(self):
        origins = self.origins()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.json")
            E._dump(path, origins)
            with open(path, encoding="utf-8") as f:
                parsed = json.load(f)
        self.assertEqual(origins, parsed)
        for role, engine in (("prefill", self.PREFILL), ("decode", self.DECODE)):
            self.assertEqual(E.leaf_paths(engine), E.leaf_paths(parsed["pool_engine_origin"][role]))


class PoolVariantLabels(unittest.TestCase):
    def test_arms_that_differ_only_in_a_pools_engine_setting_are_told_apart(self):
        a = {"prefill_tp": 4, "decode_tp": 4, "prefill_max_num_seqs": 32, "decode_max_num_seqs": 64}
        b = {**a, "decode_max_num_seqs": 128}
        base = E.sweep_baseline([a, a, b])
        self.assertEqual("", E.variant_of(a, "x", base))
        self.assertEqual("decode max seqs 128", E.variant_of(b, "x", base))


class NestedSpeculativeLabels(unittest.TestCase):
    def test_arms_that_differ_only_in_a_native_speculative_field_are_told_apart(self):
        cands = [c for c in E._engine_candidates(SWEEP["engine"]) if c["num_speculative_tokens"] == 3]
        base = E.sweep_baseline(cands)
        labels = sorted(E.variant_of(c, "x", base) for c in cands)
        self.assertEqual(2, len(set(labels)), labels)
        self.assertEqual(1, sum("speculative decoding" in lab for lab in labels), labels)

    def test_a_flat_grid_has_no_speculative_block_to_label(self):
        cands = E._engine_candidates({"speculative": {"method": ["dspark"], "num_spec_tokens": [1, 3]}})
        base = E.sweep_baseline(cands)
        self.assertEqual(["3 draft tokens"], [E.variant_of(c, "x", base) for c in cands
                                              if c["num_speculative_tokens"] == 3])


class SglangSpeculativeSettings(unittest.TestCase):
    """SGLang states drafting as flat, engine-named settings, not as vLLM's nested block."""

    def setUp(self):
        self.addCleanup(settings.use_engine, "vllm")
        settings.use_engine("sglang")
        self.vocab = settings.Vocabulary(
            parallel={"tp", "pp", "dp"}, flat={"chunked_prefill_size", "speculative_algorithm",
                                              "speculative_num_steps"},
            blocks={}, release="0.5.23", source="a test field list")

    def test_the_runners_own_drafting_keys_are_refused_for_sglang_with_the_engines_names(self):
        for key, value in (("num_speculative_tokens", 3), ("speculative_method", "mtp"),
                           ("speculative_draft_model", "x")):
            with self.subTest(key=key):
                with self.assertRaises(run_blis.ConfigError) as cm:
                    run_blis.classify_config({"tp": 8, key: value}, self.vocab)
                self.assertIn(key, str(cm.exception))
                self.assertIn("speculative_algorithm", str(cm.exception))
                self.assertIn("speculative_num_steps", str(cm.exception))

    def test_the_same_keys_still_fold_into_the_nested_block_for_vllm(self):
        settings.use_engine("vllm")
        vocab = settings.vocabulary(BLIS)
        engine = run_blis.classify_config({"tp": 8, "num_speculative_tokens": 3,
                                           "speculative_method": "dspark"}, vocab)[1]
        self.assertEqual({"method": "dspark", "num_spec_tokens": 3}, engine["speculative"])

    def test_flat_speculative_settings_keep_their_own_origin(self):
        shared = {"speculative_algorithm": "EAGLE", "chunked_prefill_size": 512}
        cand = {"speculative_num_steps": 3}
        engine = run_blis.classify_config({"tp": 8, **cand}, self.vocab)[1]
        self.assertEqual(3, engine["speculative_num_steps"])
        origin = E.engine_origin(E.merge_nested(shared, engine), {(k,) for k in shared},
                                 {("speculative_num_steps",)})
        self.assertEqual({"speculative_algorithm": "scenario", "chunked_prefill_size": "scenario",
                          "speculative_num_steps": "sweep"}, origin)


if __name__ == "__main__":
    unittest.main()
