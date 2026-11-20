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


if __name__ == "__main__":
    unittest.main()
