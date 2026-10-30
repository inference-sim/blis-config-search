#!/usr/bin/env python3
"""Tests for the Experiment Spec pipeline: validator, grid expansion, router overrides, ranking.

    python3 -I scripts/test_experiment.py

Hermetic: no BLIS, no network. Everything here is a pure function of a spec, which is exactly
the property that lets CI run the sweeper unattended -- so these tests are the guard on it.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, ".claude", "skills", "config-search", "scripts"))

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

import run_experiment as E          # noqa: E402
import validate_experiment as V     # noqa: E402
import run_blis                     # noqa: E402

EXPERIMENTS = os.path.join(ROOT, "experiments")
ROUTING = os.path.join(ROOT, "routing")


def spec(**over):
    """A minimal valid spec, overridable per test."""
    base = {
        "name": "t",
        "scenario": {"model": "qwen3-14b", "hardware": "h100", "workload": "chatbot",
                     "gpus": 16, "slo": {"itl_p99": "20ms"}, "concurrency": [1, 2]},
        "sweeps": [{"name": "s", "num-instances": [2], "parallel": {"tp": [8]}}],
    }
    base.update(over)
    return base


# --- the committed fixtures: RFC #21's examples ------------------------------------------------


class CommittedSpecsValidate(unittest.TestCase):
    """Every spec in experiments/ must validate -- the acceptance criterion from #21."""

    def test_all_fixtures_valid(self):
        paths = V.spec_paths(EXPERIMENTS)
        self.assertGreaterEqual(len(paths), 5, "the 5 #21 examples should be committed")
        for path in paths:
            if path.endswith("schema.json"):
                continue
            doc, err = V.load(path)
            self.assertIsNone(err, f"{path}: {err}")
            self.assertEqual([], V.validate(doc, ROUTING), f"{path} should validate")

    def test_every_fixture_expands_to_runs(self):
        """A spec that validates but expands to nothing is a spec nobody can learn from."""
        for path in V.spec_paths(EXPERIMENTS):
            doc, _ = V.load(path)
            gpus = doc["scenario"]["gpus"]
            for sweep in doc["sweeps"]:
                kept, _dropped = E.expand_sweep(sweep, gpus)
                self.assertTrue(kept, f"{path}: sweep {sweep['name']} expands to no configs")

    def test_schema_is_valid_json(self):
        with open(os.path.join(EXPERIMENTS, "schema.json"), encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual("Experiment Spec", doc["title"])
        self.assertIn("sweeps", doc["required"])

    def test_every_fixture_carries_narratives(self):
        """The narrative is the hypothesis the report prints beside the result; a sweep without
        one reads as a bare grid."""
        for path in V.spec_paths(EXPERIMENTS):
            doc, _ = V.load(path)
            for sweep in doc["sweeps"]:
                self.assertTrue((sweep.get("narrative") or "").strip(),
                                f"{path}: sweep {sweep['name']} has no narrative")


# --- validator --------------------------------------------------------------------------------


class ValidatorRejects(unittest.TestCase):
    def errs(self, doc):
        return V.validate(doc, ROUTING)

    def test_scalar_knob_is_refused(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": 8}}])
        self.assertTrue(any("must be a list" in e for e in self.errs(bad)))

    def test_unknown_engine_knob_is_refused_with_suggestion(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "engine": {"max_num_seq": [64]}}])
        errs = self.errs(bad)
        self.assertTrue(any("max_num_seqs" in e for e in errs), errs)

    def test_unknown_parallel_knob_is_refused(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2],
                            "parallel": {"tp": [8], "tensor_parallel": [2]}}])
        self.assertTrue(any("tensor_parallel" in e for e in self.errs(bad)))

    def test_gpu_math_that_cannot_be_satisfied(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [3], "parallel": {"tp": [8]}}])
        errs = self.errs(bad)
        self.assertTrue(any("totals the scenario's 16 GPUs" in e for e in errs), errs)

    def test_slo_without_a_unit_is_refused(self):
        bad = spec()
        bad["scenario"]["slo"] = {"itl_p99": 20}
        self.assertTrue(any("explicit unit" in e for e in self.errs(bad)))

    def test_unknown_slo_is_refused(self):
        bad = spec()
        bad["scenario"]["slo"] = {"p99": "20ms"}
        self.assertTrue(any("unknown SLO" in e for e in self.errs(bad)))

    def test_descending_concurrency_is_refused(self):
        bad = spec()
        bad["scenario"]["concurrency"] = [8, 2]
        self.assertTrue(any("ascending" in e for e in self.errs(bad)))

    def test_missing_router_file_is_refused(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "router": {"endpoint_picker_config": ["nope.yaml"]}}])
        errs = self.errs(bad)
        self.assertTrue(any("no such file routing/nope.yaml" in e for e in errs), errs)

    def test_router_path_traversal_is_refused(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "router": {"endpoint_picker_config": ["../pins.env"]}}])
        self.assertTrue(any("plain file name" in e for e in self.errs(bad)))

    def test_colocated_and_disaggregated_cannot_mix(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "prefill-instances": [1], "decode-instances": [1]}])
        self.assertTrue(any("one or the other" in e for e in self.errs(bad)))

    def test_prefill_without_decode_is_refused(self):
        bad = spec(sweeps=[{"name": "s", "prefill-instances": [1],
                            "prefill": {"parallel": {"tp": [8]}},
                            "decode": {"parallel": {"tp": [8]}}}])
        self.assertTrue(any("no counterpart pool" in e or "decode-instances" in e
                            for e in self.errs(bad)))

    def test_speculative_needs_both_fields(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "engine": {"speculative": {"num_spec_tokens": [3]}}}])
        self.assertTrue(any("needs both method and num_spec_tokens" in e for e in self.errs(bad)))

    def test_pd_pools_must_share_dp(self):
        bad = spec()
        bad["scenario"]["gpus"] = 64
        bad["sweeps"] = [{"name": "s", "prefill-instances": [2], "decode-instances": [2],
                          "prefill": {"parallel": {"tp": [8], "dp": [2]}},
                          "decode": {"parallel": {"tp": [8], "dp": [1]}}}]
        self.assertTrue(any("ONE replica factor" in e for e in self.errs(bad)))

    def test_shared_engine_takes_a_single_value(self):
        bad = spec()
        bad["scenario"]["engine"] = {"max_model_len": [262144]}
        self.assertTrue(any("takes ONE value" in e for e in self.errs(bad)))

    def test_duplicate_sweep_names_refused(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]}},
                           {"name": "s", "num-instances": [2], "parallel": {"tp": [8]}}])
        self.assertTrue(any("duplicate sweep name" in e for e in self.errs(bad)))

    def test_overrides_need_exactly_one_base(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "router": {"endpoint_picker_config": ["queue.yaml", "random.yaml"],
                                       "plugins": {"load-aware-scorer":
                                                   {"parameters": {"threshold": [1]}}}}}])
        self.assertTrue(any("apply to ONE base file" in e for e in self.errs(bad)))

    def test_undocumented_plugin_parameter_refused(self):
        bad = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "router": {"endpoint_picker_config": ["load-aware.yaml"],
                                       "plugins": {"load-aware-scorer":
                                                   {"parameters": {"thresh": [1]}}}}}])
        self.assertTrue(any("not a documented parameter" in e for e in self.errs(bad)))

    def test_unknown_top_level_key_refused(self):
        bad = spec(sweeep=[])
        self.assertTrue(any("unknown top-level key" in e for e in self.errs(bad)))

    def test_valid_spec_has_no_errors(self):
        self.assertEqual([], self.errs(spec()))

    def test_duration_parsing(self):
        self.assertEqual(20.0, V.parse_duration_ms("20ms"))
        self.assertEqual(5000.0, V.parse_duration_ms("5s"))
        self.assertEqual(0.5, V.parse_duration_ms("500us"))
        self.assertIsNone(V.parse_duration_ms("20"))
        self.assertIsNone(V.parse_duration_ms("20 ms"))

    def test_knob_vocabulary_reports_its_source(self):
        par, eng, source = V.knob_vocabulary()
        self.assertIn("tp", par)
        self.assertIn("max_num_seqs", eng)
        self.assertTrue(source)


# --- grid expansion ---------------------------------------------------------------------------


class GridExpansion(unittest.TestCase):
    def test_product_is_sorted_and_stable(self):
        g = E.grid({"b": [1, 2], "a": ["x"]})
        self.assertEqual([{"a": "x", "b": 1}, {"a": "x", "b": 2}], g)
        self.assertEqual(g, E.grid({"a": ["x"], "b": [1, 2]}),
                         "key order in the spec must not change the expansion")

    def test_empty_grid_is_one_empty_config(self):
        self.assertEqual([{}], E.grid({}))

    def test_gpu_filter_keeps_only_exact_budget(self):
        sweep = {"name": "s", "num-instances": [1, 2, 4],
                 "parallel": {"tp": [4, 8], "dp": [1, 2]}}
        kept, dropped = E.expand_sweep(sweep, 16)
        for c in kept:
            self.assertEqual(16, c["tp"] * c.get("dp", 1) * c["num_instances"])
        self.assertTrue(dropped, "off-budget combos should be recorded, not vanish")
        self.assertEqual(12, len(kept) + len(dropped), "every combo is kept or dropped")

    def test_speculative_expands_to_flat_candidate_keys(self):
        sweep = {"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                 "engine": {"speculative": {"method": ["dspark"], "num_spec_tokens": [1, 3]}}}
        kept, _ = E.expand_sweep(sweep, 16)
        self.assertEqual(2, len(kept))
        for c in kept:
            self.assertEqual("dspark", c["speculative_method"])
            self.assertIn(c["num_speculative_tokens"], (1, 3))

    def test_engine_knobs_cross_with_topology(self):
        sweep = {"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                 "engine": {"max_num_seqs": [64, 256], "cache_dtype": ["auto", "fp8"]}}
        kept, _ = E.expand_sweep(sweep, 16)
        self.assertEqual(4, len(kept))

    def test_disaggregated_counts_both_pools(self):
        sweep = {"name": "s", "prefill-instances": [2, 3, 4], "decode-instances": [4, 5, 6],
                 "prefill": {"parallel": {"tp": [8]}}, "decode": {"parallel": {"tp": [8]}}}
        kept, _ = E.expand_sweep(sweep, 64)
        pairs = sorted((c["prefill_instances"], c["decode_instances"]) for c in kept)
        self.assertEqual([(2, 6), (3, 5), (4, 4)], pairs,
                         "only pairs totalling 8 instances x TP8 = 64 GPUs survive")

    def test_disaggregated_role_knobs_are_prefixed(self):
        sweep = {"name": "s", "prefill-instances": [4], "decode-instances": [4],
                 "prefill": {"parallel": {"tp": [8]}, "engine": {"max_num_seqs": [64]}},
                 "decode": {"parallel": {"tp": [8]}}}
        kept, _ = E.expand_sweep(sweep, 64)
        self.assertEqual(1, len(kept))
        self.assertEqual(8, kept[0]["prefill_tp"])
        self.assertEqual(64, kept[0]["prefill_max_num_seqs"])

    def test_router_arms_multiply_the_grid(self):
        sweep = {"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                 "router": {"endpoint_picker_config": ["queue.yaml", "random.yaml"]}}
        kept, _ = E.expand_sweep(sweep, 16)
        self.assertEqual(["queue.yaml", "random.yaml"],
                         sorted(c["router_config"] for c in kept))

    def test_no_router_block_is_one_default_arm(self):
        sweep = {"name": "s", "num-instances": [2], "parallel": {"tp": [8]}}
        kept, _ = E.expand_sweep(sweep, 16)
        self.assertEqual(1, len(kept))
        self.assertNotIn("router_config", kept[0])

    def test_expansion_is_deterministic(self):
        sweep = {"name": "s", "num-instances": [1, 2], "parallel": {"tp": [8, 16]},
                 "engine": {"max_num_seqs": [64, 256]}}
        a, _ = E.expand_sweep(sweep, 16)
        b, _ = E.expand_sweep(sweep, 16)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


# --- router overrides -------------------------------------------------------------------------


class RouterOverrides(unittest.TestCase):
    def base(self, name="queue-kv.yaml"):
        with open(os.path.join(ROUTING, name), encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_override_combos_are_a_product(self):
        router = {"endpoint_picker_config": ["queue-kv.yaml"],
                  "schedulingProfiles": {"default": {"queue-scorer": {"weight": [1, 2]}}},
                  "plugins": {"load-aware-scorer": {"parameters": {"threshold": [1, 2, 3]}}}}
        self.assertEqual(6, len(E._override_combos(router)))

    def test_no_overrides_is_one_empty_combo(self):
        self.assertEqual([{}], E._override_combos({"endpoint_picker_config": ["queue.yaml"]}))

    def test_scheduling_profile_weight_is_applied(self):
        doc = E.apply_overrides(
            self.base(), {("schedulingProfiles", "default", "queue-scorer", "weight"): 4})
        entry = next(p for p in doc["schedulingProfiles"][0]["plugins"]
                     if p.get("pluginRef") == "queue-scorer")
        self.assertEqual(4, entry["weight"])

    def test_plugin_parameter_is_applied(self):
        doc = E.apply_overrides(
            self.base("pd-prefix.yaml"),
            {("plugins", "prefix-based-pd-decider", "parameters", "nonCachedTokens"): 4096})
        entry = next(p for p in doc["plugins"] if p.get("type") == "prefix-based-pd-decider")
        self.assertEqual(4096, entry["parameters"]["nonCachedTokens"])

    def test_base_is_not_mutated(self):
        base = self.base()
        before = json.dumps(base, sort_keys=True)
        E.apply_overrides(base, {("schedulingProfiles", "default", "queue-scorer", "weight"): 9})
        self.assertEqual(before, json.dumps(base, sort_keys=True))

    def test_unknown_plugin_is_refused(self):
        with self.assertRaises(KeyError):
            E.apply_overrides(self.base(),
                              {("plugins", "nope", "parameters", "weight"): 1})

    def test_unknown_profile_is_refused(self):
        with self.assertRaises(KeyError):
            E.apply_overrides(self.base(),
                              {("schedulingProfiles", "nope", "queue-scorer", "weight"): 1})

    def test_parameter_absent_from_base_is_refused(self):
        """An override may only CHANGE a field the base already sets (#21's stated limitation)."""
        with self.assertRaises(KeyError):
            E.apply_overrides(
                self.base("load-aware.yaml"),
                {("plugins", "load-aware-scorer", "parameters", "threshold"): 1})

    def test_materialize_writes_one_file_per_combo_named_by_content(self):
        sweep = {"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                 "router": {"endpoint_picker_config": ["pd-prefix.yaml"],
                            "plugins": {"prefix-based-pd-decider":
                                        {"parameters": {"nonCachedTokens": [16, 4096]}}}}}
        kept, _ = E.expand_sweep(sweep, 16)
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for cand in kept:
                resolved, desc = E.materialize_router(cand, ROUTING, d)
                paths.append(resolved["router_config"])
                self.assertNotIn("router_overrides", resolved)
                self.assertIn("nonCachedTokens", desc)
            self.assertEqual(2, len(set(paths)), "each combo gets its own file")
            # Re-materializing the same combo must land on the same path (content-hashed).
            again, _ = E.materialize_router(kept[0], ROUTING, d)
            self.assertEqual(paths[0], again["router_config"])
            with open(paths[0], encoding="utf-8") as f:
                body = f.read()
            self.assertIn("GENERATED by scripts/run_experiment.py", body)
            self.assertIn("nonCachedTokens: 16", body)

    def test_materialize_without_overrides_points_at_the_committed_file(self):
        cand = {"router_config": "queue.yaml"}
        resolved, desc = E.materialize_router(cand, ROUTING, "/nonexistent-should-not-be-written")
        self.assertEqual(os.path.join(ROUTING, "queue.yaml"), resolved["router_config"])
        self.assertEqual("queue", desc)


# --- labelling and ranking --------------------------------------------------------------------


class Labelling(unittest.TestCase):
    def test_colocated_label(self):
        self.assertEqual("8x TP4xDP2 + EP", E.label_of(
            {"tp": 4, "dp": 2, "num_instances": 8, "enable_expert_parallel": True}))

    def test_disaggregated_label_names_both_pools(self):
        self.assertEqual("3P/5D TP8 + EP", E.label_of(
            {"prefill_instances": 3, "decode_instances": 5, "prefill_tp": 8,
             "prefill_enable_expert_parallel": True}))

    def test_variant_describes_only_what_differs_from_the_sweep_baseline(self):
        cands = [{"max_num_seqs": 256}, {"max_num_seqs": 256}, {"max_num_seqs": 64}]
        base = E.sweep_baseline(cands)
        self.assertEqual("", E.variant_of(cands[0], "", base))
        self.assertEqual("max seqs 64", E.variant_of(cands[2], "", base))

    def test_variant_names_the_routing_arm(self):
        cands = [{"router_config": "a.yaml"}, {"router_config": "b.yaml"},
                 {"router_config": "b.yaml"}]
        base = E.sweep_baseline(cands)
        self.assertEqual("prefix-affinity routing",
                         E.variant_of(cands[0], "prefix-affinity", base))


class KnobSurfaceStaysInSync(unittest.TestCase):
    """The three lists describing the engine surface must not drift apart.

    Each drift has a specific, silent failure mode, so each gets a test rather than a comment.
    """

    def test_every_runner_knob_can_be_labelled(self):
        """A swept knob missing from VARIANT_KEYS leaves every arm labelled "(baseline)".

        That is how a cudagraph_mode sweep spanning 147 to 263 tok/s/GPU first rendered as four
        indistinguishable rows: the runner carried the knob, the report could not name it.
        """
        missing = sorted(set(run_blis.SCENARIO_ENGINE) - set(E.VARIANT_KEYS))
        self.assertEqual([], missing,
                         f"engine knobs the runner carries but the report cannot name: {missing}")

    def test_every_runner_knob_has_a_readable_name(self):
        """A knob with no KNOB_NAMES entry prints its raw field name in a variant label."""
        import topologies
        bare = [k for k in run_blis.SCENARIO_ENGINE
                if k not in topologies.KNOB_NAMES and k != "speculative"]
        self.assertEqual([], bare, f"engine knobs with no readable name: {bare}")

    def test_validator_accepts_every_knob_the_runner_can_carry(self):
        """A knob the runner carries but the validator rejects is unreachable by any spec."""
        unreachable = sorted(set(run_blis.SCENARIO_ENGINE) - set(V.ENGINE_KNOBS))
        self.assertEqual([], unreachable,
                         f"runner carries but validator rejects: {unreachable}")

    def test_validator_rejects_every_knob_the_runner_cannot_carry(self):
        """The converse: a name that validates but dies at run time wastes the whole sweep.

        The validator's name list is vLLM v0.32.0's (~51 fields), while the pinned stack's
        blis-schemas has 31, so the difference MUST be covered by ENGINE_NOT_WIRED.
        """
        carried = set(run_blis.SCENARIO_ENGINE) | {"speculative"}
        gap = sorted(set(V.ENGINE_KNOBS) - carried - set(V.ENGINE_NOT_WIRED))
        self.assertEqual([], gap,
                         f"these validate but the runner would refuse them at run time: {gap}")

    def test_not_wired_names_are_refused_by_the_runner_too(self):
        """Both layers must agree a knob is unavailable, and say why."""
        for knob in sorted(V.ENGINE_NOT_WIRED):
            with self.assertRaises(run_blis.ConfigError, msg=knob):
                run_blis.classify_config({"tp": 8, "num_instances": 1, knob: 1})

    def test_enum_values_are_read_from_the_rules_pack_when_present(self):
        enums, source = V.engine_enums()
        self.assertIn("cudagraph_mode", enums)
        self.assertIn("FULL_AND_PIECEWISE", enums["cudagraph_mode"])
        self.assertIn("fp8_ds_mla", enums["cache_dtype"])
        self.assertIn("deepep_low_latency", enums["all2all_backend"])
        self.assertEqual(("fcfs", "priority"), tuple(enums["scheduling_policy"]))
        self.assertTrue(source)

    def test_enumerated_knobs_are_all_knobs_the_runner_carries(self):
        """An enum for a knob no spec can sweep is dead weight and misleads a reader.

        "Sweepable" includes the inner fields of the nested blocks: `speculative.method` is a
        real swept value even though `speculative_method` is not a top-level engine key.
        """
        inner = {f for fields in V.NESTED_ENGINE_FIELDS.values() for f in fields}
        inner |= {f"speculative_{f}" for f in V.NESTED_ENGINE_FIELDS["speculative"]}
        stray = sorted(set(V.ENGINE_ENUMS) - set(run_blis.SCENARIO_ENGINE) - inner)
        self.assertEqual([], stray, f"value sets for unsweepable knobs: {stray}")

    def test_draft_methods_are_validated_without_a_blis_vllm_checkout(self):
        """CI has no sibling checkout, so the PINNED set must carry the draft methods too --
        otherwise a typo'd method validates in CI and fails every run."""
        self.assertIn("speculative_method", V.ENGINE_ENUMS)
        for real in ("dspark", "glm5_next_mtp", "mtp", "eagle3"):
            self.assertIn(real, V.ENGINE_ENUMS["speculative_method"])


class EngineValueChecking(unittest.TestCase):
    def errs(self, engine):
        doc = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "engine": engine}])
        return V.validate(doc, ROUTING)

    def test_wrong_case_enum_is_refused(self):
        """`cudagraph_mode: [full]` is a natural thing to write; the engine's enum is upper-case."""
        errs = self.errs({"cudagraph_mode": ["full"]})
        self.assertTrue(any("not a value the engine accepts" in e for e in errs), errs)

    def test_near_miss_enum_gets_a_suggestion(self):
        errs = self.errs({"all2all_backend": ["deepep_lowlatency"]})
        self.assertTrue(any("deepep_low_latency" in e for e in errs), errs)

    def test_wrong_dtype_spelling_is_refused(self):
        errs = self.errs({"cache_dtype": ["fp16"]})
        self.assertTrue(any("not a value" in e for e in errs), errs)

    def test_valid_enum_values_pass(self):
        self.assertEqual([], self.errs({"cudagraph_mode": ["NONE", "FULL_AND_PIECEWISE"],
                                        "cache_dtype": ["auto", "fp8"],
                                        "scheduling_policy": ["fcfs", "priority"]}))

    def test_non_boolean_for_a_tristate_knob_is_refused(self):
        errs = self.errs({"enable_prefix_caching": ["no"]})
        self.assertTrue(any("not a boolean" in e for e in errs), errs)

    def test_boolean_knobs_accept_booleans(self):
        self.assertEqual([], self.errs({"enable_prefix_caching": [True, False],
                                        "async_scheduling": [True]}))

    def test_list_valued_knob_needs_a_list_of_lists(self):
        errs = self.errs({"cudagraph_capture_sizes": [1, 2, 4]})
        self.assertTrue(any("list of lists" in e for e in errs), errs)

    def test_list_valued_knob_accepts_a_list_of_lists(self):
        self.assertEqual([], self.errs({"cudagraph_capture_sizes": [[1, 2, 4], [1, 2, 4, 8]]}))

    def test_shared_scenario_engine_values_are_checked_too(self):
        doc = spec()
        doc["scenario"]["engine"] = {"cudagraph_mode": "full"}
        self.assertTrue(any("not a value the engine accepts" in e
                            for e in V.validate(doc, ROUTING)))

    def test_nested_block_unknown_inner_field_is_refused(self):
        """`eplb` carries `enable_eplb`, not `enabled`; the engine decodes strictly."""
        errs = self.errs({"eplb": {"enabled": [True]}})
        self.assertTrue(any("not a field of this block" in e for e in errs), errs)

    def test_nested_block_inner_values_must_be_lists(self):
        errs = self.errs({"eplb": {"enable_eplb": True}})
        self.assertTrue(any("must be a list" in e for e in errs), errs)

    def test_nested_block_boolean_field_is_checked(self):
        errs = self.errs({"eplb": {"enable_eplb": ["yes"]}})
        self.assertTrue(any("not a boolean" in e for e in errs), errs)

    def test_valid_nested_blocks_pass(self):
        """A WELL-FORMED nested block passes. It must also be a sensible one: EPLB needs an MoE
        model and a divisible expert count, and an enabled DBO needs a stated threshold -- the
        cross-knob rules check both, so this fixture states a real deployment rather than a
        syntactically valid impossible one."""
        doc = spec(sweeps=[{"name": "s", "num-instances": [1],
                            "parallel": {"tp": [4], "dp": [2],
                                         "enable_expert_parallel": [True]},
                            "engine": {
                                "eplb": {"enable_eplb": [True],
                                         "num_redundant_experts": [0, 32]},
                                "dbo": {"enable_dbo": [True, False],
                                        "dbo_decode_token_threshold": [256]},
                                "engram_config": {"cpu_offload": [True, False]}}}])
        doc["scenario"].update({"model": "deepseek-v2-lite", "gpus": 8})
        self.assertEqual([], V.validate(doc, ROUTING))

    def test_nested_block_expands_to_one_candidate_per_combination(self):
        sweep = {"name": "s", "num-instances": [1], "parallel": {"tp": [8]},
                 "engine": {"eplb": {"enable_eplb": [True],
                                     "num_redundant_experts": [0, 32, 64]}}}
        kept, _ = E.expand_sweep(sweep, 8)
        self.assertEqual(3, len(kept))
        self.assertEqual([0, 32, 64],
                         sorted(c["eplb"]["num_redundant_experts"] for c in kept))
        self.assertTrue(all(c["eplb"]["enable_eplb"] is True for c in kept))

    def test_nested_block_arms_get_distinguishable_variants(self):
        sweep = {"name": "s", "num-instances": [1], "parallel": {"tp": [8]},
                 "engine": {"eplb": {"enable_eplb": [True],
                                     "num_redundant_experts": [0, 32]}}}
        kept, _ = E.expand_sweep(sweep, 8)
        base = E.sweep_baseline(kept)
        variants = [E.variant_of(c, "", base) for c in kept]
        self.assertEqual(len(variants), len(set(variants)),
                         f"arms must be distinguishable, got {variants}")
        self.assertTrue(any("redundant experts 32" in v for v in variants), variants)

    def test_nested_block_renders_as_a_sub_block(self):
        """The scenario wants a real YAML sub-block, not a JSON-flow dict."""
        par, eng, _cli, _r = run_blis.classify_config(
            {"tp": 8, "num_instances": 1,
             "eplb": {"enable_eplb": True, "num_redundant_experts": 32}})
        doc = run_blis.scenario_yaml("deepseek-v2-lite", "h100", "0.29.0", par, eng, "x", 1)
        self.assertIn("      eplb:", doc)
        self.assertIn("        enable_eplb: true", doc)
        self.assertIn("        num_redundant_experts: 32", doc)

    def test_newly_wired_knobs_expand_and_label(self):
        """The knobs this change added must survive expansion AND get a readable variant."""
        sweep = {"name": "s", "num-instances": [1], "parallel": {"tp": [8]},
                 "engine": {"cudagraph_mode": ["PIECEWISE", "FULL", "NONE"]}}
        kept, _ = E.expand_sweep(sweep, 8)
        self.assertEqual(3, len(kept))
        base = E.sweep_baseline(kept)
        variants = {E.variant_of(c, "", base) for c in kept}
        self.assertIn("graph capture full", variants)
        self.assertIn("graph capture none", variants)


class CrossKnobConstraints(unittest.TestCase):
    """Constraints that span two knobs, or a knob and the model.

    These exist because BLIS does NOT check them: the pinned simulator depends on
    blis-latency-kernel and blis-schemas only, never on blis-vllm, so the 32 rules in
    blis-vllm's v0_32 pack are not evaluated at run time. Measured: BLIS accepted a
    long_prefill_token_threshold of 999,999 against a 32k context, which real vLLM refuses at
    startup -- a spec like that simulates fine and describes an unlaunchable deployment.
    """

    def errs(self, engine, model="qwen3-14b", scenario_engine=None, parallel=None):
        sweep = {"name": "s", "num-instances": [2],
                 "parallel": parallel or {"tp": [8]}, "engine": engine}
        doc = spec(sweeps=[sweep])
        doc["scenario"]["model"] = model
        if scenario_engine:
            doc["scenario"]["engine"] = scenario_engine
        return V.validate(doc, ROUTING)

    def test_long_prefill_above_default_context_is_refused(self):
        errs = self.errs({"long_prefill_token_threshold": [999999]})
        self.assertTrue(any("exceeds the default max_model_len" in e for e in errs), errs)

    def test_long_prefill_above_swept_context_is_refused(self):
        errs = self.errs({"long_prefill_token_threshold": [65536],
                          "max_model_len": [32768]})
        self.assertTrue(any("exceeds every max_model_len" in e for e in errs), errs)

    def test_long_prefill_within_context_passes(self):
        self.assertEqual([], self.errs({"long_prefill_token_threshold": [2048, 4096],
                                        "max_model_len": [32768]}))

    def test_capture_maximum_must_be_the_largest_captured_size(self):
        errs = self.errs({"cudagraph_capture_sizes": [[1, 2, 64]],
                          "max_cudagraph_capture_size": [32]})
        self.assertTrue(any("not the largest of" in e for e in errs), errs)

    def test_capture_maximum_matching_the_largest_passes(self):
        self.assertEqual([], self.errs({"cudagraph_capture_sizes": [[1, 2, 64]],
                                        "max_cudagraph_capture_size": [64]}))

    def test_capture_plan_entirely_above_the_token_budget_is_refused(self):
        errs = self.errs({"cudagraph_capture_sizes": [[16384, 32768]],
                          "max_num_batched_tokens": [2048]})
        self.assertTrue(any("would capture no graph" in e for e in errs), errs)

    def test_capture_sizes_with_mode_none_everywhere_is_refused(self):
        errs = self.errs({"cudagraph_mode": ["NONE"],
                          "cudagraph_capture_sizes": [[1, 2, 4]]})
        self.assertTrue(any("captures no graph" in e for e in errs), errs)

    def test_capture_sizes_with_mode_none_among_others_is_fine(self):
        """Sweeping NONE against real modes is exactly how you measure capture's worth."""
        self.assertEqual([], self.errs({"cudagraph_mode": ["NONE", "FULL"]}))

    def test_max_model_len_beyond_the_catalog_context_is_refused(self):
        errs = self.errs({"max_model_len": [999999999]}, model="deepseek-v2-lite")
        if errs:                      # needs a catalog checkout; skip the assertion without one
            self.assertTrue(any("catalog context window" in e for e in errs), errs)

    def test_moe_knobs_on_a_dense_model_are_refused(self):
        errs = self.errs({"all2all_backend": ["deepep_low_latency"]})
        if V.catalog_entry("qwen3-14b"):
            self.assertTrue(any("DENSE model" in e for e in errs), errs)

    def test_eplb_on_a_dense_model_is_refused(self):
        errs = self.errs({"eplb": {"enable_eplb": [True]}})
        if V.catalog_entry("qwen3-14b"):
            self.assertTrue(any("no experts to balance" in e for e in errs), errs)

    def test_mamba_knobs_on_a_non_recurrent_model_are_refused(self):
        errs = self.errs({"mamba_cache_mode": ["align"]})
        if V.catalog_entry("qwen3-14b"):
            self.assertTrue(any("no recurrent layers" in e for e in errs), errs)

    def test_dbo_enabled_without_a_threshold_is_refused(self):
        errs = self.errs({"dbo": {"enable_dbo": [True]}})
        self.assertTrue(any("16x" in e for e in errs), errs)

    def test_dbo_enabled_with_a_threshold_passes(self):
        errs = self.errs({"dbo": {"enable_dbo": [True],
                                  "dbo_decode_token_threshold": [256, 512]}})
        self.assertEqual([], errs)

    def test_eplb_redundancy_breaking_expert_divisibility_is_refused(self):
        """deepseek-v2-lite has 64 experts; at EP width 8, 64+5 does not divide."""
        if not V.catalog_entry("deepseek-v2-lite"):
            self.skipTest("needs a ../blis-catalog checkout")
        errs = self.errs({"eplb": {"enable_eplb": [True], "num_redundant_experts": [5]}},
                         model="deepseek-v2-lite",
                         parallel={"tp": [4], "dp": [2], "enable_expert_parallel": [True]})
        self.assertTrue(any("not divisible by 8" in e for e in errs), errs)
        self.assertTrue(any("nearest fixes" in e for e in errs), errs)

    def test_eplb_redundancy_that_divides_passes(self):
        if not V.catalog_entry("deepseek-v2-lite"):
            self.skipTest("needs a ../blis-catalog checkout")
        doc = spec(sweeps=[{"name": "s", "num-instances": [1],
                            "parallel": {"tp": [4], "dp": [2],
                                         "enable_expert_parallel": [True]},
                            "engine": {"eplb": {"enable_eplb": [True],
                                                "num_redundant_experts": [0, 8, 32]}}}])
        doc["scenario"].update({"model": "deepseek-v2-lite", "gpus": 8})
        self.assertEqual([], V.validate(doc, ROUTING))

    def test_a_knob_fixed_in_scenario_engine_counts_for_the_cross_check(self):
        """scenario.engine pins a value for every run, so it participates in the constraint."""
        errs = self.errs({"long_prefill_token_threshold": [65536]},
                         scenario_engine={"max_model_len": 32768})
        self.assertTrue(any("exceeds every max_model_len" in e for e in errs), errs)

    def test_committed_fixtures_satisfy_every_cross_knob_rule(self):
        for path in V.spec_paths(EXPERIMENTS):
            doc, err = V.load(path)
            self.assertIsNone(err, f"{path}: {err}")
            self.assertEqual([], V.validate(doc, ROUTING), path)


class NarrativeLength(unittest.TestCase):
    """The narrative is the ONE line the report prints above a sweep's measured result.

    Unbounded, it drifted to 600-670 characters per sweep -- five sentences of hypothesis,
    rationale, prior measurements and design justification in the field meant for the prediction
    alone, which pushed the numbers off the screen. Advisory rather than an error: an over-long
    narrative still describes a runnable experiment, so it must not fail a build.
    """

    def test_long_narrative_is_flagged(self):
        doc = spec()
        doc["sweeps"][0]["narrative"] = "x" * (V.NARRATIVE_CHARS + 1)
        notes = V.advisories(doc)
        self.assertTrue(any("narrative" in n for n in notes), notes)

    def test_short_narrative_is_not_flagged(self):
        doc = spec()
        doc["sweeps"][0]["narrative"] = "Wider TP should win, because collectives stay on-node."
        self.assertEqual([], V.advisories(doc))

    def test_length_is_measured_on_collapsed_whitespace(self):
        """YAML folded scalars carry newlines and indentation; those are not content."""
        doc = spec()
        doc["sweeps"][0]["narrative"] = "word " * 20 + "\n\n   " + "more"
        collapsed = len(" ".join(doc["sweeps"][0]["narrative"].split()))
        notes = V.advisories(doc)
        if collapsed <= V.NARRATIVE_CHARS:
            self.assertEqual([], notes)
        else:
            self.assertTrue(any(str(collapsed) in n for n in notes), notes)

    def test_a_long_narrative_is_not_a_validation_error(self):
        doc = spec()
        doc["sweeps"][0]["narrative"] = "x" * 900
        self.assertEqual([], V.validate(doc, ROUTING),
                         "prose length must not fail a build")

    def test_our_own_fixtures_respect_the_limit(self):
        """Every spec this repo ships is an example a contributor copies, so each must meet
        the limit it documents."""
        for path in V.spec_paths(EXPERIMENTS):
            doc, err = V.load(path)
            self.assertIsNone(err, f"{path}: {err}")
            for sw in doc.get("sweeps") or []:
                text = " ".join(str(sw.get("narrative") or "").split())
                self.assertLessEqual(
                    len(text), V.NARRATIVE_CHARS,
                    f"{os.path.basename(path)}:{sw.get('name')} narrative is {len(text)} chars; "
                    f"move the rationale into a # comment")


class ReplayComesFromTheCatalog(unittest.TestCase):
    """A replay workload is fully described by blis-catalog/workloads/<name>.yaml.

    Verified against the pinned binary: `blis replay --workload-spec <descriptor>` with NO
    --corpus-sha256, --benchmark-duration, --grace-period or --max-context-length ran to
    completion, and still refused a corpus whose bytes did not match the descriptor's digest
    ("sha256 50dde578... does not match --corpus-sha256 e39cd2ff..."). So a spec restating any
    of that holds a copy that can only drift from the catalog.
    """

    CATALOG = os.path.join(ROOT, os.pardir, "blis-catalog")

    def have_catalog(self):
        return os.path.isfile(os.path.join(self.CATALOG, "workloads", "agentx-mvp.yaml"))

    def test_descriptor_is_read_from_the_catalog(self):
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        d = E.replay_descriptor("agentx-mvp", self.CATALOG)
        self.assertEqual("inferencex-agentx-mvp", d["scenario"])
        self.assertEqual(900, d["benchmark_duration_s"])
        self.assertEqual("coding", d["category"])
        self.assertTrue(d["corpus"]["sha256"])

    def test_a_shape_workload_has_no_descriptor(self):
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        self.assertIsNone(E.replay_descriptor("chatbot", self.CATALOG))
        self.assertIsNone(E.resolve_replay("chatbot", self.CATALOG, None))

    def test_resolve_needs_only_the_local_corpus_path(self):
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        out = E.resolve_replay("agentx-mvp", self.CATALOG, {"corpus": ".corpus/traces.jsonl"})
        self.assertEqual("inferencex-agentx-mvp", out["scenario"])
        self.assertEqual("900s", out["benchmark_duration"])
        self.assertEqual("max_model_len", out["max_context_length"])
        self.assertEqual("coding", out["category"])
        self.assertTrue(out["corpus_sha256"], "the digest comes from the catalog")
        self.assertNotIn("_overrides", out)

    def test_a_spec_override_is_recorded_as_one(self):
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        out = E.resolve_replay("agentx-mvp", self.CATALOG,
                               {"corpus": ".corpus/traces.jsonl", "grace_period": "inf"})
        self.assertEqual("inf", out["grace_period"])
        self.assertIn("grace_period", out["_overrides"])
        self.assertEqual("30s", out["_overrides"]["grace_period"]["catalog"])

    def test_a_replay_without_a_corpus_path_fails_loudly(self):
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        with self.assertRaises(SystemExit) as e:
            E.resolve_replay("agentx-mvp", self.CATALOG, None)
        self.assertIn("local corpus file", str(e.exception))

    def test_dry_run_tolerates_a_missing_corpus_path(self):
        """--dry-run expands grids without running, so it must not demand a fetched corpus."""
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        out = E.resolve_replay("agentx-mvp", self.CATALOG, None, dry_run=True)
        self.assertEqual("inferencex-agentx-mvp", out["scenario"])

    def test_restating_catalog_fields_is_flagged(self):
        doc = spec()
        doc["scenario"].update({"workload": "agentx-mvp", "replay": {
            "scenario": "inferencex-agentx-mvp", "corpus": ".corpus/traces.jsonl",
            "corpus_sha256": "deadbeef", "benchmark_duration": "900s"}})
        notes = V.advisories(doc)
        for field in ("scenario", "corpus_sha256", "benchmark_duration"):
            self.assertTrue(any(f"replay.{field}:" in n for n in notes), (field, notes))

    def test_a_minimal_replay_block_is_not_flagged(self):
        doc = spec()
        doc["scenario"].update({"workload": "agentx-mvp", "replay": {
            "corpus": ".corpus/traces.jsonl", "corpus_cache": "/tmp/c",
            "grace_period": "inf"}})
        self.assertEqual([], [n for n in V.advisories(doc) if "replay" in n])

    def test_absolute_corpus_path_is_flagged(self):
        doc = spec()
        doc["scenario"].update({"workload": "agentx-mvp",
                                "replay": {"corpus": "/Users/someone/traces.jsonl"}})
        self.assertTrue(any("absolute path" in n for n in V.advisories(doc)))

    def test_committed_replay_specs_state_only_local_fields(self):
        for path in V.spec_paths(EXPERIMENTS):
            doc, err = V.load(path)
            self.assertIsNone(err, f"{path}: {err}")
            rep = (doc.get("scenario") or {}).get("replay") or {}
            stray = sorted(set(rep) & V.CATALOG_OWNED_REPLAY_FIELDS)
            self.assertEqual([], stray,
                             f"{os.path.basename(path)} restates catalog data: {stray}")
            if isinstance(rep.get("corpus"), str):
                self.assertFalse(os.path.isabs(rep["corpus"]),
                                 f"{os.path.basename(path)} pins an absolute corpus path")


class KnobProfile(unittest.TestCase):
    """The per-combination profile: what APPLIES here, with what values, read from the sources.

    Exists because the catalog ROW alone cannot answer it -- it carries no speculator, no
    sliding window, no KV-dtype table and no capability gating. Those live in the model's
    graph.yaml and in blis-vllm's architecture-keyed rules.

    The headline case, verified against the pinned binary: deepseek-v4.1-flash prices
    nvfp4_ds_mla only under capability family 10, so it is a real KV axis on b200 and refused
    on h100 ("nvfp4_ds_mla needs the V4 ... on SM90").
    """

    CATALOG = os.path.join(ROOT, os.pardir, "blis-catalog")

    def setUp(self):
        if not os.path.isdir(os.path.join(self.CATALOG, "models")):
            self.skipTest("needs a ../blis-catalog checkout")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import knob_profile
        self.K = knob_profile

    def verdict(self, prof, knob):
        return next((k for k in prof["knobs"] if k["knob"].strip() == knob), None)

    def test_graph_supplies_the_speculator_the_catalog_row_lacks(self):
        g = self.K.graph_facts(self.CATALOG, "deepseek-v4.1-flash")
        self.assertEqual("dspark", g["speculator"])
        self.assertEqual(5, g["max_spec_tokens"])
        self.assertEqual("DeepseekV41ForCausalLM", g["architecture"])

    def test_architecture_falls_back_to_config_json(self):
        """Only models with architecture-specific rules state it in graph.yaml's global."""
        g = self.K.graph_facts(self.CATALOG, "qwen3-14b")
        self.assertEqual("Qwen3ForCausalLM", g["architecture"])
        self.assertEqual("config.json", g["architecture_source"])

    def test_kv_dtype_table_is_read_with_its_capability_gate(self):
        """The table lives under arch_overrides[].caches[], keyed by capability_family."""
        g = self.K.graph_facts(self.CATALOG, "deepseek-v4.1-flash")
        self.assertIn("nvfp4_ds_mla", g["kv_dtypes_priced"])
        self.assertEqual([10], g["kv_dtype_families"]["nvfp4_ds_mla"])

    def test_capability_family_comes_from_compute_capability(self):
        self.assertEqual(10, self.K.hardware_facts(self.CATALOG, "b200")["capability_family"])
        self.assertEqual(9, self.K.hardware_facts(self.CATALOG, "h100")["capability_family"])

    def test_arch_rules_read_both_resolution_rows(self):
        """An earlier regex cut at the first `},` and lost the family-9 row entirely."""
        ar = self.K.arch_rules("DeepseekV41ForCausalLM")
        fams = sorted(r["capability_family"] for r in ar["cache_dtype_resolutions"])
        self.assertEqual([9, 10, 10], fams)
        self.assertTrue(ar["refuses_mtp"])
        self.assertTrue(ar["engram"])

    def test_kv_precision_is_an_axis_on_b200_and_not_on_hopper(self):
        b = self.verdict(self.K.profile("deepseek-v4.1-flash", "b200", "agentx-mvp",
                                        self.CATALOG), "cache_dtype")
        h = self.verdict(self.K.profile("deepseek-v4.1-flash", "h100", "agentx-mvp",
                                        self.CATALOG), "cache_dtype")
        self.assertIn("nvfp4_ds_mla", b["values"])
        self.assertNotIn("nvfp4_ds_mla", h["values"],
                         "nvfp4_ds_mla is family-10 only; BLIS refuses it on SM90")
        self.assertIn("fp8_ds_mla", h["values"])

    def test_dense_model_rules_out_every_expert_knob(self):
        p = self.K.profile("qwen3-14b", "h100", "chatbot", self.CATALOG)
        for knob in ("dp + enable_expert_parallel", "all2all_backend", "moe_backend", "eplb"):
            v = self.verdict(p, knob)
            self.assertFalse(v["applies"], knob)
            self.assertIn("DENSE", v["why"])

    def test_drafting_applies_only_with_a_declared_speculator(self):
        moe = self.verdict(self.K.profile("deepseek-v4.1-flash", "h100", "agentx-mvp",
                                          self.CATALOG), "speculative")
        self.assertTrue(moe["applies"])
        self.assertIn("dspark", " ".join(str(v) for v in moe["values"]))
        self.assertIn("REFUSES every MTP-family", moe["why"])
        dense = self.verdict(self.K.profile("qwen3-14b", "h100", "chatbot", self.CATALOG),
                             "speculative")
        self.assertFalse(dense["applies"])

    def test_draft_depth_is_capped_by_the_graph(self):
        p = self.K.profile("glm-5.3-flash", "h100", "chatbot", self.CATALOG)
        v = self.verdict(p, "speculative")
        self.assertIn("[1]", " ".join(str(x) for x in v["values"]),
                      "glm-5.3-flash declares num_spec_tokens 1, so K must not exceed it")

    def test_prefix_caching_applies_only_to_a_replay(self):
        rep = self.verdict(self.K.profile("deepseek-v4.1-flash", "h100", "agentx-mvp",
                                          self.CATALOG), "enable_prefix_caching")
        shape = self.verdict(self.K.profile("qwen3-14b", "h100", "chatbot", self.CATALOG),
                             "enable_prefix_caching")
        self.assertTrue(rep["applies"])
        self.assertFalse(shape["applies"])

    def test_quantization_is_not_an_axis_on_a_quantized_checkpoint(self):
        fp8 = self.verdict(self.K.profile("deepseek-v4.1-flash", "h100", "agentx-mvp",
                                          self.CATALOG), "quantization")
        bf16 = self.verdict(self.K.profile("qwen3-14b", "h100", "chatbot", self.CATALOG),
                            "quantization")
        self.assertFalse(fp8["applies"], "the fp8 checkpoint already fixes the weight format")
        self.assertTrue(bf16["applies"])

    def test_eplb_verdict_states_the_divisibility_constraint(self):
        v = self.verdict(self.K.profile("deepseek-v4.1-flash", "h100", "agentx-mvp",
                                        self.CATALOG), "eplb")
        self.assertTrue(v["applies"])
        self.assertIn("128", v["why"])
        self.assertIn("divides", v["why"])

    def test_every_profiled_knob_is_one_the_runner_can_carry(self):
        """A profile that recommends an unsweepable knob would send a designer into a refusal."""
        p = self.K.profile("deepseek-v4.1-flash", "b200", "agentx-mvp", self.CATALOG)
        carried = (set(run_blis.SCENARIO_ENGINE) | set(run_blis.SCENARIO_PARALLEL)
                   | {"speculative", "router.endpoint_picker_config",
                      "dp + enable_expert_parallel", "mamba_cache_*"})
        for k in p["knobs"]:
            name = k["knob"].strip()
            self.assertTrue(name in carried or name.startswith("mamba_cache"),
                            f"profile names {name!r}, which the runner cannot carry")


class HeaderLength(unittest.TestCase):
    """A spec header is rationale that never reaches the report -- but it must not bury the grid.

    Measured: a generated spec opened with 36 comment lines (duplicated paragraph included)
    before `name:`, so the sweeps started below the first screen. Advisory, like the narrative
    limit: prose length should not fail a build.
    """

    def test_long_header_is_flagged(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("# c\n" * (V.HEADER_LINES + 5) + "name: t\n")
            path = f.name
        try:
            self.assertGreater(V.header_comment_lines(path), V.HEADER_LINES)
            notes = V.advisories({"sweeps": []}, path)
            self.assertTrue(any(n.startswith("header:") for n in notes), notes)
        finally:
            os.unlink(path)

    def test_short_header_is_not_flagged(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("# one line\n# two\nname: t\n")
            path = f.name
        try:
            self.assertEqual(2, V.header_comment_lines(path))
            self.assertEqual([], V.advisories({"sweeps": []}, path))
        finally:
            os.unlink(path)

    def test_a_long_header_is_not_a_validation_error(self):
        doc = spec()
        self.assertEqual([], V.validate(doc, ROUTING))

    def test_shipped_fixtures_respect_the_limit(self):
        """Every spec this repo ships is an example a contributor copies, so each must meet
        the limit it documents."""
        for path in V.spec_paths(EXPERIMENTS):
            n = V.header_comment_lines(path)
            self.assertLessEqual(n, V.HEADER_LINES,
                                 f"{os.path.basename(path)} opens with {n} comment lines")


class ExtendedValueSets(unittest.TestCase):
    def errs(self, engine):
        return V.validate(spec(sweeps=[{"name": "s", "num-instances": [2],
                                        "parallel": {"tp": [8]}, "engine": engine}]), ROUTING)

    def test_quantization_is_an_enum(self):
        errs = self.errs({"quantization": ["fp_8"]})
        self.assertTrue(any("fp8" in e for e in errs), errs)

    def test_real_quantization_values_pass(self):
        self.assertEqual([], self.errs({"quantization": ["fp8", "modelopt_fp4", "mxfp4"]}))

    def test_speculative_method_is_checked_against_the_engine_set(self):
        errs = self.errs({"speculative": {"method": ["dspar"], "num_spec_tokens": [3]}})
        self.assertTrue(any("dspark" in e for e in errs), errs)

    def test_speculative_rejects_fields_the_pinned_schema_lacks(self):
        """v0.32.0 has 7 speculative fields; pinned blis-schemas v0.5.0 holds 2."""
        errs = self.errs({"speculative": {"method": ["dspark"], "num_spec_tokens": [3],
                                          "draft_sample_method": ["greedy"]}})
        self.assertTrue(any("pinned stack carries" in e for e in errs), errs)

    def test_valid_speculative_passes(self):
        self.assertEqual([], self.errs({"speculative": {"method": ["dspark"],
                                                        "num_spec_tokens": [1, 3, 5]}}))

    def test_catalog_entry_is_optional(self):
        """CI validates specs with no catalog checkout, so the model-aware rules must skip."""
        self.assertIsNone(V.catalog_entry("qwen3-14b", catalog="/nonexistent-catalog"))


class Ranking(unittest.TestCase):
    def rows(self):
        return [
            {"label": "a", "variant": "", "concurrency": 8, "servable": True,
             "tokens_per_sec_per_gpu": 10.0, "tokens_per_sec": 160.0},
            {"label": "b", "variant": "", "concurrency": 8, "servable": True,
             "tokens_per_sec_per_gpu": 20.0, "tokens_per_sec": 320.0},
            {"label": "c", "variant": "", "concurrency": 8, "servable": False,
             "tokens_per_sec_per_gpu": 99.0, "tokens_per_sec": 999.0},
        ]

    def test_best_is_most_per_gpu_among_servable(self):
        self.assertEqual("b", E.best_of(self.rows())["label"])

    def test_unservable_rows_cannot_win(self):
        """A run that misses the SLO is not a candidate however fast it is."""
        self.assertNotEqual("c", E.best_of(self.rows())["label"])

    def test_none_when_nothing_is_servable(self):
        rows = [dict(r, servable=False) for r in self.rows()]
        self.assertIsNone(E.best_of(rows))

    def test_fleet_throughput_breaks_a_per_gpu_tie(self):
        rows = [{"label": "a", "variant": "", "concurrency": 8, "servable": True,
                 "tokens_per_sec_per_gpu": 20.0, "tokens_per_sec": 100.0},
                {"label": "b", "variant": "", "concurrency": 8, "servable": True,
                 "tokens_per_sec_per_gpu": 20.0, "tokens_per_sec": 200.0}]
        self.assertEqual("b", E.best_of(rows)["label"])

    def test_lower_concurrency_breaks_a_full_tie(self):
        rows = [{"label": "a", "variant": "", "concurrency": 64, "servable": True,
                 "tokens_per_sec_per_gpu": 20.0, "tokens_per_sec": 200.0},
                {"label": "a", "variant": "", "concurrency": 8, "servable": True,
                 "tokens_per_sec_per_gpu": 20.0, "tokens_per_sec": 200.0}]
        self.assertEqual(8, E.best_of(rows)["concurrency"])

    def test_ranking_is_deterministic_regardless_of_input_order(self):
        rows = self.rows()
        a = E.best_of(rows)
        b = E.best_of(list(reversed(rows)))
        self.assertEqual(a["label"], b["label"])


class SloHandling(unittest.TestCase):
    def test_slo_limits_converts_units(self):
        self.assertEqual({"itl_p99": 20.0, "ttft_p99": 5000.0},
                         E.slo_limits({"itl_p99": "20ms", "ttft_p99": "5s"}))

    def test_floor_rate_matches_the_itl_cap(self):
        self.assertEqual(50.0, E.floor_rate_from({"itl_p99": 20.0}))
        self.assertIsNone(E.floor_rate_from({}))

    def rec(self, **over):
        base = {"returncode": 0, "error": None,
                "metrics": {"tokens_per_sec": 100.0, "itl_p99_ms": 10.0, "ttft_p99_ms": 500.0},
                "metrics_all_instances": [
                    {"instance_id": "i0", "itl_p99_ms": 10.0, "tok_per_s_per_user": 100.0}],
                "deployment": {}}
        base.update(over)
        return base

    def test_failed_run_is_not_servable(self):
        ok, why = E.servable(self.rec(returncode=1, error="boom"), {"itl_p99": 20.0})
        self.assertFalse(ok)
        self.assertIn("boom", why)

    def test_incomplete_run_is_not_servable(self):
        ok, _ = E.servable(self.rec(error="incomplete: injected=5 != completed=4"),
                           {"itl_p99": 20.0})
        self.assertFalse(ok, "an overloaded/incomplete run is not a capacity point")

    def test_itl_over_the_cap_fails(self):
        rec = self.rec(metrics_all_instances=[
            {"instance_id": "i0", "itl_p99_ms": 50.0, "tok_per_s_per_user": 20.0}])
        ok, why = E.servable(rec, {"itl_p99": 20.0})
        self.assertFalse(ok)
        self.assertIn("itl_p99", why)

    def test_itl_under_the_cap_passes(self):
        ok, why = E.servable(self.rec(), {"itl_p99": 20.0})
        self.assertTrue(ok, why)

    def test_declared_but_unmeasured_slo_is_a_violation_not_a_pass(self):
        rec = self.rec()
        rec["metrics"].pop("ttft_p99_ms")
        ok, why = E.servable(rec, {"itl_p99": 20.0, "ttft_p99": 100.0})
        self.assertFalse(ok)
        self.assertIn("not measured", why)

    def test_ttft_over_the_cap_fails(self):
        ok, why = E.servable(self.rec(), {"ttft_p99": 100.0})
        self.assertFalse(ok)
        self.assertIn("ttft_p99", why)


# --- P/D support in the runner ----------------------------------------------------------------


class PdRunner(unittest.TestCase):
    """The P/D half of run_blis: constraints BLIS would refuse must be refused here first."""

    def cfg(self, **over):
        base = {"prefill_instances": 3, "decode_instances": 5,
                "prefill_tp": 8, "decode_tp": 8}
        base.update(over)
        return base

    def test_pools_resolve_with_instance_counts_and_footprints(self):
        par, _eng, cli, _router = run_blis.classify_config(self.cfg())
        pd = par["pd"]
        self.assertEqual(3, pd["prefill"]["instances"])
        self.assertEqual(5, pd["decode"]["instances"])
        self.assertEqual(64, run_blis.pd_total_gpus(pd))
        self.assertEqual(8, cli["num_instances"], "num_instances is derived from the two counts")

    def test_counts_must_add_up_to_num_instances(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(num_instances=7))
        self.assertIn("must add up", str(e.exception))

    def test_pool_without_counterpart_is_refused(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"prefill_instances": 2, "prefill_tp": 8})

    def test_pools_must_agree_on_dp(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(prefill_dp=2, decode_dp=1,
                                              prefill_instances=1, decode_instances=1))
        self.assertIn("replica factor", str(e.exception))

    def test_pools_must_agree_on_block_size(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(prefill_block_size=64, decode_block_size=16))
        self.assertIn("whole KV blocks", str(e.exception))

    def test_pools_must_agree_on_cache_dtype(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(prefill_cache_dtype="fp8",
                                              decode_cache_dtype="auto"))
        self.assertIn("cache formats must agree", str(e.exception))

    def test_only_nixl_connectors_are_accepted(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(pd_connector="LMCache"))
        self.assertIn("NIXL", str(e.exception))
        par, _e, _c, _r = run_blis.classify_config(self.cfg(pd_connector="NixlPullConnector"))
        self.assertEqual("NixlPullConnector", par["pd"]["connector"])

    def test_connector_is_not_emitted_as_a_cli_flag(self):
        _p, _e, cli, _r = run_blis.classify_config(self.cfg(pd_connector="NixlConnector"))
        self.assertNotIn("pd_connector", cli)

    def test_per_pool_drafting_is_refused_with_the_real_reason(self):
        """BLIS applies ONE speculative config per run, so a per-pool drafting key is a
        misunderstanding of where drafting lives -- it must say so, not "unknown key"."""
        for key in ("prefill_num_speculative_tokens", "decode_speculative_method"):
            with self.assertRaises(run_blis.ConfigError) as e:
                run_blis.classify_config(self.cfg(**{key: 3}))
            msg = str(e.exception)
            self.assertIn("one configuration per run", msg)
            self.assertNotIn("unknown key", msg)

    def test_straddling_pool_footprint_is_refused(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(prefill_tp=4, prefill_dp=3,
                                              decode_tp=4, decode_dp=3,
                                              prefill_instances=1, decode_instances=1))
        self.assertIn("whole", str(e.exception))

    def test_scenario_yaml_renders_two_pools_and_pd_transfer(self):
        par, eng, cli, _router = run_blis.classify_config(self.cfg())
        doc = run_blis.scenario_yaml("kimi-k2.5", "h200", "0.29.0", par, eng, "x",
                                     cli["num_instances"])
        self.assertIn("role: prefill", doc)
        self.assertIn("role: decode", doc)
        self.assertIn("pd_transfer", doc)
        self.assertIn("connector: NixlConnector", doc)
        self.assertIn("fabric:", doc, "a P/D cluster always spans nodes, so it needs a fabric")
        self.assertNotIn("role: colocated", doc)
        self.assertNotIn("pd:", doc, "`pd` is internal bookkeeping, not a schema field")
        # Both pools must be sized in whole nodes and sum to the cluster.
        self.assertIn("nodes: 3", doc)
        self.assertIn("nodes: 5", doc)
        self.assertIn("nodes: 8", doc)

    def test_colocated_scenario_is_unchanged(self):
        """The P/D path must not alter the colocated rendering every committed report used."""
        par, eng, _cli, _r = run_blis.classify_config({"tp": 8, "num_instances": 2})
        doc = run_blis.scenario_yaml("glm-5.3-flash", "h100", "0.29.0", par, eng, "x", 2)
        self.assertIn("role: colocated", doc)
        self.assertNotIn("pd_transfer", doc)

    def test_encode_and_shared_roles_are_refused(self):
        for key in ("encode_instances", "prefill_decode_instances"):
            with self.assertRaises(run_blis.ConfigError):
                run_blis.classify_config({key: 2})


# --- assembled report -------------------------------------------------------------------------


class DryRunPlan(unittest.TestCase):
    def test_dry_run_plans_without_touching_blis(self):
        report = E.run_experiment(spec(), blis=None, catalog=None, registry=None,
                                  out_root="/nonexistent-should-not-be-written",
                                  dry_run=True, verbose=False)
        self.assertEqual("t", report["name"])
        self.assertEqual(1, report["totals"]["configs"])
        self.assertEqual(2, report["totals"]["runs"])
        self.assertEqual(50.0, report["interactivity_floor"]["tok_per_s_per_user"])

    def test_dry_run_of_every_committed_fixture(self):
        for path in V.spec_paths(EXPERIMENTS):
            doc, _ = V.load(path)
            report = E.run_experiment(doc, None, None, None,
                                      "/nonexistent-should-not-be-written",
                                      dry_run=True, verbose=False)
            self.assertGreater(report["totals"]["runs"], 0, path)
            self.assertEqual(len(doc["sweeps"]), len(report["sweeps"]), path)
            for s in report["sweeps"]:
                self.assertTrue(s["narrative"], f"{path}: {s['name']} lost its narrative")

    def test_scenario_json_carries_the_experiment_narrative(self):
        doc, _ = V.load(os.path.join(EXPERIMENTS,
                                     "qwen3-14b-h100-chatbot-kv-precision.yaml"))
        report = E.run_experiment(doc, None, None, None, "/nonexistent", dry_run=True,
                                  verbose=False)
        sc = E._scenario_json(doc, report)
        self.assertEqual("qwen3-14b", sc["model"])
        self.assertEqual(16, sc["gpu_budget"])
        self.assertEqual({"preset": "chatbot"}, sc["workload"])
        self.assertEqual(2, len(sc["experiment"]["sweeps"]))
        self.assertTrue(sc["experiment"]["sweeps"][0]["narrative"])

    def test_pareto_json_shape(self):
        report = {"name": "x", "sweeps": [
            {"name": "s", "narrative": "", "dropped": [], "rows": [
                {"label": "a", "variant": "", "concurrency": 8, "servable": True,
                 "tokens_per_sec": 100.0, "tokens_per_sec_per_gpu": 10.0,
                 "tok_per_s_per_user": 60.0, "itl_p99_ms": 12.0, "ttft_p99_ms": 100.0}],
             "best": {"label": "a", "variant": "", "concurrency": 8, "sweep": "s",
                      "tokens_per_sec": 100.0, "tokens_per_sec_per_gpu": 10.0,
                      "tok_per_s_per_user": 60.0, "itl_p99_ms": 12.0}}],
            "best_overall": {"label": "a", "variant": "", "sweep": "s", "concurrency": 8,
                             "tokens_per_sec": 100.0, "tokens_per_sec_per_gpu": 10.0}}
        out = E._pareto_json(report)
        self.assertEqual("a", out["winner"]["chosen"])
        self.assertEqual(1, len(out["finalists"]))
        self.assertEqual(1, len(out["finalists"][0]["curve"]))
        self.assertIn("tok/s/GPU", out["winner"]["reason"])

    def test_recorded_commands_do_not_depend_on_where_the_sweep_ran(self):
        """A re-run's ledger must differ only when a RESULT differs.

        `cmd` recorded absolute --scenarios/--registry paths, so one spec swept into two output
        directories produced ledgers differing in those strings while every measured number was
        identical -- a diff that means "ran somewhere else", not "measured something else".
        """
        rec = {"cmd": ["/opt/blis", "run", "--scenarios",
                       "/tmp/out/exp/scenarios", "--registry", "/elsewhere/blis-registry",
                       "--seed", "42"]}
        E._relativize_cmd(rec, "/tmp/out/exp")
        self.assertEqual(os.path.join(".", "scenarios"), rec["cmd"][3])
        self.assertEqual("/elsewhere/blis-registry", rec["cmd"][5],
                         "a path outside the repo and run dir is left intact")
        self.assertEqual(["--seed", "42"], rec["cmd"][6:], "flags and values are preserved")

    def test_relativize_tolerates_a_row_with_no_command(self):
        rec = {"cmd": None}
        E._relativize_cmd(rec, "/tmp/out")
        self.assertIsNone(rec["cmd"])

    def test_max_parallel_does_not_change_the_ledger(self):
        """--max-parallel must be a SPEED knob, never a result knob.

        A report is a pure function of the spec plus the pinned stack -- that is what lets CI
        re-run a spec and get the same bytes. Running the grid concurrently must therefore keep
        the expansion order, not the completion order. The stub finishes the FIRST work item
        last (descending sleep), so a build that appended results as they completed would
        reverse the ledger and fail here.
        """
        import threading, time as _t
        spec = {
            "name": "parallel-order", "sweeps": [{
                "name": "topo", "narrative": "n",
                "num-instances": [2, 4],
                "parallel": {"tp": [4, 8], "dp": [1, 2, 4], "enable_expert_parallel": [True]},
            }],
            "scenario": {"model": "m", "hardware": "h", "workload": "chatbot", "gpus": 32,
                         "slo": {"itl_p99": "20ms"}, "concurrency": [8, 16, 32]},
        }
        seen_threads = set()
        calls = {"n": 0}
        lock = threading.Lock()
        # Expansion order == the order the sweeper itemizes the grid: candidates in
        # expand_sweep order, each crossed with the concurrency ladder.
        kept, _dropped = E.expand_sweep(spec["sweeps"][0], 32)
        expected_rank = {}
        for cand in kept:
            for c in spec["scenario"]["concurrency"]:
                key = (cand["num_instances"], cand["tp"], cand["dp"], c)
                expected_rank[key] = len(expected_rank)

        def fake_run_one(*a, **kw):
            cfg = kw.get("config") or a[11]   # run_one(...,horizon,config,...)
            with lock:
                seen_threads.add(threading.current_thread().name)
                order = calls["n"]; calls["n"] += 1
            # The metric is derived from the CONFIG, not from call order, so it identifies the
            # work item no matter which thread picked it up. Sleeping longer for items handed
            # out EARLY makes completion order differ from expansion order.
            key = (cfg["num_instances"], cfg["tp"], cfg["dp"], cfg["concurrency"])
            _t.sleep(max(0.0, (12 - order) * 0.004))
            return {"metrics": {"tokens_per_sec": float(expected_rank[key]) + 1.0, "itl_p99_ms": 1.0},
                    "cmd": None, "config": dict(cfg)}

        real = E.run_blis.run_one
        E.run_blis.run_one = fake_run_one
        try:
            import tempfile as _tf
            with _tf.TemporaryDirectory() as out:
                out1 = E.run_experiment(dict(spec), "/opt/blis", "/c", "/r", out,
                                        verbose=False, max_parallel=1)
                calls["n"] = 0
                out8 = E.run_experiment(dict(spec), "/opt/blis", "/c", "/r", out,
                                        verbose=False, max_parallel=8)
        finally:
            E.run_blis.run_one = real

        self.assertGreater(len(seen_threads), 1, "max_parallel 8 should use several threads")
        # tokens_per_sec encodes the work-item index (100 + i), and the stub hands out indices
        # in CALL order while finishing in reverse. So this sequence is the real probe: it is
        # 100..111 only if results were re-sorted into expansion order.
        seq1 = [r["tokens_per_sec"] for r in out1["sweeps"][0]["rows"]]
        seq8 = [r["tokens_per_sec"] for r in out8["sweeps"][0]["rows"]]
        self.assertEqual(seq1, seq8, "row order must be expansion order, not completion order")
        self.assertEqual(sorted(seq8), seq8,
                         "parallel results must be re-sorted back into expansion order")
        self.assertEqual(out1["sweeps"][0]["best"], out8["sweeps"][0]["best"])
        self.assertEqual(out1.get("winner"), out8.get("winner"))

    def test_report_json_quotes_each_sweep_hypothesis(self):
        """The narrative seam must carry the spec's PRE-RUN hypothesis beside the outcome; this
        path has no LLM, so it must quote rather than invent."""
        report = {
            "name": "x", "description": "",
            "scenario": {"model": "m", "hardware": "h100", "workload": "chatbot", "gpus": 8,
                         "concurrency": [4], "seed": 42, "replay": None},
            "totals": {"configs": 1, "runs": 1, "servable": 1},
            "sweeps": [{"name": "s", "narrative": "A testable claim.", "configs": 1, "runs": 1,
                        "dropped": [], "rows": [],
                        "best": {"label": "2x TP4", "variant": "", "concurrency": 4,
                                 "tokens_per_sec": 100.0, "tokens_per_sec_per_gpu": 12.5}}],
            "best_overall": {"label": "2x TP4", "variant": "", "sweep": "s", "concurrency": 4,
                             "tokens_per_sec": 100.0, "tokens_per_sec_per_gpu": 12.5},
        }
        out = E._report_json(report)
        self.assertIn("A testable claim.", out["sections"][0]["prose"])
        self.assertTrue(any("Measured:" in p for p in out["sections"][0]["prose"]))
        self.assertEqual(3, len(out["hero"]))
        self.assertTrue(out["hero"][-1].get("run"))
        self.assertNotIn("operating_points", out,
                         "the table is derived from runs.jsonl, not authored")
        self.assertNotIn("glossary", out, "the shared lexicon owns the vocabulary")

    def test_report_json_states_when_nothing_was_servable(self):
        report = {
            "name": "x", "description": "", "totals": {"configs": 1, "runs": 1, "servable": 0},
            "scenario": {"model": "m", "hardware": "h100", "workload": "chatbot", "gpus": 8,
                         "concurrency": [4], "seed": 42, "replay": None},
            "sweeps": [{"name": "s", "narrative": "", "configs": 1, "runs": 1, "dropped": [],
                        "rows": [], "best": None}],
            "best_overall": None,
        }
        out = E._report_json(report)
        self.assertIn("No configuration held", out["hero"][1]["html"])

    def test_finalists_carry_the_fields_the_gallery_reads(self):
        """build_gallery reads finalist.config.tp / .num_instances / .gpus for the winner card.

        Without them the card rendered "tp 0, 0 instances" beside a "2x TP4" label.
        """
        report = {"name": "x", "sweeps": [
            {"name": "s", "narrative": "", "dropped": [], "rows": [],
             "best": {"label": "2x TP4", "variant": "", "concurrency": 16, "sweep": "s",
                      "gpus": 8, "config": {"tp": 4, "num_instances": 2},
                      "tokens_per_sec": 2016.8, "tokens_per_sec_per_gpu": 252.1,
                      "tok_per_s_per_user": 89.0, "itl_p99_ms": 11.2}}],
            "best_overall": None}
        f = E._pareto_json(report)["finalists"][0]
        self.assertEqual(4, f["config"]["tp"])
        self.assertEqual(2, f["config"]["num_instances"])
        self.assertEqual(8, f["gpus"])

    def test_disaggregated_config_exposes_flat_topology_for_the_card(self):
        """A P/D config has no plain tp/num_instances, so they are derived for the summary card."""
        out = E._jsonable({"prefill_instances": 3, "decode_instances": 5, "prefill_tp": 8})
        self.assertEqual(8, out["tp"])
        self.assertEqual(8, out["num_instances"], "both pools' instances")
        self.assertEqual(3, out["prefill_instances"], "the per-pool truth is preserved")

    def test_colocated_config_is_not_rewritten(self):
        out = E._jsonable({"tp": 4, "num_instances": 2})
        self.assertEqual({"tp": 4, "num_instances": 2}, out)

    def test_winner_name_carries_the_variant(self):
        report = {"name": "x", "sweeps": [], "best_overall": {
            "label": "2x TP8", "variant": "max seqs 64", "sweep": "s", "concurrency": 8,
            "tokens_per_sec": 100.0, "tokens_per_sec_per_gpu": 10.0}}
        out = E._pareto_json(report)
        self.assertEqual("2x TP8 (max seqs 64)", out["winner"]["chosen"])


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
