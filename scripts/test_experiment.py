#!/usr/bin/env python3
"""Tests for the Experiment Spec pipeline: validator, grid expansion, router overrides, ranking.

    python3 -I scripts/test_experiment.py

Hermetic: no BLIS, no network. Everything here is a pure function of a spec, which is exactly
the property that lets CI run the sweeper unattended -- so these tests are the guard on it.
"""
import copy
import glob
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

import run_experiment as E          # noqa: E402
import validate_experiment as V     # noqa: E402
import run_blis                     # noqa: E402
import blis_cli                     # noqa: E402
import settings                     # noqa: E402
import fetch_corpus as FC           # noqa: E402

# TEMPORARY: the inference-sim commands, from scripts/testdata/blis-m1 (inference-sim#2077).
BLIS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "blis-m1", "blis")

EXPERIMENTS = os.path.join(ROOT, "experiments")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
ROUTING = os.path.join(ROOT, "routing")


def spec(**over):
    """A minimal valid spec, overridable per test."""
    base = {
        "name": "t",
        "scenario": {"model": "qwen3-14b", "hardware": "h100", "workload": "chatbot",
                     "gpus": 16, "slo": {"itl_p99": "20ms"}, "concurrency": [1, 2],
                     "cluster": {"gpus_per_node": 8}},
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
            self.assertEqual([], V.validate(doc, BLIS, ROUTING), f"{path} should validate")

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


# --- schema / validator agreement -------------------------------------------------------------


def _structurally_bad():
    """Specs broken in ways a JSON Schema CAN see, each one mutation off a real shipped spec.

    Structural only: no typo'd knob name, no illegal enum, no impossible GPU arithmetic. Those
    are the three classes schema.json's own description says it cannot reach, so requiring the
    schema to catch them would be asserting a contract nobody signed.
    """
    base, _ = V.load(os.path.join(FIXTURES, "engine-spec.yaml"))

    def mutate(fn):
        doc = copy.deepcopy(base)
        fn(doc)
        return doc

    def sweep0(doc):
        return doc["sweeps"][0]

    cases = {
        "no name": lambda d: d.pop("name"),
        "no scenario": lambda d: d.pop("scenario"),
        "no sweeps": lambda d: d.pop("sweeps"),
        "empty sweeps": lambda d: d.__setitem__("sweeps", []),
        "sweeps not a list": lambda d: d.__setitem__("sweeps", {"name": "s"}),
        "illegal name chars": lambda d: d.__setitem__("name", "Not A Valid Name!"),
        "unknown top-level key": lambda d: d.__setitem__("bogus", 1),
        "unknown scenario key": lambda d: d["scenario"].__setitem__("bogus", 1),
        "unknown sweep key": lambda d: sweep0(d).__setitem__("bogus", 1),
        "sweep without a name": lambda d: sweep0(d).pop("name", None),
        "gpus as a string": lambda d: d["scenario"].__setitem__("gpus", "32"),
        "gpus = 0": lambda d: d["scenario"].__setitem__("gpus", 0),
        "slo value not a number": lambda d: d["scenario"].__setitem__("slo_ms", {"ttft_p99": "fast"}),
        "scalar parallel knob": lambda d: sweep0(d).setdefault("parallel", {}).__setitem__("tp", 8),
        "scalar engine knob": lambda d: sweep0(d).setdefault("engine", {}).__setitem__(
            "max_num_seqs", 256),
        "empty knob list": lambda d: sweep0(d).setdefault("parallel", {}).__setitem__("tp", []),
        "colocated AND disaggregated": lambda d: sweep0(d).update(
            {"prefill-instances": [2], "decode-instances": [2]}),
        "cluster not a mapping": lambda d: d["scenario"].__setitem__("cluster", "roce-200g"),
        "unknown cluster key": lambda d: d["scenario"].__setitem__("cluster", {"fabric": "ib-400g",
                                                                               "bogus": 1}),
    }
    return {name: mutate(fn) for name, fn in cases.items()}


class SchemaAgreesWithValidator(unittest.TestCase):
    """experiments/schema.json is published to contributors as the structural contract, and an
    editor red-underlines a spec against it -- but NOTHING enforces it: the validator does all
    its structural checking in hand-written Python and never loads the file. Two independent
    definitions of the same structure, only one of them executed, is a drift generator
    (susiejojo review).

    The harmful direction is the schema REJECTING a spec the validator accepts: a contributor's
    editor calls a valid spec broken, and the authority that says otherwise is a script they may
    not run. These tests pin both directions so drift cannot land silently.

    Needs `jsonschema`, which is not a repo dependency (pyyaml is the only one) -- skipped when
    absent rather than added to all six workflows, and installed in the one CI job that runs
    these tests.
    """

    def setUp(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("needs `pip install jsonschema`")
        self.jsonschema = jsonschema
        with open(os.path.join(EXPERIMENTS, "schema.json"), encoding="utf-8") as f:
            self.schema = json.load(f)
        self.validator = jsonschema.Draft202012Validator(self.schema)

    def errors(self, doc):
        return sorted(self.validator.iter_errors(doc), key=lambda e: list(e.path))

    def test_the_schema_itself_is_a_valid_2020_12_schema(self):
        self.jsonschema.Draft202012Validator.check_schema(self.schema)

    def test_no_keyword_is_silently_misspelled(self):
        """`check_schema` does NOT catch a typo'd keyword: per 2020-12 an unrecognised keyword is
        an ANNOTATION, not an error, so `maxLenght: 12` validates fine and constrains nothing.
        Verified -- the committed schema with that one edit passes check_schema and every other
        test in this class, because a rule that quietly stopped existing rejects nothing.

        So the keywords are checked against the dialect's own vocabulary. Anything not a known
        keyword must be a schema-defined name: a `$defs` entry, or a property name (the keys
        under a `properties` object are spec fields, not keywords).
        """
        known = set(self.jsonschema.Draft202012Validator.VALIDATORS) | {
            "$schema", "$id", "$defs", "$ref", "$comment", "title", "description", "default",
            "examples", "deprecated", "readOnly", "writeOnly", "definitions",
        }
        # Keywords whose KEYS are spec field names rather than keywords. Their values differ:
        # a subschema per key for `properties`, but a plain list of names for `dependentRequired`.
        NAME_KEYED_SUBSCHEMAS = {"properties", "$defs", "definitions", "patternProperties",
                                 "dependentSchemas"}
        NAME_KEYED_PLAIN = {"dependentRequired"}

        def walk(node, path):
            if isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{path}[{i}]")
                return
            if not isinstance(node, dict):
                return
            for key, value in node.items():
                self.assertTrue(
                    key in known,
                    f"schema.json: {path}.{key} is not a JSON Schema 2020-12 keyword -- a "
                    f"misspelled keyword is an annotation that silently constrains nothing")
                if key in NAME_KEYED_PLAIN:
                    continue                         # values are name lists, not subschemas
                if key in NAME_KEYED_SUBSCHEMAS:
                    for name, sub in (value or {}).items():
                        walk(sub, f"{path}.{key}.{name}")
                    continue
                walk(value, f"{path}.{key}")

        walk(self.schema, "<root>")

    def test_every_shipped_spec_passes_the_published_schema(self):
        """The harmful direction. Each of these validates, so each must also satisfy the schema
        a contributor's editor reads -- otherwise the editor contradicts the authority."""
        for path in V.spec_paths(EXPERIMENTS):
            doc, err = V.load(path)
            self.assertIsNone(err, f"{path}: {err}")
            self.assertEqual([], V.validate(doc, BLIS, ROUTING), f"{path} should validate")
            errs = self.errors(doc)
            self.assertEqual(
                [], errs,
                f"{os.path.basename(path)} passes the validator but the published schema "
                f"rejects it: " + "; ".join(
                    f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errs[:3]))

    def test_both_reject_every_structurally_bad_spec(self):
        """The drift direction: a structural rule the validator enforces and the schema has lost
        (or never had). Asserted per case so a failure names the rule that drifted."""
        for name, doc in _structurally_bad().items():
            with self.subTest(case=name):
                self.assertTrue(self.errors(doc),
                                f"schema.json accepts a structurally-bad spec ({name}) that the "
                                f"validator rejects -- the schema has drifted behind")
                self.assertTrue(V.validate(copy.deepcopy(doc), BLIS, ROUTING),
                                f"the validator accepts a structurally-bad spec ({name})")

    def test_the_corpus_is_one_mutation_off_a_spec_that_really_passes(self):
        """Guards the corpus, not the schema: if the base spec stopped validating, every case
        above would 'pass' for the wrong reason and the suite would go quietly blind."""
        base, err = V.load(os.path.join(FIXTURES, "engine-spec.yaml"))
        self.assertIsNone(err)
        self.assertEqual([], V.validate(copy.deepcopy(base), BLIS, ROUTING))
        self.assertEqual([], self.errors(base))

    def test_the_schema_stays_looser_only_where_it_says_it_is(self):
        """The three classes schema.json documents as out of reach really are out of reach, and
        the validator really does cover them. Pins the division of labour in both directions: if
        the schema ever grew a knob-name enum, this says so instead of leaving two places to
        maintain the vocabulary; if the validator ever stopped checking, it says that too."""
        base, _ = V.load(os.path.join(FIXTURES, "engine-spec.yaml"))

        def typo(d):
            d["sweeps"][0].setdefault("engine", {})["max_num_seq"] = [256]

        def bad_gpus(d):
            d["scenario"]["gpus"] = 8
            for sweep in d["sweeps"]:
                sweep["num-instances"] = [3]
                sweep.setdefault("parallel", {})["tp"] = [3]

        for name, fn in (("knob name", typo), ("GPU arithmetic", bad_gpus)):
            with self.subTest(case=name):
                doc = copy.deepcopy(base)
                fn(doc)
                self.assertEqual(
                    [], self.errors(doc),
                    f"schema.json now catches {name}, which its description says it cannot "
                    f"reach -- either the description or the schema is wrong")
                self.assertTrue(V.validate(copy.deepcopy(doc), BLIS, ROUTING),
                                f"the validator must be the one catching {name}")

    def test_the_schema_says_it_is_not_the_authority(self):
        """The README calls schema.json the structural contract, so the file itself has to point
        at what checks the rest -- a contributor reading only the schema should not think a
        green editor means a runnable spec."""
        desc = (self.schema.get("description") or "").lower()
        self.assertIn("validate_experiment.py", desc,
                      "schema.json's description should name the script that is the authority")
        self.assertIn("structure", desc,
                      "schema.json's description should say it checks STRUCTURE only")


# --- validator --------------------------------------------------------------------------------


class ClusterPackaging(unittest.TestCase):
    """The spec's node size against the chip's catalog entry (design §2.2, SUB-3), read through
    `blis catalog resolve --chip` (TEMPORARY fixture, inference-sim#2077). The GPU arithmetic
    reads the resolved node size."""

    def errs(self, doc):
        return V.validate(doc, BLIS, ROUTING)

    def gb200(self, cluster=None, tp=8):
        doc = spec()
        doc["scenario"]["hardware"] = "gb200-nvl72"
        doc["scenario"].pop("cluster")
        if cluster is not None:
            doc["scenario"]["cluster"] = cluster
        doc["sweeps"] = [{"name": "s", "num-instances": [2], "parallel": {"tp": [tp]}}]
        return doc

    def test_gb200_needs_no_node_size(self):
        self.assertEqual([], self.errs(self.gb200()))
        self.assertEqual([], self.errs(self.gb200({"gpus_per_node": 4})))   # restated unchanged

    def test_gb200_contradicting_its_catalog_entry_is_refused(self):
        errs = self.errs(self.gb200({"gpus_per_node": 8}))
        self.assertTrue(any("hardware/gb200-nvl72.yaml" in e and "at 4" in e for e in errs), errs)

    def test_a_rack_is_never_the_specs(self):
        errs = self.errs(self.gb200({"gpus_per_rack": 72}))
        self.assertTrue(any("scenario.cluster.gpus_per_rack: unknown key" in e for e in errs), errs)

    def test_h100_without_a_node_size_is_refused(self):
        doc = spec()
        doc["scenario"].pop("cluster")
        errs = self.errs(doc)
        self.assertTrue(any("scenario.cluster.gpus_per_node: required" in e for e in errs), errs)
        self.assertEqual([], self.errs(spec()))                               # 8, stated

    def test_whole_node_arithmetic_reads_the_resolved_node_size(self):
        # 2 x TP6 = 12 GPUs on 4-GPU trays straddles a tray; on stated 8-GPU nodes TP6 fits one.
        bad = self.gb200(tp=6)
        bad["scenario"]["gpus"] = 12
        self.assertTrue(any("straddles a node boundary" in e and "4-GPU" in e for e in self.errs(bad)))
        ok = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [6]}}])
        ok["scenario"]["gpus"] = 12
        self.assertFalse([e for e in self.errs(ok) if "straddles" in e])

    def test_committed_specs_state_the_node_size_only_where_the_catalog_does_not(self):
        """X19's stated-values test for the cluster: a generated scenario's packaging is either
        the spec's statement or the catalog's, so a GB200 spec states neither field and every
        other spec states its node size."""
        for path in V.spec_paths(EXPERIMENTS):
            doc, _ = V.load(path)
            cluster = doc["scenario"].get("cluster") or {}
            fixed = blis_cli.packaging(BLIS, doc["scenario"]["hardware"])["gpus_per_node"]
            if fixed is not None:
                self.assertNotIn("gpus_per_node", cluster, path)
            else:
                self.assertIsInstance(cluster.get("gpus_per_node"), int, path)


class ValidatorRejects(unittest.TestCase):
    def errs(self, doc):
        return V.validate(doc, BLIS, ROUTING)

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

    def test_throughput_slo_is_a_positive_number(self):
        ok = spec()
        ok["scenario"]["slo"] = {"throughput_per_gpu": 500}
        self.assertFalse([e for e in self.errs(ok) if "slo" in e])
        bad = spec()
        bad["scenario"]["slo"] = {"throughput_per_gpu": "500ms"}
        self.assertTrue(any("positive tok/s" in e for e in self.errs(bad)))

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

    def test_cluster_must_be_a_mapping_with_only_fabric(self):
        bad = spec()
        bad["scenario"]["cluster"] = "roce-200g"
        self.assertTrue(any("scenario.cluster:" in e for e in self.errs(bad)))
        bad2 = spec()
        bad2["scenario"]["cluster"] = {"fabric": "ib-400g", "bogus": 1}
        self.assertTrue(any("scenario.cluster.bogus" in e for e in self.errs(bad2)))

    def test_fabric_omitted_or_empty_cluster_is_structural_error(self):
        # Present but empty is a mistake, not the IB default (which comes from OMITTING cluster).
        bad = spec()
        bad["scenario"]["cluster"] = {}
        self.assertTrue(any("scenario.cluster:" in e for e in self.errs(bad)))

    def test_duration_parsing(self):
        self.assertEqual(20.0, V.parse_duration_ms("20ms"))
        self.assertEqual(5000.0, V.parse_duration_ms("5s"))
        self.assertEqual(0.5, V.parse_duration_ms("500us"))
        self.assertIsNone(V.parse_duration_ms("20"))
        self.assertIsNone(V.parse_duration_ms("20 ms"))

    def test_knob_vocabulary_reports_its_source(self):
        vocab = V.knob_vocabulary(BLIS)
        self.assertIn("tp", vocab.parallel)
        self.assertIn("max_num_seqs", vocab.engine_names())
        self.assertIn("blis fields", vocab.source)


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

    def test_router_filename_is_a_content_hash_not_a_semantic_one(self):
        """The generated name hashes BYTES, and the header must not promise more than that.

        Reordering keys in the base file produces equivalent YAML with a different digest, so the
        arm is renamed. That is acceptable -- the base files are committed, so their key order is
        fixed, and the name only has to be stable across runs of the same inputs -- but the header
        used to read "the same overrides always produce this file", which overstates it.
        """
        overrides = {("plugins", "prefix-based-pd-decider", "parameters",
                      "nonCachedTokens"): 4096}
        plugin = ("plugins:\n- type: prefix-based-pd-decider\n"
                  "  parameters:\n    nonCachedTokens: 16\n")
        with tempfile.TemporaryDirectory() as d:
            # The same mapping, written with its two top-level keys in each order.
            with open(os.path.join(d, "a.yaml"), "w", encoding="utf-8") as f:
                f.write("apiVersion: inference.networking.x-k8s.io/v1alpha1\n" + plugin)
            with open(os.path.join(d, "b.yaml"), "w", encoding="utf-8") as f:
                f.write(plugin + "apiVersion: inference.networking.x-k8s.io/v1alpha1\n")
            out = os.path.join(d, "out")
            paths = []
            for base in ("a.yaml", "b.yaml"):
                resolved, _ = E.materialize_router(
                    {"router_config": base, "router_overrides": dict(overrides)}, d, out)
                paths.append(resolved["router_config"])
            self.assertEqual(2, len(set(paths)),
                             "byte-level hashing renames the arm when base key order changes")
            for path in paths:
                with open(path, encoding="utf-8") as f:
                    self.assertIn("not a semantic hash", f.read(),
                                  "the header must not promise more than a byte hash")

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

    def test_colocated_label_shows_pipeline_parallelism(self):
        # pp is a GPU-footprint factor (tp*pp*dp), so a pp>1 shape must not read like pp=1.
        self.assertEqual("4x TP8xPP2 + EP", E.label_of(
            {"tp": 8, "pp": 2, "num_instances": 4, "enable_expert_parallel": True}))
        self.assertEqual("4x TP4xPP4xDP2", E.label_of(
            {"tp": 4, "pp": 4, "dp": 2, "num_instances": 4}))

    def test_pipeline_parallelism_hidden_when_one(self):
        # pp=1 (the default) stays suppressed, like dp=1, so plain shapes are uncluttered.
        self.assertEqual("8x TP8 + EP", E.label_of(
            {"tp": 8, "pp": 1, "num_instances": 8, "enable_expert_parallel": True}))

    def test_disaggregated_label_names_both_pools(self):
        # Matching pool shapes collapse to one shape after the P/D instance counts.
        self.assertEqual("3P/5D TP8 + EP", E.label_of(
            {"prefill_instances": 3, "decode_instances": 5,
             "prefill_tp": 8, "prefill_enable_expert_parallel": True,
             "decode_tp": 8, "decode_enable_expert_parallel": True}))

    def test_disaggregated_label_splits_when_pools_differ(self):
        # A wide-decode sweep runs prefill tp8 against decode tp16; one shape would hide the other.
        self.assertEqual("2P TP8 + EP / 3D TP16 + EP", E.label_of(
            {"prefill_instances": 2, "decode_instances": 3,
             "prefill_tp": 8, "prefill_enable_expert_parallel": True,
             "decode_tp": 16, "decode_enable_expert_parallel": True}))

    def test_disaggregated_label_shows_pool_pipeline_parallelism(self):
        # pp>1 is priced on each pool of a disaggregated run too (kernel v0.5.0), so label it.
        # Here only the prefill pool is pipelined, so the pools differ and the label splits.
        self.assertEqual("2P TP8xPP2 + EP / 6D TP8 + EP", E.label_of(
            {"prefill_instances": 2, "decode_instances": 6,
             "prefill_tp": 8, "prefill_pp": 2, "prefill_enable_expert_parallel": True,
             "decode_tp": 8, "decode_enable_expert_parallel": True}))

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


class KnobSurfaceComesFromTheSimulator(unittest.TestCase):
    """The validator and the runner read one vocabulary, from the simulator (CFG-3)."""

    def test_the_vocabulary_is_the_field_lists(self):
        vocab = V.knob_vocabulary(BLIS)
        fields = blis_cli.fields(BLIS, "vllm")
        config = {f["name"] for f in fields["fields"] if f["source"] == "config"}
        for name in vocab.flat - set(settings.ENGINE_BLOCK_NAMES):
            self.assertIn(name, config, name)
        self.assertEqual("0.32.0", vocab.release)
        self.assertIn("blis fields", vocab.source)

    def test_renamed_fields_are_known_by_the_engine_block_name_only(self):
        """The engine block refuses vLLM's name for these four, so a spec must not use it."""
        vocab = V.knob_vocabulary(BLIS)
        self.assertIn("cache_dtype", vocab.flat)
        self.assertNotIn("kv_cache_dtype", vocab.flat)
        self.assertIn("method", vocab.blocks["speculative"])
        self.assertNotIn("speculative_config", vocab.engine_names() - set(vocab.blocks))
        self.assertIn("frontend", vocab.flat, "frontend is the engine block's VLLM_USE_RUST_FRONTEND")

    def test_the_validator_and_the_runner_agree(self):
        """A name the validator accepts is one the runner can carry, and the converse."""
        vocab = V.knob_vocabulary(BLIS)
        for name in sorted(vocab.flat)[:200]:
            run_blis.classify_config({"tp": 8, name: 1}, vocab, 8)
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "max_num_seq": 1}, vocab, 8)

    def test_the_repository_holds_no_copied_settings_list(self):
        """CFG-3's check: no hand-kept list of engine settings, and no vllm_arg_policy.yaml."""
        for path in glob.glob(os.path.join(ROOT, "**", "vllm_arg_policy*"), recursive=True):
            self.fail(f"{path}: a copied settings list; read `blis fields` instead")
        for mod in (V, run_blis, E):
            for gone in ("ENGINE_KNOBS", "PARALLEL_KNOBS", "ENGINE_ENUMS", "SCENARIO_ENGINE",
                         "ENGINE_NOT_WIRED", "ENGINE_DECODED_BUT_INERT", "NESTED_ENGINE_FIELDS"):
                self.assertFalse(hasattr(mod, gone), f"{mod.__name__}.{gone}")

    def test_every_variant_key_has_a_readable_name(self):
        """A key with no KNOB_NAMES entry prints its raw field name in a variant label."""
        import topologies
        bare = [k for k in E.VARIANT_KEYS if k not in topologies.KNOB_NAMES]
        self.assertEqual([], bare, f"variant keys with no readable name: {bare}")

    def test_a_simulator_without_the_commands_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "blis")
            with open(fake, "w") as f:
                f.write("#!/bin/sh\necho 'unknown command' >&2\nexit 1\n")
            os.chmod(fake, 0o755)
            with self.assertRaises(blis_cli.BlisCommandError) as e:
                V.knob_vocabulary(fake)
            self.assertIn("unknown command", str(e.exception))

class EngineNameChecking(unittest.TestCase):
    def errs(self, engine):
        doc = spec(sweeps=[{"name": "s", "num-instances": [2], "parallel": {"tp": [8]},
                            "engine": engine}])
        return V.validate(doc, BLIS, ROUTING)

    def test_list_valued_knob_needs_a_list_of_lists(self):
        errs = self.errs({"cudagraph_capture_sizes": [1, 2, 4]})
        self.assertTrue(any("list of lists" in e for e in errs), errs)

    def test_list_valued_knob_accepts_a_list_of_lists(self):
        self.assertEqual([], self.errs({"cudagraph_capture_sizes": [[1, 2, 4], [1, 2, 4, 8]]}))

    def test_shared_scenario_engine_names_are_checked_too(self):
        doc = spec()
        doc["scenario"]["engine"] = {"max_num_seq": 64}
        self.assertTrue(any("scenario.engine.max_num_seq" in e
                            for e in V.validate(doc, BLIS, ROUTING)))

    def test_nested_block_unknown_inner_field_is_refused(self):
        errs = self.errs({"engram_config": {"cpu_offlaod": [True]}})
        self.assertTrue(any("cpu_offload" in e for e in errs), errs)

    def test_nested_block_inner_values_must_be_lists(self):
        errs = self.errs({"engram_config": {"cpu_offload": True}})
        self.assertTrue(any("must be a list" in e for e in errs), errs)

    def test_valid_nested_block_passes(self):
        self.assertEqual([], self.errs({"engram_config": {"cpu_offload": [True, False]}}))

    def test_an_inert_setting_is_refused_by_its_field_list_status(self):
        """The field list says which settings change no decision (VLM-3 statuses). Sweeping one
        reports the same number in every arm, so the validator refuses it with that status."""
        status = settings.inert_reason(V.knob_vocabulary(BLIS), "stream_interval")
        self.assertIsNone(status, "stream_interval is not_modelled in the pinned field list")
        inert = sorted(n for n, e in V.knob_vocabulary(BLIS).status.items()
                       if e["status"] == "inert_unproven" and "." not in n)
        self.assertTrue(inert, "the pinned field list has inert_unproven fields")
        errs = self.errs({inert[0]: [1, 2]})
        self.assertTrue(any("inert_unproven" in e for e in errs), errs)

    def test_values_are_left_to_the_engine(self):
        """Value sets are the engine's (X15): it refuses a bad value at start-up with its own
        message, so the validator keeps no copy of them."""
        self.assertEqual([], self.errs({"cudagraph_mode": ["full"]}))

    def test_nested_block_expands_to_one_candidate_per_combination(self):
        kept, _ = E.expand_sweep({"name": "s", "num-instances": [1], "parallel": {"tp": [8]},
                                  "engine": {"engram_config": {"cpu_offload": [True, False]}}}, 8)
        self.assertEqual([{"cpu_offload": True}, {"cpu_offload": False}],
                         [c["engram_config"] for c in kept])

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
        return V.validate(doc, BLIS, ROUTING)

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

    def test_a_knob_fixed_in_scenario_engine_counts_for_the_cross_check(self):
        """scenario.engine pins a value for every run, so it participates in the constraint."""
        errs = self.errs({"long_prefill_token_threshold": [65536]},
                         scenario_engine={"max_model_len": 32768})
        self.assertTrue(any("exceeds every max_model_len" in e for e in errs), errs)

    def test_committed_fixtures_satisfy_every_cross_knob_rule(self):
        for path in V.spec_paths(EXPERIMENTS):
            doc, err = V.load(path)
            self.assertIsNone(err, f"{path}: {err}")
            self.assertEqual([], V.validate(doc, BLIS, ROUTING), path)


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
        self.assertEqual([], V.validate(doc, BLIS, ROUTING),
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

    def test_every_shipped_replay_spec_can_actually_run(self):
        """A committed replay spec must be runnable, not merely valid.

        The gap this closes: both AgentX examples validated and dry-ran clean while omitting
        `replay.corpus`, so the post-merge sweep would SystemExit before a single BLIS run -- the
        one path run-experiment.yml exists to drive (qa-review on #22, F1/G7). --dry-run cannot
        catch it, because dry runs deliberately tolerate an unfetched corpus.

        Resolving each spec in non-dry mode is the check: it is what the sweeper does, and it
        raises on a replay with no corpus path.
        """
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        checked = 0
        for path in V.spec_paths(EXPERIMENTS):
            doc, err = V.load(path)
            self.assertIsNone(err, f"{path}: {err}")
            sc = doc.get("scenario") or {}
            if E.replay_descriptor(sc.get("workload"), self.CATALOG) is None:
                continue
            checked += 1
            try:
                out = E.resolve_replay(sc.get("workload"), self.CATALOG, sc.get("replay"))
            except SystemExit as exc:
                self.fail(f"{os.path.basename(path)} is a replay spec that cannot run: {exc}")
            corpus = out.get("corpus") or ""
            self.assertTrue(corpus, f"{os.path.basename(path)}: no corpus path")
            self.assertFalse(os.path.isabs(corpus),
                             f"{os.path.basename(path)}: corpus path {corpus!r} is absolute, "
                             f"which pins the spec to one machine")
        self.assertTrue(checked, "no replay spec was exercised -- has the fixture set changed?")

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


class CorpusProvisioning(unittest.TestCase):
    """scripts/fetch_corpus.py provisions the one input a replay spec cannot carry.

    The corpus is 568 MB and deliberately not vendored, so CI has to fetch it and check it
    against the catalog's digest before any BLIS run (qa-review on #22, F1/G7). The dataset,
    the PINNED revision and the sha256 are read back out of the descriptor rather than restated
    in the workflow, so there is no second copy of the catalog pin to drift.
    """

    CATALOG = os.path.join(ROOT, os.pardir, "blis-catalog")

    def have_catalog(self):
        return os.path.isfile(os.path.join(self.CATALOG, "workloads", "agentx-mvp.yaml"))

    def test_a_generated_shape_needs_no_corpus(self):
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        self.assertIsNone(FC._descriptor("chatbot", self.CATALOG))

    def test_the_url_is_built_from_the_catalogs_pinned_revision(self):
        """The revision must come from the descriptor -- an unpinned URL is not reproducible."""
        if not self.have_catalog():
            self.skipTest("needs a ../blis-catalog checkout")
        desc = FC._descriptor("agentx-mvp", self.CATALOG)
        source = desc["corpus"]["source"]
        url = FC.HF_URL.format(dataset=source["huggingface"], revision=source["revision"],
                               path=source["file"])
        self.assertIn(source["revision"], url, "the pinned revision must appear in the URL")
        self.assertNotIn("/main/", url, "a branch name would not be reproducible")
        self.assertEqual(64, len(desc["corpus"]["sha256"]))

    def test_a_digest_mismatch_is_refused_and_leaves_nothing_behind(self):
        """A truncated transfer must never be renamed into place as the corpus."""
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "traces.jsonl")
            src = os.path.join(d, "served.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                f.write("truncated")
            with self.assertRaises(SystemExit) as e:
                FC.fetch(dest, "file://" + src, "0" * 64)
            self.assertIn("digest mismatch", str(e.exception))
            self.assertFalse(os.path.exists(dest), "a bad download must not land at the dest")
            self.assertFalse(os.path.exists(dest + ".part"), "the partial must be cleaned up")

    def test_an_already_verified_corpus_is_left_alone(self):
        """The step runs on every sweep, so a present, correct corpus must be a no-op."""
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "traces.jsonl")
            with open(dest, "w", encoding="utf-8") as f:
                f.write("the corpus")
            sha = FC.digest(dest)
            # An unreachable URL proves no download was attempted.
            self.assertFalse(FC.fetch(dest, "file:///nonexistent-xyz", sha))
            with open(dest, encoding="utf-8") as f:
                self.assertEqual("the corpus", f.read())

    def test_a_verified_download_is_renamed_into_place(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "served.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                f.write("real corpus bytes")
            dest = os.path.join(d, "nested", "traces.jsonl")
            self.assertTrue(FC.fetch(dest, "file://" + src, FC.digest(src)))
            with open(dest, encoding="utf-8") as f:
                self.assertEqual("real corpus bytes", f.read())

    def test_the_workflow_fetches_before_it_sweeps(self):
        """Ordering is the whole point: a fetch after the sweeper would be useless."""
        with open(os.path.join(ROOT, ".github", "workflows", "run-experiment.yml")) as f:
            wf = f.read()
        self.assertIn("fetch_corpus.py", wf, "the workflow must provision the corpus")
        self.assertLess(wf.index("fetch_corpus.py"), wf.index("Run the sweeper"),
                        "the corpus must be fetched before the sweeper runs")


class KnobProfile(unittest.TestCase):
    """The per-combination profile: what APPLIES here, with what values, read from the sources.

    Exists because the catalog ROW alone cannot answer it -- it carries no speculator, no
    sliding window, no KV-dtype table and no capability gating. Those live in the model's
    graph.yaml.

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

    def test_kv_precision_is_an_axis_on_b200_and_not_on_hopper(self):
        b = self.verdict(self.K.profile("deepseek-v4.1-flash", "b200", "agentx-mvp",
                                        self.CATALOG), "cache_dtype")
        h = self.verdict(self.K.profile("deepseek-v4.1-flash", "h100", "agentx-mvp",
                                        self.CATALOG), "cache_dtype")
        self.assertIn("nvfp4_ds_mla", b["values"])
        self.assertNotIn("nvfp4_ds_mla", h["values"],
                         "nvfp4_ds_mla is family-10 only; BLIS refuses it on SM90")

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
        vocab = V.knob_vocabulary(BLIS)
        carried = (vocab.engine_names() | vocab.parallel
                   | {"router.endpoint_picker_config", "dp + enable_expert_parallel",
                      "mamba_cache_*"})
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
        self.assertEqual([], V.validate(doc, BLIS, ROUTING))

    def test_shipped_fixtures_respect_the_limit(self):
        """Every spec this repo ships is an example a contributor copies, so each must meet
        the limit it documents."""
        for path in V.spec_paths(EXPERIMENTS):
            n = V.header_comment_lines(path)
            self.assertLessEqual(n, V.HEADER_LINES,
                                 f"{os.path.basename(path)} opens with {n} comment lines")


class SpeculativeBlock(unittest.TestCase):
    def errs(self, engine):
        return V.validate(spec(sweeps=[{"name": "s", "num-instances": [2],
                                        "parallel": {"tp": [8]}, "engine": engine}]), BLIS, ROUTING)

    def test_an_invented_field_is_refused(self):
        errs = self.errs({"speculative": {"method": ["dspark"], "num_spec_tokens": [3],
                                          "draft_smaple_method": ["greedy"]}})
        self.assertTrue(any("draft_smaple_method" in e for e in errs), errs)

    def test_valid_speculative_passes(self):
        self.assertEqual([], self.errs({"speculative": {"method": ["dspark"],
                                                        "num_spec_tokens": [1, 3, 5]}}))

class ScenarioIdentityExists(unittest.TestCase):
    """model and hardware must name catalog entries, by `blis catalog resolve` (design §5)."""

    def errs(self, **scenario):
        sc = dict(spec()["scenario"])
        sc.update(scenario)
        return V.scenario_identity_errors(sc, BLIS)

    def test_a_real_scenario_is_accepted(self):
        self.assertEqual([], self.errs())

    def test_an_unknown_hardware_is_refused_with_the_commands_reason(self):
        errs = self.errs(hardware="h200x")
        self.assertEqual(1, len(errs), errs)
        self.assertIn("scenario.hardware", errs[0])
        self.assertIn("no hardware entry", errs[0])

    def test_both_are_reported_at_once(self):
        self.assertEqual(2, len(self.errs(model="nope-x", hardware="nope-y")))

    def test_a_missing_identifier_is_left_to_the_required_check(self):
        self.assertEqual([], V.scenario_identity_errors({}, BLIS))

    def test_validate_refuses_a_typod_identifier_end_to_end(self):
        doc = spec()
        doc["scenario"]["hardware"] = "h200x"
        self.assertTrue(any("scenario.hardware" in e for e in V.validate(doc, BLIS, ROUTING)))

    def test_the_validator_reads_no_catalog_file(self):
        """Design §5: config search's scripts never read catalog or registry files directly."""
        with open(os.path.join(ROOT, "scripts", "validate_experiment.py"), encoding="utf-8") as f:
            src = f.read()
        for needle in ("build_catalog", "graph.yaml", "models/", "hardware/"):
            self.assertNotIn(needle, src)

class RegistryFigures(unittest.TestCase):
    """`scenario.checks` and `scenario.objective` name a metric by registry identifier and scope
    (CFG-4, design §8.7), validated against `blis metrics`. Applied from M2; until then a real
    run refuses them and `slo` applies."""

    def errs(self, **scenario):
        doc = spec()
        doc["scenario"].update(scenario)
        return V.validate(doc, BLIS, ROUTING)

    def test_registry_figures_validate(self):
        self.assertEqual([], self.errs(
            checks=[{"metric": "ttft_p95_ms", "scope": "cluster", "max": 2000},
                    {"metric": "ttft_p99_ms", "scope": "instance", "aggregate": "max", "max": 5000},
                    {"metric": "interactivity_p50", "scope": "cluster", "min": 50}],
            objective={"metric": "output_tok_s_per_gpu", "scope": "cluster",
                       "direction": "maximize"}))

    def test_an_unknown_metric_is_refused_with_a_suggestion(self):
        errs = self.errs(checks=[{"metric": "ttft_p95", "scope": "cluster", "max": 1}])
        self.assertTrue(any("not a metric in `blis metrics`" in e and "ttft_p95_ms" in e
                            for e in errs), errs)

    def test_an_unknown_scope_is_refused(self):
        errs = self.errs(checks=[{"metric": "ttft_p95_ms", "scope": "node", "max": 1}])
        self.assertTrue(any(".scope:" in e for e in errs), errs)

    def test_an_instance_scope_needs_an_aggregate_and_the_cluster_takes_none(self):
        self.assertTrue(any(".aggregate:" in e for e in self.errs(
            checks=[{"metric": "ttft_p95_ms", "scope": "instance", "max": 1}])))
        self.assertTrue(any(".aggregate:" in e for e in self.errs(
            checks=[{"metric": "ttft_p95_ms", "scope": "cluster", "aggregate": "max", "max": 1}])))

    def test_a_check_states_exactly_one_bound(self):
        self.assertTrue(self.errs(checks=[{"metric": "ttft_p95_ms", "scope": "cluster"}]))
        self.assertTrue(self.errs(checks=[{"metric": "ttft_p95_ms", "scope": "cluster",
                                           "max": 1, "min": 0}]))

    def test_an_objective_states_its_direction(self):
        self.assertTrue(any("direction" in e for e in self.errs(
            objective={"metric": "output_tok_s", "scope": "cluster"})))

    def test_a_spec_in_the_current_format_needs_no_registry(self):
        from unittest import mock
        with mock.patch.object(blis_cli, "metrics", side_effect=AssertionError("called")):
            self.assertEqual([], V.validate(spec(), BLIS, ROUTING))

    def test_a_real_run_refuses_figures_until_they_are_applied(self):
        doc = spec()
        doc["scenario"]["checks"] = [{"metric": "ttft_p95_ms", "scope": "cluster", "max": 1}]
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(SystemExit) as e:
                E.run_experiment(doc, BLIS, None, "/r", out, verbose=False)
        self.assertIn("M2", str(e.exception))
        report = E.run_experiment(doc, BLIS, None, None, "/nonexistent", dry_run=True,
                                  verbose=False)
        self.assertGreater(report["totals"]["runs"], 0)


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

    def test_slo_limits_keeps_the_throughput_floor(self):
        self.assertEqual({"itl_p99": 20.0, "throughput_per_gpu": 500.0},
                         E.slo_limits({"itl_p99": "20ms", "throughput_per_gpu": 500}))

    def test_throughput_floor_is_judged_per_gpu(self):
        # 100 tok/s across 2 GPUs is 50 per GPU.
        self.assertTrue(E.servable(self.rec(), {"throughput_per_gpu": 50.0}, gpus=2)[0])
        ok, why = E.servable(self.rec(), {"throughput_per_gpu": 60.0}, gpus=2)
        self.assertFalse(ok)
        self.assertIn("throughput_per_gpu 50.0", why)
        ok, why = E.servable(self.rec(), {"throughput_per_gpu": 1.0})
        self.assertFalse(ok)
        self.assertIn("GPU count is unknown", why)

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

    # --- every SLO on ONE basis: the worst instance (susiejojo review) -------------------------
    #
    # ITL went through worst_itl_p99 while TTFT/E2E read the cluster aggregate, so a run could be
    # marked servable with an instance over the cap. report.py's _feasible already judged EVERY
    # declared SLO on the worst instance, so the two surfaces disagreed -- the exact split the
    # sweeper's docstring promises cannot happen. Each test below fails on the aggregate basis.

    def test_ttft_passing_in_the_aggregate_fails_on_the_worst_instance(self):
        """The committed multidoc-prefill shape: aggregate under a 10s cap, one instance over."""
        rec = self.rec(metrics={"tokens_per_sec": 100.0, "itl_p99_ms": 10.0,
                                "ttft_p99_ms": 9981.2},
                       metrics_all_instances=[
                           {"instance_id": "i0", "itl_p99_ms": 10.0, "ttft_p99_ms": 9000.0},
                           {"instance_id": "i1", "itl_p99_ms": 10.0, "ttft_p99_ms": 10010.1}])
        ok, why = E.servable(rec, {"ttft_p99": 10000.0})
        self.assertFalse(ok, "an instance at 10010 ms breaches a 10000 ms TTFT cap")
        self.assertIn("10010", why)

    def test_e2e_passing_in_the_aggregate_fails_on_the_worst_instance(self):
        rec = self.rec(metrics={"tokens_per_sec": 100.0, "itl_p99_ms": 10.0, "e2e_p99_ms": 900.0},
                       metrics_all_instances=[
                           {"instance_id": "i0", "itl_p99_ms": 10.0, "e2e_p99_ms": 900.0},
                           {"instance_id": "i1", "itl_p99_ms": 10.0, "e2e_p99_ms": 1500.0}])
        ok, why = E.servable(rec, {"e2e_p99": 1000.0})
        self.assertFalse(ok)
        self.assertIn("1500", why)

    def test_an_idle_instance_reporting_zero_cannot_make_a_run_servable(self):
        """BLIS reports 0, not null, for an instance that served nothing. 0 is the floor, not a
        measurement, so it must never pull the worst case down below a real breach."""
        rec = self.rec(metrics={"tokens_per_sec": 100.0, "itl_p99_ms": 10.0,
                                "ttft_p99_ms": 400.0},
                       metrics_all_instances=[
                           {"instance_id": "i0", "itl_p99_ms": 10.0, "ttft_p99_ms": 0},
                           {"instance_id": "i1", "itl_p99_ms": 10.0, "ttft_p99_ms": 900.0}])
        ok, why = E.servable(rec, {"ttft_p99": 500.0})
        self.assertFalse(ok)
        self.assertIn("900", why)

    def test_the_sweeper_and_the_report_agree_on_every_declared_slo(self):
        """The guarantee itself: no run may be servable here and infeasible in the report."""
        import report as rep
        slos = {"ttft_p99_ms": 10000.0, "e2e_p99_ms": 20000.0}
        rec = self.rec(metrics={"tokens_per_sec": 100.0, "itl_p99_ms": 10.0,
                                "ttft_p99_ms": 9981.2, "e2e_p99_ms": 15000.0},
                       metrics_all_instances=[
                           {"instance_id": "i0", "itl_p99_ms": 10.0, "ttft_p99_ms": 10010.1,
                            "e2e_p99_ms": 15000.0}])
        ok, _ = E.servable(rec, {"ttft_p99": 10000.0, "e2e_p99": 20000.0})
        feasible, viol = rep._feasible(rec, {}, slos)
        self.assertEqual(ok, feasible, f"sweeper={ok} report={feasible} ({viol})")
        self.assertFalse(ok)

    def test_ttft_is_recorded_on_the_basis_it_is_judged_on(self):
        """A ledger row showing a passing aggregate beside an infeasible verdict is unreadable."""
        rec = self.rec(metrics={"tokens_per_sec": 100.0, "itl_p99_ms": 10.0,
                                "ttft_p99_ms": 9981.2},
                       metrics_all_instances=[
                           {"instance_id": "i0", "itl_p99_ms": 10.0, "ttft_p99_ms": 10010.1}])
        row = E.run_row(rec, {}, "label", "", "sweep", 8, 8)
        self.assertEqual(10010.1, row["ttft_p99_ms"], "the row must carry the worst instance")
        self.assertEqual(9981.2, row["ttft_p99_ms_aggregate"], "the aggregate stays, beside it")


# --- P/D support in the runner ----------------------------------------------------------------


class PdRunner(unittest.TestCase):
    """The P/D half of run_blis: constraints BLIS would refuse must be refused here first."""

    def cfg(self, **over):
        base = {"prefill_instances": 3, "decode_instances": 5,
                "prefill_tp": 8, "decode_tp": 8}
        base.update(over)
        return base

    def test_pools_resolve_with_instance_counts_and_footprints(self):
        par, _eng, cli, _router = run_blis.classify_config(self.cfg(), V.knob_vocabulary(BLIS), 8)
        pd = par["pd"]
        self.assertEqual(3, pd["prefill"]["instances"])
        self.assertEqual(5, pd["decode"]["instances"])
        self.assertEqual(64, run_blis.pd_total_gpus(pd))
        self.assertEqual(8, cli["num_instances"], "num_instances is derived from the two counts")

    def test_counts_must_add_up_to_num_instances(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(num_instances=7), V.knob_vocabulary(BLIS), 8)
        self.assertIn("must add up", str(e.exception))

    def test_pool_without_counterpart_is_refused(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"prefill_instances": 2, "prefill_tp": 8}, V.knob_vocabulary(BLIS), 8)

    def test_pools_must_agree_on_dp(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(prefill_dp=2, decode_dp=1,
                                              prefill_instances=1, decode_instances=1), V.knob_vocabulary(BLIS), 8)
        self.assertIn("replica factor", str(e.exception))

    def test_pools_must_agree_on_block_size(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(prefill_block_size=64, decode_block_size=16), V.knob_vocabulary(BLIS), 8)
        self.assertIn("whole KV blocks", str(e.exception))

    def test_pools_must_agree_on_cache_dtype(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(prefill_cache_dtype="fp8",
                                              decode_cache_dtype="auto"), V.knob_vocabulary(BLIS), 8)
        self.assertIn("cache formats must agree", str(e.exception))

    def test_only_nixl_connectors_are_accepted(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(pd_connector="LMCache"), V.knob_vocabulary(BLIS), 8)
        self.assertIn("NIXL", str(e.exception))
        par, _e, _c, _r = run_blis.classify_config(self.cfg(pd_connector="NixlPullConnector"), V.knob_vocabulary(BLIS), 8)
        self.assertEqual("NixlPullConnector", par["pd"]["connector"])

    def test_connector_is_not_emitted_as_a_cli_flag(self):
        _p, _e, cli, _r = run_blis.classify_config(self.cfg(pd_connector="NixlConnector"), V.knob_vocabulary(BLIS), 8)
        self.assertNotIn("pd_connector", cli)

    def test_per_pool_drafting_is_refused_with_the_real_reason(self):
        """BLIS applies ONE speculative config per run, so a per-pool drafting key is a
        misunderstanding of where drafting lives -- it must say so, not "unknown key"."""
        for key in ("prefill_num_speculative_tokens", "decode_speculative_method"):
            with self.assertRaises(run_blis.ConfigError) as e:
                run_blis.classify_config(self.cfg(**{key: 3}), V.knob_vocabulary(BLIS), 8)
            msg = str(e.exception)
            self.assertIn("one configuration per run", msg)
            self.assertNotIn("unknown key", msg)

    def test_straddling_pool_footprint_is_refused(self):
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config(self.cfg(prefill_tp=4, prefill_dp=3,
                                              decode_tp=4, decode_dp=3,
                                              prefill_instances=1, decode_instances=1), V.knob_vocabulary(BLIS), 8)
        self.assertIn("whole", str(e.exception))

    def test_scenario_yaml_renders_two_pools_and_pd_transfer(self):
        par, eng, cli, _router = run_blis.classify_config(self.cfg(), V.knob_vocabulary(BLIS), 8)
        doc = run_blis.scenario_yaml("kimi-k2.5", "h200", "0.29.0", par, eng, "x",
                                     cli["num_instances"], packaging=run_blis.LEGACY_PACKAGING)
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
        par, eng, _cli, _r = run_blis.classify_config({"tp": 8, "num_instances": 2}, V.knob_vocabulary(BLIS), 8)
        doc = run_blis.scenario_yaml("glm-5.3-flash", "h100", "0.29.0", par, eng, "x", 2, packaging=run_blis.LEGACY_PACKAGING)
        self.assertIn("role: colocated", doc)
        self.assertNotIn("pd_transfer", doc)

    def test_encode_and_shared_roles_are_refused(self):
        for key in ("encode_instances", "prefill_decode_instances"):
            with self.assertRaises(run_blis.ConfigError):
                run_blis.classify_config({key: 2}, V.knob_vocabulary(BLIS), 8)


# --- assembled report -------------------------------------------------------------------------


class DryRunPlan(unittest.TestCase):
    def test_dry_run_plans_without_touching_blis(self):
        report = E.run_experiment(spec(), blis=BLIS, catalog=None, registry=None,
                                  out_root="/nonexistent-should-not-be-written",
                                  dry_run=True, verbose=False)
        self.assertEqual("t", report["name"])
        self.assertEqual(1, report["totals"]["configs"])
        self.assertEqual(2, report["totals"]["runs"])
        self.assertEqual(50.0, report["interactivity_floor"]["tok_per_s_per_user"])

    def test_runs_and_report_use_the_simulators_engine_release(self):
        seen = []

        def fake_run_one(*a, **kw):
            seen.append(a[6])   # run_one(blis, catalog, registry, dir, model, hw, engine_version, ...)
            return {"metrics": {"tokens_per_sec": 1.0, "itl_p99_ms": 1.0}, "cmd": None,
                    "config": dict(kw.get("config") or a[11])}

        real = E.run_blis.run_one
        E.run_blis.run_one = fake_run_one
        try:
            with tempfile.TemporaryDirectory() as out:
                report = E.run_experiment(spec(), BLIS, "/c", "/r", out, verbose=False)
        finally:
            E.run_blis.run_one = real
        pinned = blis_cli.engine_release(BLIS, "vllm")
        self.assertTrue(seen)
        self.assertEqual({pinned}, set(seen))
        self.assertEqual(pinned, report["scenario"]["engine_version"])

    def test_dry_run_of_every_committed_fixture(self):
        for path in V.spec_paths(EXPERIMENTS):
            doc, _ = V.load(path)
            report = E.run_experiment(doc, BLIS, None, None,
                                      "/nonexistent-should-not-be-written",
                                      dry_run=True, verbose=False)
            self.assertGreater(report["totals"]["runs"], 0, path)
            self.assertEqual(len(doc["sweeps"]), len(report["sweeps"]), path)
            for s in report["sweeps"]:
                self.assertTrue(s["narrative"], f"{path}: {s['name']} lost its narrative")

    def test_scenario_json_carries_the_experiment_narrative(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("name: kv\nscenario: {model: qwen3-14b, hardware: h100, workload: chatbot,"
                    " gpus: 16, slo: {itl_p99: 20ms}, concurrency: [8]}\nsweeps:\n"
                    "  - {name: a, narrative: bf16 KV., num-instances: [16],"
                    " engine: {cache_dtype: [auto]}}\n"
                    "  - {name: b, narrative: fp8 KV., num-instances: [16],"
                    " engine: {cache_dtype: [fp8]}}\n")
        self.addCleanup(os.remove, f.name)
        doc, _ = V.load(f.name)
        report = E.run_experiment(doc, BLIS, None, None, "/nonexistent", dry_run=True,
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
        self.assertIn("tok/s/accelerator", out["winner"]["reason"])

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

    def test_recorded_commands_survive_a_symlinked_run_directory(self):
        """A symlinked base must not leak an absolute path into the ledger.

        The matching was purely lexical, so whenever the run directory or the checkout was reached
        through a symlink -- `/tmp` resolves to `/private/tmp` on macOS, and a container commonly
        mounts the workspace at a linked path -- the resolved form failed to match and stayed
        absolute and host-specific, which is the one thing this function exists to prevent. The
        symlink can sit on either side, so all four pairings have to relativize.
        """
        with tempfile.TemporaryDirectory() as real, tempfile.TemporaryDirectory() as holder:
            link = os.path.join(holder, "link")
            os.symlink(real, link)
            for arg_base, out_dir in ((real, link), (link, real), (link, link), (real, real)):
                rec = {"cmd": ["/opt/blis", "run", "--scenarios",
                               os.path.join(arg_base, "scenarios"), "--seed", "42"]}
                E._relativize_cmd(rec, out_dir)
                self.assertEqual(
                    os.path.join(".", "scenarios"), rec["cmd"][3],
                    f"arg under {'link' if arg_base == link else 'real'}, "
                    f"out_dir {'link' if out_dir == link else 'real'}")
                self.assertEqual(["--seed", "42"], rec["cmd"][4:])

    def test_relativize_tolerates_a_row_with_no_command(self):
        rec = {"cmd": None}
        E._relativize_cmd(rec, "/tmp/out")
        self.assertIsNone(rec["cmd"])

    def test_parallel_replay_warms_the_corpus_cache_serially(self):
        """The first replay run that reaches BLIS builds the corpus snapshot alone, before the pool."""
        import threading, time as _t
        spec = {
            "name": "warm", "sweeps": [
                {"name": "a", "narrative": "n", "parallel": {"tp": [8]}},
                {"name": "b", "narrative": "n", "parallel": {"tp": [8]}}],
            "scenario": {"model": "m", "hardware": "h", "workload": "w", "gpus": 8,
                         "slo": {"itl_p99": "20ms"}, "concurrency": [1, 2, 4, 8]},
        }
        lock, state, events = threading.Lock(), {"live": 0, "peak": 0}, []

        def fake_run_one(*a, **kw):
            with lock:
                state["live"] += 1
                state["peak"] = max(state["peak"], state["live"])
                events.append("start")
            _t.sleep(0.02)
            with lock:
                state["live"] -= 1
                events.append("end")
                refused = events.count("end") == 1
            # The first run is refused before BLIS starts, so it must not count as the warm-up.
            return {"metrics": {"tokens_per_sec": 1.0, "itl_p99_ms": 1.0}, "cmd": None,
                    "returncode": None if refused else 0}

        real_run, real_resolve = E.run_blis.run_one, E.resolve_replay
        E.run_blis.run_one = fake_run_one
        E.resolve_replay = lambda *a, **kw: {"scenario": "s", "corpus": "c"}
        try:
            import tempfile as _tf
            with _tf.TemporaryDirectory() as out:
                E.run_experiment(spec, BLIS, None, "/r", out, verbose=False, max_parallel=4)
        finally:
            E.run_blis.run_one, E.resolve_replay = real_run, real_resolve
        self.assertEqual(16, len(events))
        self.assertEqual(["start", "end"] * 2, events[:4],
                         "warm-up must continue past a refused run until one really launches")
        self.assertGreater(state["peak"], 1, "the rest should run in parallel")

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
                out1 = E.run_experiment(dict(spec), BLIS, "/c", "/r", out,
                                        verbose=False, max_parallel=1)
                calls["n"] = 0
                out8 = E.run_experiment(dict(spec), BLIS, "/c", "/r", out,
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


class PinCheck(unittest.TestCase):
    """An off-pin simulator is reported by SOURCE commit, never by binary digest (#23)."""
    PINS = {"BLIS_SHA": "f2ba6bb1c189ed5a56933cfa3d0927eae3836ee9"}

    def stack(self, commit=None, describe=None, sha256="ab" * 32):
        blis = {"sha256": sha256}
        if commit:
            blis["commit"] = commit
        if describe:
            blis["describe"] = describe
        return {"blis": "v0.14.0", "resolved": {"blis": blis}}

    def test_on_pin_is_silent(self):
        st = self.stack(self.PINS["BLIS_SHA"], "v0.14.0")
        self.assertEqual("", E.blis_pin_warning(st, self.PINS))

    def test_off_pin_names_both_commits(self):
        w = E.blis_pin_warning(self.stack("f199223d701d96dddeb1c314509bed003515b599", "v0.12.0"),
                               self.PINS)
        for part in ("f2ba6bb1c189", "f199223d701d", "v0.12.0"):
            self.assertIn(part, w)

    def test_another_build_of_the_pinned_commit_is_not_drift(self):
        st = self.stack(self.PINS["BLIS_SHA"], "v0.14.0", sha256="1feaa390a486" + "0" * 52)
        self.assertEqual("", E.blis_pin_warning(st, self.PINS))

    def test_dirty_checkout_at_the_pin_warns(self):
        st = self.stack(self.PINS["BLIS_SHA"], "v0.14.0-dirty")
        self.assertIn("UNCOMMITTED", E.blis_pin_warning(st, self.PINS))

    def test_unverifiable_binary_is_silent(self):
        for st in (self.stack(), {"blis": "v0.14.0"}, {"blis": "v0.14.0", "resolved": {}}):
            self.assertEqual("", E.blis_pin_warning(st, self.PINS))
        self.assertEqual("", E.blis_pin_warning(self.stack("c" * 40), {}))

    def test_source_commit_needs_the_simulators_own_tree(self):
        import shutil
        import subprocess
        d = tempfile.mkdtemp()
        try:
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                            "commit", "-q", "--allow-empty", "-m", "x"], cwd=d, check=True)
            fake = os.path.join(d, "blis")
            with open(fake, "wb") as f:
                f.write(b"not the simulator")
            self.assertIsNone(E._blis_source_commit(fake)[1], "no main.go: not the simulator")
            open(os.path.join(d, "main.go"), "w").close()
            self.assertEqual(40, len(E._blis_source_commit(fake)[1] or ""))
            blis = E._pinned_stack(fake, None, None, "0.32.0")["resolved"]["blis"]
            self.assertTrue(blis.get("sha256") and blis.get("commit"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_pinned_stack_invents_nothing_when_nothing_is_on_disk(self):
        ps = E._pinned_stack(None, None, None, "0.32.0")
        self.assertIn("blis", ps)
        self.assertNotIn("resolved", ps)


class ProvenanceSplit(unittest.TestCase):
    """A ledger whose rows span seeds or stacks is flagged, not rendered as one comparison."""

    def test_split_is_detected(self):
        import report
        seeds = [{"cmd": ["blis", "run", "--seed", "0"]}, {"cmd": ["blis", "run", "--seed", "42"]}]
        self.assertIn("2 seeds", report.provenance_split(seeds))
        one = [{"cmd": ["blis", "run", "--seed", "42"]}] * 2
        self.assertEqual("", report.provenance_split(one))
        stacks = [{"pinned_stack": {"registry": "v0.1.1"}}, {"pinned_stack": {"registry": "v0.2.0"}}]
        self.assertIn("distinct resolved stacks", report.provenance_split(stacks))


class PerUserRateUnderDrafting(unittest.TestCase):
    """Under speculative decoding BLIS logs ITL per VERIFICATION STEP, and a step emits AL tokens,
    so the naive 1000/ITL is a step rate. report._per_user_rate must return the AL-scaled per-token
    rate, withhold it when AL is unknown, and give the direct rate without drafting -- the one
    definition _feasible also judges on. (Ported from the retired test_report.py, issue #69, and
    rewritten to call the kept helper directly rather than the removed build_records.)"""

    ITL = 32.995

    def _run(self, *, k=0, al=None):
        dep = {"parallel": {"tp": 4}, "num_instances": 1, "gpus": 4,
               "engine": {"max_num_batched_tokens": 8192, "max_num_seqs": 256}}
        if k:
            dep["engine"]["speculative"] = {"method": "dspark", "num_spec_tokens": k}
            if al:
                dep["acceptance"] = {"acceptance_length": al, "source": "test (b200)"}
        return {"concurrency": 8, "config": {"tp": 4}, "deployment": dep, "returncode": 0,
                "metrics": {"tokens_per_sec": 1041.5, "itl_p99_ms": self.ITL,
                            "injected_requests": 40, "completed_requests": 40}}

    def test_no_drafting_uses_the_direct_rate(self):
        import report
        pu, basis = report._per_user_rate(self._run())
        self.assertAlmostEqual(pu, 1000 / self.ITL, places=1)
        self.assertEqual(basis, "")

    def test_drafting_with_acceptance_scales_by_al(self):
        import report
        pu, basis = report._per_user_rate(self._run(k=5, al=3.51))
        self.assertAlmostEqual(pu, 3.51 * 1000 / self.ITL, places=1)
        self.assertEqual(basis, "derived-from-acceptance")

    def test_the_derived_rate_is_above_the_naive_one(self):
        """The whole point: the naive axis would reject this config against a 50 tok/s/user floor
        while the correct one clears it comfortably."""
        import report
        pu, _ = report._per_user_rate(self._run(k=5, al=3.51))
        self.assertLess(1000 / self.ITL, 50)      # naive: looks like a violation
        self.assertGreater(pu, 50)                # derived: clears the floor

    def test_drafting_without_acceptance_withholds_the_rate(self):
        """An unknown acceptance factor means an unknown multiple -- publish nothing."""
        import report
        pu, basis = report._per_user_rate(self._run(k=5, al=None))
        self.assertIsNone(pu)
        self.assertEqual(basis, "step-estimate-withheld")


class ReportSchemaDoc(unittest.TestCase):
    """report/schema.md documents report.json and must describe the per-user rate on the p99 basis,
    not a mean-ITL basis (review F3). (Ported from the retired test_report.py, issue #69.)"""

    def test_docs_describe_pu_as_p99(self):
        import report
        schema = os.path.join(report.REPORT_DIR, "schema.md")
        text = open(schema, encoding="utf-8").read().lower()
        self.assertNotIn("mean inter-token latency", text, "schema.md still says mean ITL for pu")
        self.assertNotIn("1000 / mean itl", text, "schema.md still says 1000/mean ITL for pu")
        self.assertNotIn("1000/mean-itl", text, "schema.md still says 1000/mean-ITL for pu")


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
