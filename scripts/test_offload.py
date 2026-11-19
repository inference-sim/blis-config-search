#!/usr/bin/env python3
"""Tests of the sweepable CPU KV tier (a sweep's `offload:` block, #94). Hermetic.

Run: python3 -I scripts/test_offload.py
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_cli  # noqa: E402
import run_blis  # noqa: E402
import run_experiment as E  # noqa: E402
import settings  # noqa: E402
import test_experiment as TE  # noqa: E402,F401  (points ROUTER_CONFIGS_DIR at its hermetic router fixture)
import topologies  # noqa: E402
import validate_experiment as V  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLIS = os.path.join(HERE, "testdata", "blis-m1", "blis")
ROUTING = os.path.join(os.path.dirname(HERE), "routing")


def doc(offload=None, **sweep):
    s = {"name": "s", "num-instances": [2], "parallel": {"tp": [8]}}
    if offload is not None:
        s["offload"] = offload
    s.update(sweep)
    return {"name": "t", "scenario": {"model": "qwen3-14b", "hardware": "h100", "workload": "chatbot",
                                      "cluster": {"gpus_per_node": 8}, "gpus": 16, "concurrency": [2]},
            "sweeps": [s]}


class SizesAndValidation(unittest.TestCase):
    def test_sizes_are_zero_bytes_or_a_number_with_a_unit(self):
        self.assertEqual(0, V.parse_size(0))
        self.assertEqual(549755813888, V.parse_size("512GiB"))
        self.assertEqual(549755813888, V.parse_size("512 GiB"))
        self.assertEqual(600 * 10**9, V.parse_size("600GB"))
        for bad in ("512", "GiB", "-1GiB", "1.5.2GiB", True, None, -4):
            self.assertIsNone(V.parse_size(bad), bad)

    def test_a_good_offload_block_validates(self):
        d = doc({"cpu_bytes": [0, "512GiB"], "eviction_policy": ["lru"]})
        self.assertEqual([], V.validate(copy.deepcopy(d), BLIS, ROUTING))

    def test_each_way_the_block_can_be_wrong_is_named(self):
        for offload, want in (({"cpu_bytes": ["512"]}, "is not a size"),
                              ({"cpu_bytes": []}, "non-empty list"),
                              ({"cpu_bytes": [0], "tier": ["x"]}, "unknown offload option"),
                              ({"eviction_policy": ["lru"]}, "needs cpu_bytes"),
                              ({}, "non-empty mapping")):
            errs = V.validate(copy.deepcopy(doc(offload)), BLIS, ROUTING)
            self.assertTrue(any(want in e for e in errs), (offload, errs))

    def test_a_disaggregated_sweep_has_no_offload_key_yet(self):
        d = doc({"cpu_bytes": ["512GiB"]})
        sw = d["sweeps"][0]
        del sw["num-instances"], sw["parallel"]
        sw.update({"prefill-instances": [1], "decode-instances": [1],
                   "prefill": {"parallel": {"tp": [8]}}, "decode": {"parallel": {"tp": [8]}}})
        errs = V.validate(d, BLIS, ROUTING)
        self.assertTrue(any("no per-pool offload" in e and "#307" in e for e in errs), errs)


def fake_engine(directory, refuses, min_tp=1, packaging=None):
    """A simulator that opens every scenario except one naming a policy in `refuses` or one with tp below `min_tp`.
    With `packaging`, a scenario that states `gpus_per_node: N` for another N is refused for its packaging."""
    import stat
    path = os.path.join(directory, "blis-fake")
    real = BLIS
    with open(path, "w") as f:
        f.write(f"#!/bin/sh\n"
                f"if [ \"$1\" = fields ] && [ \"$2\" = --scenario ] && grep -q eviction_policy \"$5/$3\"; then\n"
                f"  if [ {min_tp} -gt \"$(sed -n 's/^ *tp: *//p' \"$5/$3\" | head -1)\" ]; then\n"
                f"    echo 'Error: the model does not fit at this tp' >&2; exit 1\n  fi\n"
                + (f"  if grep -q 'gpus_per_node:' \"$5/$3\" && ! grep -q 'gpus_per_node: {packaging}$' \"$5/$3\"; then\n"
                   f"    echo 'Error: cluster: gpus_per_node contradicts the catalog' >&2; exit 1\n  fi\n" if packaging else "") +
                f"  if grep -q 'eviction_policy: \"{refuses}\"' \"$5/$3\"; then\n"
                f"    echo 'Error: offloading: eviction_policy \"{refuses}\" is not modelled, only lru' >&2; exit 1\n"
                f"  fi\n  exit 0\nfi\nexec {real} \"$@\"\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


class PolicyNames(unittest.TestCase):
    def test_a_policy_the_engine_refuses_is_a_validation_error_naming_it(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fake = fake_engine(d, "arc")
            ok = doc({"cpu_bytes": ["512GiB"], "eviction_policy": ["lru"]})
            self.assertEqual([], V.validate(copy.deepcopy(ok), fake, ROUTING))
            bad = doc({"cpu_bytes": ["512GiB"], "eviction_policy": ["lru", "arc"]})
            errs = V.validate(copy.deepcopy(bad), fake, ROUTING)
            self.assertTrue(any("refuses 'arc'" in e and "not modelled" in e for e in errs), errs)
            self.assertFalse(any("refuses 'lru'" in e for e in errs), errs)


    def test_the_policy_is_probed_at_the_largest_tp_not_the_first(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fake = fake_engine(d, "arc", min_tp=4)
            bad = doc({"cpu_bytes": ["512GiB"], "eviction_policy": ["arc"]})
            bad["sweeps"][0]["parallel"]["tp"] = [1, 4]
            bad["scenario"]["gpus"] = 8
            errs = V.validate(copy.deepcopy(bad), fake, ROUTING)
            self.assertTrue(any("refuses 'arc'" in e for e in errs), errs)
            good = doc({"cpu_bytes": ["512GiB"], "eviction_policy": ["lru"]})
            good["sweeps"][0]["parallel"]["tp"] = [1, 4]
            good["scenario"]["gpus"] = 8
            self.assertEqual([], V.validate(copy.deepcopy(good), fake, ROUTING))


class ProbePackaging(unittest.TestCase):
    """The policy probe uses the hardware's packaging, not an assumed eight GPUs a node (#154)."""

    def spec(self, policy):
        d = doc({"cpu_bytes": ["512GiB"], "eviction_policy": [policy]})
        d["scenario"]["hardware"] = "gb200-nvl72"      # the catalog fixes 4 GPUs a node
        del d["scenario"]["cluster"]
        d["sweeps"][0]["parallel"]["tp"] = [4]
        d["scenario"]["gpus"] = 8
        return d

    def test_a_bad_policy_is_found_on_hardware_whose_packaging_is_not_eight(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fake = fake_engine(d, "arc", packaging=4)
            errs = V.offload_policy_errors(self.spec("arc"), fake)
            self.assertTrue(any("refuses 'arc'" in e for e in errs), errs)
            self.assertEqual([], V.offload_policy_errors(self.spec("lru"), fake))

    def test_a_chip_with_no_packaging_anywhere_is_refused_not_probed_on_eight(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fake = fake_engine(d, "arc")
            sp = self.spec("arc")
            sp["scenario"]["hardware"] = "h100"        # the catalog fixes nothing; the spec states nothing
            errs = V.offload_policy_errors(sp, fake)
            self.assertTrue(any("packaging" in e for e in errs), errs)


def pool_offload_blis(directory):
    """The fixture simulator, but its deployment schema lets a pool state `offload` and `pd_transfer`
    (what blis-substrate#307 adds), as a build with per-pool offload prints it."""
    import json
    import stat
    fixture = os.path.join(HERE, "testdata", "blis-m1")
    with open(os.path.join(fixture, "deployment.schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    props = schema["$defs"]["deployment.Pool"]["properties"]
    props["offload"] = {"$ref": "#/$defs/deployment.Offload"}
    props["pd_transfer"] = {"$ref": "#/$defs/deployment.PDTransfer"}
    schema_path = os.path.join(directory, "deployment.schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f)
    path = os.path.join(directory, "blis-pool-offload")
    with open(path, "w") as f:
        f.write(f"#!/bin/sh\nif [ \"$1\" = schema ] && [ \"$2\" = deployment ]; then cat '{schema_path}'; exit 0; fi\n"
                f"exec '{BLIS}' \"$@\"\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def pd_doc(**sweep):
    d = doc()
    d["scenario"]["gpus"] = 16
    sw = d["sweeps"][0]
    del sw["num-instances"], sw["parallel"]
    sw.update({"prefill-instances": [1], "decode-instances": [1],
               "prefill": {"parallel": {"tp": [8]}}, "decode": {"parallel": {"tp": [8]}}})
    sw.update(sweep)
    return d


class PerPoolOffload(unittest.TestCase):
    """A disaggregated sweep's CPU tier, per pool (#94; the schema side is blis-substrate#307)."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.capable = pool_offload_blis(self.tmp.name)
        self.vocab = settings.vocabulary(BLIS)

    def scenario(self, config, pool_offload=True):
        base = {"prefill_instances": 1, "decode_instances": 1, "prefill_tp": 8, "decode_tp": 8}
        parallel, engine, cli, router = run_blis.classify_config({**base, **config}, self.vocab, 8, pool_offload)
        return run_blis.scenario_yaml("qwen3-14b", "h100", "0.33.0", parallel, engine, "x", node_size=8)

    def pool(self, text, role):
        return text.split(f"  - role: {role}\n")[1].split("\n  - role:")[0].split("\npd_transfer:")[0]

    def test_the_capability_is_read_from_the_deployment_schema(self):
        self.assertFalse(blis_cli.supports_pool_offload(BLIS))
        self.assertTrue(blis_cli.supports_pool_offload(self.capable))

    def test_a_tier_on_the_prefill_pool_alone_renders_on_that_pool_with_its_own_connector(self):
        text = self.scenario({"prefill_offload_cpu_bytes": 2**39, "prefill_offload_eviction_policy": "lru"})
        pf, dc = self.pool(text, "prefill"), self.pool(text, "decode")
        self.assertIn("tiers:\n        - {tier: cpu_dram, bytes: 549755813888}", pf)
        self.assertIn('eviction_policy: "lru"', pf)
        self.assertIn("pd_transfer: {connector: MultiConnector}", pf)
        self.assertNotIn("offload", dc)
        self.assertNotIn("pd_transfer", dc)
        self.assertTrue(text.rstrip().endswith("pd_transfer:\n  connector: NixlConnector"), text[-80:])
        self.assertIn("cost-model-offload-tiers", text)

    def test_the_sweep_level_block_is_the_default_for_both_pools_and_a_pool_can_opt_out(self):
        both = self.scenario({"offload_cpu_bytes": 2**38})
        self.assertIn("tier: cpu_dram", self.pool(both, "prefill"))
        self.assertIn("tier: cpu_dram", self.pool(both, "decode"))
        decode_off = self.scenario({"offload_cpu_bytes": 2**38, "decode_offload_cpu_bytes": 0})
        self.assertIn("tier: cpu_dram", self.pool(decode_off, "prefill"))
        self.assertNotIn("offload", self.pool(decode_off, "decode"))
        self.assertNotIn("pd_transfer", self.pool(decode_off, "decode"))

    def test_a_deployment_connector_of_multi_gives_a_pool_without_a_tier_nixl(self):
        text = self.scenario({"prefill_offload_cpu_bytes": 2**38, "pd_connector": "MultiConnector"})
        self.assertNotIn("pd_transfer", self.pool(text, "prefill"))
        self.assertIn("pd_transfer: {connector: NixlConnector}", self.pool(text, "decode"))

    def test_a_disaggregated_run_without_a_tier_renders_as_before(self):
        text = self.scenario({"prefill_offload_cpu_bytes": 0})
        self.assertNotIn("offload", text)
        self.assertEqual(1, text.count("pd_transfer"))

    def test_a_simulator_without_per_pool_offload_refuses_the_tier_naming_the_schema(self):
        with self.assertRaises(run_blis.ConfigError) as cm:
            self.scenario({"prefill_offload_cpu_bytes": 2**38}, pool_offload=False)
        self.assertIn("blis-substrate#307", str(cm.exception))

    def test_a_pool_block_on_a_colocated_candidate_is_refused(self):
        with self.assertRaises(run_blis.ConfigError) as cm:
            run_blis.classify_config({"tp": 8, "prefill_offload_cpu_bytes": 2**38}, self.vocab, 8, True)
        self.assertIn("disaggregated", str(cm.exception))

    def test_the_validator_accepts_a_pool_block_only_where_the_schema_has_it(self):
        spec = pd_doc(offload={"cpu_bytes": [0, "256GiB"]}, prefill={
            "parallel": {"tp": [8]}, "offload": {"cpu_bytes": ["512GiB"], "eviction_policy": ["lru"]}})
        self.assertEqual([], V.validate(copy.deepcopy(spec), self.capable, ROUTING))
        errs = V.validate(copy.deepcopy(spec), BLIS, ROUTING)
        self.assertTrue(any("sweeps[0](s).prefill.offload" in e and "#307" in e for e in errs), errs)
        bad = pd_doc(decode={"parallel": {"tp": [8]}, "offload": {"eviction_policy": ["lru"]}})
        errs = V.validate(bad, self.capable, ROUTING)
        self.assertTrue(any("decode.offload.eviction_policy: needs cpu_bytes" in e for e in errs), errs)

    def test_a_disaggregated_sweep_at_data_parallel_8_validates(self):
        """The validator once refused every P/D pool at dp>1 (blis-vllm#44, closed): the simulator runs it."""
        spec = pd_doc(prefill={"parallel": {"tp": [1], "dp": [8]}, "offload": {"cpu_bytes": ["512GiB"]}},
                      decode={"parallel": {"tp": [1], "dp": [8]}})
        self.assertEqual([], V.validate(spec, self.capable, ROUTING))

    def test_a_pools_eviction_policy_is_probed_with_the_engine(self):
        with __import__("tempfile").TemporaryDirectory() as d:
            fake = fake_engine(d, "arc")
            errs = V.offload_policy_errors(pd_doc(prefill={
                "parallel": {"tp": [8]}, "offload": {"cpu_bytes": ["512GiB"], "eviction_policy": ["arc"]}}), fake)
            self.assertTrue(any("refuses 'arc'" in e for e in errs), errs)

    def test_the_expansion_crosses_each_pools_sizes_and_leaves_other_sweeps_alone(self):
        sw = pd_doc(prefill={"parallel": {"tp": [8]}, "offload": {"cpu_bytes": [0, "256GiB"], "eviction_policy": ["lru"]}},
                    decode={"parallel": {"tp": [8]}, "offload": {"cpu_bytes": [0, "512GiB"]}})["sweeps"][0]
        kept, _ = E.expand_sweep(sw, 16)
        arms = [(c.get("prefill_offload_cpu_bytes"), c.get("decode_offload_cpu_bytes")) for c in kept]
        self.assertEqual([(0, 0), (0, 512 * 2**30), (256 * 2**30, 0), (256 * 2**30, 512 * 2**30)], arms)
        self.assertEqual("lru", kept[2]["prefill_offload_eviction_policy"])
        plain, _ = E.expand_sweep(pd_doc()["sweeps"][0], 16)
        self.assertTrue(plain and not any("offload" in k for c in plain for k in c))

    @unittest.skipUnless(os.environ.get("BLIS_POOL_OFFLOAD"), "needs a simulator with per-pool offload (BLIS_POOL_OFFLOAD=path, BLIS_POOL_OFFLOAD_FIELDS=fields.json)")
    def test_the_real_simulator_opens_the_scenario(self):
        import subprocess
        import tempfile
        real = os.environ["BLIS_POOL_OFFLOAD"]
        fields = os.environ.get("BLIS_POOL_OFFLOAD_FIELDS")     # the engine's field list, until `blis fields` has it
        vocab = settings.vocabulary(real, fields)
        base = {"prefill_instances": 1, "decode_instances": 1, "prefill_tp": 8, "decode_tp": 8,
                "prefill_offload_cpu_bytes": 2**39}
        parallel, engine, cli, router = run_blis.classify_config(base, vocab, 8, blis_cli.supports_pool_offload(real))
        for role in run_blis.PD_ROLES:       # run_one states the block size before it renders
            parallel["pd"][role]["engine"].setdefault("block_size", 64)
        text = run_blis.scenario_yaml("qwen3-14b", "h100", blis_cli.engine_release(real, "vllm", fields), parallel, engine,
                                      "x", node_size=8)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "x.yaml"), "w") as f:
                f.write(text)
            pr = subprocess.run([real, "fields", "--scenario", "x.yaml", "--scenarios", d],
                                capture_output=True, text=True)
        self.assertEqual(0, pr.returncode, pr.stderr[-400:])


class ArmsDiffer(unittest.TestCase):
    def test_the_grid_has_a_no_tier_control_and_one_arm_per_size_and_policy(self):
        sw = doc({"cpu_bytes": [0, "256GiB", "512GiB"], "eviction_policy": ["lru", "fifo"]})["sweeps"][0]
        kept, _ = E.expand_sweep(sw, 16)
        arms = [(c["offload_cpu_bytes"], c.get("offload_eviction_policy")) for c in kept]
        self.assertEqual([(0, None), (256 * 2**30, "lru"), (256 * 2**30, "fifo"),
                          (512 * 2**30, "lru"), (512 * 2**30, "fifo")], arms)

    def test_a_sweep_without_the_block_is_unchanged(self):
        kept, _ = E.expand_sweep(doc()["sweeps"][0], 16)
        self.assertTrue(kept and all(not any(k.startswith("offload_") for k in c) for c in kept))

    def test_the_scenario_carries_the_tier_and_its_coefficient_set_only_when_on(self):
        vocab = settings.vocabulary(BLIS)
        texts = {}
        for size in (0, 2**39):
            parallel, engine, cli, router = run_blis.classify_config(
                {"tp": 4, "offload_cpu_bytes": size, "offload_eviction_policy": "lru"}, vocab)
            texts[size] = run_blis.scenario_yaml("llama-3.1-70b-instruct", "h200", "0.33.0", parallel, engine, "x")
        self.assertNotIn("offload", texts[0])
        self.assertNotIn("cost-model-offload-tiers", texts[0])
        self.assertIn("tiers:\n    - {tier: cpu_dram, bytes: 549755813888}", texts[2**39])
        self.assertIn("eviction_policy: \"lru\"", texts[2**39])
        self.assertIn("cost-model-offload-tiers", texts[2**39])
        self.assertNotEqual(texts[0], texts[2**39])

    def test_a_disaggregated_candidate_with_a_tier_is_refused_by_name(self):
        vocab = settings.vocabulary(BLIS)
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config({"prefill_instances": 1, "decode_instances": 1, "prefill_tp": 4,
                                      "decode_tp": 4, "offload_cpu_bytes": 2**39}, vocab)
        self.assertIn("#94", str(e.exception))

    def test_the_arms_are_told_apart_in_the_report(self):
        self.assertEqual("CPU KV tier off", topologies.knob_phrase("offload_cpu_bytes", 0))
        self.assertEqual("CPU KV tier 512 GiB", topologies.knob_phrase("offload_cpu_bytes", 2**39))
        self.assertIn("offload_cpu_bytes", E.VARIANT_KEYS)
        cands = [{"tp": 4, "offload_cpu_bytes": 0}, {"tp": 4, "offload_cpu_bytes": 2**39}]
        base = E.sweep_baseline(cands + [dict(cands[0])])
        self.assertEqual("", E.variant_of(cands[0], "", base))
        self.assertIn("CPU KV tier 512 GiB", E.variant_of(cands[1], "", base))

    def test_the_tiers_hit_rate_is_read_from_the_result(self):
        doc_ = {"metrics": {"output_tok_s": {"value": 10.0, "unit": "tok/s"}},
                "reports": {"engine": {"cluster": {"cpu_tier_hit_rate": 0.25, "cache_hit_rate": 0.5}}}}
        flat = run_blis._flatten_result(doc_)
        self.assertEqual(0.25, flat["cpu_tier_hit_rate"])
        self.assertEqual(0.5, flat["cache_hit_rate"])


if __name__ == "__main__":
    unittest.main()
