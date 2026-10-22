#!/usr/bin/env python3
"""Unit tests for the config-search engine helpers. No network / no BLIS binary needed.

    python3 scripts/test_run_blis.py          # or: python3 -m unittest -v  (from this dir)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_blis
import sweep_pool


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
        par, eng, cli = run_blis.classify_config(
            {"tp": 8, "enable_expert_parallel": True, "block_size": 32, "num_instances": 2, "concurrency": 64})
        self.assertEqual((par["tp"], par["pp"], par["dp"]), (8, 1, 1))
        self.assertTrue(par["enable_expert_parallel"])
        self.assertEqual(eng["block_size"], 32)
        self.assertEqual(cli, {"num_instances": 2, "concurrency": 64})

    def test_precision_not_defaulted(self):
        _, eng, _ = run_blis.classify_config({"tp": 8})
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
        _, eng, _ = run_blis.classify_config({"tp": 8, "block_size": 64})
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
        self.assertIn("block_size: 64", run_blis.scenario_yaml("glm-5.3", "h200", "0.29.0", par,
                                                               eng, "n", 1))

    def test_precision_passes_through_when_set(self):
        _, eng, _ = run_blis.classify_config({"tp": 8, "quantization": "fp8", "cache_dtype": "fp8"})
        self.assertEqual(eng["quantization"], "fp8")

    def test_policy_knobs_go_to_cli(self):
        _, _, cli = run_blis.classify_config({"tp": 8, "scheduler": "fcfs", "routing_policy": "round-robin"})
        self.assertEqual(cli, {"scheduler": "fcfs", "routing_policy": "round-robin"})

    def test_rejects_unknown_key(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "totally_made_up": 1})

    def test_rejects_pd_keys(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "prefill_tp": 4, "decode_tp": 8})

    def test_rejects_long_prefill(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "long_prefill_token_threshold": 4096})

    def test_rejects_pp_gt_1(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "pp": 2})


class Footprint(unittest.TestCase):
    def test_tp_pp_dp(self):
        self.assertEqual(run_blis.footprint_gpus({"tp": 8, "pp": 1, "dp": 2}), 16)
        self.assertEqual(run_blis.footprint_gpus({"tp": 4, "pp": 1, "dp": 1}), 4)

    def test_within_node_dp_allowed(self):
        _, _, _ = run_blis.classify_config({"tp": 4, "dp": 2})          # footprint 8 == one node, OK

    def test_multinode_single_instance_allowed_on_whole_nodes(self):
        # A multi-node single instance is REAL in BLIS: kimi-k3 needs a 32-GPU footprint to fit its
        # weights (332.3 GiB/rank at tp=8 on a 141 GiB device) and runs at tp=32 across 4 nodes
        # once the scenario names a fabric. The runner used to refuse it, which made the largest
        # models unreachable rather than slow.
        par, _, _ = run_blis.classify_config({"tp": 32, "enable_expert_parallel": True})
        self.assertEqual(run_blis.footprint_gpus(par), 32)
        par2, _, _ = run_blis.classify_config({"tp": 8, "dp": 2})
        self.assertEqual(run_blis.footprint_gpus(par2), 16)

    def test_multinode_footprint_must_fill_whole_nodes(self):
        # A footprint that straddles a node boundary (e.g. 12 GPUs = 1.5 nodes) describes no real
        # placement: the pool is allocated in whole nodes, so the remainder would sit idle and the
        # cross-node cost would be mispriced. Refuse it rather than silently round the pool up.
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 2, "dp": 6})                # 12 GPUs: 1.5 nodes

    def test_multinode_scenario_names_a_fabric(self):
        # Cross-node cost resolves from the fabric, not the chip; BLIS refuses a multi-node
        # cluster that does not name one.
        par, eng, _ = run_blis.classify_config({"tp": 32, "enable_expert_parallel": True})
        eng.setdefault("block_size", 64)
        y = run_blis.scenario_yaml("kimi-k3", "h200", "0.29.0", par, eng, "s", 1)
        self.assertIn("fabric:", y)
        self.assertIn("nodes: 4", y)                                    # 32 GPUs / 8 per node


class ScenarioYaml(unittest.TestCase):
    def test_omits_unset_precision_lists_memory_coeff(self):
        par, eng, _ = run_blis.classify_config({"tp": 8})
        y = run_blis.scenario_yaml("qwen3-14b", "h100", "0.29.0", par, eng, "s")
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
        many = run_blis.scenario_yaml("m", "h100", "0.29.0", par, eng, "n", num_instances=16)
        self.assertIn("nodes: 16", many)
        self.assertIn("fabric: ", many)                     # required once nodes > 1
        one = run_blis.scenario_yaml("m", "h100", "0.29.0", par, eng, "n", num_instances=1)
        self.assertIn("nodes: 1", one)
        self.assertNotIn("fabric:", one)                    # single node prices no cross-node traffic

    def test_instance_count_changes_the_scenario_identity(self):
        """Two runs differing only in instance count need different files, else the second reuses
        a pool too small for it."""
        import tempfile
        d = tempfile.mkdtemp()
        par, eng = {"tp": 4, "pp": 1, "dp": 1}, {"block_size": 16}
        a = run_blis.write_scenario(d, "m", "h100", "0.29.0", par, eng, 1)
        b = run_blis.write_scenario(d, "m", "h100", "0.29.0", par, eng, 8)
        self.assertNotEqual(a, b)


class BuildCmd(unittest.TestCase):
    def test_kernel_flags_and_cli_knobs(self):
        cmd = run_blis.build_cmd("blis", "cat", "reg", "scn", "s.yaml", "chatbot", 100, 0, None, None,
                                 {"num_instances": 2, "concurrency": 64, "scheduler": "fcfs"})
        # v0.10.0 selects the kernel backend from the scenario's `coefficients:` list, NOT a
        # `--latency-model` CLI flag (the flag was removed and is rejected as unknown).
        self.assertNotIn("--latency-model", cmd)
        self.assertNotIn("blis-latency-kernel", cmd)
        for a, b in (("--scenario", "s.yaml"), ("--scenarios", "scn"), ("--registry", "reg"),
                     ("--num-instances", "2"), ("--concurrency", "64"), ("--scheduler", "fcfs")):
            self.assertEqual(cmd[cmd.index(a) + 1], b)
        for refused in ("--tp", "--model", "--hardware", "--enable-expert-parallel", "--dp"):
            self.assertNotIn(refused, cmd)


class RunOneRejects(unittest.TestCase):
    def test_unsupported_config_is_an_error_row_not_a_crash(self):
        rec = run_blis.run_one("blis", "cat", "reg", "/tmp/scn-x", "m", "h100", "0.29.0",
                               None, None, 0, None, {"tp": 8, "prefill_tp": 4})
        self.assertIsNone(rec["returncode"])
        self.assertIn("unsupported config", rec["error"])
        self.assertEqual(rec["metrics"], {})
        # same row SHAPE as a real run so sweep_pool summary + the report never trip on missing keys
        for k in ("wall_s", "deployment", "metrics_all_instances", "stderr_tail", "cmd"):
            self.assertIn(k, rec)
        self.assertEqual(rec["wall_s"], 0.0)


class ComputeCap(unittest.TestCase):
    def setUp(self):
        self._cpus, self._mem = sweep_pool.available_cpus, sweep_pool.mem_available_mib

    def tearDown(self):
        sweep_pool.available_cpus, sweep_pool.mem_available_mib = self._cpus, self._mem

    def _patch(self, cpus, mem):
        sweep_pool.available_cpus = lambda: cpus
        sweep_pool.mem_available_mib = lambda: mem

    def test_cores_bound(self):
        self._patch(10, 10 ** 9)
        self.assertEqual(sweep_pool.compute_cap(0.8, 0.8, 1024.0, None)[:2], (8, "cores"))

    def test_memory_bound(self):
        self._patch(64, 4096)
        self.assertEqual(sweep_pool.compute_cap(0.8, 0.8, 1024.0, None)[:2], (3, "memory"))

    def test_max_parallel_tightens(self):
        self._patch(10, 10 ** 9)
        self.assertEqual(sweep_pool.compute_cap(0.8, 0.8, 1024.0, 2)[:2], (2, "explicit --max-parallel"))

    def test_max_parallel_equal_not_relabeled(self):
        self._patch(10, 10 ** 9)
        self.assertEqual(sweep_pool.compute_cap(0.8, 0.8, 1024.0, 8)[:2], (8, "cores"))

    def test_mem_unknown_conservative_cap(self):
        self._patch(10, None)                     # core_cap 8, no /proc/meminfo
        cap, bound, info = sweep_pool.compute_cap(0.8, 0.8, 1024.0, None)
        self.assertEqual(cap, sweep_pool.UNKNOWN_MEM_CAP)     # conservative fallback, not 8
        self.assertEqual(bound, "memory-unknown-fallback")
        self.assertIsNone(info["mem_available_mib"])

    def test_mem_unknown_but_max_parallel_wins(self):
        self._patch(10, None)
        self.assertEqual(sweep_pool.compute_cap(0.8, 0.8, 1024.0, 2)[:2], (2, "explicit --max-parallel"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
