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

    def test_multinode_single_instance_rejected(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "dp": 2})                # footprint 16 > one node


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
