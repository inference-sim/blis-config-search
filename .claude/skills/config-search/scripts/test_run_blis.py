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
        s = ('=== Simulation Metrics ===\n'
             '{"instance_id":"i0","tokens_per_sec":10.0}\n'
             '{"instance_id":"cluster","tokens_per_sec":42.0}\n')
        self.assertEqual(run_blis.parse_metrics(s).get("tokens_per_sec"), 42.0)

    def test_braces_inside_string_values(self):
        s = '{"instance_id":"cluster","tokens_per_sec":7.0,"path":"a{b}c","err":"oops}"}'
        self.assertEqual(run_blis.parse_metrics(s).get("tokens_per_sec"), 7.0)

    def test_nested_envelope(self):
        s = 'prefix {"result":{"instance_id":"cluster","tokens_per_sec":5.0}} suffix'
        self.assertEqual(run_blis.parse_metrics(s).get("tokens_per_sec"), 5.0)

    def test_no_metrics_returns_empty(self):
        self.assertEqual(run_blis.parse_metrics("no json {not valid} here"), {})

    def test_all_instances_excludes_cluster(self):
        s = ('{"instance_id":"i0","tokens_per_sec":1.0}\n'
             '{"instance_id":"i1","tokens_per_sec":2.0}\n'
             '{"instance_id":"cluster","tokens_per_sec":3.0}')
        got = [o["instance_id"] for o in run_blis.parse_all_instances(s)]
        self.assertEqual(got, ["i0", "i1"])


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


class Deployment(unittest.TestCase):
    def test_split_parallel_engine_cli(self):
        parallel, engine = run_blis._deployment(
            {"tp": 8, "enable_expert_parallel": True, "block_size": 32, "num_instances": 2, "concurrency": 64})
        self.assertEqual(parallel["tp"], 8)
        self.assertTrue(parallel["enable_expert_parallel"])
        self.assertEqual(engine["block_size"], 32)
        self.assertEqual(engine["max_num_seqs"], 256)          # default filled
        self.assertNotIn("num_instances", {**parallel, **engine})  # CLI, not scenario
        self.assertNotIn("concurrency", {**parallel, **engine})


class ScenarioYaml(unittest.TestCase):
    def test_renders_lowercase_bool_and_both_docs(self):
        parallel, engine = run_blis._deployment({"tp": 8, "enable_expert_parallel": True})
        y = run_blis.scenario_yaml("glm-5.3-flash", "h100", "0.29.0", parallel, engine, "s")
        self.assertIn("kind: Scenario", y)
        self.assertIn("kind: Deployment", y)
        self.assertIn("model: glm-5.3-flash", y)
        self.assertIn("hardware: h100", y)
        self.assertIn("enable_expert_parallel: true", y)       # not Python "True"
        self.assertIn("cost-model-memory", y)                  # required coefficient set listed


class BuildCmd(unittest.TestCase):
    def test_kernel_flags_and_cli_knobs(self):
        cmd = run_blis.build_cmd("blis", "cat", "reg", "scn", "s.yaml", "chatbot", 100, 0, None, None,
                                 {"num_instances": 2, "concurrency": 64})
        self.assertEqual(cmd[:3], ["blis", "run", "--latency-model"])
        self.assertIn("blis-latency-kernel", cmd)
        for pair in (("--scenario", "s.yaml"), ("--scenarios", "scn"), ("--registry", "reg"),
                     ("--num-instances", "2"), ("--concurrency", "64")):
            self.assertEqual(cmd[cmd.index(pair[0]) + 1], pair[1])
        # the kernel REFUSES these -- the runner must never emit them
        for refused in ("--tp", "--model", "--hardware", "--enable-expert-parallel", "--dp"):
            self.assertNotIn(refused, cmd)


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
        self._patch(64, 4096)  # mem_cap = floor(4096*0.8/1024) = 3
        self.assertEqual(sweep_pool.compute_cap(0.8, 0.8, 1024.0, None)[:2], (3, "memory"))

    def test_max_parallel_tightens(self):
        self._patch(10, 10 ** 9)
        self.assertEqual(sweep_pool.compute_cap(0.8, 0.8, 1024.0, 2)[:2], (2, "explicit --max-parallel"))

    def test_max_parallel_equal_not_relabeled(self):
        self._patch(10, 10 ** 9)
        self.assertEqual(sweep_pool.compute_cap(0.8, 0.8, 1024.0, 8)[:2], (8, "cores"))

    def test_mem_unknown_falls_back_to_cores(self):
        self._patch(10, None)
        cap, _, info = sweep_pool.compute_cap(0.8, 0.8, 1024.0, None)
        self.assertEqual(cap, 8)
        self.assertIsNone(info["mem_available_mib"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
