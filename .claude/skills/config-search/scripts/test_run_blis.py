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
        # a hand-rolled brace counter would desync on the '}' inside these strings
        s = '{"instance_id":"cluster","tokens_per_sec":7.0,"path":"a{b}c","err":"oops}"}'
        self.assertEqual(run_blis.parse_metrics(s).get("tokens_per_sec"), 7.0)

    def test_nested_envelope(self):
        s = 'prefix {"result":{"instance_id":"cluster","tokens_per_sec":5.0}} suffix'
        self.assertEqual(run_blis.parse_metrics(s).get("tokens_per_sec"), 5.0)

    def test_no_metrics_returns_empty(self):
        self.assertEqual(run_blis.parse_metrics("no json {not valid} here"), {})

    def test_first_metrics_when_no_cluster(self):
        self.assertEqual(run_blis.parse_metrics('{"instance_id":"i0","tokens_per_sec":3.0}')
                         .get("tokens_per_sec"), 3.0)


class BuildCmd(unittest.TestCase):
    def _cmd(self, config):
        return run_blis.build_cmd("blis", "cat", "m", "H100", None, None, 0, None, config, None)

    def test_listed_bool_true_is_presence_only(self):
        c = self._cmd({"enable_expert_parallel": True})
        self.assertIn("--enable-expert-parallel", c)
        self.assertNotIn("True", c)

    def test_unlisted_bool_false_is_omitted(self):
        c = self._cmd({"enable_foo": False})
        self.assertNotIn("--enable-foo", c)
        self.assertNotIn("False", c)

    def test_unlisted_bool_true_is_presence_only(self):
        c = self._cmd({"enable_foo": True})
        self.assertIn("--enable-foo", c)
        self.assertNotIn("True", c)

    def test_scalar_knob_takes_a_value(self):
        c = self._cmd({"tp": 8})
        self.assertEqual(c[c.index("--tp") + 1], "8")


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
        cap, bound, _ = sweep_pool.compute_cap(0.8, 0.8, 1024.0, None)
        self.assertEqual((cap, bound), (8, "cores"))

    def test_memory_bound(self):
        self._patch(64, 4096)  # mem_cap = floor(4096*0.8/1024) = 3
        cap, bound, _ = sweep_pool.compute_cap(0.8, 0.8, 1024.0, None)
        self.assertEqual((cap, bound), (3, "memory"))

    def test_max_parallel_tightens(self):
        self._patch(10, 10 ** 9)
        cap, bound, _ = sweep_pool.compute_cap(0.8, 0.8, 1024.0, 2)
        self.assertEqual((cap, bound), (2, "explicit --max-parallel"))

    def test_max_parallel_equal_not_relabeled(self):
        self._patch(10, 10 ** 9)  # core_cap = 8
        cap, bound, _ = sweep_pool.compute_cap(0.8, 0.8, 1024.0, 8)
        self.assertEqual((cap, bound), (8, "cores"))

    def test_mem_unknown_falls_back_to_cores(self):
        self._patch(10, None)
        cap, bound, info = sweep_pool.compute_cap(0.8, 0.8, 1024.0, None)
        self.assertEqual(cap, 8)
        self.assertIsNone(info["mem_available_mib"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
