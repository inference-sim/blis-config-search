#!/usr/bin/env python3
"""Tests of scripts/shard_experiment.py: bounded shards whose runs are the parent's runs. Hermetic.

Run: python3 -I scripts/test_shard.py
"""
import collections
import copy
import glob
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiment as E  # noqa: E402
import shard_experiment as S  # noqa: E402
import test_experiment as TE  # noqa: E402,F401  (points ROUTER_CONFIGS_DIR at its hermetic router fixture)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spec_of(*sweeps, gpus=16, loads=(1, 2, 4)):
    return {"name": "parent", "scenario": {"model": "m", "hardware": "h100", "workload": "chatbot",
                                           "gpus": gpus, "concurrency": list(loads)},
            "sweeps": list(sweeps)}


COLOCATED = {"name": "colocated", "num-instances": [1, 2, 4, 8],
             "parallel": {"tp": [1, 2, 4, 8], "pp": [1, 2]},
             "engine": {"max_num_seqs": [64, 128, 256], "block_size": [16, 32]}}
OFFLOAD = {"name": "offload", "num-instances": [2, 4], "parallel": {"tp": [2, 4, 8]},
           "offload": {"cpu_bytes": [0, "64GiB", "128GiB"], "eviction_policy": ["lru", "fifo"]},
           "engine": {"max_num_seqs": [64, 128]}}
SPECULATIVE = {"name": "speculative", "num-instances": [2, 4], "parallel": {"tp": [2, 4, 8]},
               "engine": {"speculative": {"num_spec_tokens": [0, 1, 3], "method": ["eagle3", "dspark"],
                                          "draft_model": ["a", "b"]},
                          "max_num_seqs": [64, 128]}}
DISAGGREGATED = {"name": "pd", "prefill-instances": [1, 2, 3], "decode-instances": [1, 2, 4],
                 "prefill": {"parallel": {"tp": [2, 4]}, "engine": {"max_num_seqs": [32, 64]}},
                 "decode": {"parallel": {"tp": [4, 8]}, "engine": {"max_num_seqs": [64, 128]}}}
ROUTED = {"name": "routed", "num-instances": [2, 4], "parallel": {"tp": [2, 4, 8]},
          "router": {"endpoint_picker_config": ["default.yaml", "round-robin.yaml"],
                     "plugins": {"prefix-cache-scorer": {"parameters": {"blockSize": [16, 64],
                                                                       "lruCapacityPerServer": [100, 400]}}}}}


def canon(value):
    if isinstance(value, dict):
        return tuple(sorted(((repr(k), canon(v)) for k, v in value.items())))
    return repr(value)


def multiset(sweeps, gpus, loads):
    seen = collections.Counter()
    for sweep in sweeps:
        kept, dropped = E.expand_sweep(sweep, gpus)
        for cand in kept:
            for load in loads:
                seen[("kept", sweep["name"], canon(cand), load)] += 1
        for cand, total in dropped:
            for load in loads:
                seen[("dropped", sweep["name"], canon(cand), total, load)] += 1
    return seen


def shards_of(spec, max_runs):
    plans = S.plans_for(spec, max_runs)
    total = sum(plan["shards"] for plan in plans)
    return total, [S.compile_shard(spec, i, max_runs, plans) for i in range(total)]


class ShardedEqualsUnsharded(unittest.TestCase):
    def check(self, spec, max_runs):
        gpus, loads = spec["scenario"]["gpus"], spec["scenario"]["concurrency"]
        total, shards = shards_of(spec, max_runs)
        sharded = collections.Counter()
        for shard in shards:
            self.assertEqual(1, len(shard["sweeps"]))
            sharded += multiset(shard["sweeps"], gpus, loads)
        self.assertEqual(multiset(spec["sweeps"], gpus, loads), sharded)
        return total, shards

    def test_every_kind_of_sweep_at_several_bounds(self):
        for sweep in (COLOCATED, OFFLOAD, SPECULATIVE, DISAGGREGATED, ROUTED):
            for max_runs in (27, 30, 60, 100, 100000):
                with self.subTest(sweep=sweep["name"], max_runs=max_runs):
                    self.check(spec_of(sweep), max_runs)

    def test_a_spec_with_several_sweeps(self):
        total, shards = self.check(spec_of(COLOCATED, OFFLOAD, SPECULATIVE), 30)
        self.assertEqual(["colocated", "offload", "speculative"],
                         list(dict.fromkeys(s["sweeps"][0]["name"] for s in shards)))

    def test_budget_refusals_are_kept_not_filtered(self):
        spec = spec_of(COLOCATED)
        dropped = E.expand_sweep(COLOCATED, 16)[1]
        self.assertTrue(dropped)
        total, shards = self.check(spec, 9)
        self.assertEqual(len(dropped) * 3,
                         sum(1 for k in multiset([s["sweeps"][0] for s in shards], 16, [1, 2, 4])
                             if k[0] == "dropped"))

    def test_every_committed_experiment(self):
        paths = sorted(glob.glob(os.path.join(ROOT, "experiments", "*.yaml")))
        self.assertTrue(paths)
        for path in paths:
            with open(path, encoding="utf-8") as f:
                spec = yaml.safe_load(f)
            with self.subTest(spec=os.path.basename(path)):
                self.check(spec, 40 * len(spec["scenario"]["concurrency"]))

    def test_axes_that_expand_together_are_never_cut(self):
        spec = spec_of(SPECULATIVE)
        total, shards = shards_of(spec, 27)
        self.assertGreater(total, 1)
        for shard in shards:
            block = shard["sweeps"][0]["engine"]["speculative"]
            self.assertEqual(SPECULATIVE["engine"]["speculative"], block)
        total, shards = shards_of(spec_of(OFFLOAD), 15)
        for shard in shards:
            self.assertEqual(OFFLOAD["offload"], shard["sweeps"][0]["offload"])

    def test_cutting_a_coupled_axis_would_not_be_exact(self):
        """The reason the plan probes: slicing the speculative method alone repeats the K=0 arm."""
        cut = copy.deepcopy(SPECULATIVE)
        cut["engine"]["speculative"]["method"] = ["eagle3"]
        other = copy.deepcopy(SPECULATIVE)
        other["engine"]["speculative"]["method"] = ["dspark"]
        naive = multiset([cut, other], 16, [1])
        naive = collections.Counter({k[2:]: v for k, v in naive.items()})
        whole = collections.Counter({k[2:]: v for k, v in multiset([SPECULATIVE], 16, [1]).items()})
        self.assertNotEqual(whole, naive)


class Bounded(unittest.TestCase):
    def test_no_shard_holds_more_runs_than_the_bound(self):
        spec = spec_of(COLOCATED, OFFLOAD, DISAGGREGATED, ROUTED)
        for max_runs in (27, 33, 60, 500):
            _, shards = shards_of(spec, max_runs)
            for shard in shards:
                kept, dropped = E.expand_sweep(shard["sweeps"][0], 16)
                self.assertLessEqual((len(kept) + len(dropped)) * 3, max_runs)

    def test_a_bound_below_the_loads_is_refused(self):
        with self.assertRaisesRegex(ValueError, "every declared load"):
            S.plans_for(spec_of(COLOCATED), 2)

    def test_a_bound_below_the_coupled_group_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "expand together"):
            S.plans_for(spec_of(OFFLOAD), 3 * 2)

    def test_the_index_is_checked_against_the_shard_count(self):
        spec = spec_of(COLOCATED)
        plans = S.plans_for(spec, 12)
        total = sum(plan["shards"] for plan in plans)
        S.compile_shard(spec, total - 1, 12, plans)
        for bad in (-1, total, total + 5):
            with self.assertRaises(IndexError):
                S.compile_shard(spec, bad, 12, plans)

    def test_a_shard_is_a_new_name_that_still_validates_as_one(self):
        _, shards = shards_of(spec_of(COLOCATED), 12)
        names = [s["name"] for s in shards]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertRegex(name, S.NAME.pattern)
        too_long = spec_of(COLOCATED)
        too_long["name"] = "a" * 60
        with self.assertRaisesRegex(ValueError, "shorten"):
            S.compile_shard(too_long, 0, 12)


class PlanCostsTheProbesNotTheGrid(unittest.TestCase):
    def test_a_two_million_point_grid_is_planned_without_expanding_it(self):
        big = {"name": "big", "num-instances": list(range(1, 11)),
               "parallel": {"tp": [1, 2, 4, 8], "pp": [1, 2], "dp": [1, 2]},
               "engine": {"max_num_seqs": list(range(8, 8 + 10)), "block_size": [16, 32],
                          "max_num_batched_tokens": list(range(100, 110)),
                          "gpu_memory_utilization": [0.8, 0.85, 0.9, 0.95],
                          "enable_prefix_caching": [True, False],
                          "long_prefill_token_threshold": [0, 512, 1024, 2048, 4096]},
               "simulator": {"seed_offset": [0, 1, 2, 3, 4, 5, 6]}}
        spec = spec_of(big, loads=(1,))
        points = 10 * 4 * 2 * 2 * 10 * 2 * 10 * 4 * 2 * 5 * 7
        self.assertGreaterEqual(points, 2_000_000)
        sizes = []
        real = E.expand_sweep

        def watch(sweep, gpus):
            kept, dropped = real(sweep, gpus)
            sizes.append(len(kept) + len(dropped))
            return kept, dropped
        with mock.patch.object(S.run_experiment, "expand_sweep", watch):
            plan = S.plans_for(spec, 50)
        self.assertEqual(points, plan[0]["points"])
        self.assertLess(max(sizes), 1000)
        self.assertLess(sum(sizes), 20000)
        self.assertGreater(plan[0]["shards"], points // 50)
        with mock.patch.object(S.run_experiment, "expand_sweep", watch):
            shard = S.compile_shard(spec, plan[0]["shards"] - 1, 50, plan)
        kept, dropped = real(shard["sweeps"][0], 16)
        self.assertLessEqual(len(kept) + len(dropped), 50)

    def test_a_coupling_no_pair_shows_is_caught_when_the_shard_is_compiled(self):
        sweep = {"name": "triple", "num-instances": [1, 2], "parallel": {"tp": [1, 2], "pp": [1, 2]}}
        real = E.expand_sweep

        def hidden(sw, gpus):
            kept, dropped = real(sw, gpus)
            seen = {(c["num_instances"], c["tp"], c["pp"]) for c in kept} | {
                (c["num_instances"], c["tp"], c["pp"]) for c, _ in dropped}
            if (2, 2, 2) in seen:
                kept = [c for c in kept if (c["num_instances"], c["tp"], c["pp"]) != (2, 2, 2)]
                dropped = [d for d in dropped if (d[0]["num_instances"], d[0]["tp"], d[0]["pp"]) != (2, 2, 2)]
            return kept, dropped
        spec = spec_of(sweep, gpus=4, loads=(1,))
        with mock.patch.object(S.run_experiment, "expand_sweep", hidden):
            plans = S.plans_for(spec, 8)
            with self.assertRaisesRegex(ValueError, "pairwise probes missed"):
                for i in range(plans[0]["shards"]):
                    S.compile_shard(spec, i, 8, plans)


class CommandLine(unittest.TestCase):
    def test_the_plan_and_a_shard_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "spec.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(spec_of(COLOCATED), f)
            out = os.path.join(d, "shard.yaml")
            with mock.patch("sys.stdout") as stdout:
                S.main([path, "--max-runs", "12"])
            printed = json.loads("".join(c.args[0] for c in stdout.write.call_args_list))
            self.assertEqual(12, printed["max_runs_per_shard"])
            self.assertEqual(S.plans_for(spec_of(COLOCATED), 12)[0]["shards"], printed["shards"])
            S.main([path, "--max-runs", "12", "--index", "1", "--out", out])
            with open(out, encoding="utf-8") as f:
                self.assertEqual(S.compile_shard(spec_of(COLOCATED), 1, 12), yaml.safe_load(f))
            with self.assertRaises(FileExistsError):
                S.main([path, "--max-runs", "12", "--index", "1", "--out", out])


if __name__ == "__main__":
    unittest.main(verbosity=2)
