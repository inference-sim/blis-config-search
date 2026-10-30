#!/usr/bin/env python3
"""Unit tests for the config-search engine helpers. No network / no BLIS binary needed.

    python3 scripts/test_run_blis.py          # or: python3 -m unittest -v  (from this dir)
"""
import os
import sys
import tempfile
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
        par, eng, cli, _ = run_blis.classify_config(
            {"tp": 8, "enable_expert_parallel": True, "block_size": 32, "num_instances": 2, "concurrency": 64})
        self.assertEqual((par["tp"], par["pp"], par["dp"]), (8, 1, 1))
        self.assertTrue(par["enable_expert_parallel"])
        self.assertEqual(eng["block_size"], 32)
        self.assertEqual(cli, {"num_instances": 2, "concurrency": 64})

    def test_precision_not_defaulted(self):
        _, eng, _, _ = run_blis.classify_config({"tp": 8})
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
        _, eng, _, _ = run_blis.classify_config({"tp": 8, "block_size": 64})
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
        _, eng, _, _ = run_blis.classify_config({"tp": 8, "quantization": "fp8", "cache_dtype": "fp8"})
        self.assertEqual(eng["quantization"], "fp8")

    def test_policy_knobs_go_to_cli(self):
        _, _, cli, _ = run_blis.classify_config(
            {"tp": 8, "scheduler": "fcfs", "preemption_policy": "fcfs"})
        self.assertEqual(cli, {"scheduler": "fcfs", "preemption_policy": "fcfs"})

    def test_rejects_routing_flags_removed_in_v0_12_0(self):
        """v0.12.0 moved routing into the embedded llm-d router, so the old flags are gone.

        They must fail LOUDLY with the migration path: silently dropping them would run an
        all-defaults router while the ledger row still claimed a routing policy.
        """
        for key in ("routing_policy", "routing_scorers", "admission_policy"):
            with self.assertRaises(run_blis.ConfigError) as cm:
                run_blis.classify_config({"tp": 8, key: "least-loaded"})
            self.assertIn("router_config", str(cm.exception))

    def test_router_config_goes_to_the_scenario(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml") as epp:
            _, _, cli, router = run_blis.classify_config({"tp": 8, "router_config": epp.name})
            self.assertEqual(router, {"router_config": epp.name})
            self.assertEqual(cli, {})            # it is a SCENARIO key, never a CLI flag

    def test_router_config_must_exist(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "router_config": "/nonexistent/epp.yaml"})

    def test_rejects_unknown_key(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "totally_made_up": 1})

    def test_pd_keys_need_both_instance_counts(self):
        """P/D is supported now, but a candidate naming per-role layout without the instance
        counts describes no deployment -- BLIS refuses a pool with no counterpart."""
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"prefill_tp": 4, "decode_tp": 8})

    def test_accepts_long_prefill(self):
        """`long_prefill_token_threshold` IS a field of the pinned blis-schemas (v0.5.0) Engine
        struct, and a run naming it completes and moves the number (measured 1983.2 -> 1973.9
        tok/s at 256 on chatbot tp8 c=8). It was listed unsupported on the strength of a stale
        note about the kernel, which cost the search a real prefill-admission knob."""
        _par, engine, _cli, _router = run_blis.classify_config(
            {"tp": 8, "long_prefill_token_threshold": 4096})
        self.assertEqual(4096, engine["long_prefill_token_threshold"])

    def test_rejects_knobs_the_pinned_schema_has_no_field_for(self):
        """vLLM v0.32.0 defines ~51 engine fields; the pinned blis-schemas v0.5.0 has 31. A key
        outside those fails the scenario's strict decode, so it is refused here with that reason
        rather than written into a scenario that cannot load."""
        with self.assertRaises(run_blis.ConfigError) as e:
            run_blis.classify_config({"tp": 8, "max_num_active_seqs": 128})
        self.assertIn("pinned BLIS", str(e.exception))

    def test_rejects_pp_gt_1(self):
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 8, "pp": 2})

    def test_carries_the_wider_engine_surface(self):
        """Every knob in SCENARIO_ENGINE must reach the scenario's engine block.

        Verified against the pinned binary: a scenario carrying all of these loads and runs.
        """
        cfg = {"tp": 8, "num_instances": 1, "cudagraph_mode": "FULL",
               "max_cudagraph_capture_size": 256, "enable_prefix_caching": False,
               "scheduling_policy": "priority", "async_scheduling": True,
               "allreduce_backend": "nccl", "disable_cascade_attn": True,
               "all2all_backend": "deepep_low_latency", "moe_backend": "triton",
               "long_prefill_token_threshold": 2048}
        _par, engine, _cli, _router = run_blis.classify_config(cfg)
        for key, want in cfg.items():
            if key in ("tp", "num_instances"):
                continue
            self.assertEqual(want, engine[key], key)

    def test_booleans_and_int_lists_render_as_valid_yaml(self):
        """json.dumps gives YAML-valid `true`/`false` and `[1, 2, 4]`; a Python repr would not."""
        par, engine, _cli, _r = run_blis.classify_config(
            {"tp": 8, "enable_prefix_caching": False, "async_scheduling": True,
             "cudagraph_capture_sizes": [1, 2, 4]})
        doc = run_blis.scenario_yaml("qwen3-14b", "h100", "0.29.0", par, engine, "x", 1)
        self.assertIn("enable_prefix_caching: false", doc)
        self.assertIn("async_scheduling: true", doc)
        self.assertIn("cudagraph_capture_sizes: [1, 2, 4]", doc)
        self.assertNotIn("False", doc)
        self.assertNotIn("True", doc)


class Footprint(unittest.TestCase):
    def test_tp_pp_dp(self):
        self.assertEqual(run_blis.footprint_gpus({"tp": 8, "pp": 1, "dp": 2}), 16)
        self.assertEqual(run_blis.footprint_gpus({"tp": 4, "pp": 1, "dp": 1}), 4)

    def test_within_node_dp_allowed(self):
        _, _, _, _ = run_blis.classify_config({"tp": 4, "dp": 2})          # footprint 8 == one node, OK

    def test_multinode_single_instance_allowed_on_whole_nodes(self):
        # A multi-node single instance is REAL in BLIS: kimi-k3 needs a 32-GPU footprint to fit its
        # weights (332.3 GiB/rank at tp=8 on a 141 GiB device) and runs at tp=32 across 4 nodes
        # once the scenario names a fabric. The runner used to refuse it, which made the largest
        # models unreachable rather than slow.
        par, _, _, _ = run_blis.classify_config({"tp": 32, "enable_expert_parallel": True})
        self.assertEqual(run_blis.footprint_gpus(par), 32)
        par2, _, _, _ = run_blis.classify_config({"tp": 8, "dp": 2})
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
        par, eng, _, _ = run_blis.classify_config({"tp": 32, "enable_expert_parallel": True})
        eng.setdefault("block_size", 64)
        y = run_blis.scenario_yaml("kimi-k3", "h200", "0.29.0", par, eng, "s", 1)
        self.assertIn("fabric:", y)
        self.assertIn("nodes: 4", y)                                    # 32 GPUs / 8 per node


class ScenarioYaml(unittest.TestCase):
    def test_omits_unset_precision_lists_memory_coeff(self):
        par, eng, _, _ = run_blis.classify_config({"tp": 8})
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


def _write_catalog(root, model, arch=None, sparse_mla=False, hardware=None, cc=None,
                   workloads=None):
    """A minimal on-disk catalog: only the bytes these helpers actually scan."""
    mdir = os.path.join(root, "models", model)
    os.makedirs(mdir, exist_ok=True)
    lines = ["kind: ModelGraph", f"name: {model}", "global:"]
    if arch:
        lines.append(f"  architecture: {arch}")
    if sparse_mla:
        lines.append("  attention_kind: sparse_mla")
    with open(os.path.join(mdir, "graph.yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if hardware:
        hdir = os.path.join(root, "hardware")
        os.makedirs(hdir, exist_ok=True)
        with open(os.path.join(hdir, hardware + ".yaml"), "w") as f:
            f.write('_comment_cc: "ComputeCapability is prose that must NOT be parsed"\n')
            if cc:
                f.write(f'ComputeCapability: "{cc}"\n')
    for name, body in (workloads or {}).items():
        wdir = os.path.join(root, "workloads")
        os.makedirs(wdir, exist_ok=True)
        with open(os.path.join(wdir, name + ".yaml"), "w") as f:
            f.write(body)


class BlockSizeIsModelAndPartAware(unittest.TestCase):
    """block_size cannot be one constant: the kernel's rule depends on BOTH the architecture and
    the part (blis-latency-kernel sparse_mla.go `v4BlockSize`). A flat 64 made every
    DeepSeek-V4.1 Blackwell run fail; a flat 16 made every DSA model fail."""

    def _bs(self, **kw):
        wl = kw.pop("workloads", None)
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m", workloads=wl, **kw)
            run_blis._ARCH_CACHE.clear(); run_blis._CC_CACHE.clear(); run_blis._DSA_CACHE.clear()
            return run_blis.default_block_size(d, "m", kw.get("hardware"))

    def test_v41_on_hopper_is_64(self):
        self.assertEqual(self._bs(arch="DeepseekV41ForCausalLM", hardware="h100", cc="9.0"), 64)

    def test_v41_on_blackwell_is_128(self):
        self.assertEqual(self._bs(arch="DeepseekV41ForCausalLM", hardware="b200", cc="10.0"), 128)
        self.assertEqual(self._bs(arch="DeepseekV41ForCausalLM", hardware="b300", cc="10.3"), 128)

    def test_v4_is_256_on_every_part(self):
        self.assertEqual(self._bs(arch="DeepseekV4ForCausalLM", hardware="h200", cc="9.0"), 256)
        self.assertEqual(self._bs(arch="DeepseekV4ForCausalLM", hardware="b200", cc="10.0"), 256)

    def test_v41_unstated_capability_falls_back_to_hopper(self):
        """The kernel infers the family and records the assumption rather than failing, so the
        runner matches it instead of guessing Blackwell."""
        self.assertEqual(self._bs(arch="DeepseekV41ForCausalLM", hardware="mystery"), 64)

    def test_dsa_non_v4_is_64_and_plain_model_is_16(self):
        self.assertEqual(self._bs(sparse_mla=True, hardware="h100", cc="9.0"), 64)
        self.assertEqual(self._bs(hardware="h100", cc="9.0"), 16)

    def test_comment_prose_is_not_read_as_the_capability(self):
        """hardware/*.yaml carries a _comment_cc line containing the word ComputeCapability; a
        looser scan would read the prose and mis-size the block."""
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m", arch="DeepseekV41ForCausalLM", hardware="b200", cc="10.0")
            run_blis._CC_CACHE.clear()
            self.assertEqual(run_blis.compute_capability_major(d, "b200"), 10)


class WorkloadShapeIsPassedExplicitly(unittest.TestCase):
    """`--workload <preset>` is IGNORED under --concurrency (cmd/root.go takes the concurrency
    branch before the preset loader), so the preset's shape must reach BLIS as explicit flags or
    every run is priced on the 512/512 defaults while claiming the preset."""

    SHAPE = ("prompt:\n  tokens: 10240\n  tokens_stdev: 1200\n  tokens_min: 500\n"
             "  tokens_max: 20480\noutput:\n  tokens: 1536\n  tokens_stdev: 300\n"
             "  tokens_min: 50\n  tokens_max: 4096\n")

    def test_shape_preset_becomes_token_flags(self):
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m", workloads={"multidoc": self.SHAPE})
            run_blis._PRESET_CACHE.clear()
            shape = run_blis.load_workload_shape(d, "multidoc")
        self.assertEqual(shape["--prompt-tokens"], 10240)
        self.assertEqual(shape["--output-tokens"], 1536)
        self.assertEqual(shape["--prompt-tokens-max"], 20480)

    def test_replay_preset_is_refused_not_silently_defaulted(self):
        """A replay has no ISL/OSL by construction. Returning {} would let `blis run` serve the
        CLI defaults under a replay's name -- the exact mislabelling this guards."""
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m", workloads={"agentx-mvp": "replay:\n  scenario: x\n"})
            run_blis._PRESET_CACHE.clear()
            with self.assertRaises(run_blis.ConfigError) as cm:
                run_blis.load_workload_shape(d, "agentx-mvp")
            self.assertIn("REPLAY", str(cm.exception))

    def test_unknown_preset_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _write_catalog(d, "m")
            run_blis._PRESET_CACHE.clear()
            with self.assertRaises(run_blis.ConfigError):
                run_blis.load_workload_shape(d, "no-such-preset")

    def test_build_cmd_emits_the_shape_flags(self):
        cmd = run_blis.build_cmd("/b", None, "/reg", "/sc", "s.yaml", "multidoc", 100, 0, None,
                                 None, {}, {"--prompt-tokens": 10240, "--output-tokens": 1536})
        self.assertIn("--prompt-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--prompt-tokens") + 1], "10240")


class ReplayCommand(unittest.TestCase):
    REPLAY = {"scenario": "inferencex-agentx-mvp", "corpus": "/c/traces.jsonl",
              "corpus_sha256": "abc123", "max_context_length": "max_model_len",
              "benchmark_duration": "900s", "corpus_cache": "/cache"}

    def test_uses_the_replay_subcommand_with_the_corpus_digest(self):
        cmd = run_blis.build_replay_cmd("/b", "/cat", "/reg", "/sc", "s.yaml", 42, None,
                                        {"concurrency": 16, "num_instances": 2}, self.REPLAY)
        self.assertEqual(cmd[1], "replay")
        self.assertEqual(cmd[cmd.index("--agentic-scenario") + 1], "inferencex-agentx-mvp")
        # The digest is REQUIRED upstream: the harness reads the dataset's default branch, so only
        # the file's bytes tie a run to its corpus.
        self.assertEqual(cmd[cmd.index("--corpus-sha256") + 1], "abc123")
        self.assertEqual(cmd[cmd.index("--concurrency") + 1], "16")

    def test_passes_no_num_requests(self):
        """A replay is bounded by its sending window, not a request count; --num-requests is not
        even a replay flag."""
        cmd = run_blis.build_replay_cmd("/b", "/cat", "/reg", "/sc", "s.yaml", 0, None, {},
                                        self.REPLAY)
        self.assertNotIn("--num-requests", cmd)
        self.assertNotIn("--workload", cmd)

    def test_requires_corpus_and_digest(self):
        class A:
            agentic_scenario = "inferencex-agentx-mvp"
            corpus = None
            corpus_sha256 = None
        with self.assertRaises(ValueError):
            run_blis.replay_from_args(A())

    def test_none_without_an_agentic_scenario(self):
        class A:
            agentic_scenario = None
        self.assertIsNone(run_blis.replay_from_args(A()))


class RouterBlockInScenario(unittest.TestCase):
    def test_router_is_deployment_level_not_pool_level(self):
        """blis-schemas puts Router on Deployment (a sibling of pools:), so a pool-indented block
        would be rejected as an unknown pool field."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as epp:
            y = run_blis.scenario_yaml("m", "h100", "0.29.0", dict(run_blis.PARALLEL_DEFAULTS),
                                       {"block_size": 64}, "n", 1, run_blis.DEFAULT_FABRIC,
                                       {"router_config": epp.name})
        self.assertRegex(y, r"(?m)^router:$")
        self.assertRegex(y, r"(?m)^  endpoint_picker_config: ")

    def test_absent_router_emits_no_block(self):
        y = run_blis.scenario_yaml("m", "h100", "0.29.0", dict(run_blis.PARALLEL_DEFAULTS),
                                   {"block_size": 16}, "n")
        self.assertNotIn("router:", y)

    def test_router_participates_in_the_scenario_hash(self):
        """Two routing arms differ ONLY in the router block: sharing a file would run the second
        arm with the first arm's router."""
        import shutil
        d = tempfile.mkdtemp()
        try:
            with tempfile.NamedTemporaryFile(suffix=".yaml") as a, \
                 tempfile.NamedTemporaryFile(suffix=".yaml") as b:
                args = ("m", "h100", "0.29.0", dict(run_blis.PARALLEL_DEFAULTS),
                        {"block_size": 64}, 1, run_blis.DEFAULT_FABRIC)
                n1 = run_blis.write_scenario(d, *args, {"router_config": a.name})
                n2 = run_blis.write_scenario(d, *args, {"router_config": b.name})
                n0 = run_blis.write_scenario(d, *args, None)
            self.assertEqual(len({n0, n1, n2}), 3)
        finally:
            shutil.rmtree(d)


class SpeculativeDecoding(unittest.TestCase):
    """Drafting is priced by v0.12.0 + registry v0.2.0; the runner used to refuse it outright as
    "not yet validated on the kernel backend", which was true of v0.10.0 only."""

    def test_depth_becomes_a_nested_engine_block(self):
        _, eng, _, _ = run_blis.classify_config({"tp": 4, "num_speculative_tokens": 5})
        self.assertEqual(eng["speculative"], {"num_spec_tokens": 5})

    def test_method_is_carried_when_stated(self):
        _, eng, _, _ = run_blis.classify_config(
            {"tp": 4, "num_speculative_tokens": 3, "speculative_method": "dspark"})
        self.assertEqual(eng["speculative"], {"method": "dspark", "num_spec_tokens": 3})

    def test_zero_depth_is_drafting_off(self):
        _, eng, _, _ = run_blis.classify_config({"tp": 4, "num_speculative_tokens": 0})
        self.assertNotIn("speculative", eng)

    def test_method_without_depth_is_refused(self):
        """Naming a drafter while drafting 0 tokens would record a drafter that never ran."""
        with self.assertRaises(run_blis.ConfigError):
            run_blis.classify_config({"tp": 4, "speculative_method": "dspark"})

    def test_acceptance_selectors_go_to_the_cli(self):
        _, _, cli, _ = run_blis.classify_config(
            {"tp": 4, "num_speculative_tokens": 5,
             "speculative_category": "coding", "speculative_thinking": "on"})
        self.assertEqual(cli, {"speculative_category": "coding", "speculative_thinking": "on"})
        cmd = run_blis.build_cmd("/b", None, "/r", "/s", "s.yaml", None, None, 0, None, None, cli)
        self.assertEqual(cmd[cmd.index("--speculative-category") + 1], "coding")
        self.assertEqual(cmd[cmd.index("--speculative-thinking") + 1], "on")

    def test_nested_block_is_rendered_as_yaml_not_json_flow(self):
        """json.dumps on the dict would emit flow style, which the strict loader rejects."""
        y = run_blis.scenario_yaml("m", "b200", "0.29.0", dict(run_blis.PARALLEL_DEFAULTS),
                                   {"block_size": 128,
                                    "speculative": {"method": "dspark", "num_spec_tokens": 5}},
                                   "n")
        self.assertIn("      speculative:\n", y)
        self.assertIn('        method: "dspark"', y)
        self.assertIn("        num_spec_tokens: 5", y)
        self.assertNotIn('{"method"', y)

    def test_drafter_is_read_from_the_models_own_graph(self):
        """The right drafter is a property of the MODEL (dspark here, mtp on v4-pro), so it is
        read from graph.yaml rather than hardcoded or demanded from every caller."""
        with tempfile.TemporaryDirectory() as d:
            mdir = os.path.join(d, "models", "m")
            os.makedirs(mdir)
            with open(os.path.join(mdir, "graph.yaml"), "w") as f:
                f.write("kind: ModelGraph\nspeculator:\n  method: dspark\n  num_spec_tokens: 5\n"
                        "stack:\n  pattern: [a]\n")
            run_blis._SPEC_CACHE.clear()
            self.assertEqual(run_blis.graph_speculator(d, "m"), "dspark")

    def test_model_with_no_speculator_reads_empty(self):
        with tempfile.TemporaryDirectory() as d:
            mdir = os.path.join(d, "models", "m")
            os.makedirs(mdir)
            with open(os.path.join(mdir, "graph.yaml"), "w") as f:
                f.write("kind: ModelGraph\nstack:\n  pattern: [a]\n")
            run_blis._SPEC_CACHE.clear()
            self.assertEqual(run_blis.graph_speculator(d, "m"), "")


class AcceptanceLength(unittest.TestCase):
    """The acceptance length is what makes the per-user axis correct under drafting, and BLIS
    never echoes it, so the runner resolves and records it."""

    PROFILES = ("kind: AcceptanceProfileSet\nname: test-al\nprofiles:\n"
                "- model: deepseek-v4.1-flash\n  speculative_method: dspark\n"
                "  num_spec_tokens: 5\n  thinking: 'on'\n  category: coding\n"
                "  acceptance_length: 3.51\n  provenance:\n    hardware: b200\n")

    def _reg(self, d):
        os.makedirs(os.path.join(d, "acceptance"), exist_ok=True)
        with open(os.path.join(d, "acceptance", "p.yaml"), "w") as f:
            f.write(self.PROFILES)
        run_blis._AL_CACHE.clear()

    def test_resolves_the_measured_value_with_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            self._reg(d)
            al, src = run_blis.acceptance_length(
                d, "deepseek-v4.1-flash", "dspark", 5, "on", "coding")
        self.assertEqual(al, 3.51)
        self.assertIn("b200", src)

    def test_unmatched_key_resolves_to_none(self):
        """An unknown acceptance factor must yield None, so the report withholds `pu` rather than
        publishing a rate wrong by an unknown multiple."""
        with tempfile.TemporaryDirectory() as d:
            self._reg(d)
            al, _ = run_blis.acceptance_length(
                d, "deepseek-v4.1-flash", "dspark", 4, "on", "coding")   # K=4 absent
        self.assertIsNone(al)

    def test_registry_without_acceptance_dir_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            run_blis._AL_CACHE.clear()
            self.assertEqual(run_blis.acceptance_length(d, "m", "dspark", 5, "on", "coding"),
                             (None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
