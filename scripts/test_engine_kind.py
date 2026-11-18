#!/usr/bin/env python3
"""Tests of the engine kind of a spec (scenario.engine_kind) and of the specs waiting for the pin.

Hermetic: the stand-in simulator, a synthetic SGLang field list, fake binaries for the registered kinds.
Run: python3 -I scripts/test_engine_kind.py
"""
import copy
import json
import os
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_cli  # noqa: E402
import test_experiment  # noqa: E402,F401  (points ROUTER_CONFIGS_DIR at its hermetic router fixture)
import check_pending_pin  # noqa: E402
import run_blis  # noqa: E402
import run_experiment as E  # noqa: E402
import settings  # noqa: E402
import validate_experiment as V  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLIS = os.path.join(HERE, "testdata", "blis-m1", "blis")
ROUTING = os.path.join(ROOT, "routing")
PENDING = os.path.join(ROOT, "experiments", "pending-pin")
SGLANG_FIELDS = {"kind": "FieldList", "engine": {"kind": "sglang", "version": "0.5.23"}, "fields": [
    {"name": n, "source": "config", "status": "modelled"}
    for n in ("chunked_prefill_size", "mem_fraction_static", "context_length", "max_running_requests",
              "kv_cache_dtype")] + [{"name": "max_num_seqs", "source": "config", "status": "not_modelled"}]}


def sglang_spec(**engine):
    return {"name": "t", "scenario": {"model": "glm-5", "hardware": "h200", "workload": "chatbot",
                                      "engine_kind": "sglang", "cluster": {"gpus_per_node": 8}, "gpus": 8,
                                      "concurrency": [4], "engine": {"mem_fraction_static": 0.85}},
            "sweeps": [{"name": "s", "num-instances": [1], "parallel": {"tp": [8]}, "engine": engine}]}


class EngineKind(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fields = os.path.join(self.tmp.name, "sglang-fields.json")
        with open(self.fields, "w") as f:
            json.dump(SGLANG_FIELDS, f)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(settings.use_engine, "vllm")

    def test_the_default_kind_is_vllm_and_an_unknown_one_is_refused(self):
        self.assertEqual("vllm", settings.spec_engine_kind({"scenario": {}}))
        with self.assertRaises(ValueError):
            settings.use_engine("tgi")
        errs = V.validate(dict(sglang_spec(), scenario={**sglang_spec()["scenario"], "engine_kind": "tgi"}),
                          BLIS, ROUTING, None, self.fields)
        self.assertTrue(any("engine_kind" in e and "tgi" in e for e in errs), errs)

    def test_a_sglang_spec_sweeps_sglangs_own_settings_and_none_of_vllms(self):
        good = sglang_spec(chunked_prefill_size=[512, 2048])
        self.assertEqual([], V.validate(copy.deepcopy(good), BLIS, ROUTING, None, self.fields))
        bad = sglang_spec(max_num_batched_tokens=[8192, 16384])
        errs = V.validate(copy.deepcopy(bad), BLIS, ROUTING, None, self.fields)
        self.assertTrue(any("max_num_batched_tokens" in e for e in errs), errs)

    def test_the_same_setting_is_unknown_to_the_kind_that_lacks_it(self):
        spec = sglang_spec(chunked_prefill_size=[512])
        spec["scenario"].pop("engine_kind")
        errs = V.validate(spec, BLIS, ROUTING)         # vllm: the stand-in's list has no such setting
        self.assertTrue(any("chunked_prefill_size" in e for e in errs), errs)

    def test_the_runner_states_no_vllm_default_and_keeps_the_engines_names(self):
        settings.use_engine("sglang")
        vocab = settings.vocabulary(BLIS, self.fields)
        parallel, engine, cli, router = run_blis.classify_config(
            {"tp": 8, "chunked_prefill_size": 2048, "kv_cache_dtype": "fp8"}, vocab)
        self.assertNotIn("max_model_len", engine)
        self.assertEqual({"chunked_prefill_size": 2048, "kv_cache_dtype": "fp8"}, engine)
        body = run_blis.scenario_yaml("glm-5", "h200", "0.5.23", parallel, engine, "x", node_size=8)
        self.assertIn("kind: sglang", body)
        self.assertIn('version: "0.5.23"', body)
        self.assertIn("kv_cache_dtype: \"fp8\"", body)          # vLLM would have said kv_cache_dtype too; cache_dtype below
        self.assertNotIn("block_size", body)
        settings.use_engine("vllm")
        self.assertEqual({"max_model_len": 32768}, run_blis.classify_config({"tp": 8}, settings.vocabulary(BLIS))[1])

    def test_vllms_block_renames_do_not_touch_another_engine(self):
        settings.use_engine("sglang")
        self.assertEqual({"cache_dtype": "x"}, run_blis.to_vllm_block({"cache_dtype": "x"}))
        settings.use_engine("vllm")
        self.assertEqual({"kv_cache_dtype": "x"}, run_blis.to_vllm_block({"cache_dtype": "x"}))

    def test_the_dry_run_carries_the_release_of_the_engines_own_field_list(self):
        """The release in the report is read from the field list the engine's kind supplied: a list at
        another release gives that release, so the report cannot state a release it did not read."""
        spec = sglang_spec(chunked_prefill_size=[512, 2048, 8192])
        rep = E.run_experiment(copy.deepcopy(spec), BLIS, None, None, "/nonexistent", dry_run=True,
                               verbose=False, fields=self.fields)
        self.assertEqual("0.5.23", rep["scenario"]["engine_version"])
        self.assertEqual("0.5.23", rep["pinned_stack"]["engine_version"])
        self.assertEqual("sglang", settings.ENGINE_KIND)
        other = os.path.join(self.tmp.name, "other-fields.json")
        with open(other, "w") as f:
            json.dump(dict(SGLANG_FIELDS, engine={"kind": "sglang", "version": "0.5.99"}), f)
        rep = E.run_experiment(copy.deepcopy(spec), BLIS, None, None, "/nonexistent", dry_run=True,
                               verbose=False, fields=other)
        self.assertEqual("0.5.99", rep["scenario"]["engine_version"])

    def test_the_dry_run_expands_one_arm_per_chunk_size(self):
        spec = sglang_spec(chunked_prefill_size=[512, 2048, 8192])
        rep = E.run_experiment(spec, BLIS, None, None, "/nonexistent", dry_run=True, verbose=False, fields=self.fields)
        self.assertEqual({"configs": 3, "runs": 3}, {k: rep["totals"][k] for k in ("configs", "runs")})


def fake_blis(directory, kinds):
    path = os.path.join(directory, "blis")
    with open(path, "w") as f:
        f.write("#!/bin/sh\necho 'Error: scenario \"kind-probe.yaml\" names engine kind \"kind-probe\"; "
                f"this build runs the kinds {kinds}' >&2\nexit 1\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


class PendingPin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_the_kinds_a_build_runs_are_read_from_its_refusal(self):
        self.assertEqual(["vllm"], blis_cli.registered_kinds(fake_blis(self.tmp.name, "vllm")))
        self.assertEqual(["sglang", "vllm"], blis_cli.registered_kinds(fake_blis(self.tmp.name, "vllm, sglang")))

    def test_only_the_refusal_line_itself_names_the_kinds(self):
        """A line that merely mentions the phrase, or a refusal with trailing words, is not the list."""
        for text in ("hint: this build runs the kinds fake, other",
                     'Error: scenario "x" names engine kind "k"; this build runs the kinds vllm and some words'):
            path = os.path.join(self.tmp.name, "blis")
            with open(path, "w") as f:
                f.write(f"#!/bin/sh\necho '{text}' >&2\nexit 1\n")
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
            with self.assertRaises(blis_cli.BlisCommandError):
                blis_cli.registered_kinds(path)

    def test_the_committed_pending_specs_are_pending_on_a_build_without_the_kind(self):
        self.assertEqual([], check_pending_pin.problems(fake_blis(self.tmp.name, "vllm"), PENDING))

    def test_a_pending_spec_fails_the_day_the_pinned_build_runs_its_kind(self):
        found = check_pending_pin.problems(fake_blis(self.tmp.name, "sglang, vllm"), PENDING)
        self.assertTrue(found and all("move it from pending-pin/ to experiments/" in p for p in found), found)

    def test_a_vllm_spec_is_never_pending(self):
        d = os.path.join(self.tmp.name, "p")
        os.mkdir(d)
        with open(os.path.join(d, "x.yaml"), "w") as f:
            f.write("name: x\nscenario: {model: m, hardware: h, workload: w, gpus: 1, concurrency: [1]}\nsweeps: []\n")
        found = check_pending_pin.problems(fake_blis(self.tmp.name, "vllm"), d)
        self.assertTrue(any("never pending" in p for p in found), found)

    def test_every_pending_spec_names_a_known_engine_kind_and_parses(self):
        for name in os.listdir(PENDING):
            if name.endswith(".yaml"):
                spec, err = V.load(os.path.join(PENDING, name))
                self.assertIsNone(err, name)
                self.assertIn(settings.spec_engine_kind(spec), settings.ENGINE_KINDS, name)


if __name__ == "__main__":
    unittest.main()
