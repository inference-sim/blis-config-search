#!/usr/bin/env python3
"""Tests that a generated scenario names no `engine.version` (M3.2, design LIF-6).

From M3.2 the build fixes the engine release: `blis fields` prints it and every Result records it, and a
scenario that names `engine: {kind, version}` is refused ("field version not found in type scenario.Engine").
Config search reads the release from the field list, never from a scenario.
Hermetic: a fake binary that refuses a version line as the M3.2 build does.
Run: python3 -I scripts/test_engine_version.py
"""
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blis_cli  # noqa: E402
import run_blis  # noqa: E402
import settings  # noqa: E402

RELEASE = "0.34.0+0fda86a35419"
FAKE = r'''#!/usr/bin/env python3
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
with open(os.path.join(HERE, "calls.log"), "a") as f:
    f.write(" ".join(args) + "\n")


def refuse_a_version(text):
    if "version:" in text:
        sys.stderr.write("Error: field version not found in type scenario.Engine\n")
        sys.exit(1)


if args[:3] == ["schema", "validate", "scenario"]:
    refuse_a_version(open(args[3]).read())
elif args[:1] == ["fields"]:
    text = open(os.path.join(args[args.index("--scenarios") + 1], args[args.index("--scenario") + 1])).read()
    refuse_a_version(text)
    if "model: kind-probe" in text:
        sys.stderr.write('Error: scenario p.yaml: engine vllm %(release)s: model "kind-probe" is not in the data\n')
        sys.exit(1)
    print(json.dumps({"kind": "FieldList", "engine": {"kind": "vllm", "version": "%(release)s"},
                      "fields": [{"name": "max_num_seqs", "source": "config", "status": "modelled"}]}))
'''


def fake_binary(test):
    d = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, d)
    path = os.path.join(d, "blis")
    with open(path, "w") as f:
        f.write(FAKE % {"release": RELEASE})
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    open(os.path.join(d, "calls.log"), "w").close()
    return path


PARALLEL = {"tp": 8, "pp": 1, "dp": 1}
PD = {"prefill": {"parallel": PARALLEL, "engine": {}, "instances": 1},
      "decode": {"parallel": PARALLEL, "engine": {}, "instances": 1}}


class NoVersionLine(unittest.TestCase):
    def setUp(self):
        self.addCleanup(settings.use_engine, "vllm")

    def test_both_writers_state_the_engine_kind_only(self):
        single = run_blis.scenario_yaml("qwen3-14b", "h100", PARALLEL, {}, "x")
        pd = run_blis.scenario_yaml("qwen3-14b", "h100", dict(PARALLEL, pd=PD), {}, "x")
        for text in (single, pd):
            self.assertIn("engine:\n  kind: vllm\nmodel: qwen3-14b\n", text)
            self.assertNotIn("version", text)

    def test_the_other_engine_kind_follows_the_same_rule(self):
        settings.use_engine("sglang")
        text = run_blis.scenario_yaml("glm-5", "h200", PARALLEL, {}, "x")
        self.assertIn("engine:\n  kind: sglang\nmodel: glm-5\n", text)
        self.assertNotIn("version", text)

    def test_the_scenario_name_still_hashes_the_release(self):
        # The file name a committed report recorded is the hash of the release it states: unchanged.
        self.assertEqual("qwen3-14b-h100-1fd5de1fbc",
                         run_blis.scenario_name("qwen3-14b", "h100", "0.32.0", PARALLEL, {}))
        self.assertNotEqual(run_blis.scenario_name("qwen3-14b", "h100", "0.32.0", PARALLEL, {}),
                            run_blis.scenario_name("qwen3-14b", "h100", "0.33.0", PARALLEL, {}))

    def test_the_probe_scenario_names_only_the_engine(self):
        text = blis_cli._engine_probe("vllm")
        self.assertIn("engine:\n  kind: vllm\nmodel: kind-probe", text)
        self.assertNotIn("version", text)


class ReleaseFromTheFieldList(unittest.TestCase):
    def test_a_written_scenario_is_accepted_by_the_validator(self):
        blis = fake_binary(self)
        with tempfile.TemporaryDirectory() as d:
            name = run_blis.write_scenario(blis, d, "qwen3-14b", "h100", RELEASE, dict(PARALLEL), {})
            with open(os.path.join(d, name)) as f:
                self.assertNotIn("version", f.read())

    def test_a_spec_probe_names_no_version_and_the_release_is_the_field_lists(self):
        blis = fake_binary(self)
        sc = {"model": "qwen3-14b", "hardware": "h100", "cluster": {"gpus_per_node": 8}}
        orig = blis_cli.chip_packaging
        blis_cli.chip_packaging = lambda *a, **k: {"gpus_per_node": 8}
        self.addCleanup(setattr, blis_cli, "chip_packaging", orig)
        doc = run_blis.ask_field_list(blis, None, sc)
        self.assertEqual(RELEASE, doc["engine"]["version"])
        self.assertEqual(RELEASE, blis_cli.engine_release(blis, "vllm"))
        with open(os.path.join(os.path.dirname(blis), "calls.log")) as f:
            asked = [c for c in f.read().splitlines() if c.startswith("fields")]
        self.assertEqual(1, len(asked))

    def test_without_a_spec_a_build_that_needs_a_model_refuses_the_probe_for_the_model(self):
        blis = fake_binary(self)
        with self.assertRaises(blis_cli.BlisCommandError) as e:
            blis_cli.fields(blis, "vllm")
        self.assertNotIn("version not found", str(e.exception))
        self.assertIn("kind-probe", str(e.exception))


if __name__ == "__main__":
    unittest.main()
