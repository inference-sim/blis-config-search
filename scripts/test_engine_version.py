#!/usr/bin/env python3
"""Tests of the `engine.version` line of a generated scenario (M3.2, design LIF-6).

A build before M3.2 requires `engine: {kind, version}`; from M3.2 the build fixes the release and refuses a
scenario that names one. Config search reads `blis schema scenario` and writes the line only where the schema
has it. Hermetic: a fake binary whose schema, validator and `fields` behave as each kind of build does.
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
TAKES = %(takes)s
args = sys.argv[1:]
with open(os.path.join(HERE, "calls.log"), "a") as f:
    f.write(" ".join(args) + "\n")
if args[:2] == ["schema", "scenario"]:
    props = {"kind": {"type": "string"}}
    if TAKES:
        props["version"] = {"type": "string"}
    print(json.dumps({"$defs": {"scenario.Engine": {"properties": props}}}))
elif args[:3] == ["schema", "validate", "scenario"]:
    text = open(args[3]).read()
    if not TAKES and "version:" in text:
        sys.stderr.write("field version not found in type scenario.Engine\n")
        sys.exit(1)
elif args[:1] == ["fields"]:
    text = open(os.path.join(args[args.index("--scenarios") + 1], args[args.index("--scenario") + 1])).read()
    if not TAKES and "version:" in text:
        sys.stderr.write("Error: field version not found in type scenario.Engine\n")
        sys.exit(1)
    if TAKES and 'version: "0"' in text:
        sys.stderr.write('Error: names engine vllm 0; this build runs vllm 0.32.0, not 0\n')
        sys.exit(1)
    if "model: kind-probe" in text:
        sys.stderr.write('Error: scenario p.yaml: engine vllm %(release)s: vllmengine: model "kind-probe" '
                         'is not in the data in use\n')
        sys.exit(1)
    print(json.dumps({"kind": "FieldList", "engine": {"kind": "vllm", "version": "%(release)s"},
                      "fields": [{"name": "max_num_seqs", "source": "config", "status": "modelled"}]}))
'''


def fake_binary(test, takes):
    d = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, d)
    path = os.path.join(d, "blis")
    with open(path, "w") as f:
        f.write(FAKE % {"takes": takes, "release": RELEASE})
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    open(os.path.join(d, "calls.log"), "w").close()
    return path


def calls(blis):
    with open(os.path.join(os.path.dirname(blis), "calls.log")) as f:
        return f.read().splitlines()


PARALLEL = {"tp": 8, "pp": 1, "dp": 1}
PD = {"prefill": {"parallel": PARALLEL, "engine": {}, "instances": 1},
      "decode": {"parallel": PARALLEL, "engine": {}, "instances": 1}}


class VersionLine(unittest.TestCase):
    def setUp(self):
        self.addCleanup(settings.use_engine, "vllm")

    def single(self, **kw):
        return run_blis.scenario_yaml("qwen3-14b", "h100", "0.32.0", PARALLEL, {}, "x", **kw)

    def pd(self, **kw):
        return run_blis.scenario_yaml("qwen3-14b", "h100", "0.32.0", dict(PARALLEL, pd=PD), {}, "x", **kw)

    def test_a_build_that_takes_a_version_gets_it_by_default(self):
        for text in (self.single(), self.pd()):
            self.assertIn('engine:\n  kind: vllm\n  version: "0.32.0"\nmodel:', text)

    def test_a_build_that_takes_none_is_written_none_by_both_writers(self):
        for text in (self.single(takes_engine_version=False), self.pd(takes_engine_version=False)):
            self.assertIn("engine:\n  kind: vllm\nmodel:", text)
            self.assertNotIn("version", text)

    def test_the_other_engine_kind_follows_the_same_rule(self):
        settings.use_engine("sglang")
        self.assertIn('kind: sglang\n  version: "0.5.23"\n', run_blis.scenario_yaml(
            "glm-5", "h200", "0.5.23", PARALLEL, {}, "x"))
        self.assertNotIn("version", run_blis.scenario_yaml(
            "glm-5", "h200", "0.5.23", PARALLEL, {}, "x", takes_engine_version=False))

    def test_the_scenario_name_is_the_one_a_committed_report_recorded(self):
        # Hash of [model, hardware, release, parallel, engine, ...] as written before this change.
        self.assertEqual("qwen3-14b-h100-1fd5de1fbc",
                         run_blis.scenario_name("qwen3-14b", "h100", "0.32.0", PARALLEL, {}))


class CapabilityProbe(unittest.TestCase):
    def test_reads_whether_the_scenario_schema_has_a_version(self):
        self.assertTrue(blis_cli.scenario_takes_engine_version(fake_binary(self, True)))
        self.assertFalse(blis_cli.scenario_takes_engine_version(fake_binary(self, False)))

    def test_is_asked_once_per_binary(self):
        blis = fake_binary(self, False)
        for _ in range(3):
            blis_cli.scenario_takes_engine_version(blis)
        self.assertEqual(1, [c for c in calls(blis)].count("schema scenario --json"))

    def test_a_schema_without_an_engine_definition_is_an_error_not_a_guess(self):
        blis = fake_binary(self, True)
        with open(blis) as f:
            body = f.read().replace("scenario.Engine", "scenario.Other")
        with open(blis, "w") as f:
            f.write(body)
        with self.assertRaises(blis_cli.BlisCommandError):
            blis_cli.scenario_takes_engine_version(blis)

    def test_the_real_pinned_stand_in_takes_a_version(self):
        stand_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "blis-m1", "blis")
        self.assertTrue(blis_cli.scenario_takes_engine_version(stand_in))


class ProbeAndRelease(unittest.TestCase):
    def test_the_probe_names_a_version_only_when_given_one(self):
        self.assertIn('version: "0.32.0"', blis_cli._engine_probe("vllm", "0.32.0"))
        no = blis_cli._engine_probe("vllm", None)
        self.assertNotIn("version", no)
        self.assertIn("engine:\n  kind: vllm\nmodel: kind-probe", no)

    def test_build_release_of_a_build_that_takes_a_version_reads_its_refusal(self):
        self.assertEqual("0.32.0", blis_cli.build_release(fake_binary(self, True), "vllm"))

    def test_build_release_of_a_build_that_takes_none_reads_what_the_build_prints(self):
        blis = fake_binary(self, False)
        self.assertEqual(RELEASE, blis_cli.build_release(blis, "vllm"))
        self.assertEqual(RELEASE, blis_cli.build_release(blis, "vllm"))
        self.assertEqual(1, len([c for c in calls(blis) if c.startswith("fields")]))

    def test_the_field_list_probe_of_a_build_that_takes_none_names_no_version(self):
        blis = fake_binary(self, False)
        with self.assertRaises(blis_cli.BlisCommandError) as e:
            blis_cli.fields(blis, "vllm")          # no spec: the model-less probe is refused for its model
        self.assertNotIn("version not found", str(e.exception))
        self.assertIn("kind-probe", str(e.exception))

    def test_a_spec_probe_asks_a_build_that_takes_none_without_a_version(self):
        blis = fake_binary(self, False)
        sc = {"model": "qwen3-14b", "hardware": "h100", "cluster": {"gpus_per_node": 8}}
        orig = blis_cli.chip_packaging
        blis_cli.chip_packaging = lambda *a, **k: {"gpus_per_node": 8}
        self.addCleanup(setattr, blis_cli, "chip_packaging", orig)
        doc = run_blis.ask_field_list(blis, None, sc)
        self.assertEqual(RELEASE, doc["engine"]["version"])


class WrittenScenario(unittest.TestCase):
    def write(self, takes):
        blis = fake_binary(self, takes)
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        name = run_blis.write_scenario(blis, d, "qwen3-14b", "h100", RELEASE, dict(PARALLEL), {})
        with open(os.path.join(d, name)) as f:
            return f.read()

    def test_a_build_that_takes_none_validates_what_is_written(self):
        self.assertNotIn("version", self.write(False))      # the fake validator refuses a version line

    def test_a_build_that_takes_one_validates_what_is_written(self):
        self.assertIn(f'version: "{RELEASE}"', self.write(True))


if __name__ == "__main__":
    unittest.main()
