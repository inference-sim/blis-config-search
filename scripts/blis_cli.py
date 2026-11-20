#!/usr/bin/env python3
"""The one place config search calls the simulator's own commands.

Design §5 (blis-config-search card): config search reads settings lists through `blis fields`,
catalog identities through `blis catalog resolve`, the metric registry through `blis metrics`, and the schemas of its inputs and outputs
through `blis schema`, and validates every scenario and Result with `blis schema validate`
(CFG-3, SUB-19, X7). Each function here runs one of those commands on the binary it is given
and returns the parsed output.

Library code: a failure raises BlisCommandError (or DocumentInvalid), never exits the process.
"""
import json
import os
import re
import subprocess
import tempfile


class BlisCommandError(RuntimeError):
    """A command could not run, failed, or printed something other than its documented shape."""


class DocumentInvalid(ValueError):
    """`blis schema validate` refused a document. The message is the validator's own."""


def _run(blis, args, timeout):
    if not blis:
        raise BlisCommandError("no simulator binary: pass --blis, or set BLIS")
    try:
        return subprocess.run([blis, *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise BlisCommandError(f"`{blis} {' '.join(args)}`: {e}") from e


def _stderr(pr):
    lines = [ln for ln in (pr.stderr or "").splitlines() if ln.strip()]
    return lines[-1] if lines else f"exit {pr.returncode}, no message"


def _json(blis, args, timeout=120):
    pr = _run(blis, [*args, "--json"], timeout)
    if pr.returncode != 0:
        raise BlisCommandError(f"`blis {' '.join(args)} --json` failed: {_stderr(pr)}")
    try:
        return json.loads(pr.stdout)
    except ValueError as e:
        raise BlisCommandError(f"`blis {' '.join(args)} --json` printed no JSON: {e}") from e


_CACHE = {}


def _cached(key, load):
    if key not in _CACHE:
        _CACHE[key] = load()
    return _CACHE[key]


def _key(blis, *rest):
    return (os.path.realpath(blis) if blis else None, *rest)


def _field_list_json(blis, engine, path, scenario, catalog=None):
    """The engine's field list as a parsed document: the file `path` names (--fields), or what
    `blis fields --scenario` prints for the scenario text `scenario()` returns, built over `catalog`."""
    if not path:
        # Without a spec's scenario, one that names only the engine; a build that needs a model refuses it.
        text = scenario() if scenario else _engine_probe(
            engine, build_release(blis, engine) if scenario_takes_engine_version(blis) else None)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "fields-probe.yaml"), "w", encoding="utf-8") as f:
                f.write(text)
            pr = _run(blis, ["fields", "--scenario", "fields-probe.yaml", "--scenarios", d]
                      + (["--catalog", catalog] if catalog else []), 120)
        if pr.returncode != 0:
            raise BlisCommandError(f"`blis fields --scenario` for engine {engine} failed: {_stderr(pr)}"
                                   + ("" if scenario else "; pass --fields FILE, or run from a spec"))
        try:
            return json.loads(pr.stdout)
        except ValueError as e:
            raise BlisCommandError(f"`blis fields --scenario` printed no JSON: {e}") from e
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        # `path` is the vLLM list (scripts/fetch_fields.py). Another engine's list sits beside it,
        # with `.json` replaced by `.<engine>.json`.
        if isinstance(doc, dict) and (doc.get("engine") or {}).get("kind") not in (None, engine):
            sibling = re.sub(r"\.json$", "", path) + f".{engine}.json"
            if os.path.exists(sibling):
                with open(sibling, encoding="utf-8") as f:
                    doc = json.load(f)
        return doc
    except (OSError, ValueError) as e:
        raise BlisCommandError(f"the field list of engine {engine!r} ({path}): {e}") from e


def fields(blis, engine, path=None, scenario=None, catalog=None):
    """The engine's field list (fields.json) at the release this build runs: the file `path` names,
    or `blis fields --scenario` on the text `scenario()` returns. The list is the same for every
    scenario of one engine release, so the binary is asked once and later calls without a path reuse it."""
    def load():
        doc = _field_list_json(blis, engine, path, scenario, catalog)
        if not isinstance(doc, dict) or doc.get("kind") != "FieldList":
            raise BlisCommandError(f"`blis fields` for engine {engine} printed no FieldList")
        got = (doc.get("engine") or {}).get("kind")
        if got != engine:
            raise BlisCommandError(f"`blis fields` for engine {engine} printed the field list of "
                                   f"engine {got!r}")
        if not (doc.get("engine") or {}).get("version"):
            raise BlisCommandError(f"`blis fields` for engine {engine} names no engine version")
        rows = doc.get("fields")
        if not isinstance(rows, list) or not rows:
            raise BlisCommandError(f"`blis fields` for engine {engine} lists no fields")
        for row in rows:
            if not (isinstance(row, dict) and row.get("name") and row.get("source") in
                    ("config", "env") and row.get("status")):
                raise BlisCommandError(f"`blis fields` for engine {engine}: malformed entry {row!r}")
        return doc
    return _cached(_key(blis, "fields", engine, path), load)


def scenario_takes_engine_version(blis):
    """Whether a scenario of this build may name `engine.version`, read from `blis schema scenario`.

    A build before M3.2 requires it; from M3.2 the build fixes the engine release (LIF-6) and a scenario
    that names one is refused ("field version not found in type scenario.Engine"). Asked once per binary."""
    def load():
        defs = schema(blis, "scenario").get("$defs") or {}
        engine = defs.get("scenario.Engine")
        if not isinstance(engine, dict):
            raise BlisCommandError("`blis schema scenario` defines no scenario.Engine, so the scenario's "
                                   "engine block cannot be written")
        return "version" in (engine.get("properties") or {})
    return _cached(_key(blis, "scenario_takes_engine_version"), load)


def build_release(blis, engine):
    """The upstream release of `engine` this build runs. A build whose scenario names a version says it in
    the refusal of a scenario naming another: `names engine vllm 0; this build runs vllm 0.33.0, ...`.
    A build that takes none prints it with the field list, or in the first words of any refusal: `engine
    vllm 0.34.0+0fda86a35419: ...`. It lets config search write the probe scenario `blis fields` needs
    before it has the field list."""
    def load():
        takes = scenario_takes_engine_version(blis)
        text = _engine_probe(engine, "0" if takes else None)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "release-probe.yaml"), "w", encoding="utf-8") as f:
                f.write(text)
            pr = _run(blis, ["fields", "--scenario", "release-probe.yaml", "--scenarios", d], 120)
        said = (pr.stderr or "") + "\n" + (pr.stdout or "")
        if takes:
            m = re.search(rf'names engine {re.escape(engine)} 0; this build runs {re.escape(engine)} (\S+?),', said)
            release = m.group(1) if m else None
        else:
            release = None
            if pr.returncode == 0:
                try:
                    release = (json.loads(pr.stdout).get("engine") or {}).get("version")
                except (ValueError, AttributeError):
                    pass
            if not release:
                m = re.search(rf'\bengine {re.escape(engine)} (\S+?): ', said)
                release = m.group(1) if m else None
        if not release:
            raise BlisCommandError(f"`blis fields` did not say which {engine} release the build runs: "
                                   + _stderr(pr))
        return release
    return _cached(_key(blis, "build_release", engine), load)


def engine_release(blis, engine, fields_path=None):
    """The upstream release of `engine` this build runs, for example "0.32.0" (LIF-6)."""
    return fields(blis, engine, fields_path)["engine"]["version"]


def schema(blis, document):
    """The JSON Schema of a BLIS document, as this build's substrate release publishes it."""
    def load():
        doc = _json(blis, ["schema", document])
        if not isinstance(doc, dict) or "$defs" not in doc:
            raise BlisCommandError(f"`blis schema {document}` printed no JSON Schema")
        return doc
    return _cached(_key(blis, "schema", document), load)


def metric_registry(blis):
    """The metric registry this build pins, as `blis metrics --json` prints it: {id: record}, each
    record with the registry's own keys (formula, statistic, unit, scopes, aggregates, ...)."""
    def load():
        doc = _json(blis, ["metrics"])
        recs = doc.get("metrics") if isinstance(doc, dict) else None
        if not isinstance(recs, list) or not all(isinstance(r, dict) and r.get("id") for r in recs):
            raise BlisCommandError("`blis metrics --json` printed no metric list")
        return {r["id"]: r for r in recs}
    return _cached(_key(blis, "metrics"), load)


def supports_pool_offload(blis):
    """True when this build's deployment schema lets a pool state its own `offload` and
    `pd_transfer` (blis-substrate#307). A build without it takes one offload block for the whole
    deployment, so a disaggregated deployment cannot give only some pools a CPU tier."""
    pool = (schema(blis, "deployment").get("$defs") or {}).get("deployment.Pool") or {}
    return "offload" in (pool.get("properties") or {})


def resolve(blis, model=None, chip=None, catalog=None):
    """(identity, reason) for a model or chip name: `blis catalog resolve`.

    identity is the catalog's name for it, or None when the catalog has no such entry.
    `catalog` names a catalog directory; None means the catalog the build embeds.
    """
    if (model is None) == (chip is None):
        raise ValueError("resolve: give exactly one of model and chip")
    args = ["catalog", "resolve", "--model" if model is not None else "--chip",
            model if model is not None else chip]
    if catalog:
        args += ["--catalog", catalog]

    def load():
        doc = _json(blis, args)
        if not isinstance(doc, dict) or "identity" not in doc:
            raise BlisCommandError(f"`blis {' '.join(args)}` printed no identity")
        return doc.get("identity"), doc.get("reason") or ""
    return _cached(_key(blis, *args), load)


def chip_packaging(blis, chip, catalog=None):
    """{gpus_per_node, gpus_per_rack} the catalog entry of `chip` fixes, each None where the
    scenario states it (design §2.2): `blis catalog resolve --chip`.

    A chip the catalog does not know is an error, not an empty packaging.
    """
    args = ["catalog", "resolve", "--chip", chip]
    if catalog:
        args += ["--catalog", catalog]

    def load():
        doc = _json(blis, args)
        if not isinstance(doc, dict) or doc.get("identity") is None:
            raise BlisCommandError(f"`blis {' '.join(args)}`: {doc.get('reason') if isinstance(doc, dict) else 'no answer'}")
        pk = doc.get("packaging")
        if not isinstance(pk, dict) or not {"gpus_per_node", "gpus_per_rack"} <= set(pk):
            raise BlisCommandError(f"`blis {' '.join(args)}` printed no packaging")
        return {k: pk[k] for k in ("gpus_per_node", "gpus_per_rack")}
    return _cached(_key(blis, "packaging", *args), load)


def validate_document(blis, document, path):
    """Validate the file at `path` as a BLIS `document` with the Go validators.

    Raises DocumentInvalid with the validator's message when the document is refused (exit 1),
    and BlisCommandError when the command itself cannot run (exit 2 or anything else).
    """
    pr = _run(blis, ["schema", "validate", document, path], timeout=120)
    if pr.returncode == 0:
        return
    if pr.returncode == 1:
        raise DocumentInvalid(f"{document} {os.path.basename(path)}: {_stderr(pr)}")
    raise BlisCommandError(f"`blis schema validate {document}` failed: {_stderr(pr)}")


_PROBE = """kind: Scenario
name: kind-probe
engine:
  kind: kind-probe
  version: "0"
model: kind-probe
cluster:
  hardware: kind-probe
  nodes: 1
---
kind: Deployment
name: kind-probe
pools:
  - role: colocated
    nodes: 1
"""


def _engine_probe(engine, version):
    """The probe scenario naming `engine`; `version` None writes no engine.version line (a build that takes none)."""
    text = _PROBE.replace("kind: kind-probe", f"kind: {engine}", 1)
    if version is None:
        return text.replace('  version: "0"\n', "", 1)
    return text.replace('version: "0"', f'version: "{version}"', 1)


def registered_kinds(blis):
    """The engine kinds this build registers, from the refusal it prints for a kind it lacks:
    `names engine kind "x"; this build runs the kinds a, b`. TEMPORARY until `blis fields` lists them."""
    import re
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "kind-probe.yaml"), "w", encoding="utf-8") as f:
            f.write(_PROBE)
        pr = _run(blis, ["fields", "--scenario", "kind-probe.yaml", "--scenarios", d], 120)
    # The whole refusal line, not a phrase found anywhere in the output: a line that merely mentions
    # "this build runs the kinds" (a hint, a log line) must not be read as the list.
    m = re.search(r'^.*names engine kind "[^"]*"; this build runs the kinds? ([\w-]+(?:, ?[\w-]+)*)\s*$',
                  (pr.stderr or "") + "\n" + (pr.stdout or ""), re.M)
    if not m:
        raise BlisCommandError("`blis fields` did not say which engine kinds the build runs: "
                               + _stderr(pr))
    return sorted(k.strip() for k in m.group(1).split(",") if k.strip())


def engine_refusal(blis, scenario_text, word=None, timeout=120):
    """What the engine says when it opens the scenario, as its last stderr line, or None if it opens.

    `blis fields --scenario` builds the engine the scenario names, so a setting the engine cannot model
    (an eviction policy it does not simulate) is refused here, as it would be by a run. With `word`, a
    refusal that does not mention it is not this check's and returns None.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "probe.yaml"), "w", encoding="utf-8") as f:
            f.write(scenario_text)
        pr = _run(blis, ["fields", "--scenario", "probe.yaml", "--scenarios", d], timeout)
    if pr.returncode == 0:
        return None
    line = _stderr(pr)
    return line if word is None or word in (pr.stderr or "") else None
