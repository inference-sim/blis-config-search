#!/usr/bin/env python3
"""The one place config search calls the simulator's own commands.

Design §5 (blis-config-search card): config search reads settings lists through `blis fields`,
catalog identities through `blis catalog resolve`, and the schemas of its inputs and outputs
through `blis schema`, and validates every scenario and Result with `blis schema validate`
(CFG-3, SUB-19, X7). Each function here runs one of those commands on the binary it is given
and returns the parsed output.

Library code: a failure raises BlisCommandError (or DocumentInvalid), never exits the process.
"""
import json
import os
import subprocess


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


def _pinned_release():
    """VLLM_REF from pins.env: the blis-vllm release the pinned simulator is built with."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pins.env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("VLLM_REF="):
                return line.split("=", 1)[1].strip()
    raise BlisCommandError("pins.env has no VLLM_REF")


def _field_list_json(blis, engine):
    """The engine's field list as a parsed document.

    TEMPORARY until the M2 substrate release lets the engine report its own list through
    `blis fields`: the pinned simulator has no such command, so the list is the pinned
    blis-vllm release's fields.json, which BLIS_FIELDS names (a path) or, when it is unset,
    which is read from that release on GitHub. A build that does have the command is asked.
    """
    if engine != "vllm":
        return _json(blis, ["fields", "--engine", engine])
    path = os.environ.get("BLIS_FIELDS")
    try:
        if path:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        import urllib.request
        url = f"https://raw.githubusercontent.com/inference-sim/blis-vllm/{_pinned_release()}/fields.json"
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (a fixed https URL)
            return json.load(r)
    except (OSError, ValueError) as e:
        raise BlisCommandError(f"the field list of engine {engine!r}: {e}") from e


def fields(blis, engine):
    """The engine's field list (fields.json) at the release this build runs: `blis fields`."""
    def load():
        doc = _field_list_json(blis, engine)
        if not isinstance(doc, dict) or doc.get("kind") != "FieldList":
            raise BlisCommandError(f"`blis fields --engine {engine}` printed no FieldList")
        got = (doc.get("engine") or {}).get("kind")
        if got != engine:
            raise BlisCommandError(f"`blis fields --engine {engine}` printed the field list of "
                                   f"engine {got!r}")
        if not (doc.get("engine") or {}).get("version"):
            raise BlisCommandError(f"`blis fields --engine {engine}` names no engine version")
        rows = doc.get("fields")
        if not isinstance(rows, list) or not rows:
            raise BlisCommandError(f"`blis fields --engine {engine}` lists no fields")
        for row in rows:
            if not (isinstance(row, dict) and row.get("name") and row.get("source") in
                    ("config", "env") and row.get("status")):
                raise BlisCommandError(f"`blis fields --engine {engine}`: malformed entry {row!r}")
        return doc
    return _cached(_key(blis, "fields", engine, os.environ.get("BLIS_FIELDS")), load)


def engine_release(blis, engine):
    """The upstream release of `engine` this build runs, for example "0.32.0" (LIF-6)."""
    return fields(blis, engine)["engine"]["version"]


def schema(blis, document):
    """The JSON Schema of a BLIS document, as this build's substrate release publishes it."""
    def load():
        doc = _json(blis, ["schema", document])
        if not isinstance(doc, dict) or "$defs" not in doc:
            raise BlisCommandError(f"`blis schema {document}` printed no JSON Schema")
        return doc
    return _cached(_key(blis, "schema", document), load)


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
