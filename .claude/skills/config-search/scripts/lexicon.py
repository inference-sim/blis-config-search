#!/usr/bin/env python3
# lexicon.py -- the shared vocabulary both report surfaces read.
#
# config-search renders the same runs twice: the standalone HTML report (report.py + figures.py)
# and the React dashboard (web/src). They had grown two vocabularies for one set of numbers and,
# worse, two ranking objectives for "What to run" -- the report ranked operating points by
# per-user rate while the dashboard ranked them by tok/s per GPU, so at high load the two tables
# recommended DIFFERENT configurations for the same scenario (issue #13).
#
# Both surfaces now read their labels, their captions and their ranking objective from
# report/lexicon.json. This module is the Python reader; web/src/lexicon.json is a byte-identical
# mirror the Vite build imports, and scripts/test_build.py fails if the two drift.
#
# Pure data: no I/O beyond reading the one JSON file, no network, no BLIS.
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.normpath(os.path.join(HERE, "..", "report", "lexicon.json"))


def _resolve(node, root):
    """Expand "@a.b" references in-place-ish, so a figure's axis label IS the shared axis string
    rather than a copy that can drift from it."""
    if isinstance(node, str) and node.startswith("@"):
        cur = root
        for part in node[1:].split("."):
            cur = cur[part]
        return cur
    if isinstance(node, dict):
        return {k: _resolve(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, root) for v in node]
    return node


def load(path=PATH):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return _resolve(raw, raw)


LEX = load()

OBJECTIVE = LEX["objective"]
METRICS = LEX["metrics"]
FACETS = LEX["facets"]
AXES = LEX["axes"]
FIGURES = LEX["figures"]
OPERATING_COLUMNS = LEX["operating_columns"]

# The deploy flags, as (record key, flag) pairs in the fixed order a reader copies them in.
FLAG_LABELS = [(f["key"], f["flag"]) for f in LEX["flags"]]
# Which flag keys print bare, with no value (`--enable-expert-parallel`, not "... 8").
BOOL_FLAGS = {f["key"] for f in LEX["flags"] if f.get("bool")}
# Values worth no ink: a knob at its default or an "off" state. A `None` is always skipped, so the
# JSON lists only the real defaults; callers add None themselves.
FLAG_SKIP = {k: tuple(v) + (None,) for k, v in LEX["flag_skip"].items()}


def metric(key, field="label"):
    """One metric's label / unit / digits / help, by RECORDS key."""
    return METRICS[key][field]


def label(key):
    """The display label for a metric OR a facet key -- the two namespaces do not overlap."""
    if key in METRICS:
        return METRICS[key]["label"]
    return FACETS[key]["label"]


def figure(name, field):
    """One figure's title / caption / axis labels / how-to-read text."""
    return FIGURES[name][field]
