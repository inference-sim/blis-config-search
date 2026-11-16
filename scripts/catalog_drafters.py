#!/usr/bin/env python3
"""A model's drafters and their measured acceptance, read from a blis-substrate checkout.

TEMPORARY (inference-sim/inference-sim#2141): the pinned simulator's `blis catalog resolve` does not
list a model's drafters, and it has no `blis acceptance` command to run D8's lookup without a
run. Until it has both, this module reads the checkout's catalog/models/*/model.yaml and
graph.yaml and registry/acceptance/*.yaml, the same data the release embeds (pins.env
SUBSTRATE_REF). It finds only profiles with the exact key; the borrowing steps are the
simulator's, and a drafting run's Result records which one it took.
"""
import os

import yaml

# The word a target's `drafters:` list uses for its graph's own speculator (blis-substrate
# contracts/spec/model/identity.go InGraphSpeculator).
IN_GRAPH = "speculator"


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _identity(catalog, name):
    return _load(os.path.join(catalog, "models", name, "model.yaml"))


def _speculator(catalog, name):
    spec = _load(os.path.join(catalog, "models", name, "graph.yaml")).get("speculator") or {}
    return spec.get("method"), spec.get("num_spec_tokens")


def drafters(catalog, model):
    """Every drafter of `model`, each a dict:

    name             "speculator" for the in-graph one, else the drafter's catalog name
    draft_model      the value of speculative.draft_model that selects it ("" for in-graph)
    kind             "in_graph" or "separate"
    method, max_spec_tokens   from the drafter's graph speculator
    linked           True when the link holds both ways: the model lists it, and a separate
                     drafter names the model in `targets:`
    why              the reason when linked is False
    """
    listed = list(_identity(catalog, model).get("drafters") or [])
    out = []
    method, k = _speculator(catalog, model)
    if IN_GRAPH in listed or method:
        ok = IN_GRAPH in listed and bool(method)
        why = ("" if ok else
               f"{model}'s graph declares no speculator, but its model.yaml lists one" if not method
               else f"{model}'s graph declares speculator {method!r}, but its model.yaml does not "
                    f"list it under drafters")
        out.append({"name": IN_GRAPH, "draft_model": "", "kind": "in_graph", "method": method,
                    "max_spec_tokens": k, "linked": ok, "why": why})
    claims = set()
    models_dir = os.path.join(catalog, "models")
    try:
        names = sorted(os.listdir(models_dir))
    except OSError:
        names = []
    for name in names:
        if model in (_identity(catalog, name).get("targets") or []):
            claims.add(name)
    for name in sorted((set(listed) - {IN_GRAPH}) | claims):
        method, k = _speculator(catalog, name)
        if name not in listed:
            why = f"{name} names {model} in targets, but {model}'s model.yaml does not list it"
        elif name not in claims:
            why = (f"{model} lists {name} under drafters, but {name}'s model.yaml does not name "
                   f"{model} in targets")
        elif not method:
            why = f"{name}'s graph declares no speculator"
        else:
            why = ""
        out.append({"name": name, "draft_model": name, "kind": "separate", "method": method,
                    "max_spec_tokens": k, "linked": not why, "why": why})
    return out


def _thinking(v):
    if isinstance(v, bool):
        return "on" if v else "off"
    return None if v is None else str(v)


def exact_profiles(registry, model, drafter, method, category=None, thinking=None):
    """The acceptance profiles whose key is exactly (model, drafter, method), at the workload's
    category and thinking mode when they are given, as dicts sorted by K:
    {num_spec_tokens, acceptance_length, set, source, hardware}.

    `drafter` is "" for the model's in-graph speculator, as a profile without `drafter:` means.
    """
    adir = os.path.join(registry or "", "acceptance")
    try:
        files = sorted(n for n in os.listdir(adir) if n.endswith((".yaml", ".yml")))
    except OSError:
        return []
    out = []
    for fn in files:
        doc = _load(os.path.join(adir, fn))
        for p in doc.get("profiles") or []:
            if (p.get("model") != model or (p.get("drafter") or "") != (drafter or "")
                    or p.get("speculative_method") != method):
                continue
            if category is not None and p.get("category") != category:
                continue
            if thinking is not None and _thinking(p.get("thinking")) != _thinking(thinking):
                continue
            prov = p.get("provenance") or {}
            out.append({"num_spec_tokens": p.get("num_spec_tokens"),
                        "acceptance_length": p.get("acceptance_length"),
                        "category": p.get("category"), "thinking": _thinking(p.get("thinking")),
                        "set": doc.get("name") or fn, "source": prov.get("source"),
                        "hardware": prov.get("hardware")})
    return sorted(out, key=lambda r: (r["num_spec_tokens"] or 0, r["set"]))
