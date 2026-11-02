#!/usr/bin/env python3
"""Project one experiment run dir into the explorer's compact explore.json.

    python3 -I scripts/explore_data.py reports/<run_id>      # prints the JSON

The explorer is a client-side crossfilter over every run, so it ships rows as arrays under one
shared key list (a 3,880-run report is ~1 MB instead of ~6 MB of objects). Router configs are
parsed here into the plugin chain the page draws, so the page needs no YAML parser.
"""
import json
import os
import re
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# One summary per base routing file; a generated arm says which parameter it overrides.
ROUTER_SUMMARY = {
    "random": "Spreads requests at random; the baseline every scorer has to beat.",
    "queue": "Sends each request to the instance with the shortest queue.",
    "load-aware": "Sends each request to the least-loaded instance.",
    "queue-kv": "Weighs queue depth and free KV cache equally.",
    "prefix-affinity": "Keeps requests that share a prefix on the instance that served it first.",
    "pd-always": "Disaggregated: every request prefills on the prefill pool, then decodes on the decode pool.",
    "pd-prefix": "Disaggregated: a request goes to the prefill pool only when enough of its prompt is uncached.",
    "default": "llm-d-router's built-in config, used when a run names none: queue depth, KV cache and prefix scorers, then the highest score.",
}
# What llm-d-router loads when BLIS passes it no config (loadDefaultConfig in pkg/epp/config/loader
# at the pinned fork, v1.0.0-blis.2). Kept here so the page can draw it like any other router.
DEFAULT_ROUTER = """apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: prefix-cache-scorer
- type: metrics-data-source
- type: core-metrics-extractor
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
  - pluginRef: prefix-cache-scorer
    weight: 3
"""
# (short key the page uses, the experiment.json row field), in column order
METRICS = (("tpg", "tokens_per_sec_per_gpu"), ("cl", "tokens_per_sec"), ("pu", "tok_per_s_per_user"),
           ("itl", "itl_p99_ms"), ("ttft", "ttft_p99_ms"))
CATEGORIES = [  # plugin type suffix -> the stage it plays in the request path
    (r"-producer$|-source$|-extractor$", "data"), (r"-filter$", "filter"),
    (r"-scorer$", "scorer"), (r"-picker$", "picker"), (r"-profile-handler$", "handler"),
    (r"-decider$", "decider"), (r"-detector$|-admitter$", "gate"), (r"-policy$", "policy"),
    (r"-parser$", "parser"),
]


def _category(ptype):
    return next((c for pat, c in CATEGORIES if re.search(pat, ptype)), "other")


def _strip_comments(text):
    return "\n".join(l for l in text.splitlines() if not l.startswith("#")).strip() + "\n"


def _strings(v):
    if isinstance(v, str):
        return {v}
    items = v.values() if isinstance(v, dict) else v if isinstance(v, list) else ()
    return set().union(*map(_strings, items)) if items else set()


def parse_router(text):
    """The EndpointPickerConfig as the stages a request passes through."""
    cfg = yaml.safe_load(text) or {}
    plugins = [{"name": p.get("name") or p["type"], "type": p["type"],
                "category": _category(p["type"]), "params": p.get("parameters") or {}}
               for p in cfg.get("plugins") or []]
    by_name = {p["name"]: p for p in plugins}
    profiles = []
    for prof in cfg.get("schedulingProfiles") or []:
        steps = [{"ref": s["pluginRef"], "weight": s.get("weight")} for s in prof.get("plugins") or []]
        profiles.append({"name": prof.get("name", "default"), "steps": steps})
    flow = cfg.get("flowControl") if "flowControl" in (cfg.get("featureGates") or []) else None
    gate = []
    if flow and (flow.get("saturationDetector") or {}).get("pluginRef") in by_name:
        sat = by_name[flow["saturationDetector"]["pluginRef"]]
        gate = [sat["name"]] + [d for d in (sat["params"].get("detectors") or []) if d in by_name]
    # A data plugin feeds whichever plugins name it as a parameter value.
    feeds = {d["name"]: [p["name"] for p in plugins if p is not d and d["name"] in _strings(p["params"])]
             for d in plugins if d["category"] == "data"}
    return {"plugins": plugins, "profiles": profiles, "flow_control": bool(flow),
            "gate": gate, "feeds": feeds}


def _router_entry(path, run_dir):
    """(id, entry) for one router_config value; generated arms live in <run_dir>/routing/."""
    base = os.path.basename(path)
    rid = base[:-5] if base.endswith(".yaml") else base
    local = os.path.join(run_dir, "routing", base)
    src = local if os.path.exists(local) else os.path.join(ROOT, path)
    family = re.sub(r"-[0-9a-f]{10}$", "", rid)
    if not os.path.exists(src):  # a shared file renamed or removed since the sweep
        return rid, {"path": path, "family": family, "override": "",
                     "summary": ROUTER_SUMMARY.get(family, ""), "yaml": "", "chain": None}
    with open(src, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^# Overrides:\s*(.+)$", text, re.M)
    return rid, {
        "path": path, "family": family,
        "override": m.group(1).strip() if m else "",
        "summary": ROUTER_SUMMARY.get(family, ""),
        "yaml": _strip_comments(text),
        "chain": parse_router(text),
    }


def build(run_dir):
    """The explorer data for one run dir, or None when it has no sweeps to explore."""
    path = os.path.join(run_dir, "experiment.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        exp = json.load(f)
    if not exp.get("sweeps"):
        return None
    spec, spec_text = {}, ""
    if os.path.exists(os.path.join(run_dir, "spec.yaml")):
        with open(os.path.join(run_dir, "spec.yaml"), encoding="utf-8") as f:
            spec_text = f.read()
        spec = yaml.safe_load(spec_text) or {}
    spec_sweeps = {s["name"]: s for s in spec.get("sweeps") or []}

    rows_in = [(si, r) for si, s in enumerate(exp["sweeps"]) for r in s["rows"]]
    keys = sorted({k for _, r in rows_in for k in r["config"] if k != "concurrency"})
    routers, rows, seen, variants = {}, [], {}, {}
    for si, r in rows_in:
        variants.setdefault(r.get("variant") or "", len(variants))
        cfg = dict(r["config"])
        if cfg.get("router_config"):
            path = cfg["router_config"]
            if path not in seen:  # parse each router file once, not once per run
                seen[path] = _router_entry(path, run_dir)
            rid, entry = seen[path]
            routers[rid] = entry
            cfg["router_config"] = rid
        rows.append([si, r["concurrency"], *[cfg.get(k) for k in keys],
                     *[r.get(field) for _, field in METRICS], 1 if r.get("servable") else 0])

    if any(r["config"].get("router_config") is None for _, r in rows_in):
        routers["default"] = {"path": "built into llm-d-router (no config passed)", "family": "default",
                              "override": "", "summary": ROUTER_SUMMARY["default"], "yaml": DEFAULT_ROUTER,
                              "chain": parse_router(DEFAULT_ROUTER)}
    sc = exp["scenario"]
    sweeps = [{"name": s["name"], "narrative": s.get("narrative", ""), "configs": s.get("configs", 0),
               "dropped": s.get("dropped") or [],
               "spec": {k: v for k, v in spec_sweeps.get(s["name"], {}).items()
                        if k not in ("name", "narrative")}}
              for s in exp["sweeps"]]
    return {
        "run_id": exp["name"],
        "slo": sc.get("slo") or {},
        "concurrency": sc.get("concurrency") or [],
        "fixed_engine": sc.get("engine") or {},
        "totals": exp.get("totals") or {},
        "sweeps": sweeps,
        "keys": keys,
        "metrics": [short for short, _ in METRICS],
        "rows": rows,
        # the dashboard's caption for each row, as an index into one list of distinct strings
        "variants": list(variants),
        "variant": [variants[r.get("variant") or ""] for _, r in rows_in],
        "routers": routers,
        "spec_yaml": spec_text,
    }


def _match_key(sweep, conc, cfg, drop):
    cfg = {k: v for k, v in cfg.items() if k not in drop}
    # runs.jsonl keeps the sweeping host's absolute path for shared routing/*.yaml files
    if cfg.get("router_config"):
        cfg["router_config"] = os.path.basename(cfg["router_config"])
    return (sweep, conc, json.dumps(cfg, sort_keys=True))


def commands(run_dir):
    """Each explore row's recorded `blis run` line and the deployment it priced, in row order.

    The per-run scenario YAMLs are regenerated, not committed, so the deployment block (what the
    scenario file held) ships beside the command. None for a row runs.jsonl has no record of.
    """
    with open(os.path.join(run_dir, "experiment.json"), encoding="utf-8") as f:
        exp = json.load(f)
    by_key = {}
    path = os.path.join(run_dir, "runs.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cfg = {k: v for k, v in rec["config"].items() if k != "concurrency"}
                by_key[_match_key(rec.get("sweep"), rec["concurrency"], cfg, ())] = rec
    out = []
    for s in exp.get("sweeps") or []:
        for r in s["rows"]:
            cfg = {k: v for k, v in r["config"].items() if k != "concurrency"}
            rec = by_key.get(_match_key(s["name"], r["concurrency"], cfg, ()))
            if rec is None:
                # experiment.json also gives P/D rows the shared tp/num_instances runs.jsonl leaves out
                rec = by_key.get(_match_key(s["name"], r["concurrency"], cfg, ("tp", "num_instances")))
            out.append([" ".join(rec["cmd"]), rec.get("deployment")] if rec and rec.get("cmd") else None)
    return out


if __name__ == "__main__":
    json.dump(build(sys.argv[1]), sys.stdout, sort_keys=True, separators=(",", ":"))  # null if no sweeps
