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
import shlex
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_catalog  # noqa: E402
import run_blis  # noqa: E402

# What the engine does with a knob a run leaves unset and the ledger does not record
# (blis-vllm rules/engineconfig/engine.go and rules/v0_32/pack.go at the pinned stack).
ENGINE_UNSET = {
    "all2all_backend": "allgather_reducescatter",
    "enable_prefix_caching": True,
    "long_prefill_token_threshold": 0,
    "cache_dtype": "auto",
    "quantization": "checkpoint",
    "max_cudagraph_capture_size": "auto",
    "eplb": "off",
    "engram_config": {"cpu_offload": True},
    "num_speculative_tokens": 0,
    "speculative_method": "off",
}
TOPOLOGY_KEY = re.compile(r"^((prefill|decode)_)?(tp|dp|pp|enable_expert_parallel|instances|num_instances)$|^router_config$")
# Knobs the scenario stores under another name.
NESTED = {"speculative_method": ("speculative", "method"),
          "num_speculative_tokens": ("speculative", "num_spec_tokens")}
# One summary per base routing file; a generated arm says which parameter it overrides.
ROUTER_SUMMARY = {
    "random": "Spreads requests at random; the baseline every scorer has to beat.",
    "queue": "Sends each request to the instance with the shortest queue.",
    "load-aware": "Sends each request to the least-loaded instance.",
    "queue-kv": "Weighs queue depth and free KV cache equally.",
    "prefix-affinity": "Keeps requests that share a prefix on the instance that served it first.",
    "pd-always": "Disaggregated: every request prefills on the prefill pool, then decodes on the decode pool.",
    "pd-prefix": "Disaggregated: a request goes to the prefill pool only when enough of its prompt is uncached.",
    "pd-prefix-based": "Disaggregated, the router's own sample: a request goes to the prefill pool only when enough of its prompt is uncached.",
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


def _resolved(dep, key):
    """The value a recorded deployment ran `key` at, or None when it does not state one."""
    m = re.match(r"(prefill|decode)_(.+)$", key)
    par, eng = dep.get("parallel") or {}, dep.get("engine") or {}
    if m:
        pool = ((par.get("pd") or {}).get(m.group(1))) or {}
        key, par, eng = m.group(2), pool.get("parallel") or {}, pool.get("engine") or {}
    if key in NESTED:
        block, field = NESTED[key]
        return (eng.get(block) or {}).get(field)
    return eng[key] if key in eng else par.get(key)


def defaults(run_dir, keys):
    """For each knob, the value a run that left it unset actually used.

    Read from the deployments BLIS priced when they agree, else the engine's own default, so the
    page can say "8192 (default)" instead of a bare "default".
    """
    seen = {k: set() for k in keys}
    path = os.path.join(run_dir, "runs.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                dep, cfg = rec.get("deployment"), rec.get("config") or {}
                if not dep:
                    continue
                for k in keys:
                    if cfg.get(k) is None and (v := _resolved(dep, k)) is not None:
                        seen[k].add(json.dumps(v, sort_keys=True))
    out = {}
    for k in keys:
        bare = re.sub(r"^(prefill|decode)_", "", k)
        if len(seen[k]) == 1:
            out[k] = json.loads(next(iter(seen[k])))
        elif not seen[k] and bare in ENGINE_UNSET:
            out[k] = ENGINE_UNSET[bare]
    return out


GH = "https://github.com/inference-sim"
_PINS = build_catalog.read_pins()
# The catalog/ of a blis-substrate checkout at the pinned release (workload descriptors are read from it).
CATALOG_DIR = os.environ.get("SUBSTRATE_CATALOG") or os.path.join(ROOT, os.pardir, "blis-substrate", "catalog")


def _clone(repo, ref, commit, dest, *then):
    """Shallow-clone `ref`, then run `then` only if the tag still points at the recorded commit.

    One && chain, not `exit`: the script is pasted into an interactive shell, which must stay open.
    """
    steps = [f"git clone --branch {ref} --depth 1 {GH}/{repo}.git {dest}"]
    if commit:
        steps.append(f'{{ test "$(git -C {dest} rev-parse HEAD)" = {commit} || '
                     f'{{ echo "{dest}: {ref} moved off {commit[:8]}, stopping" >&2; false; }}; }}')
    return [" && \\\n  ".join([*steps, *then])]


def _corpus(workload):
    """(file, URL, sha256) of a replay workload's corpus, from the catalog checkout; None if unknown."""
    path = os.path.join(CATALOG_DIR, "workloads", f"{workload}.yaml")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        corpus = ((yaml.safe_load(f) or {}).get("replay") or {}).get("corpus") or {}
    src = corpus.get("source") or {}
    if not (src.get("huggingface") and src.get("revision") and src.get("file")):
        return None
    url = f"https://huggingface.co/datasets/{src['huggingface']}/resolve/{src['revision']}/{src['file']}"
    return src["file"], url, corpus.get("sha256")


def _wall_hours(run_dir):
    """Serial BLIS time the sweep took, summed from its ledger; None when not recorded."""
    path = os.path.join(run_dir, "runs.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        total = sum(json.loads(l).get("wall_s") or 0 for l in f)
    return round(total / 3600, 1) if total else None


def setup(exp, run_dir=None):
    """The stack that priced this report, a script installing it, and how to reproduce the sweep locally."""
    ps = exp.get("pinned_stack") or {}
    res = ps.get("resolved") or {}
    commit = lambda k: (res.get(k) or {}).get("commit")
    if not ps.get("blis"):
        return None
    sh = ["# the stack that priced this report (needs git and Go)"]
    sh += _clone("inference-sim", ps["blis"], commit("blis"), "inference-sim",
                 "(cd inference-sim && go build -o blis main.go)", 'export PATH="$PWD/inference-sim:$PATH"')
    sh.append("")
    if ps.get("substrate"):
        # the simulator prices with the data embedded in this release (CFG-1); the checkout is only
        # where the scripts read catalog and registry files from
        sh += _clone("blis-substrate", ps["substrate"], None, "blis-substrate",
                     'export SUBSTRATE_CATALOG="$PWD/blis-substrate/catalog"',
                     'export SUBSTRATE_REGISTRY="$PWD/blis-substrate/registry"') + [""]
    for key, repo, env in (("catalog", "blis-catalog", "SUBSTRATE_CATALOG"), ("registry", "blis-registry", "SUBSTRATE_REGISTRY")):
        if ps.get(key) and ps.get("data") != "embedded":     # a report from before the data was embedded
            sh += _clone(repo, ps[key], commit(key), repo, f'export {env}="$PWD/{repo}"') + [""]
    sh += ["# the commands run from this repo: router configs live here, scenarios/ is where you save each scenario",
           f"git clone --depth 1 {GH}/blis-config-search.git && cd blis-config-search && mkdir -p scenarios"]
    replay = (exp.get("scenario") or {}).get("replay") or {}
    if replay.get("corpus"):
        got = _corpus((exp.get("scenario") or {}).get("workload") or "")
        dest, sha = replay["corpus"], replay.get("corpus_sha256")
        sh += ["", "# the replay corpus (not vendored anywhere; checked against its digest)",
               f"mkdir -p {shlex.quote(os.path.dirname(dest) or '.')}"]
        sh += [f"curl -L -o {shlex.quote(dest)} {got[1]}"] if got else [f"# fetch the corpus to {dest} (see SETUP.md)"]
        if sha:
            sh.append(f'echo "{sha}  {dest}" | shasum -a 256 -c')
    rid = exp.get("name") or ""
    spec = f"reports/{rid}/spec.yaml"
    # writes to repro/, never reports/: reproducing locally does not touch the published report
    sweep = ["# in the blis-config-search checkout, after the install script",
             "python3 -m venv .venv && . .venv/bin/activate && pip install pyyaml",
             f'python3 -I scripts/run_experiment.py {spec} --dry-run --catalog "$SUBSTRATE_CATALOG"',
             f"python3 -I scripts/run_experiment.py {spec} \\",
             '  --blis "$(command -v blis)" --catalog "$SUBSTRATE_CATALOG" --registry "$SUBSTRATE_REGISTRY" \\',
             "  --out repro/"]
    stack = [{"name": "BLIS", "repo": "inference-sim", "ref": ps["blis"], "commit": commit("blis")},
             *[{"name": n, "repo": r, "ref": ps[k], "commit": commit(k)}
               for k, n, r in (("substrate", "substrate (embedded data)", "blis-substrate"),
                               ("catalog", "catalog", "blis-catalog"), ("registry", "registry", "blis-registry"))
               if ps.get(k) and not (ps.get("data") == "embedded" and k in ("catalog", "registry"))]]
    # compiled into BLIS; a ledger from before the router was recorded takes the pin of the same BLIS release
    router = ps.get("router") or (_PINS.get("ROUTER_REF") if _PINS.get("BLIS_REF") == ps["blis"] else None)
    for name, repo, ref in (("latency kernel", "blis-latency-kernel", ps.get("kernel")),
                            ("router", "llm-d-router", router)):
        if ref and ref != "unpinned":
            stack.append({"name": name, "repo": repo, "ref": ref, "commit": None})
    # a fresh checkout sweeps at today's pins.env, whose simulator fixes the engine release
    then = (ps["blis"], ps.get("substrate"))
    now = (_PINS.get("BLIS_REF"), _PINS.get("SUBSTRATE_REF"))
    return {"stack": stack, "script": "\n".join(sh) + "\n", "sweep": "\n".join(sweep) + "\n",
            "pins_moved": then != now,
            "runs": (exp.get("totals") or {}).get("runs"), "hours": _wall_hours(run_dir) if run_dir else None}


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
        # topology keys already have fixed defaults on the page (dp 1, pp 1, EP off)
        "defaults": defaults(run_dir, [k for k in keys if not TOPOLOGY_KEY.match(k)]),
        "metrics": [short for short, _ in METRICS],
        "rows": rows,
        # the dashboard's caption for each row, as an index into one list of distinct strings
        "variants": list(variants),
        "variant": [variants[r.get("variant") or ""] for _, r in rows_in],
        "routers": routers,
        "spec_yaml": spec_text,
        "setup": setup(exp, run_dir),
    }


def _match_key(sweep, conc, cfg, drop):
    cfg = {k: v for k, v in cfg.items() if k not in drop}
    # runs.jsonl keeps the sweeping host's absolute path for shared routing/*.yaml files
    if cfg.get("router_config"):
        cfg["router_config"] = os.path.basename(cfg["router_config"])
    return (sweep, conc, json.dumps(cfg, sort_keys=True))


# Flags dropped from the shown command: inert for a rerun (LoRA defaults are compiled in, the
# corpus cache changes no output) or a label BLIS ignores under --concurrency (WORKLOAD_IGNORED).
INERT_FLAGS = {"--defaults-filepath", "--corpus-cache", "--workload"}


def _scenario(run_dir, exp, dep, unset=None):
    """(file name, YAML) of the scenario a run read, regenerated from its recorded deployment.

    Only returned when it hashes to the recorded file name, so it is what BLIS got: the router path
    is rewritten relative to a `scenarios/` dir at the repo root, and `unset` (see _note_unset)
    adds comments only.
    """
    sc = exp.get("scenario") or {}
    model, hw, name = sc.get("model"), sc.get("hardware"), dep.get("scenario")
    if not (model and hw and name):
        return None
    router = dep.get("router") or {}
    # The fabric the run was priced over (issue #71): part of the scenario hash, so regenerating
    # with the wrong one would never match the recorded name. A record predating the field used IB.
    fabric = dep.get("fabric") or run_blis.DEFAULT_FABRIC
    # The node size the run was priced with (#96); a record that predates it used 8, stated.
    node_size = dep.get("gpus_per_node") or 8
    states_node_size = dep.get("states_gpus_per_node", True)
    # the engine release the ledger recorded; a ledger that records none cannot be reconstructed
    recorded = (exp.get("pinned_stack") or {}).get("engine_version")
    for ev in dict.fromkeys([v for v in (recorded, sc.get("engine_version")) if v]):
        if run_blis.scenario_name(model, hw, ev, dep["parallel"], dep["engine"], dep["num_instances"],
                                  fabric, router, node_size, states_node_size) + ".yaml" != name:
            continue
        shown = {}
        if router.get("router_config"):
            rc = router["router_config"]
            local = os.path.join("reports", exp["name"], "routing", os.path.basename(rc))
            # a generated override arm lives in the report; a stock arm (no overrides) in routing/
            rel = local if os.path.exists(os.path.join(ROOT, local)) else os.path.join("routing", os.path.basename(rc))
            shown = {"router_config": "__ROUTER__"}
        text = run_blis.scenario_yaml(model, hw, ev, json.loads(json.dumps(dep["parallel"])),
                                      dict(dep["engine"]), name[:-5], dep["num_instances"],
                                      fabric, shown or None, node_size, states_node_size)
        if shown:
            text = text.replace(json.dumps(os.path.abspath("__ROUTER__")), json.dumps("../" + rel))
        return name, _note_unset(text, unset or {})
    return None


def _unset_value(key, role, dflt):
    """The default a pool ran `key` at when unset: the role's own column, then the shared one."""
    for k in (f"{role}_{key}", key):
        if k in dflt:
            return dflt[k]
    return ENGINE_UNSET.get(key)


def _note_unset(text, dflt):
    """Comment each spec knob a pool's engine block leaves unset with the default it ran at.

    Comments keep the file equivalent to the one BLIS read, so it reruns to the same result.
    """
    bare = sorted({re.sub(r"^(prefill|decode)_", "", k) for k in dflt})
    out, role, stated = [], "", None

    def flush():
        if stated is None:
            return
        for key in bare:
            field = NESTED[key][0] if key in NESTED else key
            v = _unset_value(key, role, dflt)
            if field not in stated and v is not None:
                out.append(f"      # {key}: {json.dumps(v)}  (unset; the default it ran at)")

    for line in text.splitlines():
        if stated is not None and not line.startswith("      "):
            flush()
            stated = None
        if m := re.match(r"  - role: (\S+)", line):
            role = m.group(1)
        out.append(line)
        if line == "    engine:":
            stated = set()
        elif stated is not None and (m := re.match(r"      (\w+):", line)):
            stated.add(m.group(1))
    flush()
    return "\n".join(out) + "\n"


def _max_model_len(dep):
    """The one max_model_len every pool of a deployment ran at, else None."""
    pd = ((dep or {}).get("parallel") or {}).get("pd") or {}
    engines = [(pd[r].get("engine") or {}) for r in run_blis.PD_ROLES if r in pd] or [(dep or {}).get("engine") or {}]
    lens = {e.get("max_model_len") for e in engines}
    return lens.pop() if len(lens) == 1 else None


def _recipe(cmd, workload, dep=None):
    """The recorded command made portable: env-var roots, a repo-relative scenarios dir, one flag per line.

    `--max-context-length max_model_len` shows the number the deployment resolved it to.
    """
    args, i, lines = [str(a) for a in cmd], 2, []
    while i < len(args):
        flag = args[i]
        # assumes no BLIS flag takes a "--"-prefixed value; such a value would be read as the next flag
        val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else None
        i += 2 if val is not None else 1
        if flag in INERT_FLAGS:
            continue
        if flag == "--registry":
            val = '"$SUBSTRATE_REGISTRY"'
        elif flag == "--catalog":
            val = '"$SUBSTRATE_CATALOG"'
        elif flag == "--scenarios":
            val = "scenarios"
        elif flag == "--max-context-length" and val == "max_model_len" and _max_model_len(dep):
            val = str(_max_model_len(dep))
        elif val is not None:
            val = shlex.quote(val)
        lines.append(flag if val is None else f"{flag} {val}")
    head = f"blis {args[1]}"
    note = (f"# workload: the {workload} preset, passed as its token shape below"
            if workload and args[1] == "run" else "")
    return "\n".join(filter(None, [note, " \\\n  ".join([head, *lines])]))


def commands(run_dir):
    """Each explore row's reproducible command, plus the scenario files those commands read.

    {"scenarios": {file name: YAML}, "runs": [[command, scenario file name, deployment] | None]}
    in explore row order. The scenario YAMLs are not committed, so they are regenerated here and
    shared across the rows (one per deployment, not per load). None for a row with no record.
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
    out, scenarios = [], {}
    workload = (exp.get("scenario") or {}).get("workload")
    # every engine knob the spec sets or sweeps, with the value a run that leaves it unset used
    knobs = {k for s in exp.get("sweeps") or [] for r in s["rows"] for k in r["config"]
             if k != "concurrency" and not TOPOLOGY_KEY.match(k)}
    knobs |= set((exp.get("scenario") or {}).get("engine") or {})
    unset = defaults(run_dir, sorted(knobs))
    for s in exp.get("sweeps") or []:
        for r in s["rows"]:
            cfg = {k: v for k, v in r["config"].items() if k != "concurrency"}
            rec = by_key.get(_match_key(s["name"], r["concurrency"], cfg, ()))
            if rec is None:
                # experiment.json also gives P/D rows the shared tp/num_instances runs.jsonl leaves out
                rec = by_key.get(_match_key(s["name"], r["concurrency"], cfg, ("tp", "num_instances")))
            if not (rec and rec.get("cmd")):
                out.append(None)
                continue
            dep = rec.get("deployment") or {}
            name = dep.get("scenario")
            if name and name not in scenarios:
                got = _scenario(run_dir, exp, dep, unset) if dep.get("parallel") is not None else None
                scenarios[name] = got[1] if got else None
            # the deployment ships only when its scenario could not be regenerated exactly
            out.append([_recipe(rec["cmd"], workload, dep), name,
                        None if scenarios.get(name) else dep or None])
    return {"scenarios": {k: v for k, v in scenarios.items() if v}, "runs": out}


if __name__ == "__main__":
    json.dump(build(sys.argv[1]), sys.stdout, sort_keys=True, separators=(",", ":"))  # null if no sweeps
