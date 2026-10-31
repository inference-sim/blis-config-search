#!/usr/bin/env python3
"""Ingest the sibling BLIS repos into static catalog data for the webpage.

    python3 -I scripts/build_catalog.py                     # -> site/catalog.js
    python3 -I scripts/build_catalog.py --out site/catalog.js --siblings ../

Reads (read-only, never mutated):
  blis-catalog/models/<name>/{model.yaml,config.json,graph.yaml}   -> models
  blis-catalog/hardware/<gpu>.yaml                                 -> hardware
  blis-catalog/workloads/<name>.yaml  (a shape, or a replay)       -> workloads
  blis-catalog/networks/*.yaml, devices/storage.yaml              -> fabrics, storage
  blis-registry/coefficients/*.yaml                              -> registry sets

Emits window.CATALOG = {models, hardware, workloads, fabrics, storage, registry, versions,
stack}, where `stack` is the pinned BLIS stack as one root dependency plus the transitives
it resolves to (see _stack); `versions` stays a flat map for anything already reading it.
Pure + deterministic: entries sorted by name, JSON serialized with sort_keys. Never runs
BLIS. If a sibling repo is absent, its section is empty and `versions` records "absent" so
the webpage can show "catalog built without <repo>" instead of failing.

`versions["inference-sim"]` is NOT a sibling describe: it is BLIS_REF from pins.env, the ref
the report workflow clones and builds. Nothing here is ingested from ../inference-sim, so an
ambient checkout's HEAD is not provenance for the committed runs.

Only whitelisted, display-relevant fields are projected; unknown config keys are dropped,
never invented. The derived graph.yaml is the authoritative source for attention/MoE wiring.

A workloads/*.yaml is one of two kinds, and the projection keeps them apart under `kind`
rather than flattening both into one row of distribution columns: a SHAPE states prompt and
output token distributions, while a REPLAY (top-level `replay:`) names a load generator's
trace-replay scenario over a pinned corpus and states no distribution at all. See
_workload_entry.
"""
import argparse
import json
import os
import re

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PINS = os.path.join(ROOT, "pins.env")


def read_pins(path=PINS):
    """The pinned stack from pins.env, as a dict of KEY -> value.

    This is the authority for "which simulator priced these runs". Returns {} if the file is
    missing, so a checkout without it degrades to "unpinned" rather than failing the build.
    """
    pins = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                pins[k.strip()] = v.strip()
    except OSError:
        return {}
    return pins


def _load_yaml(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return default


def _load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _git_describe(repo):
    """Authoritative version string for a sibling repo: the exact tag if HEAD is tagged,
    else `<nearest-tag>-<n>-g<sha>` (git's own "commits ahead" form), else a short sha.
    Uses `git describe` (read-only) so packed-refs and the ahead-of-tag case are handled
    correctly — a hand-rolled ref walk missed both and could mislabel a moved HEAD as a tag."""
    if not os.path.isdir(os.path.join(repo, ".git")):
        return "absent"
    import subprocess
    try:
        out = subprocess.run(["git", "-C", repo, "describe", "--tags", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    # fallback: short HEAD sha via rev-parse
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        pass
    return "?"


# ---- config.json shape extraction (handles nested text_config/language_config) ----

def _inner_config(cfg):
    """Return the transformer sub-config, descending into a nested text/language block."""
    if not isinstance(cfg, dict):
        return {}
    for k in ("text_config", "language_config", "llm_config", "thinker_config"):
        if isinstance(cfg.get(k), dict):
            return cfg[k]
    return cfg


def _model_entry(models_dir, name):
    cfg = _load_json(os.path.join(models_dir, name, "config.json"), {}) or {}
    ident = _load_yaml(os.path.join(models_dir, name, "model.yaml"), {}) or {}
    graph = _load_yaml(os.path.join(models_dir, name, "graph.yaml"), {}) or {}
    inner = _inner_config(cfg)

    # architecture family + dense/MoE from graph.yaml layer_kinds (authoritative) or config
    layer_kinds = graph.get("layer_kinds") or []
    is_moe = False
    attn_kind = None
    experts = top_k = shared = None
    for lk in layer_kinds:
        for node in (lk.get("nodes") or []):
            op = node.get("op")
            if op == "Attention" and attn_kind is None:
                attn_kind = node.get("kind")
            if op in ("GroupedGEMM", "MoE") or node.get("role") in ("experts", "moe"):
                is_moe = True
                experts = node.get("experts", experts)
                top_k = node.get("top_k", top_k)
                shared = node.get("shared_experts", shared)
    # fall back to config fields for experts
    if experts is None:
        experts = inner.get("num_experts") or inner.get("n_routed_experts") or inner.get("num_local_experts")
        if experts:
            is_moe = True
    if top_k is None:
        top_k = inner.get("num_experts_per_tok") or inner.get("moe_top_k")

    quant = None
    qc = cfg.get("quantization_config") or inner.get("quantization_config")
    if isinstance(qc, dict):
        quant = qc.get("quant_method") or qc.get("fmt") or "quantized"
    elif inner.get("torch_dtype"):
        quant = inner["torch_dtype"]

    # param hint parsed from the name only (no numeric param field exists upstream)
    pm = re.search(r"(\d+(?:\.\d+)?)b(?:-a(\d+(?:\.\d+)?)b)?", name.lower())
    params = None
    if pm:
        params = {"total_b": float(pm.group(1))}
        if pm.group(2):
            params["active_b"] = float(pm.group(2))

    src = ident.get("source") or {}
    return {
        "name": name,
        "model_type": inner.get("model_type") or cfg.get("model_type") or "?",
        "arch": "MoE" if is_moe else "dense",
        "attn_kind": attn_kind,
        "layers": inner.get("num_hidden_layers"),
        "hidden": inner.get("hidden_size"),
        "heads_q": inner.get("num_attention_heads"),
        "heads_kv": inner.get("num_key_value_heads"),
        "head_dim": inner.get("head_dim") or None,
        "context": inner.get("max_position_embeddings"),
        "experts": experts,
        "top_k": top_k,
        "shared_experts": shared,
        "quant": quant,
        "params": params,
        "source_repo": src.get("repo"),
    }


def _hardware_entry(path, gpu):
    h = _load_yaml(path, {}) or {}
    return {
        "name": gpu,
        "memory_gib": h.get("MemoryGiB"),
        "bw_tbs": h.get("BwPeakTBs"),
        "tflops_bf16": h.get("TFlopsPeak"),
        "tflops_fp8": h.get("TFlopsFP8") or None,
        "tflops_nvfp4": h.get("TFlopsNVFP4") or None,
        "sm_count": h.get("SMCount"),
        "intra_node_bw_gbps": h.get("IntraNodeBwGBps"),
        "gpus_per_node": h.get("GPUsPerNode"),
        "gpus_per_rack": h.get("GPUsPerRack"),
        "provenance": h.get("Provenance"),
    }


def _str_or_none(v):
    """A display string for a scalar that YAML may have typed for us.

    `thinking: "on"` is a string, but YAML 1.1 reads an unquoted `on` as the boolean
    True, and `harness_version: 0.12.0` could arrive as a float. Normalize both to the
    text a reader expects rather than letting `True` reach the page."""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v)


def _workload_entry(path, name):
    """Project one blis-catalog workloads/*.yaml.

    The namespace holds two kinds of entry, and they are not interchangeable (the
    schema's workload.Binding picks exactly one arm). A SHAPE states prompt/output
    token distributions a cost model draws from. A REPLAY (top-level `replay:`, e.g.
    agentx-mvp) names a load generator's trace-replay scenario over a pinned corpus:
    its traffic is real recorded requests, so it has no ISL/OSL mean or stdev to
    state — the corpus carries per-request lengths and a prefix tree instead.

    `kind` is therefore projected explicitly rather than inferred from null columns.
    Reading absent distributions as "unknown" would be wrong twice over: it reports a
    replay as a shape with missing data, and it hides the provenance (scenario,
    corpus, revision) that is the only thing worth showing about a replay.
    """
    w = _load_yaml(path, {}) or {}
    rep = w.get("replay")
    if isinstance(rep, dict):
        corpus = rep.get("corpus") or {}
        source = corpus.get("source") or {}
        return {
            "name": name,
            "kind": "replay",
            "scenario": rep.get("scenario"),
            "harness_version": _str_or_none(rep.get("harness_version")),
            "corpus": source.get("huggingface") or corpus.get("loader"),
            # a revision pins the corpus; short form, since the table shows it inline
            "corpus_revision": (source.get("revision") or "")[:8] or None,
            "category": rep.get("category"),
            "thinking": _str_or_none(rep.get("thinking")),
            # a replay's traffic is the corpus, not a distribution — these stay absent
            "prefix_tokens": None,
            "isl_mean": None, "isl_stdev": None, "isl_min": None, "isl_max": None,
            "osl_mean": None, "osl_stdev": None, "osl_min": None, "osl_max": None,
        }
    pr = w.get("prompt") or {}
    out = w.get("output") or {}
    return {
        "name": name,
        "kind": "shape",
        "scenario": None, "harness_version": None, "corpus": None,
        "corpus_revision": None,
        "category": w.get("category"), "thinking": _str_or_none(w.get("thinking")),
        # absent prefix_tokens means no shared prefix, i.e. 0 (BLIS default); the multidoc
        # preset omits the field where the others write 0 — normalize so the tab is consistent
        "prefix_tokens": w.get("prefix_tokens", 0),
        "isl_mean": pr.get("tokens"), "isl_stdev": pr.get("tokens_stdev"),
        "isl_min": pr.get("tokens_min"), "isl_max": pr.get("tokens_max"),
        "osl_mean": out.get("tokens"), "osl_stdev": out.get("tokens_stdev"),
        "osl_min": out.get("tokens_min"), "osl_max": out.get("tokens_max"),
    }


def _priced_by(pins, reports_dir=None):
    """The simulator version to publish as having priced the committed reports.

    BLIS_REF alone is wrong the moment a pin moves: the new ref did not price the old reports.
    Each report's scenario.json records the `pinned_stack.blis` that did, so:

      * every report agrees with the pin  -> that version, plainly;
      * reports disagree with each other  -> the distinct versions, newest pin first, because a
        gallery spanning two simulators must not read as though it spans one;
      * no report records one (or none exist) -> fall back to BLIS_REF, the ref the workflow
        clones and builds, which is the best available statement.

    Deliberately NOT a re-measurement claim: it reports what the artifacts say priced them.
    """
    want = pins.get("BLIS_REF")
    rdir = reports_dir or os.path.join(ROOT, "reports")
    found = []
    if os.path.isdir(rdir):
        for name in sorted(os.listdir(rdir)):
            sc = _load_json(os.path.join(rdir, name, "scenario.json"), {}) or {}
            got = ((sc.get("pinned_stack") or {}).get("blis"))
            if got and got not in found:
                found.append(got)
    if not found:
        return want or "unpinned (no BLIS_REF in pins.env)"
    if found == [want]:
        return want
    # Pin first when it is among them, so the newest stack leads; then the rest, stably.
    ordered = ([want] if want in found else []) + [v for v in found if v != want]
    return ", ".join(ordered)


def _stack(pins, versions):
    """The pinned stack as ONE dependency with its transitives, not a flat row of peers.

    The footer used to list blis-catalog / blis-registry / inference-sim as three flat pills,
    which reads as "this site has three dependencies of equal standing". It has one: the BLIS
    simulator binary. The catalog and registry are data that binary is run against, and the
    latency kernel is pinned inside inference-sim's own go.mod -- recorded here, never cloned
    by this repo. So `root` is BLIS and `deps` is what it resolves to, and the webpage can show
    the root alone until asked to expand.

    KERNEL_REF is carried through even though `versions` never had it: it prices every run (it
    is the cost model), so omitting it from the displayed stack understated what pins a number.

    Every row is sourced from pins.env, NOT from _git_describe of a sibling checkout, and that
    is a bug fix rather than a preference. `versions` reads the catalog and registry off
    ../blis-catalog and ../blis-registry, which do not exist under actions/checkout -- so on the
    deployed site both pills have always read the literal string "absent", exactly the failure
    already documented above for inference-sim ("the pill read absent even though every run was
    priced by the pin"). That one row was moved to pins.env; these two were left behind. pins.env
    is committed, so it is the only source that survives a fresh clone.

    The describe is kept as a FALLBACK, for the case pins.env omits a ref: then an ambient
    checkout is still a better answer than "absent". Where both exist the pin wins, because the
    footer's job is provenance for the committed reports and the pin is the authority on that --
    a developer's sibling checkout sitting ahead of the pin (0.5.0 vs a pinned 0.4.0) must not be
    published as having priced reports it never saw.

    The rows come from two kinds of pin, which is why some are conditional:

      * DERIVED (kernel, schemas, vllm) -- Go dependencies compiled into the binary, written to
        pins.env from inference-sim's go.mod by scripts/sync_pins.py. A derived key can be legally
        ABSENT: blis-vllm is not a dependency before BLIS v0.14.0, so at an older pin there is no
        row for it rather than a row reading "absent". Omitting beats inventing.
      * HAND-PINNED (catalog, registry) -- data directories passed to the binary, not Go modules,
        so BLIS_REF can never determine them. Always shown.
    """
    def dep(name, version, note):
        return {"name": name, "version": version or "absent", "note": note}

    deps = [dep("blis-latency-kernel", pins.get("KERNEL_REF"),
                "cost model, derived from inference-sim's go.mod at the pinned ref")]
    # Present only from BLIS v0.14.0, where the engine surface moved into blis-vllm and it became
    # a real dependency; before that it is a reference consulted at authoring time, not something
    # that priced a run, so it earns no provenance row.
    if pins.get("VLLM_REF"):
        deps.append(dep("blis-vllm", pins["VLLM_REF"],
                        "engine config surface (vLLM's startup rules), derived from go.mod"))
    if pins.get("SCHEMAS_REF"):
        deps.append(dep("blis-schemas", pins["SCHEMAS_REF"],
                        "deployment/engine structs the scenario decodes against, derived from go.mod"))
    deps += [
        dep("blis-catalog", pins.get("CATALOG_REF") or versions.get("blis-catalog"),
            "models, hardware, workloads the binary is run against"),
        dep("blis-registry", pins.get("REGISTRY_REF") or versions.get("blis-registry"),
            "cost-model coefficients and speculative-decoding profiles"),
    ]
    return {
        "root": dep("inference-sim", versions.get("inference-sim"),
                    "the BLIS simulator whose binary prices every run"),
        "deps": deps,
    }


def build(siblings):
    catalog = os.path.join(siblings, "blis-catalog")
    registry = os.path.join(siblings, "blis-registry")

    models, hardware, workloads, fabrics, storage, reg_sets = [], [], [], [], [], []

    mdir = os.path.join(catalog, "models")
    if os.path.isdir(mdir):
        for name in sorted(os.listdir(mdir)):
            if os.path.isfile(os.path.join(mdir, name, "model.yaml")):
                models.append(_model_entry(mdir, name))

    hdir = os.path.join(catalog, "hardware")
    if os.path.isdir(hdir):
        for fn in sorted(os.listdir(hdir)):
            if fn.endswith(".yaml"):
                hardware.append(_hardware_entry(os.path.join(hdir, fn), fn[:-5]))

    wdir = os.path.join(catalog, "workloads")
    if os.path.isdir(wdir):
        for fn in sorted(os.listdir(wdir)):
            if fn.endswith(".yaml"):
                workloads.append(_workload_entry(os.path.join(wdir, fn), fn[:-5]))

    ndir = os.path.join(catalog, "networks")
    if os.path.isdir(ndir):
        for fn in sorted(os.listdir(ndir)):
            if fn.endswith(".yaml"):
                n = _load_yaml(os.path.join(ndir, fn), {}) or {}
                fabrics.append({"name": fn[:-5], "inter_node_bw_gbps": n.get("InterNodeBwGBps")})

    sfile = os.path.join(catalog, "devices", "storage.yaml")
    sdata = _load_yaml(sfile, {}) or {}
    if isinstance(sdata, dict):
        for tier, v in sorted(sdata.items()):
            if isinstance(v, dict):
                storage.append({"name": tier, "read_mbs": v.get("read_mbps") or v.get("read_MBs"),
                                "write_mbs": v.get("write_mbps") or v.get("write_MBs"),
                                "base_latency_us": v.get("base_latency_us")})

    cdir = os.path.join(registry, "coefficients")
    if os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".yaml"):
                c = _load_yaml(os.path.join(cdir, fn), {}) or {}
                coeffs = c.get("coefficients") or []
                reg_sets.append({"name": c.get("name") or fn[:-5], "entries": len(coeffs)})

    # `inference-sim` is the version of the simulator that PRICED the committed runs, so it comes
    # from pins.env -- the ref the report workflow actually clones and builds -- NOT from
    # _git_describe() of a sibling checkout. Nothing is ingested from ../inference-sim (unlike the
    # catalog and registry, which are read file-by-file), so an ambient clone's HEAD says nothing
    # about the reports: whatever branch a developer happened to leave it on would be published as
    # provenance, and in CI (which clones the pinned ref to ./upstream, not ../inference-sim) the
    # pill read "absent" even though every run was priced by the pin.
    # ... but BLIS_REF is the CURRENT pin, which is only the right answer while no committed
    # report predates it. A pin bump makes the two diverge, and then publishing BLIS_REF would
    # claim the new simulator priced runs it never saw. So the reports get asked first: each
    # scenario.json records the `pinned_stack.blis` that actually priced it.
    pins = read_pins()
    versions = {
        "blis-catalog": _git_describe(catalog),
        "blis-registry": _git_describe(registry),
        "inference-sim": _priced_by(pins),
    }
    return {"models": models, "hardware": hardware, "workloads": workloads,
            "fabrics": fabrics, "storage": storage, "registry": reg_sets, "versions": versions,
            "stack": _stack(pins, versions)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest sibling BLIS repos -> site/catalog.js")
    ap.add_argument("--siblings", default=os.path.join(ROOT, os.pardir),
                    help="dir containing blis-catalog / blis-registry (inference-sim is not "
                         "read: its version comes from pins.env)")
    ap.add_argument("--out", default=os.path.join(ROOT, "site", "catalog.js"))
    args = ap.parse_args(argv)
    cat = build(os.path.abspath(args.siblings))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("/* generated by build_catalog.py — do not edit */\nwindow.CATALOG = "
                + json.dumps(cat, indent=2, sort_keys=True) + ";\n")
    print(f"catalog.js: {len(cat['models'])} models, {len(cat['hardware'])} GPUs, "
          f"{len(cat['workloads'])} workloads, {len(cat['registry'])} coeff sets "
          f"-> {os.path.relpath(args.out, ROOT)}")
    print("  versions: " + ", ".join(f"{k}={v}" for k, v in cat["versions"].items()))
    return cat


if __name__ == "__main__":
    main()
