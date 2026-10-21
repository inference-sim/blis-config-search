#!/usr/bin/env python3
"""Ingest the sibling BLIS repos into static catalog data for the webpage.

    python3 -I scripts/build_catalog.py                     # -> site/catalog.js
    python3 -I scripts/build_catalog.py --out site/catalog.js --siblings ../

Reads (read-only, never mutated):
  blis-catalog/models/<name>/{model.yaml,config.json,graph.yaml}   -> models
  blis-catalog/hardware/<gpu>.yaml                                 -> hardware
  blis-catalog/workloads/<name>.yaml                              -> workloads
  blis-catalog/networks/*.yaml, devices/storage.yaml              -> fabrics, storage
  blis-registry/coefficients/*.yaml                              -> registry sets

Emits window.CATALOG = {models, hardware, workloads, fabrics, storage, registry, versions}.
Pure + deterministic: entries sorted by name, JSON serialized with sort_keys. Never runs
BLIS. If a sibling repo is absent, its section is empty and `versions` records "absent" so
the webpage can show "catalog built without <repo>" instead of failing.

Only whitelisted, display-relevant fields are projected; unknown config keys are dropped,
never invented. The derived graph.yaml is the authoritative source for attention/MoE wiring.
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
    """Best-effort tag/commit of a sibling repo, without shelling into it unsafely."""
    head = os.path.join(repo, ".git", "HEAD")
    if not os.path.exists(head):
        return "absent"
    # resolve HEAD -> ref -> sha; keep it simple and dependency-free
    try:
        with open(head, encoding="utf-8") as f:
            ref = f.read().strip()
        if ref.startswith("ref: "):
            refpath = os.path.join(repo, ".git", ref[5:])
            sha = open(refpath, encoding="utf-8").read().strip()[:10] if os.path.exists(refpath) else "?"
        else:
            sha = ref[:10]
    except OSError:
        sha = "?"
    # look for a tag pointing at this sha
    tags_dir = os.path.join(repo, ".git", "refs", "tags")
    tag = None
    if os.path.isdir(tags_dir):
        for t in sorted(os.listdir(tags_dir)):
            try:
                if open(os.path.join(tags_dir, t), encoding="utf-8").read().strip()[:10] == sha:
                    tag = t
                    break
            except OSError:
                pass
    return tag or sha


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


def _workload_entry(path, name):
    w = _load_yaml(path, {}) or {}
    pr = w.get("prompt") or {}
    out = w.get("output") or {}
    return {
        "name": name,
        "prefix_tokens": w.get("prefix_tokens"),
        "isl_mean": pr.get("tokens"), "isl_stdev": pr.get("tokens_stdev"),
        "isl_min": pr.get("tokens_min"), "isl_max": pr.get("tokens_max"),
        "osl_mean": out.get("tokens"), "osl_stdev": out.get("tokens_stdev"),
        "osl_min": out.get("tokens_min"), "osl_max": out.get("tokens_max"),
    }


def build(siblings):
    catalog = os.path.join(siblings, "blis-catalog")
    registry = os.path.join(siblings, "blis-registry")
    sim = os.path.join(siblings, "inference-sim")

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

    versions = {
        "blis-catalog": _git_describe(catalog),
        "blis-registry": _git_describe(registry),
        "inference-sim": _git_describe(sim),
    }
    return {"models": models, "hardware": hardware, "workloads": workloads,
            "fabrics": fabrics, "storage": storage, "registry": reg_sets, "versions": versions}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest sibling BLIS repos -> site/catalog.js")
    ap.add_argument("--siblings", default=os.path.join(ROOT, os.pardir),
                    help="dir containing blis-catalog / blis-registry / inference-sim")
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
