#!/usr/bin/env python3
"""Pin-bump automation: move the pinned inference-sim, then regenerate every report it priced.

Three workflows drive this:

    upstream release --bump-blis-pin.yml-->  DRAFT pin PR
        pin-compat.yml gates it (drift + smoke on the new binary); compatibility fixes are
        pushed onto that same branch until the gate is green
    --merge-->  regen-reports.yml  --one job per report-->  one PR per batch  --merge-->  published

The pin PR is the compatibility PR on purpose: a release can need code changes here (PR #25 moved
14 files for v0.14.0), and those changes must land in the same merge as the pin, because merging
the pin is what starts hours of regeneration against it.

  python3 -I scripts/blis_pin.py latest                       # newest upstream release tag
  python3 -I scripts/blis_pin.py bump v0.15.0                 # rewrite pins.env + SETUP.md
  python3 -I scripts/blis_pin.py changed OLD_PINS.env         # which pins a push moved
  python3 -I scripts/blis_pin.py plan [--only ID,...]         # what to regenerate, in batches
  python3 -I scripts/blis_pin.py regen ID --blis B --catalog C --registry R --out DIR
  python3 -I scripts/blis_pin.py summary --old DIR --new DIR ID [ID ...]
  python3 -I scripts/blis_pin.py drift --siblings DIR         # knob tables vs the new stack
  python3 -I scripts/blis_pin.py smoke [ID ...] --blis B --catalog C --registry R [--corpus F]

Why every report, and why not in place: a report's numbers are a function of (its source, the
pins). Move a pin and every committed report is now labelled with a stack the site no longer
runs, so `build_catalog._priced_by` starts publishing two simulator versions side by side. The
only honest fix is to re-run each report from its source -- never to relabel it.

What "its source" is, per report directory (the `kind` in a plan):
  * experiment -- the dir has experiment.json, so run_experiment.py swept a spec. The spec is
    experiments/<name>.yaml, falling back to the spec.yaml the run dir carries.
  * submission -- submissions/<run_id>.yaml exists, so run_search.py searched it. An AgentX
    replay dir also carried a report.json, which write_agentx_report.py re-derives from the data.
  * orphan     -- neither. It cannot be regenerated, so the plan lists it for a human instead of
    guessing; a PR that silently skipped it would leave one stale report looking current.

Nothing here uses an LLM: both runners are deterministic at a fixed seed, which is what lets the
regeneration run unattended and still be reviewed as a plain diff of numbers.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SKILL = os.path.join(ROOT, ".claude", "skills", "config-search", "scripts")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, SKILL)

PINS = os.path.join(ROOT, "pins.env")
SETUP = os.path.join(ROOT, ".claude", "skills", "config-search", "SETUP.md")
UPSTREAM = "inference-sim/inference-sim"
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
# The keys whose VALUE says which stack priced a report. A comment-only edit to pins.env moves
# none of them and must not queue hours of sweeps.
STACK_KEYS = ("BLIS_REF", "BLIS_SHA", "KERNEL_REF", "CATALOG_REF", "REGISTRY_REF")
# A batch is one PR. Small enough to review the numbers, large enough not to open a PR per report.
MAX_BATCH = 4


# ---------------------------------------------------------------------------------- pins

def read_pins(path=PINS):
    import build_catalog
    return build_catalog.read_pins(path)


def semver(tag):
    """(major, minor, patch) for a plain release tag, or None for anything else (rc, dev, ...)."""
    m = TAG_RE.match(tag or "")
    return tuple(int(x) for x in m.groups()) if m else None


def newer(candidate, current):
    """True when `candidate` is a strictly later release than `current`. An unparseable tag on
    either side is never "newer": the bot opens a PR only for a release it can order."""
    a, b = semver(candidate), semver(current)
    return bool(a and b and a > b)


def _http_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _http_text(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def latest_release():
    """The newest published, non-prerelease upstream release, as {tag, name, url, body}.

    /releases/latest already excludes drafts and prereleases, so a release candidate never
    becomes the pin."""
    rel = _http_json(f"https://api.github.com/repos/{UPSTREAM}/releases/latest")
    return {"tag": rel["tag_name"], "name": rel.get("name") or rel["tag_name"],
            "url": rel.get("html_url", ""), "body": rel.get("body") or ""}


def resolve_tag_sha(tag):
    """The COMMIT `tag` points at. ls-remote lists an annotated tag twice; the peeled `^{}` row
    is the commit, which is what BLIS_SHA pins and what CI compares `git rev-parse HEAD` to."""
    out = subprocess.run(["git", "ls-remote", f"https://github.com/{UPSTREAM}.git",
                          f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
                         check=True, capture_output=True, text=True).stdout
    rows = dict(reversed(line.split("\t")) for line in out.splitlines() if "\t" in line)
    sha = rows.get(f"refs/tags/{tag}^{{}}") or rows.get(f"refs/tags/{tag}")
    if not sha:
        raise SystemExit(f"{UPSTREAM} has no tag {tag}")
    return sha


def go_mod_versions(go_mod):
    """{module-suffix: version} for the blis-* modules inference-sim's go.mod requires."""
    out = {}
    for m in re.finditer(r"github\.com/inference-sim/(blis-[a-z-]+)\s+(v[0-9][^\s]*)", go_mod):
        out.setdefault(m.group(1), m.group(2))
    return out


def rewrite_pins(text, updates):
    """`text` with each KEY=VALUE assignment in `updates` replaced, comments untouched.

    Refuses a key that is not already assigned: pins.env is the schema, and a bump that invented
    a new pin would be read by nothing."""
    out, seen = [], set()
    for line in text.splitlines(keepends=True):
        m = re.match(r"^([A-Z_]+)=(.*?)(\r?\n)?$", line)
        if m and m.group(1) in updates:
            seen.add(m.group(1))
            line = f"{m.group(1)}={updates[m.group(1)]}{m.group(3) or ''}"
        out.append(line)
    missing = set(updates) - seen
    if missing:
        raise ValueError(f"pins.env has no assignment for {', '.join(sorted(missing))}")
    return "".join(out)


def rewrite_setup(text, old_ref, new_ref, new_sha, note):
    """SETUP.md's step-1 clone block, moved from old_ref to new_ref.

    Only the four instructional lines change -- the clone, the "tag is NOT immutable" warning, the
    EXPECT sha and the abort message. Every other mention of old_ref in SETUP.md is a historical
    statement ("routing moved in v0.12.0") and stays true after the bump."""
    lines, hits = text.splitlines(keepends=True), 0
    for i, line in enumerate(lines):
        if line.startswith("EXPECT="):
            lines[i] = f"EXPECT={new_sha}   # {note}\n"
            hits += 1
        elif (line.startswith("git clone --branch ") and "inference-sim.git" in line) \
                or "tag is NOT configured immutable" in line \
                or "no longer resolves to $EXPECT" in line:
            lines[i] = line.replace(old_ref, new_ref)
            hits += 1
    if hits != 4:
        raise ValueError(f"SETUP.md's clone block changed shape: expected 4 pin lines, found {hits}")
    return "".join(lines)


def bump(tag, sha=None, go_mod=None, release_name=None, pins_path=PINS, setup_path=SETUP):
    """Move the pin to `tag`, in pins.env and SETUP.md. Returns {key: (old, new)} for what moved.

    KERNEL_REF moves with it because inference-sim's go.mod pins the kernel. CATALOG_REF and
    REGISTRY_REF are NOT inferred: they are separate repos and no file upstream says which ones a
    release needs, so the PR asks a reviewer instead of guessing."""
    pins = read_pins(pins_path)
    old_ref = pins.get("BLIS_REF")
    if not newer(tag, old_ref):
        raise SystemExit(f"{tag} is not newer than the pinned {old_ref}; nothing to bump")
    sha = sha or resolve_tag_sha(tag)
    if go_mod is None:
        go_mod = _http_text(f"https://raw.githubusercontent.com/{UPSTREAM}/{sha}/go.mod")
    mods = go_mod_versions(go_mod)
    updates = {"BLIS_REF": tag, "BLIS_SHA": sha}
    if mods.get("blis-latency-kernel"):
        updates["KERNEL_REF"] = mods["blis-latency-kernel"]
    with open(pins_path) as f:
        pins_text = f.read()
    with open(setup_path) as f:
        setup_text = f.read()
    extras = ", ".join(f"{k.removeprefix('blis-')} {v}" for k, v in sorted(mods.items()))
    note = f"release {tag}" + (f", \"{release_name}\"" if release_name and release_name != tag
                               else "") + (f" ({extras} in go.mod)" if extras else "")
    new_pins = rewrite_pins(pins_text, updates)
    new_setup = rewrite_setup(setup_text, old_ref, tag, sha, note)
    with open(pins_path, "w") as f:
        f.write(new_pins)
    with open(setup_path, "w") as f:
        f.write(new_setup)
    return {k: (pins.get(k), v) for k, v in updates.items() if pins.get(k) != v}


def changed(old_pins, new_pins):
    """The stack pins whose value differs between two parsed pins.env files."""
    return {k: (old_pins.get(k), new_pins.get(k)) for k in STACK_KEYS
            if old_pins.get(k) != new_pins.get(k)}


# ---------------------------------------------------------------------------------- plan

def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def classify(run_id, root=ROOT):
    """How to regenerate reports/<run_id>/ -- see the module docstring for the three kinds."""
    rdir = os.path.join(root, "reports", run_id)
    scen = _load_json(os.path.join(rdir, "scenario.json"), {}) or {}
    entry = {"id": run_id, "model": scen.get("model") or "unknown",
             "hardware": scen.get("hardware") or "", "replay": False}
    if os.path.isfile(os.path.join(rdir, "experiment.json")):
        src = os.path.join("experiments", run_id + ".yaml")
        if not os.path.isfile(os.path.join(root, src)):
            # A spec renamed or removed after its report merged: the run dir's own copy is the
            # spec that produced it (run_experiment writes it verbatim for this reason).
            src = next((os.path.relpath(p, root) for p in _spec_files(root)
                        if (_load_yaml(p).get("name") == run_id)), None)
            src = src or (os.path.join("reports", run_id, "spec.yaml")
                          if os.path.isfile(os.path.join(rdir, "spec.yaml")) else None)
        if src:
            spec = _load_yaml(os.path.join(root, src))
            entry.update(kind="experiment", source=src,
                         replay=bool((spec.get("scenario") or {}).get("replay")))
            return entry
    sub = os.path.join("submissions", run_id + ".yaml")
    if os.path.isfile(os.path.join(root, sub)):
        wl = (_load_yaml(os.path.join(root, sub)).get("workload") or {})
        entry.update(kind="submission", source=sub,
                     replay=isinstance(wl, dict) and bool(wl.get("replay")),
                     narrative=os.path.isfile(os.path.join(rdir, "report.json")))
        return entry
    entry.update(kind="orphan", source=None)
    return entry


def _spec_files(root):
    d = os.path.join(root, "experiments")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".yaml")) \
        if os.path.isdir(d) else []


def report_ids(root=ROOT):
    d = os.path.join(root, "reports")
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d)
                  if os.path.isfile(os.path.join(d, n, "scenario.json")))


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "misc"


def plan(root=ROOT, only=None, max_batch=MAX_BATCH):
    """Every committed report, classified, and grouped into PR-sized batches.

    Batches group by MODEL, then split to at most `max_batch` reports. A reviewer compares a
    model's reports against each other (the same model across hardware or budget), so keeping
    those together is what makes a regenerated batch reviewable; mixing models would not."""
    ids = report_ids(root)
    if only:
        unknown = sorted(set(only) - set(ids))
        if unknown:
            raise SystemExit(f"no committed report named {', '.join(unknown)}")
        ids = [i for i in ids if i in set(only)]
    entries = [classify(i, root) for i in ids]
    runnable = [e for e in entries if e["kind"] != "orphan"]
    by_model = {}
    for e in runnable:
        by_model.setdefault(e["model"], []).append(e["id"])
    batches = []
    for model in sorted(by_model):
        members = by_model[model]
        chunks = [members[i:i + max_batch] for i in range(0, len(members), max_batch)]
        for n, chunk in enumerate(chunks, 1):
            name = _slug(model) + (f"-{n}" if len(chunks) > 1 else "")
            batches.append({"name": name, "model": model, "ids": chunk})
    batch_of = {i: b["name"] for b in batches for i in b["ids"]}
    for e in entries:
        e["batch"] = batch_of.get(e["id"])
    return {"reports": runnable, "orphans": [e for e in entries if e["kind"] == "orphan"],
            "batches": batches}


# ---------------------------------------------------------------------------------- regen

def _run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def _replay_descriptor(catalog, harness_scenario):
    """The catalog workload whose replay block names `harness_scenario` (a submission states the
    harness scenario, a spec the catalog workload name; this maps the former to the latter)."""
    wdir = os.path.join(catalog, "workloads")
    for name in sorted(os.listdir(wdir)):
        if name.endswith(".yaml"):
            rep = (_load_yaml(os.path.join(wdir, name)).get("replay") or {})
            if rep.get("scenario") == harness_scenario:
                return rep
    raise SystemExit(f"catalog has no replay workload for scenario {harness_scenario!r}")


def fetch_submission_corpus(sub_path, catalog, dest):
    """Fetch + digest-check the corpus a REPLAY submission names, via the catalog descriptor.

    The submission's own corpus_sha256 must match the catalog's: if the catalog pin moved the
    corpus, the submission describes traffic the new stack no longer replays, and regenerating it
    would silently swap the workload under an unchanged report title."""
    import fetch_corpus
    rep = (_load_yaml(sub_path).get("workload") or {}).get("replay") or {}
    desc = _replay_descriptor(catalog, rep.get("scenario"))
    corpus = desc.get("corpus") or {}
    src = corpus.get("source") or {}
    want = corpus.get("sha256")
    if rep.get("corpus_sha256") and rep["corpus_sha256"] != want:
        raise SystemExit(f"{sub_path} pins corpus {rep['corpus_sha256'][:12]} but the catalog now "
                         f"names {str(want)[:12]}; update the submission deliberately first")
    url = fetch_corpus.HF_URL.format(dataset=src.get("huggingface"), revision=src.get("revision"),
                                     path=src.get("file", "traces.jsonl"))
    fetch_corpus.fetch(dest, url, want, token=os.environ.get("HF_TOKEN"))
    return dest


def regen(run_id, blis, catalog, registry, out, timeout=1800, max_parallel=1,
          corpus=os.path.join(ROOT, ".corpus", "traces.jsonl"),
          corpus_cache="/tmp/agx-cache"):
    """Regenerate ONE report into <out>/<run_id>/ from its source, at the pins on disk.

    Writes into a fresh `out` rather than over reports/: the caller swaps the directory in whole,
    so a file the new run no longer writes cannot survive from the old one."""
    e = classify(run_id)
    py = sys.executable
    os.makedirs(out, exist_ok=True)
    if e["kind"] == "experiment":
        spec = e["source"]
        _run([py, "-I", "scripts/validate_experiment.py", spec, "--catalog", catalog])
        if e["replay"]:
            _run([py, "-I", "scripts/fetch_corpus.py", spec, "--catalog", catalog])
        _run([py, "-I", "scripts/run_experiment.py", spec, "--blis", blis, "--catalog", catalog,
              "--registry", registry, "--out", out, "--timeout", str(timeout),
              "--max-parallel", str(max_parallel)])
        shutil.rmtree(os.path.join(out, run_id, "scenarios"), ignore_errors=True)
    elif e["kind"] == "submission":
        cmd = [py, "-I", "scripts/run_search.py", "--submission", e["source"], "--blis", blis,
               "--catalog", catalog, "--registry", registry, "--out", out,
               "--run-timeout", str(timeout), "--max-parallel", str(max_parallel)]
        if e["replay"]:
            fetch_submission_corpus(os.path.join(ROOT, e["source"]), catalog, corpus)
            cmd += ["--corpus", corpus, "--corpus-cache", corpus_cache]
        _run(cmd)
        if e["narrative"]:
            # The AgentX replay reports carry an authored-from-data narrative; re-derive it so the
            # prose quotes the NEW numbers (regen_agentx.sh step 3).
            _run([py, "-I", "scripts/write_agentx_report.py", os.path.join(out, run_id)])
    else:
        raise SystemExit(f"{run_id} has no source to regenerate from (no experiment spec, no "
                         f"submissions/{run_id}.yaml)")
    _check_coherent(os.path.join(out, run_id))
    return os.path.join(out, run_id)


def _check_coherent(run_dir):
    """A regenerated ledger must be one seed and one stack, or it is not a regeneration."""
    import report
    path = os.path.join(run_dir, "runs.jsonl")
    with open(path) as f:
        runs = [json.loads(l) for l in f if l.strip()]
    split = report.provenance_split(runs)
    if split:
        raise SystemExit(f"{run_dir}: NOT coherent after regeneration: {split}")
    print(f"{run_dir}: coherent ({len(runs)} runs, one seed, one stack)")


# ---------------------------------------------------------------------------------- summary

def headline(run_dir):
    """The few numbers a reviewer compares across a regeneration, or None if the dir is absent."""
    scen = _load_json(os.path.join(run_dir, "scenario.json"))
    if scen is None:
        return None
    pareto = _load_json(os.path.join(run_dir, "pareto.json"), {}) or {}
    win = pareto.get("winner") or {}
    runs, servable = 0, None
    exp = _load_json(os.path.join(run_dir, "experiment.json"))
    gal = _load_json(os.path.join(run_dir, "gallery.json"), {}) or {}
    if exp:
        runs, servable = exp["totals"]["runs"], exp["totals"]["servable"]
        best = exp.get("best_overall") or {}
        tpg = best.get("tokens_per_sec_per_gpu")
    else:
        runs, servable = gal.get("n_runs"), gal.get("n_servable")
        fin = (pareto.get("finalists") or [{}])[0]
        tpg = (fin.get("at_c_use") or {}).get("tokens_per_sec_per_gpu")
    return {"blis": (scen.get("pinned_stack") or {}).get("blis"),
            "winner": win.get("chosen") or win.get("label") or "none", "tok_s_gpu": tpg,
            "runs": runs, "servable": servable}


def _fmt(v):
    return "—" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))


def summary(old_root, new_root, ids, failed=()):
    """A markdown before/after table for a regeneration PR body."""
    rows = ["| report | stack | winner | tok/s/GPU | servable / runs |",
            "|---|---|---|---|---|"]
    for i in ids:
        old, new = headline(os.path.join(old_root, i)), headline(os.path.join(new_root, i))
        if new is None or i in failed:
            rows.append(f"| `{i}` | {_fmt((old or {}).get('blis'))} → **failed** | "
                        f"{_fmt((old or {}).get('winner'))} | — | — |")
            continue
        old = old or {}

        def pair(k):
            a, b = _fmt(old.get(k)), _fmt(new.get(k))
            return b if a == b else f"{a} → **{b}**"
        sr = f"{_fmt(new['servable'])} / {_fmt(new['runs'])}"
        osr = f"{_fmt(old.get('servable'))} / {_fmt(old.get('runs'))}"
        rows.append(f"| `{i}` | {pair('blis')} | {pair('winner')} | {pair('tok_s_gpu')} | "
                    + (sr if sr == osr else f"{osr} → **{sr}**") + " |")
    if failed:
        rows += ["", "**Not regenerated** (the job failed; the committed report is unchanged and "
                 "still priced by the old stack): " + ", ".join(f"`{i}`" for i in failed)]
    return "\n".join(rows)


# ---------------------------------------------------------------------------------- compat gate
#
# A new simulator can break this repo two ways, and only one of them is loud:
#   * LOUD   -- a knob name or value this repo writes is refused, a required field appears, the
#               scenario no longer decodes. Every run of a sweep fails.
#   * SILENT -- the run succeeds but a knob stopped doing anything. v0.14.0 did this to 11 engine
#               knobs (PR #25): they decoded, and measured identical numbers to the baseline, so a
#               sweep over them reported "no effect" as if it were a finding.
# Neither shows up in a dry run. Both have to be fixed on the pin PR -- as code changes to the
# validator's knob tables / run_blis.py -- BEFORE it merges, because merging is what starts the
# regeneration. So the gate runs on the pin PR, against the new binary:
#   drift  -- the validator's PINNED knob tables (what CI validates with) vs the blis-vllm and
#             blis-schemas the release's go.mod names. Any difference is code to change.
#   smoke  -- a handful of rows from each committed report's own ledger, re-run on the new
#             binary through the same run_blis.run_one the regeneration uses. A row that ran
#             before and fails now is LOUD; a knob that moved the numbers before and is now
#             byte-identical to its baseline is SILENT. Both fail the gate.

# Keys that place a deployment rather than tune it: two rows that differ in one of these are
# different shapes, not a knob and its baseline.
SHAPE_KEYS = ("tp", "dp", "pp", "dp_local", "pcp", "dcp", "enable_expert_parallel",
              "num_instances", "concurrency")
# Selectors that ride along with a drafting knob; they are not a knob of their own.
RIDERS = ("speculative_category", "speculative_thinking")
# A knob "moved" the old numbers when it changed throughput by more than this (run_search's tie
# tolerance: anything closer is modelling noise, not an effect a report would claim).
MOVED = 0.005
SMOKE_PAIRS = 6


def _is_shape(key):
    return key in SHAPE_KEYS or key.startswith(("prefill_", "decode_"))


def _ok(rec):
    m = rec.get("metrics") or {}
    return rec.get("returncode") == 0 and not rec.get("error") and bool(m.get("tokens_per_sec"))


def _tps(rec):
    return (rec.get("metrics") or {}).get("tokens_per_sec") or 0.0


def _knob_diff(a, b):
    """The tuning keys on which configs a and b differ, or None if they are different shapes."""
    ca, cb = a["config"], b["config"]
    for k in set(ca) | set(cb):
        if _is_shape(k) and ca.get(k) != cb.get(k):
            return None
    return tuple(sorted(k for k in set(ca) | set(cb)
                        if k not in RIDERS and not _is_shape(k) and ca.get(k) != cb.get(k)))


def smoke_pairs(runs, max_pairs=SMOKE_PAIRS):
    """Choose (knob, base_row, knob_row) pairs from a committed ledger, deterministically.

    One pair per knob, each differing from its base in that knob alone. The base is the row with FEWER tuning keys (the
    default the knob was swept against). Prefer the cheapest pair -- lowest concurrency -- among
    those where the knob MOVED the old numbers, since only those can show a knob going inert; a
    knob that never moved anything is still re-run, for the loud check alone."""
    ok = [r for r in runs if _ok(r)]
    best = {}
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            d = _knob_diff(a, b)
            # Exactly ONE knob, so an effect (or its absence) is attributable to it.
            if not d or len(d) != 1:
                continue
            base, knob = sorted((a, b), key=lambda r: (len(r["config"]),
                                                       json.dumps(r["config"], sort_keys=True)))
            moved = abs(_tps(knob) - _tps(base)) > MOVED * max(_tps(base), 1e-9)
            rank = (not moved, base["config"].get("concurrency") or 0,
                    json.dumps(knob["config"], sort_keys=True))
            if d not in best or rank < best[d][0]:
                best[d] = (rank, base, knob, moved)
    pairs = sorted(best.items(), key=lambda kv: (kv[1][0], kv[0]))[:max_pairs]
    return [{"knob": ",".join(d), "base": b, "row": k, "moved": mv}
            for d, (_r, b, k, mv) in pairs]


def _local_router(path, root, run_id):
    """A ledger's router_config is an absolute path on the machine that ran it; find this
    checkout's copy -- the report's own generated arms first, then the shared routing/."""
    if not path:
        return path
    name = os.path.basename(path)
    for d in (os.path.join(root, "reports", run_id, "routing"), os.path.join(root, "routing")):
        if os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    raise SystemExit(f"{run_id}: router config {name} is in neither reports/{run_id}/routing/ "
                     f"nor routing/")


def _run_context(entry, root, catalog, corpus, corpus_cache):
    """(model, hardware, engine_version, workload, seed, replay, nreq_of) for a report's source,
    built the way its runner builds them, so a smoke run is a run the regeneration would make."""
    import run_experiment
    import run_search
    src = _load_yaml(os.path.join(root, entry["source"]))
    local = {"corpus": os.path.abspath(corpus)} if corpus else {}
    if corpus and corpus_cache:
        local["corpus_cache"] = os.path.abspath(corpus_cache)
    if entry["kind"] == "experiment":
        sc = src["scenario"]
        replay = None
        if entry["replay"]:
            replay = run_experiment.resolve_replay(sc["workload"], catalog,
                                                   dict(sc.get("replay") or {}, **local))
            replay = {k: v for k, v in replay.items() if k != "_overrides"}
        return (sc["model"], sc["hardware"], sc.get("engine_version", "0.29.0"), sc["workload"],
                int(sc.get("seed", 42)), replay,
                (lambda c: None) if replay else (lambda c: max(80, c * 4)))
    wl = src.get("workload") or {}
    replay = dict(wl["replay"], **local) if isinstance(wl, dict) and wl.get("replay") else None
    return (src["model"], src["hardware"], src.get("engine_version", "0.29.0"),
            wl.get("preset") if isinstance(wl, dict) else wl, run_search.scenario_seed(src),
            replay, (lambda c: None) if replay else (lambda c: run_search.requests_for(c, 80)))


def _same_measurement(a, b):
    ma, mb = a.get("metrics") or {}, b.get("metrics") or {}
    keys = ("tokens_per_sec", "itl_p99_ms", "ttft_p99_ms", "completed_requests")
    return all(ma.get(k) == mb.get(k) for k in keys)


def ensure_corpus(entry, catalog, dest, root=ROOT):
    """Fetch + digest-check a replay report's corpus to `dest` unless it is already there."""
    import fetch_corpus
    import run_experiment
    if entry["kind"] == "submission":
        return fetch_submission_corpus(os.path.join(root, entry["source"]), catalog, dest)
    wl = (_load_yaml(os.path.join(root, entry["source"])).get("scenario") or {}).get("workload")
    corpus = (run_experiment.replay_descriptor(wl, catalog) or {}).get("corpus") or {}
    src = corpus.get("source") or {}
    url = fetch_corpus.HF_URL.format(dataset=src.get("huggingface"), revision=src.get("revision"),
                                     path=src.get("file", "traces.jsonl"))
    fetch_corpus.fetch(dest, url, corpus.get("sha256"), token=os.environ.get("HF_TOKEN"))
    return dest


def smoke(run_id, blis, catalog, registry, root=ROOT, corpus=None, corpus_cache=None,
          max_pairs=SMOKE_PAIRS, timeout=1800, max_parallel=1):
    """Re-run a few of reports/<run_id>/'s own rows on `blis`; return {id, checks, failures}."""
    import concurrent.futures
    import tempfile
    import run_blis
    entry = classify(run_id, root)
    if entry["kind"] == "orphan":
        return {"id": run_id, "checks": [], "failures": [f"{run_id}: no source to run from"]}
    if entry["replay"]:
        if not corpus:
            raise SystemExit(f"{run_id} is a replay report; smoke needs --corpus")
        ensure_corpus(entry, catalog, corpus, root)
    with open(os.path.join(root, "reports", run_id, "runs.jsonl")) as f:
        runs = [json.loads(l) for l in f if l.strip()]
    pairs = smoke_pairs(runs, max_pairs)
    if not pairs:   # a ledger with no knob pairs: still prove its cheapest row runs
        ok = sorted((r for r in runs if _ok(r)), key=lambda r: r["config"].get("concurrency") or 0)
        pairs = [{"knob": "", "base": ok[0], "row": None, "moved": False}] if ok else []
    model, hardware, ev, workload, seed, replay, nreq_of = _run_context(
        entry, root, catalog, corpus, corpus_cache)
    scen_dir = tempfile.mkdtemp(prefix="pin-smoke-")
    todo = {}
    for p in pairs:
        for rec in (p["base"], p["row"]):
            if rec is not None:
                todo.setdefault(json.dumps(rec["config"], sort_keys=True), rec)

    def _one(key):
        cfg = dict(todo[key]["config"])
        if cfg.get("router_config"):
            cfg["router_config"] = _local_router(cfg["router_config"], root, run_id)
        c = cfg.get("concurrency") or 1
        return key, run_blis.run_one(blis, catalog, registry, scen_dir, model, hardware, ev,
                                     workload, nreq_of(c), seed, None, cfg, phase="pin-smoke",
                                     timeout=timeout, replay=replay)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_parallel)) as ex:
        new = dict(ex.map(_one, sorted(todo)))
    shutil.rmtree(scen_dir, ignore_errors=True)

    checks, failures = [], []
    for key, old in sorted(todo.items()):
        n = new[key]
        checks.append({"config": old["config"], "old_tps": round(_tps(old), 1),
                       "new_tps": round(_tps(n), 1) if _ok(n) else None,
                       "error": None if _ok(n) else (n.get("error") or
                                                     (n.get("stderr_tail") or "")[-400:])})
        if not _ok(n):
            failures.append(f"LOUD: {run_id} {_cfg_phrase(old['config'])} ran on the old stack "
                            f"and fails on the new one: "
                            f"{(n.get('error') or (n.get('stderr_tail') or '')[-300:]).strip()}")
    for p in pairs:
        if p["row"] is None or not p["moved"]:
            continue
        b = new[json.dumps(p["base"]["config"], sort_keys=True)]
        k = new[json.dumps(p["row"]["config"], sort_keys=True)]
        if _ok(b) and _ok(k) and _same_measurement(b, k):
            failures.append(
                f"SILENT: {run_id} `{p['knob']}` moved throughput {_tps(p['base']):.1f} -> "
                f"{_tps(p['row']):.1f} on the old stack and now measures IDENTICALLY to its "
                f"baseline ({_tps(b):.1f}) -- the new release no longer models it. Move it to the "
                f"validator's inert table (validate_experiment.py) so specs stop sweeping it.")
    return {"id": run_id, "checks": checks, "failures": failures,
            "pairs": [{"knob": p["knob"], "moved": p["moved"]} for p in pairs]}


def _cfg_phrase(cfg):
    return "{" + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())
                           if k != "router_config") + \
        (f", router={os.path.basename(cfg['router_config'])}" if cfg.get("router_config") else "") \
        + "}"


def drift(siblings, modules=None):
    """Differences between the validator's PINNED knob tables and the upstream checkouts in
    `siblings` (blis-vllm/, blis-schemas/ at the versions the release's go.mod names).

    CI validates every spec against the pinned tables (it has no sibling checkouts), so a table
    that lags the stack either refuses a knob the new engine accepts or passes one it refuses.
    Returns a list of human-readable problems; empty means the tables match the stack.

    `modules` is the release's go.mod requirements ({"blis-vllm": "v0.32.0-blis.1", ...}). A
    module the release does not require has nothing to drift FROM -- v0.12.0 builds without
    blis-vllm -- so its checks are skipped rather than reported as an unreadable upstream. A
    module it DOES require whose file cannot be read is still a failure: that is the reader
    going stale. None (no go.mod given) checks everything."""
    import validate_experiment as V
    saved = V.SIBLINGS
    V.SIBLINGS = os.path.abspath(siblings)
    try:
        par, eng = V._names_from_blis_schemas(), V._names_from_blis_vllm()
        enums = V._enums_from_blis_vllm()
    finally:
        V.SIBLINGS = saved
    def required(mod):
        return modules is None or mod in modules
    out = []
    for what, live, pinned, table, mod in (
            ("parallel", par, V.PARALLEL_KNOBS, "PARALLEL_KNOBS", "blis-schemas"),
            ("engine", eng, V.ENGINE_KNOBS, "ENGINE_KNOBS", "blis-vllm")):
        if not required(mod):
            continue
        if live is None:
            out.append(f"the {what} knob reader found nothing upstream -- the upstream file moved "
                       f"or changed shape; fix its reader in validate_experiment.py")
            continue
        if live - pinned:
            out.append(f"{table} lacks {sorted(live - pinned)}: knobs the new stack defines. Add "
                       f"them (and decide whether run_blis.py carries them, or they belong in a "
                       f"not-wired table)")
        if pinned - live:
            out.append(f"{table} still lists {sorted(pinned - live)}, which the new stack no "
                       f"longer defines: a spec using them validates and then fails every run")
    if not required("blis-vllm"):
        return out
    if enums is None:
        out.append(f"the engine value reader found no rules/{V.RULES_PACK}/pack.go upstream; the "
                   f"release may model a newer vLLM -- move RULES_PACK and re-read the value sets")
    else:
        for knob, values in sorted(enums.items()):
            pinned = tuple(V.ENGINE_ENUMS.get(knob, ()))
            if set(values) != set(pinned):
                add, gone = sorted(set(values) - set(pinned)), sorted(set(pinned) - set(values))
                out.append(f"ENGINE_ENUMS[{knob!r}]: " + "; ".join(
                    x for x in ((f"new values {add}" if add else ""),
                                (f"values no longer accepted {gone}" if gone else "")) if x))
    rules = os.path.join(siblings, "blis-vllm", "rules")
    packs = sorted((d for d in (os.listdir(rules) if os.path.isdir(rules) else ())
                    if re.fullmatch(r"v\d+_\d+", d)),
                   key=lambda d: tuple(int(x) for x in d[1:].split("_")))
    if packs and packs[-1] != V.RULES_PACK:
        out.append(f"blis-vllm now carries rules/{packs[-1]} (newer than the {V.RULES_PACK} the "
                   f"validator reads): check which vLLM the release models, and move RULES_PACK")
    return out


def smoke_markdown(results):
    """The smoke result as markdown, for the job summary and the pin PR's comment."""
    rows = ["| report | rows re-run | knob pairs | old → new tok/s (first row) | result |",
            "|---|---|---|---|---|"]
    for r in results:
        c0 = r["checks"][0] if r["checks"] else {}
        rows.append(f"| `{r['id']}` | {len(r['checks'])} | "
                    + (", ".join(f"`{p['knob']}`" + ("" if p["moved"] else " (inert before)")
                                 for p in r.get("pairs", []) if p["knob"]) or "—")
                    + f" | {_fmt(c0.get('old_tps'))} → {_fmt(c0.get('new_tps'))} | "
                    + ("**FAIL**" if r["failures"] else "ok") + " |")
    fails = [f for r in results for f in r["failures"]]
    if fails:
        rows += ["", "**Failures** — fix these on this PR; regeneration only starts after merge:", ""]
        rows += [f"- {f}" for f in fails]
    return "\n".join(rows)


# ---------------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("latest", help="print the newest upstream release as JSON, plus `newer`")
    b = sub.add_parser("bump", help="rewrite pins.env + SETUP.md to an upstream release tag")
    b.add_argument("tag")
    b.add_argument("--release-name")
    c = sub.add_parser("changed", help="print the stack pins that differ from OLD pins.env")
    c.add_argument("old")
    p = sub.add_parser("plan", help="classify every committed report and batch it into PRs")
    p.add_argument("--only", help="comma-separated run ids (default: every committed report)")
    p.add_argument("--max-batch", type=int, default=MAX_BATCH)
    r = sub.add_parser("regen", help="regenerate one report into --out/<id>/")
    r.add_argument("id")
    r.add_argument("--blis", required=True)
    r.add_argument("--catalog", required=True)
    r.add_argument("--registry", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--max-parallel", type=int, default=1)
    m = sub.add_parser("modules", help="print the blis-* modules a go.mod requires, one per line")
    m.add_argument("go_mod")
    d = sub.add_parser("drift", help="pinned knob tables vs upstream checkouts; exit 1 on drift")
    d.add_argument("--siblings", required=True,
                   help="dir holding blis-vllm/ and blis-schemas/ at the release's go.mod versions")
    d.add_argument("--go-mod", help="the release's go.mod: a module it does not require is skipped")
    k = sub.add_parser("smoke", help="re-run a few ledger rows per report on a new binary")
    k.add_argument("ids", nargs="*", help="report ids (default: every committed report)")
    k.add_argument("--blis", required=True)
    k.add_argument("--catalog", required=True)
    k.add_argument("--registry", required=True)
    k.add_argument("--corpus", help="replay corpus (needed when any selected report is a replay)")
    k.add_argument("--corpus-cache")
    k.add_argument("--max-pairs", type=int, default=SMOKE_PAIRS)
    k.add_argument("--timeout", type=int, default=1800)
    k.add_argument("--max-parallel", type=int, default=1)
    k.add_argument("--json", dest="json_out", help="also write the full result here")
    ks = sub.add_parser("smoke-summary", help="render smoke --json files as one markdown table")
    ks.add_argument("files", nargs="+")
    s = sub.add_parser("summary", help="markdown before/after table for a regeneration PR")
    s.add_argument("ids", nargs="+")
    s.add_argument("--old", required=True)
    s.add_argument("--new", required=True)
    s.add_argument("--failed", default="", help="comma-separated ids whose regeneration failed")
    a = ap.parse_args(argv)

    if a.cmd == "latest":
        rel = latest_release()
        rel["current"] = read_pins().get("BLIS_REF")
        rel["newer"] = newer(rel["tag"], rel["current"])
        print(json.dumps(rel))
    elif a.cmd == "bump":
        moved = bump(a.tag, release_name=a.release_name)
        print(json.dumps({k: list(v) for k, v in moved.items()}))
    elif a.cmd == "changed":
        print(json.dumps({k: list(v) for k, v in changed(read_pins(a.old), read_pins()).items()}))
    elif a.cmd == "plan":
        only = [x for x in (a.only or "").replace(" ", ",").split(",") if x]
        print(json.dumps(plan(only=only or None, max_batch=a.max_batch)))
    elif a.cmd == "regen":
        regen(a.id, os.path.abspath(a.blis), os.path.abspath(a.catalog),
              os.path.abspath(a.registry), os.path.abspath(a.out), a.timeout, a.max_parallel)
    elif a.cmd == "modules":
        with open(a.go_mod) as f:
            for mod, ver in sorted(go_mod_versions(f.read()).items()):
                print(mod, ver)
    elif a.cmd == "drift":
        mods = None
        if a.go_mod:
            with open(a.go_mod) as f:
                mods = go_mod_versions(f.read())
            for mod in ("blis-schemas", "blis-vllm"):
                print(f"{mod}: " + (mods[mod] if mod in mods else
                                    "not required by this release -- its checks do not apply"))
        problems = drift(a.siblings, mods)
        for p_ in problems:
            print(f"DRIFT: {p_}")
        print("knob tables match the stack" if not problems else
              f"{len(problems)} drift(s): change the tables on this PR before merging")
        return 1 if problems else 0
    elif a.cmd == "smoke":
        results = [smoke(i, os.path.abspath(a.blis), os.path.abspath(a.catalog),
                         os.path.abspath(a.registry), corpus=a.corpus, corpus_cache=a.corpus_cache,
                         max_pairs=a.max_pairs, timeout=a.timeout, max_parallel=a.max_parallel)
                   for i in (a.ids or report_ids())]
        if a.json_out:
            with open(a.json_out, "w") as f:
                json.dump(results, f, indent=1)
        print(smoke_markdown(results))
        return 1 if any(r["failures"] for r in results) else 0
    elif a.cmd == "smoke-summary":
        results = []
        for path in a.files:
            with open(path) as f:
                results += json.load(f)
        print(smoke_markdown(results))
        return 1 if any(r["failures"] for r in results) else 0
    elif a.cmd == "summary":
        print(summary(a.old, a.new, a.ids, [x for x in a.failed.split(",") if x]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
