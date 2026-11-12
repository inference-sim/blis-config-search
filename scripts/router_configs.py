#!/usr/bin/env python3
"""The router's reference EndpointPicker configs, read from the simulator, and the sweep overrides.

The simulator embeds the llm-d router fork's reference configs and their index (`policies.json`)
and prints them with `blis router-configs --json`:

  {"router_module", "version", "index": <policies.json>, "files": {name: {"sha256", "content"}}}

Config search keeps no copy and clones no router: a spec names a file, the run uses it unchanged, and
an override edits the PARSED file and writes it back (design X1: router configuration reaches the
router parser as the router writes it). The files are therefore exactly those of the pinned simulator
(pins.env BLIS_REF/BLIS_SHA). A file whose bytes do not hash to the stated sha256 is refused.

Developer override: set $ROUTER_CONFIGS_DIR (or pass --router-configs DIR) to a router checkout to
read its reference-configs/ directory instead of the simulator's, for trying a router change before
it is embedded in a simulator build. Nothing in CI or in a committed report uses it.

Override grammar (spec `router:` block, every value a list):

  schedulingProfiles.<profile>.<plugin>.weight   a scorer weight; adds `weight` to a profile entry that has none
  plugins.<plugin>.parameters.<name>             a plugin parameter; added when the file lacks it, and the
                                                 plugin is added when the file does not declare it

For a file in the index only the index's `knobs` (hosted_effect true: each proven to move a decision
in the hosted boundary) are reachable. An `ineffective_parameters` entry is refused by name, and any
other name is refused as unlisted, so a typo or a parameter that does nothing fails before the run
instead of producing the base file and being read as "this knob does nothing". A change of P/D
decider is not an override: the index names the complete policy variant (`pd_decider.complete_variant`),
and choosing it is listing that policy's file in `endpoint_picker_config`.
"""
import atexit
import copy
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REFERENCE_DIR = "reference-configs"
HOSTED_NOTE = "accepted by the router, no effect in the hosted boundary"
COMMAND = ("router-configs", "--json")


class RouterConfigsError(Exception):
    pass


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(root, *args):
    try:
        pr = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return pr.stdout.strip() if pr.returncode == 0 else None


def _command_payload(blis):
    """The parsed output of `blis router-configs --json`; RouterConfigsError when it cannot be had."""
    try:
        pr = subprocess.run([blis, *COMMAND], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise RouterConfigsError(f"`{blis} {' '.join(COMMAND)}` could not run: {e}")
    if pr.returncode != 0:
        detail = pr.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RouterConfigsError(f"`{blis} {' '.join(COMMAND)}` failed (exit {pr.returncode}): "
                                 f"{detail[-1] if detail else 'no message'}; the simulator pinned in "
                                 f"pins.env must be one that embeds the router's reference configs")
    try:
        payload = json.loads(pr.stdout)
    except ValueError as e:
        raise RouterConfigsError(f"`{blis} {' '.join(COMMAND)}` did not print JSON: {e}")
    if not isinstance(payload, dict) or not isinstance(payload.get("index"), dict) \
            or not isinstance(payload.get("files"), dict):
        raise RouterConfigsError(f"`{blis} {' '.join(COMMAND)}`: expected an object with `index` and `files`")
    return payload


class RouterConfigs:
    """The router's reference files and index, materialised under `self.dir` (path(name) is a file).

    Built by `from_command` (the simulator's embedded copy) or `from_dir` (the developer override).
    `commit` names where they came from, for the report. `problem` is "" or one line of warning."""

    def __init__(self, index, source, directory):
        if index.get("schema") != 1:
            raise RouterConfigsError(f"{source}: index schema {index.get('schema')!r} is not one "
                                     f"this runner reads (1)")
        self.index, self.commit, self.dir, self.problem = index, source, directory, ""
        self.files = {}
        for pol in index.get("policies") or []:
            for role, name in (pol.get("files") or {}).items():
                if name:
                    self.files[name] = {
                        "policy": pol["name"], "role": role,
                        # only knobs proven to move a hosted decision are sweepable
                        "knobs": [k for k in pol.get("knobs") or [] if k.get("hosted_effect", True)],
                        "ineffective": list(pol.get("ineffective_parameters") or []),
                        "pd_decider": pol.get("pd_decider")}

    @classmethod
    def from_command(cls, blis):
        payload = _command_payload(blis)
        tmp = tempfile.mkdtemp(prefix="router-configs-")
        atexit.register(shutil.rmtree, tmp, True)
        label = f"{payload.get('router_module') or 'router'}@{payload.get('version') or 'unknown'}"
        for name, entry in sorted(payload["files"].items()):
            if os.path.basename(name) != name or name.startswith("."):
                raise RouterConfigsError(f"{label}: file name {name!r} is not a plain file name")
            body = (entry or {}).get("content")
            if not isinstance(body, str):
                raise RouterConfigsError(f"{label}: {name} has no content")
            data = body.encode("utf-8")
            if hashlib.sha256(data).hexdigest() != (entry or {}).get("sha256"):
                raise RouterConfigsError(f"{label}: {name} does not hash to the sha256 the simulator "
                                         f"states for it; refusing to use it")
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(data)
        self = cls(payload["index"], label, tmp)
        for name in self.files:
            if name not in payload["files"]:
                raise RouterConfigsError(f"{label}: the index names {name} but the simulator printed no such file")
        return self

    @classmethod
    def from_dir(cls, root):
        root = os.path.abspath(root)
        dirpath = os.path.join(root, REFERENCE_DIR)
        index_path = os.path.join(dirpath, "policies.json")
        try:
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
        except (OSError, ValueError) as e:
            raise RouterConfigsError(f"{index_path}: {e}")
        head = _git(root, "rev-parse", "HEAD")
        self = cls(index, f"{root} ({head[:12] if head else 'not a git checkout'})", dirpath)
        for name in self.files:
            if not os.path.isfile(os.path.join(dirpath, name)):
                raise RouterConfigsError(f"{dirpath}: the index names {name} but the file is missing")
        return self

    def path(self, name):
        return os.path.join(self.dir, name)

    def knobs(self, name):
        """The file's sweepable knobs and the parameters it refuses by name, for apply_overrides."""
        f = self.files[name]
        return {"knobs": f["knobs"], "ineffective": f["ineffective"]}

    def variant_of(self, name):
        """The complete policy file that is this file's decider variant, or None."""
        v = (self.files[name].get("pd_decider") or {}).get("complete_variant")
        return f"{v}.yaml" if v else None


def open_checkout(path=None, verify=True, blis=None):
    """The router's reference configs. Order: `path`, else $ROUTER_CONFIGS_DIR (developer override: a
    router checkout), else the simulator `blis` (`blis router-configs --json`). None when none of
    the three is given. `verify` is kept for callers; the simulator's files are always hash-checked."""
    path = path or os.environ.get("ROUTER_CONFIGS_DIR")
    if path:
        if not os.path.isfile(os.path.join(path, REFERENCE_DIR, "policies.json")):
            return None
        return RouterConfigs.from_dir(path)
    if blis:
        return RouterConfigs.from_command(blis)
    return None


def resolve(name, routing_dir, router):
    """(source, path, knobs) for a base file named in a spec.

    A name found in the local `routing/` directory is local (knobs None: the old rule, change what
    the file already sets). A name in the router's index is the router's file. A name in both is an
    error: a local copy must not shadow the router's file of that name. Raises RouterConfigsError."""
    local = os.path.join(routing_dir, name) if routing_dir else None
    is_local = bool(local and os.path.isfile(local))
    in_index = bool(router and name in router.files)
    if is_local and in_index:
        raise RouterConfigsError(f"{name} is both in routing/ and in the router's reference configs; "
                                 f"delete the local copy")
    if in_index:
        return "router", router.path(name), router.knobs(name)
    if is_local:
        return "local", local, None
    have = sorted(f for f in os.listdir(routing_dir) if f.endswith(".yaml")) \
        if routing_dir and os.path.isdir(routing_dir) else []
    ref = sorted(router.files) if router else []
    where = (f"available: {', '.join(have) or 'none'}; router reference configs: {', '.join(ref)}"
             if router else f"available: {', '.join(have) or 'none'}; no router configs were found "
                            f"(use a simulator that has `blis router-configs`, or set ROUTER_CONFIGS_DIR)")
    raise RouterConfigsError(f"no such file routing/{name} ({where})")


# --- overrides ---------------------------------------------------------------------------------


def override_combos(router):
    """Every combination of the router override trees, as {path-tuple: value}. [{}] when none.

    Paths: ("schedulingProfiles", profile, ref, field), ("plugins", plugin, "parameters", field)."""
    paths = {}
    for profile, refs in (router.get("schedulingProfiles") or {}).items():
        for ref, fields in (refs or {}).items():
            for field, values in (fields or {}).items():
                paths[("schedulingProfiles", profile, ref, field)] = list(values)
    for plugin, block in (router.get("plugins") or {}).items():
        for field, values in ((block or {}).get("parameters") or {}).items():
            paths[("plugins", plugin, "parameters", field)] = list(values)
    if not paths:
        return [{}]
    keys = sorted(paths)
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*[paths[k] for k in keys])]
    return combos


def _allowed(knobs):
    weights = {(k.get("profile"), k["plugin"]) for k in knobs["knobs"] if k.get("kind") == "scorer_weight"}
    params = {(k["plugin"], k["parameter"]) for k in knobs["knobs"] if k.get("kind") == "plugin_parameter"}
    ineffective = {(k["plugin"], k["parameter"]): k.get("reason", "") for k in knobs["ineffective"]}
    return weights, params, ineffective


def _same(a, b):
    """Equal values; numbers compare as numbers, so 0 equals 0.0."""
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        return float(a) == float(b)
    return a == b


def unmet_requirements(knob, doc, overrides, knobs):
    """The `requires` of an index knob that the config does not meet, as readable strings.

    A conditional knob (the router's index states `requires: [{id, value}]`) moves a decision only
    while each named knob has that value. The value a requirement sees is the one this combination
    overrides, else the one the base file sets, else the router's default from the required knob's
    own entry; nothing here knows which knobs are conditional or on what. `doc` is the base file
    parsed; `overrides` is one combination {path: value}."""
    by_id = {k.get("id"): k for k in knobs["knobs"]}
    out = []
    for req in knob.get("requires") or []:
        need = by_id.get(req.get("id"))
        if need is None or need.get("kind") != "plugin_parameter":
            out.append(f"{req.get('id')} (not a knob this index lists)")
            continue
        plugin, name = need["plugin"], need["parameter"]
        key = ("plugins", plugin, "parameters", name)
        if key in overrides:
            have = overrides[key]
        else:
            entry = _plugin(doc, plugin)
            params = (entry or {}).get("parameters") or {}
            have = params[name] if name in params else need.get("router_default")
        if not _same(have, req.get("value")):
            out.append(f"{name} must be {req.get('value')!r}, it is {have!r}")
    return out


def _plugin(doc, type_):
    return next((p for p in (doc.get("plugins") or []) if p.get("type") == type_), None)


def apply_overrides(base_doc, overrides, knobs=None):
    """A copy of `base_doc` with each override applied; raises KeyError naming what cannot be.

    `knobs` is RouterConfigs.knobs(name) for the base file ({"knobs", "ineffective"}), or None for a
    file without an index entry. With an index, a weight or parameter the file lacks is added when
    the index lists it, a plugin the file lacks (a producer) is added when the index lists a
    parameter of it, and an ineffective parameter is refused by name. Without one, only a field the
    file already sets can change (the previous rule)."""
    doc = copy.deepcopy(base_doc)
    weights, params_ok, ineffective = _allowed(knobs) if knobs is not None else (set(), set(), {})
    for path, value in sorted(overrides.items()):
        if path[0] == "schedulingProfiles":
            _, profile, ref, field = path
            profiles = doc.get("schedulingProfiles") or []
            target = next((p for p in profiles if p.get("name") == profile), None)
            if target is None:
                raise KeyError(f"schedulingProfiles has no profile named {profile!r} "
                               f"(has: {[p.get('name') for p in profiles]})")
            entry = next((p for p in (target.get("plugins") or [])
                          if p.get("pluginRef") == ref), None)
            if entry is None:
                raise KeyError(f"profile {profile!r} does not reference plugin {ref!r} "
                               f"(references: "
                               f"{[p.get('pluginRef') for p in target.get('plugins') or []]})")
            if knobs is not None and not (field == "weight" and ((profile, ref) in weights
                                                                 or (None, ref) in weights)):
                raise KeyError(f"the router index lists no {field!r} knob for {ref!r} in profile "
                               f"{profile!r} (weights: {sorted((p or 'any') + '/' + r for p, r in weights)})")
            entry[field] = value
        else:
            _, plugin, _params, field = path
            if (plugin, field) in ineffective:
                raise KeyError(f"{plugin}.{field}: {HOSTED_NOTE}. {ineffective[(plugin, field)]}")
            knob = next((k for k in (knobs or {}).get("knobs", []) if k.get("kind") == "plugin_parameter"
                         and k["plugin"] == plugin and k["parameter"] == field), None)
            unmet = unmet_requirements(knob, base_doc, overrides, knobs) if knob else []
            if unmet:
                raise KeyError(f"{plugin}.{field} changes no decision unless "
                               f"{'; '.join(unmet)} in the same configuration "
                               f"({knob.get('effective_when', '')})".rstrip(" ()"))
            entry = _plugin(doc, plugin)
            listed = (plugin, field) in params_ok
            if entry is None:
                if not listed:
                    raise KeyError(f"the base config declares no plugin of type {plugin!r} "
                                   f"(declares: {[p.get('type') for p in doc.get('plugins') or []]})")
                entry = {"type": plugin}
                doc.setdefault("plugins", []).append(entry)
            params = entry.setdefault("parameters", {})
            if field not in params and not listed:
                raise KeyError(f"plugin {plugin!r} has no parameter {field!r} in the base config "
                               f"(has: {sorted(params)}); "
                               + ("the router index does not list it for this file"
                                  if knobs is not None else
                                  "an override may only change a field the base file already sets"))
            params[field] = value
    return doc


def override_phrase(overrides):
    """A readable description of one override combination, for a config's variant label."""
    parts = []
    for path, value in sorted(overrides.items()):
        if path[0] == "schedulingProfiles":
            parts.append(f"{path[2]} {path[3]} {value}")
        else:
            parts.append(f"{path[1]} {path[3]} {value}")
    return ", ".join(parts)


def override_record(overrides):
    """The overrides as {dotted.path: value}, for the report."""
    return {".".join(path): value for path, value in sorted(overrides.items())}
