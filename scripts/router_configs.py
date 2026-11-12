#!/usr/bin/env python3
"""The router's reference EndpointPicker configs, taken at a pinned commit, and the sweep overrides.

The llm-d router fork publishes complete configuration files and an index (`policies.json`) that
names every knob a search may sweep (llm-d-router `reference-configs/`). Config search does not keep
copies: a spec names a file of that directory, the run uses the file unchanged, and an override
edits the PARSED file and writes it back (design X1: router configuration reaches the router parser
as the router writes it).

pins.env: ROUTER_CONFIGS_REF (the branch or tag cloned) and ROUTER_CONFIGS_SHA (the commit required).
A checkout at another commit is reported (`problem`), never silently used: CI clones the pin.

Override grammar (spec `router:` block, every value a list):

  schedulingProfiles.<profile>.<plugin>.weight   a scorer weight; adds `weight` to a profile entry that has none
  plugins.<plugin>.parameters.<name>             a plugin parameter; added when the file lacks it, and the
                                                 plugin is added when the file does not declare it
  decider                                        the P/D decider type (index knob pd.decider)

For a file in the index only the knobs the index lists for it are reachable, so a typo fails before
the run instead of producing the base file and being read as "this knob does nothing".
"""
import copy
import hashlib
import itertools
import json
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REFERENCE_DIR = "reference-configs"
HANDLER = "disagg-profile-handler"
ALWAYS_DECIDER = "always-disagg-pd-decider"
PREFIX_DECIDER = "prefix-based-pd-decider"


class RouterConfigsError(Exception):
    pass


def pins(path=None):
    import build_catalog
    p = build_catalog.read_pins(path) if path else build_catalog.read_pins()
    return p.get("ROUTER_CONFIGS_REF"), p.get("ROUTER_CONFIGS_SHA")


def default_dir():
    """$ROUTER_CONFIGS_DIR, else the sibling checkout ../llm-d-router."""
    return os.environ.get("ROUTER_CONFIGS_DIR") or os.path.join(os.path.dirname(ROOT), "llm-d-router")


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(root, *args):
    try:
        pr = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return pr.stdout.strip() if pr.returncode == 0 else None


class RouterConfigs:
    """A checkout of the router repository: its reference files and index.

    `problem` is "" when the checkout is the pinned commit with the reference directory unmodified,
    else one line saying how it differs (a warning locally; CI refuses by cloning the pin)."""

    def __init__(self, root, verify=True, want_sha=None):
        self.root = os.path.abspath(root)
        self.dir = os.path.join(self.root, REFERENCE_DIR)
        index_path = os.path.join(self.dir, "policies.json")
        try:
            with open(index_path, encoding="utf-8") as f:
                self.index = json.load(f)
        except (OSError, ValueError) as e:
            raise RouterConfigsError(f"{index_path}: {e}")
        if self.index.get("schema") != 1:
            raise RouterConfigsError(f"{index_path}: schema {self.index.get('schema')!r} is not one "
                                     f"this runner reads (1)")
        self.files = {}
        for pol in self.index.get("policies") or []:
            for role, name in (pol.get("files") or {}).items():
                if name:
                    self.files[name] = {"policy": pol["name"], "role": role,
                                        "knobs": pol.get("knobs") or []}
        self.commit = _git(self.root, "rev-parse", "HEAD") if verify else None
        self.want = want_sha if want_sha is not None else pins()[1]
        self.problem = ""
        if verify:
            self.problem = self._problem()

    def _problem(self):
        if not self.commit:
            return f"{self.root} is not a git checkout, so the pinned commit cannot be confirmed"
        if self.want and self.commit != self.want:
            return (f"pins.env pins the router configs at {self.want[:12]}, but {self.root} is at "
                    f"{self.commit[:12]}")
        dirty = _git(self.root, "status", "--porcelain", "--", REFERENCE_DIR)
        if dirty:
            return f"{os.path.join(self.root, REFERENCE_DIR)} has uncommitted changes"
        return ""

    def path(self, name):
        return os.path.join(self.dir, name)

    def knobs(self, name):
        return self.files[name]["knobs"]


def open_checkout(path=None, verify=True):
    """The checkout at `path` (default: default_dir()), or None when there is none there."""
    path = path or default_dir()
    if not os.path.isfile(os.path.join(path, REFERENCE_DIR, "policies.json")):
        return None
    return RouterConfigs(path, verify=verify)


def clone_hint():
    ref, sha = pins()
    return (f"clone it with `sh scripts/clone_pinned.sh {sha} https://github.com/inference-sim/llm-d-router.git "
            f"../llm-d-router` or pass --router-configs DIR" if sha else "pin ROUTER_CONFIGS_SHA in pins.env")


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
             if router else f"available: {', '.join(have) or 'none'}; no router checkout was found "
                            f"({clone_hint()})")
    raise RouterConfigsError(f"no such file routing/{name} ({where})")


# --- overrides ---------------------------------------------------------------------------------


def override_combos(router):
    """Every combination of the router override trees, as {path-tuple: value}. [{}] when none.

    Paths: ("schedulingProfiles", profile, ref, field), ("plugins", plugin, "parameters", field),
    ("decider",). A combination that selects the always-disagg decider drops the prefix-based
    decider's parameters (they cannot change anything) and is then kept once."""
    paths = {}
    for profile, refs in (router.get("schedulingProfiles") or {}).items():
        for ref, fields in (refs or {}).items():
            for field, values in (fields or {}).items():
                paths[("schedulingProfiles", profile, ref, field)] = list(values)
    for plugin, block in (router.get("plugins") or {}).items():
        for field, values in ((block or {}).get("parameters") or {}).items():
            paths[("plugins", plugin, "parameters", field)] = list(values)
    if router.get("decider"):
        paths[("decider",)] = list(router["decider"])
    if not paths:
        return [{}]
    keys = sorted(paths)
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*[paths[k] for k in keys])]
    out, seen = [], set()
    for c in combos:
        if c.get(("decider",)) == ALWAYS_DECIDER:
            c = {k: v for k, v in c.items() if not (k[0] == "plugins" and k[1] == PREFIX_DECIDER)}
        key = json.dumps(sorted((list(k), v) for k, v in c.items()), default=str)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _allowed(knobs):
    weights = {(k.get("profile"), k["plugin"]) for k in knobs if k.get("kind") == "scorer_weight"}
    params = {(k["plugin"], k["parameter"]) for k in knobs if k.get("kind") == "plugin_parameter"}
    choices = next((k.get("choices") or [] for k in knobs if k.get("id") == "pd.decider"), None)
    return weights, params, choices


def _plugin(doc, type_):
    return next((p for p in (doc.get("plugins") or []) if p.get("type") == type_), None)


def _add_picker_refs(doc, picker):
    """Reference a newly declared picker from every profile that has none, as the loader does for a
    file that names no picker."""
    for prof in doc.get("schedulingProfiles") or []:
        refs = prof.setdefault("plugins", [])
        if not any(str(r.get("pluginRef", "")).endswith("-picker") for r in refs):
            refs.append({"pluginRef": picker})


def _set_decider(doc, value, choices):
    if choices is None:
        raise KeyError("this file's index lists no decider choice (pd.decider)")
    if value not in choices:
        raise KeyError(f"decider {value!r} is not one of {choices}")
    handler = _plugin(doc, HANDLER)
    if handler is None:
        raise KeyError(f"the base config declares no {HANDLER}, so it has no decider to choose")
    if value == PREFIX_DECIDER:
        missing = []
        if _plugin(doc, "approx-prefix-cache-producer") is None:
            missing.append("plugin approx-prefix-cache-producer")
        for prof in doc.get("schedulingProfiles") or []:
            if not any(r.get("pluginRef") == "prefix-cache-scorer" for r in prof.get("plugins") or []):
                missing.append(f"prefix-cache-scorer in profile {prof.get('name')!r}")
        if missing:
            raise KeyError(f"{PREFIX_DECIDER} needs approx-prefix-cache-producer and prefix-cache-scorer "
                           f"in both profiles, and the base config lacks: {', '.join(missing)}")
    if _plugin(doc, value) is None:
        doc.setdefault("plugins", []).append({"type": value})
    handler.setdefault("parameters", {}).setdefault("deciders", {})["prefill"] = value


def apply_overrides(base_doc, overrides, knobs=None):
    """A copy of `base_doc` with each override applied; raises KeyError naming what cannot be.

    `knobs` is the index's knob list for the base file, or None for a file without an index entry.
    With an index, a weight or parameter the file lacks is added when the index lists it, and a
    plugin the file lacks (a producer, a picker) is added when the index lists a parameter of it.
    Without one, only a field the file already sets can change (the previous rule)."""
    doc = copy.deepcopy(base_doc)
    weights, params_ok, choices = _allowed(knobs) if knobs is not None else (set(), set(), None)
    for path, value in sorted(overrides.items()):
        if path[0] == "decider":
            if knobs is None:
                raise KeyError("decider can only be swept on a router reference config")
            _set_decider(doc, value, choices)
        elif path[0] == "schedulingProfiles":
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
            entry = _plugin(doc, plugin)
            listed = (plugin, field) in params_ok
            if entry is None:
                if not listed:
                    raise KeyError(f"the base config declares no plugin of type {plugin!r} "
                                   f"(declares: {[p.get('type') for p in doc.get('plugins') or []]})")
                entry = {"type": plugin}
                doc.setdefault("plugins", []).append(entry)
                if plugin.endswith("-picker"):
                    _add_picker_refs(doc, plugin)
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
        if path[0] == "decider":
            parts.append(f"decider {value}")
        elif path[0] == "schedulingProfiles":
            parts.append(f"{path[2]} {path[3]} {value}")
        else:
            parts.append(f"{path[1]} {path[3]} {value}")
    return ", ".join(parts)


def override_record(overrides):
    """The overrides as {dotted.path: value}, for the report."""
    return {".".join(path): value for path, value in sorted(overrides.items())}
