#!/usr/bin/env python3
"""Split an experiment into bounded specs whose runs, taken together, are the parent's runs.

    python3 -I scripts/shard_experiment.py SPEC --max-runs N                 # the plan
    python3 -I scripts/shard_experiment.py SPEC --max-runs N --index I --out FILE

Each shard keeps one sweep's name and every declared load, with each independent axis cut to a
contiguous slice of its values. The shards' runs are the parent's runs, as a multiset. Nothing is
priced, ranked or filtered: GPU-budget refusals stay the runner's.

The grid is not purely Cartesian (a speculative K of 0 collapses its method and drafter, a 0-byte
offload tier takes no policy, a repeated value is listed once), so cutting an axis is exact only
where expand_sweep multiplies. The plan asks expand_sweep itself, on one-point probes, which axes
couple, and never cuts those: every shard carries each coupled group whole. The plan costs the
probes, not the grid.
"""
import argparse
import copy
import itertools
import json
import math
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment

NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def axes_of(node, prefix=()):
    axes = []
    for key, value in sorted(node.items()):
        path = prefix + (key,)
        if isinstance(value, list):
            if not value:
                raise ValueError(f"empty grid axis: {'.'.join(path)}")
            axes.append((path, value))
        elif isinstance(value, dict):
            axes.extend(axes_of(value, path))
    return axes


def assign(document, path, values):
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = copy.deepcopy(values)


def count_of(sweep, gpus):
    kept, dropped = run_experiment.expand_sweep(sweep, gpus)
    return len(kept) + len(dropped)


def probe_count(sweep, axes, varied, gpus):
    probe = copy.deepcopy(sweep)
    for index, (path, values) in enumerate(axes):
        assign(probe, path, values if index in varied else values[:1])
    return count_of(probe, gpus)


def plan_sweep(sweep, gpus, loads, max_runs):
    axes = axes_of(sweep)
    if probe_count(sweep, axes, set(), gpus) != 1:
        raise ValueError("the first value of every axis does not expand to exactly one point")
    single = [probe_count(sweep, axes, {i}, gpus) for i in range(len(axes))]
    parent = list(range(len(axes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(axes)), 2):
        if probe_count(sweep, axes, {i, j}, gpus) != single[i] * single[j]:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(len(axes)):
        groups.setdefault(find(i), []).append(i)
    fixed, cut = [], []
    for members in groups.values():
        if len(members) == 1 and single[members[0]] == len(axes[members[0]][1]):
            cut.append(members[0])
        else:
            fixed.append((members, probe_count(sweep, axes, set(members), gpus)))
    fixed_points = math.prod(count for _, count in fixed)
    budget = max_runs // loads // fixed_points
    if budget < 1:
        raise ValueError(f"the axes that expand together make {fixed_points} configurations "
                         f"({fixed_points * loads} runs with {loads} loads); --max-runs {max_runs} "
                         f"cannot hold them in one shard")
    partitions = []
    for i in sorted(cut):
        values = axes[i][1]
        width = min(len(values), budget)
        partitions.append((axes[i][0], [values[start:start + width]
                                        for start in range(0, len(values), width)]))
        budget //= width
    return {"partitions": partitions, "fixed_points": fixed_points,
            "shards": math.prod(len(chunks) for _, chunks in partitions),
            "points": loads * fixed_points * math.prod(len(axes[i][1]) for i in cut)}


def plans_for(spec, max_runs):
    loads = len(spec["scenario"]["concurrency"])
    if loads == 0 or max_runs < loads:
        raise ValueError("a shard must fit every declared load for at least one configuration")
    return [plan_sweep(sweep, spec["scenario"]["gpus"], loads, max_runs) for sweep in spec["sweeps"]]


def compile_shard(spec, index, max_runs, plans=None):
    plans = plans if plans is not None else plans_for(spec, max_runs)
    total = sum(plan["shards"] for plan in plans)
    if index < 0 or index >= total:
        raise IndexError(f"shard index {index} outside [0, {total})")
    local = index
    for sweep_index, plan in enumerate(plans):
        if local >= plan["shards"]:
            local -= plan["shards"]
            continue
        sweep = copy.deepcopy(spec["sweeps"][sweep_index])
        expected = plan["fixed_points"]
        for path, chunks in reversed(plan["partitions"]):
            local, selected = divmod(local, len(chunks))
            assign(sweep, path, chunks[selected])
            expected *= len(chunks[selected])
        got = count_of(sweep, spec["scenario"]["gpus"])
        if got != expected:
            raise ValueError(f"shard {index} expands to {got} configurations, not the {expected} "
                             f"the plan counts: axes couple in a way the pairwise probes missed; "
                             f"split the sweep by hand")
        name = f"{spec['name']}-shard-{index:0{len(str(total - 1))}d}"
        if not NAME.match(name):
            raise ValueError(f"shard name {name!r} breaks the experiment name rule; shorten the "
                             f"parent's name")
        output = copy.deepcopy(spec)
        output["name"] = name
        output["sweeps"] = [sweep]
        return output
    raise AssertionError("unreachable shard index")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--max-runs", type=int, required=True,
                        help="the most runs one shard may hold (configurations x loads)")
    parser.add_argument("--index", type=int, help="write this shard; omit to print the plan")
    parser.add_argument("--out", type=Path, help="the shard's file; refused if it exists")
    args = parser.parse_args(argv)
    with args.spec.open(encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    plans = plans_for(spec, args.max_runs)
    if args.index is None:
        print(json.dumps({"shards": sum(plan["shards"] for plan in plans),
                          "points": sum(plan["points"] for plan in plans),
                          "max_runs_per_shard": args.max_runs}, sort_keys=True))
        return
    if args.out is None:
        parser.error("--out is required with --index")
    output = compile_shard(spec, args.index, args.max_runs, plans)
    with args.out.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)


if __name__ == "__main__":
    main()
