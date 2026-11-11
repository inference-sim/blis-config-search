#!/usr/bin/env python3
"""Partition native experiment grids without pricing, ranking or dropping choices."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment


def axes_of(document, prefix=()):
    axes = []
    for key, value in sorted(document.items()):
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


def plans_for(spec, max_runs):
    loads = len(spec["scenario"]["concurrency"])
    if loads == 0 or max_runs < loads:
        raise ValueError("a shard must fit every declared load for at least one configuration")
    plans = []
    for sweep in spec["sweeps"]:
        axes = axes_of(sweep)
        probe = copy.deepcopy(sweep)
        for path, values in axes:
            assign(probe, path, values[:1])
        kept, dropped = run_experiment.expand_sweep(probe, spec["scenario"]["gpus"])
        if len(kept) + len(dropped) != 1:
            raise ValueError("the native expansion is not a one-point Cartesian baseline")
        for path, values in axes:
            varied = copy.deepcopy(probe)
            assign(varied, path, values)
            kept, dropped = run_experiment.expand_sweep(varied, spec["scenario"]["gpus"])
            if len(kept) + len(dropped) != len(values):
                raise ValueError(f"axis does not expand natively: {'.'.join(path)}")
        remaining = max_runs // loads
        partitions = []
        for path, values in axes:
            width = min(len(values), remaining)
            partitions.append((path, [values[start:start + width]
                                      for start in range(0, len(values), width)]))
            remaining //= width
        plans.append({"partitions": partitions,
                      "shards": math.prod(len(chunks) for _, chunks in partitions),
                      "raw_points": loads * math.prod(len(values) for _, values in axes)})
    return plans


def compile_shard(spec, index, max_runs, plans=None):
    plans = plans if plans is not None else plans_for(spec, max_runs)
    total = sum(plan["shards"] for plan in plans)
    if index < 0 or index >= total:
        raise IndexError(f"shard index {index} outside [0, {total})")
    local_index = index
    for sweep_index, plan in enumerate(plans):
        if local_index >= plan["shards"]:
            local_index -= plan["shards"]
            continue
        sweep = copy.deepcopy(spec["sweeps"][sweep_index])
        coordinate = local_index
        for path, chunks in reversed(plan["partitions"]):
            coordinate, selected = divmod(coordinate, len(chunks))
            assign(sweep, path, chunks[selected])
        kept, dropped = run_experiment.expand_sweep(sweep, spec["scenario"]["gpus"])
        if (len(kept) + len(dropped)) * len(spec["scenario"]["concurrency"]) > max_runs:
            raise ValueError("native expansion exceeds the shard limit")
        output = copy.deepcopy(spec)
        output["name"] = f"{spec['name']}-shard-{index:012d}"
        output["sweeps"] = [sweep]
        return output
    raise AssertionError("unreachable shard index")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--max-runs", type=int, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    with args.spec.open(encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    plans = plans_for(spec, args.max_runs)
    if args.index is None:
        print(json.dumps({"shards": sum(plan["shards"] for plan in plans),
                          "raw_points": sum(plan["raw_points"] for plan in plans),
                          "max_runs_per_shard": args.max_runs}, sort_keys=True))
        return
    if args.out is None:
        parser.error("--out is required with --index")
    output = compile_shard(spec, args.index, args.max_runs, plans)
    with args.out.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)


if __name__ == "__main__":
    main()
