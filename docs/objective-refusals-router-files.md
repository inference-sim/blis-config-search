# The Pareto objective, refused candidates, and the router's files

How a sweep ranks, what it says about a candidate the simulator declines to run, and where its
router configuration comes from. Each section states the rule, the inputs, the commands to
reproduce, an expected output, the cost and the limits.

## Contents

- [The Pareto objective](#the-pareto-objective)
- [Refused candidates](#refused-candidates)
- [The router's reference files](#the-routers-reference-files)
- [Simulator switches and omitted engine knobs](#simulator-switches-and-omitted-engine-knobs)
- [What this does not show](#what-this-does-not-show)

## The Pareto objective

**Rule.** Both axes are the metric registry's definitions (design DEV-8), read from the Result's
measured window, not defined here:

| Axis | Registry metric | Meaning |
|---|---|---|
| x, tokens/s per user | `interactivity_of_mean_tpot` | 1000 / mean TPOT in ms (InferenceX `mean_intvty`). Not the mean of per-request 1/TPOT, which is `interactivity_mean` and reads 2 to 27 percent higher. |
| y, tokens/s per GPU | `total_tokens_per_second_per_gpu` | (input + output tokens)/s over all GPUs. `throughput: total`, the default. |
| y, decode form | `output_tokens_per_second_per_decode_gpu` | output tokens/s over the decode GPUs (all GPUs when there is no decode pool). `throughput: decode`. |

The Result's own `interactivity_of_mean_tpot` is the only source (a definition is read from the
registry, never rebuilt from another field). A Result that does not state it leaves the run unrated:
it is counted (`unrated`) and not placed on the front.

Only runs that held the scenario's SLOs are points; the SLO gate is a separate constraint. A point
is dominated when another is at least as good on both axes and better on one. The non-dominated set
is the front. It is computed for every sweep and for all sweeps together (a P/D sweep and an
aggregated sweep are compared in the all-sweeps front), for both throughput forms, and written to
`pareto.json` under `front`.

**No tie rule.** Points are compared exactly. Preferring P/D when it "does no worse" requires knowing
when two deployments are not distinguished, which needs paired replications under common random
numbers, an interval on the difference, three outcomes and control of the false-winner rate over all
comparisons (design DEV-11). One run per arm gives no interval, and a fixed relative tolerance would
copy another repository's noise figure (X7). So nothing displaces a point; `objective.tie_tolerance`
is an error that says so. The front has no interval and says so (`pareto.interval: null`). The P/D
preference is not delivered here; it arrives with the replications.

**Which ranking headlines.** `objective.ranking` is `scalar` (default: the previous rule, one
finalist per sweep by output tok/s per accelerator) or `pareto` (the finalists are the all-sweeps
front's members; the card's winner is the member with the most throughput per GPU, the throughput
end of the front, and the front itself is the result). `pareto.json` and the report say which was
used under `ranking`.

```yaml
objective: {ranking: pareto, throughput: total}
```

**Worked example.** Aggregated A (interactivity 100, 100 tok/s/GPU), aggregated B (99, 105) and P/D P
(99, 99): A and B are the front; P is dominated by B and is not a member.

**Reproduce.** `python3 -I scripts/test_experiment.py ParetoFront` (hermetic).

**Cost.** Quadratic in the number of servable rows per sweep; thousands of rows take well under a
second.

## Refused candidates

A run ends as one of: `served`, `slo_miss` (it ran and missed an SLO), `refused`, `overloaded` (it
ran but did not finish its requests), `timeout`, `crash` (a panic or an exit with no fatal log line),
`invalid_result`. A refusal is the simulator's fatal log line (exit 1, no Result), or this runner
declining a candidate it can show the engine would refuse (pool disagreement, a footprint that
straddles a node, a switch on the wrong deployment); the second kind never reaches the simulator.

A refused run is not evidence about speed. It is counted apart (`totals.refused`,
`totals.configs_refused`, per-row `outcome` and `refusal`), grouped by reason in `refusals`
(reason text with the instance id and generated scenario file name removed, the issue URL if the
message cites one, the number of runs and of distinct candidates) in `experiment.json` and
`pareto.json`, and named in the report's hero text. When every run is refused the text says the
simulator refused every configuration; it no longer reads as "no configuration held the SLOs".

The validator rejects a sweep when the runner would refuse every arm, and notes a sweep where
some are (`validate_experiment.py` prints `note:` lines). Refusals only the engine can make (for
example a connector beside a sliding-window cache group, which depends on the model) are found by
running, and appear as `refused` rows with the engine's reason. The simulator gives no reason code,
so the reason is its message.

A run with no winner publishes. `scenarios.json` carries `winner: null`, `state` (`all_refused` when
every run was refused, else `no_winner`), the outcome counts and the refusal reasons, and the
single-page app shows a card that says which. The all-sweeps front (`front`, exact, no interval) is
published for a run that has a winner. The explorer draws no front of its own.

**Reproduce.** `python3 -I scripts/test_experiment.py Refusals GalleryOfTheRunsOutcome`.

## The router's reference files

The router fork publishes complete EndpointPicker files and an index of their knobs
(`reference-configs/policies.json`, llm-d-router pull request 7). Config search keeps no copy: a
spec names one of those files, and `pins.env` pins the commit (`ROUTER_CONFIGS_SHA`). Fetch it
with `sh scripts/clone_pinned.sh "$ROUTER_CONFIGS_SHA" https://github.com/inference-sim/llm-d-router.git ../llm-d-router`.
A checkout at another commit is reported, and every report records the pin in `pinned_stack`.

With no override the arm runs the router's file byte for byte, staged in `reports/<name>/routing/`.
With overrides the file is parsed, edited and written back there, named by a hash of its bytes
(comments and key order of the router's file are not kept). Every run row records `router`: the
base file, its sha256 and commit, the overrides, and the file the run read with its sha256.

Every knob in the index's `knobs` (each marked `hosted_effect: true`, proven to move a decision in
the hosted boundary) is reachable: a scorer `weight` in a profile entry
(`schedulingProfiles.<profile>.<plugin>.weight`) and a plugin parameter whether or not the file sets
it (`plugins.<plugin>.parameters.<name>`, adding the plugin if the file lacks it). A parameter under
`ineffective_parameters` (`hosted_effect: false`: `maxNumOfEndpoints`, `addEstimatedOutputTokens`,
`prefixMatchInfoProducerName`, `maxPrefixBlocksToMatch`) is refused by name, "accepted by the router,
no effect in the hosted boundary", with the index's reason; any other unlisted name is refused as
unlisted. Both refusals happen in the validator, before any run.

The P/D decider is not an override. The index names the complete variant
(`pd_decider.complete_variant`, `pd-prefix-based`): replacing only the decider's name makes the file
fail to load, and the variant scores differently (prefix-cache 2, queue 1) from the files it
replaces. A sweep compares policies by listing the files, each run unedited; `router.decider` is
rejected with that instruction. The local `routing/pd-prefix.yaml` is deleted: it set
`nonCachedTokens: 256` where the router's file sets 16, so existing arms that did not sweep the
threshold now run at 16.

**Reproduce.** `ROUTER_CONFIGS_DIR=../llm-d-router python3 -I scripts/test_experiment.py RealRouterConfigs`
applies an override for every `knobs` entry of every file in the index and checks it landed, and checks that every `ineffective_parameters` entry is refused by name.

**Cost.** Parsing and writing a file takes milliseconds per arm.

## Simulator switches and omitted engine knobs

`sweeps[].simulator.pd_transfer_contention: [false, true]` sweeps `--pd-transfer-contention`
(concurrent KV handoffs share the fabric links; default off, so a handoff is priced as if alone).
The flag takes no value: it is passed when true and left off when false. The simulator refuses it
on a colocated run, so the validator and the runner do too.

An omitted engine knob takes the engine's default, as the spec guide always said. The runner used to
substitute its own `max_num_seqs`, `max_num_batched_tokens`, `cudagraph_mode` and
`gpu_memory_utilization`; it now supplies only `max_model_len` (32768), which the engine requires.
Checked with the pinned simulator on one replay-free run: the engine runs and logs the defaults it
applied as `engine assumption` lines (for example `cudagraph_mode = FULL_AND_PIECEWISE`), and the
tokens/s differ by about 1.5 percent from the old constants.

## What this does not show

- The front is as good as the runs under it: a coarse concurrency list gives a coarse front, and
  with one run per arm there is no interval, so no claim that two deployments differ or tie.
- It does not say a refused P/D arm would have won; it says nothing about it.
- A simulator build whose Result does not state `total_tokens_per_second_per_gpu` (the build
  `BLIS_SHA` names today is one) gives rows without the figure. The front is then empty, the
  `unrated` count says how many runs could not be placed, and the report says so instead of
  reporting a missed SLO. The decode form and `interactivity_of_mean_tpot` have the same rule
  (interactivity falls back to `tpot_mean_ms`; the throughput forms have no fallback).
- Committed reports were not regenerated here; the regeneration bot does that at the next pin move.

Data and revisions: the metric definitions are the substrate registry's (`contracts/metrics`); the
noise floor is the accuracy dashboard's `docs/limits.md`; the router files are llm-d-router at the
commit in `pins.env`.
