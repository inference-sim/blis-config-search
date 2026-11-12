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

When the Result states `interactivity_of_mean_tpot` it is used. Otherwise it is computed from the
Result's `tpot_mean_ms` by the registry's formula, and the row says which (`interactivity_basis`).
Example from an AgentX replay: `tpot_mean_ms` 8.7707 gives 114.0; the Result's `interactivity_mean`
was 114.9.

Only runs that held the scenario's SLOs are points; the SLO gate is a separate constraint. A point
is dominated when another is at least as good on both axes and better on one. The non-dominated set
is the front. It is computed for every sweep and for all sweeps together (a P/D sweep and an
aggregated sweep are compared in the all-sweeps front), for both throughput forms, and written to
`pareto.json` under `front`.

**Tie rule.** The team prefers P/D when it ties an aggregated deployment. Two measurements closer
than the run-to-run noise are not distinguished, so P/D point P ties aggregated front point A when
`P.x >= (1 - tol) * A.x` and `P.y >= (1 - tol) * A.y`. Then P displaces A: A is listed under
`displaced`, and P, if it is not on the exact front, is a member with `status: tie_preferred`. Two
aggregated points, or two P/D points, are compared exactly. `tol` is the spec field
`objective.tie_tolerance`, default 0.016: the accuracy dashboard's documented fallback noise floor
(1.6 percent), to be replaced when repeat noise is measured. A P/D point that is itself dominated by
another aggregated point beyond the tolerance can still be promoted by tying A; it is reported with
the points that displace it so the reader can see this.

**Which ranking headlines.** `objective.ranking` is `scalar` (default: the previous rule, one
finalist per sweep by output tok/s per accelerator) or `pareto` (the finalists are the all-sweeps
front's members; the card's winner is the member with the most throughput per GPU, the throughput
end of the front, and the front itself is the result). `pareto.json` and the report say which was
used under `ranking`.

```yaml
objective: {ranking: pareto, throughput: total, tie_tolerance: 0.016}
```

**Worked example.** Aggregated (interactivity 100, 1000 tok/s/GPU) and P/D (99, 990): P/D is 1 percent
below on both axes, inside 1.6 percent, so the front is the P/D point alone. At (90, 990) it is
not, and the front is the aggregated point.

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

**Reproduce.** `python3 -I scripts/test_experiment.py Refusals`.

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
  with one seed the tolerance, not a confidence interval, decides ties.
- It does not say a refused P/D arm would have won; it says nothing about it.
- `interactivity_of_mean_tpot` derived from `tpot_mean_ms` equals the Result's figure only when both
  use the same request population (at least two output tokens); a Result that states the figure is
  preferred.
- A simulator build whose Result does not state `total_tokens_per_second_per_gpu` (the build
  `BLIS_SHA` names today is one) gives rows without the figure. The front is then empty, the
  `unrated` count says how many runs could not be placed, and the report says so instead of
  reporting a missed SLO. The decode form and `interactivity_of_mean_tpot` have the same rule
  (interactivity falls back to `tpot_mean_ms`; the throughput forms have no fallback).
- Committed reports were not regenerated here; the regeneration bot does that at the next pin move.

Data and revisions: the metric definitions are the substrate registry's (`contracts/metrics`); the
noise floor is the accuracy dashboard's `docs/limits.md`; the router files are llm-d-router at the
commit in `pins.env`.
