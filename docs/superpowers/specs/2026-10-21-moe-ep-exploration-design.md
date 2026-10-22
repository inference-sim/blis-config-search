# MoE expert-parallel exploration + a two-part report explorer

Date: 2026-10-21

## Problem

`reports/deepseek-v4-pro-h200-chatbot/` recommends `32x TP8` on one finalist out of 28 runs, 21 of
which failed. It never tried expert parallelism, even though the skill advertises EP as a searched
axis.

The report was not written by the agentic search. Its `move_trail` says "enumerated single-node
topologies within budget" — that is `scripts/run_search.py`, the deterministic mini-search behind
the Submit button. Its enumeration (`run_search.py:36`) varies exactly one axis:

```python
for tp in VALID_TP:            # [1, 2, 4, 8]
    out.append({"tp": tp, "num_instances": gpu_budget // tp})
```

No `enable_expert_parallel`, no `dp`, for any model. So EP was unreachable from the Submit button
and from CI, for every MoE model.

### Why this matters more than a missing knob

Measured on the pinned binary (deepseek-v4-pro, h200, 4 instances, `num_requests=120`, c=32):

| shape | tok/s | ITL p99 |
|---|---|---|
| TP8, EP off | 1556.3 | 31.5 ms |
| TP8, **EP on** | **1923.7** | 31.5 ms |
| TP4xDP2, EP off | 1372.2 | 19.3 ms |
| TP4xDP2, **EP on** | **1664.6** | **16.3 ms** |
| TP2xDP4, EP on | 1315.7 | 19.4 ms |
| TP1xDP8, EP on | 955.8 | 24.9 ms |

EP is worth ~24% throughput, sometimes with a better tail. The reported winner is the weaker arm of
the knob the search never turned.

The 21 failures share one cause:

> `the kernel reports 191.2 GiB of per-rank fixed occupancy on a 141.0 GiB device
> (deepseek-v4-pro, tp=4, expert-parallel width 1)`

`expert-parallel width 1` is the diagnosis. Width is derived as `tp x pcp x dp`
(`blis-schemas@v0.2.2/spec/deployment/deployment.go:109`), so `tp=4` alone is width 1 and the
experts do not shard. The fix is footprint, not EP alone: `tp=4,dp=2` runs with EP *and* without it
(1372.2 tok/s). The missing axis is `dp` as much as `enable_expert_parallel` — what failed was every
shape with `tp*dp < 8` for a 384-expert model.

`run_blis.py` already supports both keys correctly; only the automated enumeration omits them.

## Design

### Part 1 — `scripts/topologies.py` (new, shared)

One enumeration module for both paths, so they cannot drift again. Input: `gpu_budget` and the
catalog model entry. Output: candidate dicts in a stable order.

**Dense** — unchanged: `tp in {1,2,4,8}` dividing the budget, `dp=1`, no EP. Necessary, not
conservative: BLIS rejects `dp=2` on qwen3-14b with *"states dp 2 for a dense model; DP placement is
modelled for MoE"*.

**MoE** — cross `tp x dp` footprints fitting one node (`tp*dp <= 8`) against
`enable_expert_parallel in {false, true}`, filtered to footprints the budget divides. EP-on
candidates whose derived width (`tp*dp`) would be 1 are skipped rather than run, matching the
schema's own warning that such a layout shards nothing.

MoE/dense comes from the catalog's `arch` field, which `build_catalog.py:137` derives from
`graph.yaml` `layer_kinds` — authoritative, never a name guess.

**Excluded by choice:** `all2all_backend`. The kernel does price it (allgather 1127.7 vs DeepEP
1105.0 tok/s on this model) but 2% against EP's 24% does not earn a doubling of the grid. It is
wired as a report facet so a search that *does* sweep it displays correctly.

### Part 2 — `run_search.py` consumes it

Replace `candidate_topologies`. Two defects to fix in the same pass:

1. `MAX_CANDIDATES = 8` truncates a list sorted by `-tp`, which would silently drop the EP arms.
   Ranking becomes feasibility-aware, keeping both EP arms of the widest footprints.
2. The fixed ramp `[1,2,4,8,16,24,32]` with `num_requests=80` saturates. Measured: c=128 and c=256
   return byte-identical throughput (5116.8 tok/s) because the request pool runs dry, so a large
   budget plateaus on an artifact rather than a real ceiling. Scale the ramp with the budget and
   `num_requests` with concurrency.

`pareto.json` records the EP decision in `move_trail`, and `dropped[]` gets populated — the failed
shapes should appear with their kernel error, not vanish.

### Part 3 — SKILL.md

Make the MoE knob tier explicit (EP arms, `tp x dp` footprints, the derived-width rule and the
memory failure it causes) so the agentic path prescribes what Part 1 enumerates.

### Part 4 — Explorer: histograms + MoE facets

`report/template.html` today has only click facets (9 dims, no brushing). Add:

- **Brushable histograms** over `pu` (per-user tok/s), `tpg` (output per GPU), `ttft`, `itl` — a row
  of small SVG histograms with d3 brushes feeding the same crossfilter selection as the facets.
  This is the reference report's "Filter by outcome: brush a range".
- **Facets** `epw`, `dp`, `a2a` in `ALL_DIMS`. The existing `varies()` guard auto-hides them on
  dense runs, which is already correct.
- **`tpg` in `build_records`** — output per GPU is a brush axis and today only `cl` and `gpus` exist
  separately.

The null-tolerance discipline already in the explorer (review F4) extends to the brush code: a null
metric must not place a point or break the page.

## Testing

TDD throughout; `test_run_blis.py` and `test_report.py` already exist.

- `topologies.py`: dense excludes EP and `dp`; MoE yields both EP arms; derived-width-1 skipped;
  budget divisibility; stable order.
- `build_records`: `tpg` present and null-safe.
- Determinism (INV-6): identical run-dir data still renders byte-identical HTML.
- Then regenerate the DeepSeek report from real runs and compare against the `32x TP8` claim.

## Expected outcome

`32x TP8 + EP` should displace the current winner (~24% more throughput at an equal-or-better
tail), and the finalist count should go from 1 to roughly 6-8.

## Risks

- Regenerating committed reports rewrites `reports/deepseek-v4-pro-h200-chatbot/` and any other
  re-run scenario. The working tree already has uncommitted changes there; confirm before
  overwriting.
- Runs are ~0.15s each, so the wider grid costs nothing meaningful in wall-clock.
