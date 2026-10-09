# sample-run (real end-to-end run)

A **real** run directory produced by `config-search` on the pinned stack (BLIS `kernel-exclusive`
PR #1851, blis-catalog `0.2.1`), not synthetic. Scenario: GLM-5.3-Flash on 16x H100, chatbot, hold
ITL p99 <= 20 ms (>= 50 tok/s/user), headroom 0.8. It is the committed **seam fixture** the report
step builds against.

Files:

- `scenario.json` -- the resolved Scenario that was searched.
- `runs.jsonl` -- 40 BLIS runs (probe / steer / finalist), full metrics object per run. 8 of them are
  the infeasible `4x TP4 + EP` topology BLIS refused (the feasibility filter), recorded with `error`.
- `pareto.json` -- finalists with C* / C_use / curves, the dropped topology, and the chosen winner
  with its move trail.

What the search found: `4x TP4 + EP` was dropped as infeasible; `2x TP8 + EP` reached C*=128 and
`1x TP8 + EP` C*=64 at the ITL floor; the binding signal was a decode-compute ceiling (ITL pinned
~20 ms, no preemptions), so the steering move enabled MTP-3 speculative decoding, which the simulator
predicts ~3x the output throughput at the same interactivity -- winning both the budget-max and
per-GPU axes.

These are **simulator predictions**; the flat ~19-20 ms ITL across a wide concurrency range and the
spec-decode gain are exactly the kind of thing to spot-check against the real llm-d measurements in
`curvebender-tools/reports` before acting on them. The layout is convention-based and expected to
evolve -- a starting contract, not a frozen schema.
