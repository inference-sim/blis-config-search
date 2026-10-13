# sample-run (real kernel run)

A **real** run directory produced by `config-search` on the pinned stack, priced by the
**blis-latency-kernel** backend (BLIS v0.10.0 @ 0d092d4, kernel v0.1.0, blis-catalog 0.2.1,
blis-registry v0.1.1). Scenario: GLM-5.3-Flash on 16x H100, chatbot, 200 requests, hold ITL p99
<= 20 ms (>= 50 tok/s/user), headroom 0.8. It is the committed **seam fixture** the report step
builds against.

Files:

- `scenario.json` -- the resolved search Scenario (one pinned workload for every run).
- `runs.jsonl` -- 27 kernel-priced runs (probe / bisect / finalist). Each row carries `config`, the
  resolved `deployment` (`parallel`/`engine`/`num_instances`/`gpus`/`scenario`), `metrics` (cluster
  aggregate), `metrics_all_instances` (per-instance, for idle-pool/imbalance checks), and `error`.
- `pareto.json` -- finalists with `c_star`/`c_use` and an enriched `curve[]` per point
  (`tokens_per_sec`, `itl_p99_ms`, `ttft_p99_ms`, `scheduling_delay_p99_ms`, `preemption_count`, and
  **`tok_per_s_per_user`**), plus `dropped[]` and `winner{reason, move_trail}`.

What the search found: topologies were priced on the kernel via generated scenarios and bisected to
C* at the 50 tok/s/user floor. The kernel shows a **sharp interactivity cliff** for this model, so C*
is small (1x TP8+EP C*=6, 2x TP8+EP C*=12, 4x TP4+EP C*=20). The winner on both the budget-max and
per-GPU axes is **4x TP4 + EP** (~1264 tok/s, 79/GPU at C_use=16).

> **Provenance caveat on `pareto.json` (4x TP4+EP `c_star`).** The engine's capacity probe bisects on
> the **cluster-aggregate** ITL, which stays under 20 ms through concurrency 25, so a fresh
> engine-driven search writes `c_star: 25` here. The report step (`report.py`), however, checks ITL
> **per instance** (`metrics_all_instances`), and at conc 24/25 one of the four instances is already
> at ~55 ms — so those points are not actually servable. To keep this committed fixture
> self-consistent with the (correct) per-instance report logic, `4x TP4+EP`'s `c_star` was
> **hand-corrected 25 → 20** (and `c_use` 20 → 16, `at_c_use` taken from the conc-16 run). This is the
> one hand-edit in the directory; everything else is the raw engine output. The durable fix is
> engine-side — teach the capacity probe (and `_completion_issue`) to check `metrics_all_instances`
> the way `report.py` does, then regenerate this fixture from a real run. Until then the report and a
> freshly-generated `pareto.json` would disagree on this one `c_star`.

**On `tok_per_s_per_user` (why it is a field, not `1000/itl_p99_ms`):** BLIS appends an ITL entry per
decode *step*, not per emitted token, so under speculative decoding/MTP one entry covers several
tokens and `1000/itl_p99` understates the real per-user rate. The brain emits `tok_per_s_per_user`
per point; the report plots that, not a derived axis.

These are **simulator predictions**. The kernel prices MLA and hybrid (KDA) attention from fitted
per-operator coefficients (these runs are clean -- no step-time warnings); spot-check against the real
llm-d measurements in `curvebender-tools/reports` before acting. `stderr_tail` is retained per run
(empty on a clean run) for any diagnostics BLIS does emit. The layout is convention-based and expected
to evolve.
