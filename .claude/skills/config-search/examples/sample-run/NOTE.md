# sample-run (real kernel run)

A **real** run directory produced by `config-search` on the pinned stack, priced by the
**blis-latency-kernel** backend (BLIS PR #1851 @ 99bec1b6, kernel v0.1.0, blis-catalog 0.2.1,
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
is small (1x TP8+EP C*=6, 2x TP8+EP C*=12, 4x TP4+EP C*=25). The winner on both the budget-max and
per-GPU axes is **4x TP4 + EP** (~1472 tok/s, 92/GPU at C_use=20).

**On `tok_per_s_per_user` (why it is a field, not `1000/itl_p99_ms`):** BLIS appends an ITL entry per
decode *step*, not per emitted token, so under speculative decoding/MTP one entry covers several
tokens and `1000/itl_p99` understates the real per-user rate. The brain emits `tok_per_s_per_user`
per point; the report plots that, not a derived axis.

These are **simulator predictions**, and for GLM-5.3-Flash the kernel's step-time is **pessimistic**
for MLA and hybrid (KDA) attention (BLIS warns about both on stderr). Spot-check against the real
llm-d measurements in `curvebender-tools/reports` before acting. The layout is convention-based and
expected to evolve.
