/* Shared mock dataset for blis-config-search gallery mockups.
 *
 * ONE real scenario (glm53flash-h100-chatbot, from the committed sample-run fixture) plus
 * several CLEARLY-SYNTHETIC scenarios so the crossfilter has something to filter. Synthetic
 * rows are marked `synthetic:true` and must be rendered with a "mock" affordance in any
 * mockup — we never want a fake prediction to read as a real BLIS result.
 *
 * Each scenario mirrors what the gallery would read out of a run directory:
 *   scenario.json (model, hardware, gpu_budget, workload, floor, headroom)
 *   pareto.json   (winner + finalists, each with c_use operating point)
 * The gallery links to <run_id>/index.html (the self-contained report.py output).
 */
window.SCENARIOS = [
  {
    run_id: "glm53flash-h100-chatbot-kernel-0001",
    synthetic: false,
    model: "glm-5.3-flash",
    hardware: "H100",
    gpu_budget: 16,
    workload: "chatbot",
    isl_osl: "ShareGPT",
    floor: "50 tok/s/user (ITL p99 ≤ 20 ms)",
    headroom: 0.8,
    n_runs: 27,
    n_servable: 17,
    generated: "2026-10-18",
    winner: {
      label: "4× TP4 + EP",
      tp: 4, ni: 4, ep: true, quant: "fp8",
      gpus: 16, c_use: 16, c_star: 20,
      tps: 1263.5, tps_per_gpu: 79.0, per_user: 74.2, itl_p99: 13.48,
    },
    // all finalists' usable operating points, for cross-scenario views
    finalists: [
      { label: "1× TP8 + EP", tp: 8, ni: 1, gpus: 8,  c_use: 5,  tps: 511.6,  tps_per_gpu: 64.0, per_user: 113.1, itl_p99: 8.84 },
      { label: "2× TP8 + EP", tp: 8, ni: 2, gpus: 16, c_use: 10, tps: 983.6,  tps_per_gpu: 61.5, per_user: 101.7, itl_p99: 9.83 },
      { label: "4× TP4 + EP", tp: 4, ni: 4, gpus: 16, c_use: 16, tps: 1263.5, tps_per_gpu: 79.0, per_user: 74.2,  itl_p99: 13.48 },
    ],
    reason: "Most output tok/s on the budget while holding ≥50 tok/s/user on EVERY instance.",
  },
  {
    run_id: "qwen3-14b-h100-chatbot-0002",
    synthetic: true,
    model: "qwen3-14b",
    hardware: "H100",
    gpu_budget: 8,
    workload: "chatbot",
    isl_osl: "ShareGPT",
    floor: "50 tok/s/user (ITL p99 ≤ 20 ms)",
    headroom: 0.8,
    n_runs: 22,
    n_servable: 14,
    generated: "2026-10-19",
    winner: { label: "4× TP2", tp: 2, ni: 4, ep: false, quant: "bf16", gpus: 8, c_use: 24, c_star: 30, tps: 1842.0, tps_per_gpu: 230.3, per_user: 68.0, itl_p99: 14.7 },
    finalists: [
      { label: "2× TP4", tp: 4, ni: 2, gpus: 8, c_use: 14, tps: 1190.0, tps_per_gpu: 148.8, per_user: 88.0, itl_p99: 11.4 },
      { label: "4× TP2", tp: 2, ni: 4, gpus: 8, c_use: 24, tps: 1842.0, tps_per_gpu: 230.3, per_user: 68.0, itl_p99: 14.7 },
      { label: "8× TP1", tp: 1, ni: 8, gpus: 8, c_use: 20, tps: 1610.0, tps_per_gpu: 201.3, per_user: 59.0, itl_p99: 17.2 },
    ],
    reason: "Mock scenario — narrowest TP that still fits KV spreads load across four replicas.",
  },
  {
    run_id: "glm53flash-h200-agentic-0003",
    synthetic: true,
    model: "glm-5.3-flash",
    hardware: "H200",
    gpu_budget: 8,
    workload: "agentic",
    isl_osl: "8k / 512",
    floor: "50 tok/s/user (ITL p99 ≤ 20 ms)",
    headroom: 0.8,
    n_runs: 18,
    n_servable: 9,
    generated: "2026-10-19",
    winner: { label: "2× TP4 + EP", tp: 4, ni: 2, ep: true, quant: "fp8", gpus: 8, c_use: 12, c_star: 15, tps: 940.0, tps_per_gpu: 117.5, per_user: 72.0, itl_p99: 13.0 },
    finalists: [
      { label: "1× TP8 + EP", tp: 8, ni: 1, gpus: 8, c_use: 6,  tps: 540.0, tps_per_gpu: 67.5,  per_user: 108.0, itl_p99: 9.0 },
      { label: "2× TP4 + EP", tp: 4, ni: 2, gpus: 8, c_use: 12, tps: 940.0, tps_per_gpu: 117.5, per_user: 72.0,  itl_p99: 13.0 },
    ],
    reason: "Mock scenario — H200 KV headroom lets two TP4 instances share the agentic load.",
  },
  {
    run_id: "qwen3-14b-h200-summarization-0004",
    synthetic: true,
    model: "qwen3-14b",
    hardware: "H200",
    gpu_budget: 4,
    workload: "summarization",
    isl_osl: "16k / 256",
    floor: "40 tok/s/user (ITL p99 ≤ 25 ms)",
    headroom: 0.8,
    n_runs: 15,
    n_servable: 11,
    generated: "2026-10-20",
    winner: { label: "2× TP2", tp: 2, ni: 2, ep: false, quant: "bf16", gpus: 4, c_use: 18, c_star: 22, tps: 1060.0, tps_per_gpu: 265.0, per_user: 52.0, itl_p99: 18.9 },
    finalists: [
      { label: "1× TP4", tp: 4, ni: 1, gpus: 4, c_use: 9,  tps: 610.0,  tps_per_gpu: 152.5, per_user: 71.0, itl_p99: 13.8 },
      { label: "2× TP2", tp: 2, ni: 2, gpus: 4, c_use: 18, tps: 1060.0, tps_per_gpu: 265.0, per_user: 52.0, itl_p99: 18.9 },
    ],
    reason: "Mock scenario — long-input summarization is prefill-heavy; replicas win on throughput.",
  },
  {
    run_id: "glm53flash-h100-agentic-0005",
    synthetic: true,
    model: "glm-5.3-flash",
    hardware: "H100",
    gpu_budget: 32,
    workload: "agentic",
    isl_osl: "8k / 512",
    floor: "50 tok/s/user (ITL p99 ≤ 20 ms)",
    headroom: 0.8,
    n_runs: 34,
    n_servable: 20,
    generated: "2026-10-20",
    winner: { label: "8× TP4 + EP", tp: 4, ni: 8, ep: true, quant: "fp8", gpus: 32, c_use: 32, c_star: 40, tps: 2510.0, tps_per_gpu: 78.4, per_user: 71.0, itl_p99: 13.9 },
    finalists: [
      { label: "4× TP8 + EP", tp: 8, ni: 4, gpus: 32, c_use: 20, tps: 1960.0, tps_per_gpu: 61.3, per_user: 99.0, itl_p99: 10.1 },
      { label: "8× TP4 + EP", tp: 4, ni: 8, gpus: 32, c_use: 32, tps: 2510.0, tps_per_gpu: 78.4, per_user: 71.0, itl_p99: 13.9 },
    ],
    reason: "Mock scenario — doubling the budget scales the 4× TP4 winner linearly to eight instances.",
  },
  {
    run_id: "qwen3-14b-h100-codegen-0006",
    synthetic: true,
    model: "qwen3-14b",
    hardware: "H100",
    gpu_budget: 16,
    workload: "codegen",
    isl_osl: "2k / 2k",
    floor: "60 tok/s/user (ITL p99 ≤ 16 ms)",
    headroom: 0.8,
    n_runs: 26,
    n_servable: 12,
    generated: "2026-10-20",
    winner: { label: "8× TP2", tp: 2, ni: 8, ep: false, quant: "bf16", gpus: 16, c_use: 40, c_star: 50, tps: 3120.0, tps_per_gpu: 195.0, per_user: 64.0, itl_p99: 15.2 },
    finalists: [
      { label: "4× TP4", tp: 4, ni: 4, gpus: 16, c_use: 24, tps: 2180.0, tps_per_gpu: 136.3, per_user: 78.0, itl_p99: 12.1 },
      { label: "8× TP2", tp: 2, ni: 8, gpus: 16, c_use: 40, tps: 3120.0, tps_per_gpu: 195.0, per_user: 64.0, itl_p99: 15.2 },
    ],
    reason: "Mock scenario — balanced 2k/2k codegen rewards many narrow replicas under a tight floor.",
  },
];
