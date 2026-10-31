// Data contracts for the config-search webpage. These mirror exactly what
// scripts/build_gallery.py + build_catalog.py emit and web/scripts/stage-data.mjs stages
// into web/public/data/. Keep in sync with those producers.

export interface Winner {
  /** Topology only, e.g. "64x TP4xDP2 + EP" — several configs can share this. */
  label: string
  /** Knob delta from this topology's defaults, e.g. "least-loaded routing"; "" at defaults. */
  variant?: string
  /** label + variant — what to DISPLAY, since label alone does not identify a config. */
  name?: string
  tp: number | null
  ni: number | null
  ep?: boolean
  quant?: string | null
  gpus: number | null
  /** The RECOMMENDED load, and always the load whose measurement the figures below come from. */
  c_use: number | null
  /** The 0.8xC* target, present only when the recommendation backed off below it because that
   *  load measured under the interactivity floor. */
  c_use_target?: number | null
  c_star?: number | null
  tps: number
  tps_per_gpu: number
  per_user: number
  itl_p99: number
}

export interface Finalist {
  /** Topology only; see Winner.label — a finalist is identified by label + variant. */
  label: string
  variant?: string
  name?: string
  tp: number | null
  ni: number | null
  gpus: number | null
  /** The RECOMMENDED load, and always the load the figures were measured at. */
  c_use: number | null
  /** The 0.8xC* target, when the recommendation backed off below it. */
  c_use_target?: number | null
  tps: number
  tps_per_gpu: number
  per_user: number
  itl_p99: number
}

export interface Scenario {
  run_id: string
  synthetic: boolean
  model: string
  hardware: string
  gpu_budget: number
  workload: string
  isl_osl: string
  /** A caveat about the DATA itself (not the recommendation): real BLIS runs whose measurement
   *  does not match what the scenario claims to measure. Empty for a scenario with none. */
  caveat?: string
  floor: string
  headroom: number | null
  n_runs: number
  n_servable: number
  n_records?: number
  generated: string
  winner: Winner
  finalists: Finalist[]
  reason: string
  // Per-run records for the detail explorer, inlined by stage-data.mjs (no separate fetch).
  records: RunRecord[]
}

// One run row for the per-scenario detail explorer (report.build_records projection).
export interface RunRecord {
  id: string
  src: string
  conc: number | null
  tp: number | null
  dp: number | null
  ni: number | null
  gpus: number | null
  series: string
  route: string
  pd: string
  /** Draft depth K (0 = speculative decoding off). */
  mtp: number
  /** The speculative method the model drafted with (dspark, mtp, eagle3, …); null when off. */
  spec?: string | null
  /** Acceptance length the run was priced with: mean tokens accepted per verification step. */
  al?: number | null
  /** Where that acceptance was measured, e.g. "inferencex-golden-al (b200)". */
  al_src?: string | null
  /** How `pu` was obtained: "" direct, "derived-from-acceptance" (AL-scaled — the correct
   *  per-token rate under drafting), "step-estimate-withheld" (drafting on, AL unknown). */
  pu_basis?: string
  mnbt: number | null
  mns: number | null
  epw?: number | string | null
  a2a?: string | null
  pu: number | null
  cl: number | null
  itl: number | null
  ttft: number | null
  hit: number | null
  p95: number | null
  ok: boolean
  status: string
  viol: string[]
  // --- deployment detail, for the click-to-inspect panel. build_gallery stages all of these;
  // the web app simply had no type for them. Optional because an older run directory may lack
  // any given field, and the panel must render what it has rather than assume.
  /** Knob delta from this topology's defaults, e.g. "least-loaded routing"; "" at defaults. */
  variant?: string
  /** Which knobs varied in this run's topology group, raw values (drives the panel's knob list). */
  knobs?: Record<string, string | number | boolean | null>
  pp?: number | null
  nodes?: number | null
  fabric?: string | null
  bs?: number | null
  mml?: number | null
  gmu?: number | null
  cg?: string | null
  quant?: string | null
  cdt?: string | null
  scorers?: string | null
  /** The scheduler POLICY. Not to be confused with `sched`, the scheduling-delay metric. */
  schedp?: string | null
  admit?: string | null
  /** Output per GPU, from the unrounded throughput. */
  tpg?: number | null
  e2e?: number | null
  /** Scheduling delay p99 in ms (a metric — see `schedp` for the policy). */
  sched?: number | null
  itl_mean?: number | null
  /** The generated scenario file BLIS consumed. */
  scen?: string | null
  /** The exact `blis run` invocation, shell-quoted; null for runs recorded before this existed. */
  cmd?: string | null
}

export interface ModelEntry {
  name: string
  model_type: string
  arch: 'dense' | 'MoE'
  attn_kind: string | null
  layers: number | null
  hidden: number | null
  heads_q: number | null
  heads_kv: number | null
  head_dim: number | null
  context: number | null
  experts: number | null
  top_k: number | null
  shared_experts: number | null
  quant: string | null
  params: { total_b: number; active_b?: number } | null
  source_repo: string | null
}

export interface HardwareEntry {
  name: string
  memory_gib: number | null
  bw_tbs: number | null
  tflops_bf16: number | null
  tflops_fp8: number | null
  tflops_nvfp4: number | null
  sm_count: number | null
  intra_node_bw_gbps: number | null
  gpus_per_node: number | null
  gpus_per_rack: number | null
  provenance: string | null
}

export interface WorkloadEntry {
  name: string
  /** Which arm of the catalog's workload namespace this entry is. A `shape` states
   *  prompt/output token distributions; a `replay` names a load generator's
   *  trace-replay scenario over a pinned corpus and states no distribution at all. */
  kind: 'shape' | 'replay'
  /** Replay only: the scenario, the harness that defines it, and the pinned corpus. */
  scenario: string | null
  harness_version: string | null
  corpus: string | null
  corpus_revision: string | null
  /** The acceptance profile selectors, which either arm may state. */
  category: string | null
  thinking: string | null
  /** Shape only; a replay's prefix structure lives in its corpus, not in one scalar. */
  prefix_tokens: number | null
  isl_mean: number | null
  isl_stdev: number | null
  isl_min: number | null
  isl_max: number | null
  osl_mean: number | null
  osl_stdev: number | null
  osl_min: number | null
  osl_max: number | null
}

export interface Catalog {
  models: ModelEntry[]
  hardware: HardwareEntry[]
  workloads: WorkloadEntry[]
  fabrics: { name: string; inter_node_bw_gbps: number | null }[]
  storage: { name: string; read_mbs: number | null; write_mbs: number | null; base_latency_us: number | null }[]
  registry: { name: string; entries: number }[]
  versions: Record<string, string>
  // The pinned stack as one root dependency (BLIS) plus what it resolves to. Optional: a
  // catalog.js generated before build_catalog.py grew `stack` has only the flat `versions`.
  stack?: Stack
}

export interface StackDep {
  name: string
  version: string
  note: string
}

export interface Stack {
  root: StackDep
  deps: StackDep[]
}

export type Mode = 'pages' | 'local'
