// Data contracts for the config-search webpage. These mirror exactly what
// scripts/build_gallery.py + build_catalog.py emit and web/scripts/stage-data.mjs stages
// into web/public/data/. Keep in sync with those producers.

export interface Winner {
  label: string
  tp: number | null
  ni: number | null
  ep?: boolean
  quant?: string | null
  gpus: number | null
  c_use: number | null
  c_star?: number | null
  tps: number
  tps_per_gpu: number
  per_user: number
  itl_p99: number
}

export interface Finalist {
  label: string
  tp: number | null
  ni: number | null
  gpus: number | null
  c_use: number | null
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
  floor: string
  headroom: number | null
  n_runs: number
  n_servable: number
  n_records?: number
  generated: string
  winner: Winner
  finalists: Finalist[]
  reason: string
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
  mtp: number
  mnbt: number | null
  mns: number | null
  pu: number | null
  cl: number | null
  itl: number | null
  ttft: number | null
  hit: number | null
  p95: number | null
  ok: boolean
  status: string
  viol: string[]
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
}

export type Mode = 'pages' | 'local'
