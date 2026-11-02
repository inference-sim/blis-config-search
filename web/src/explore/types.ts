// explore.json, as scripts/explore_data.py writes it: rows are arrays under one shared key list.

export type Scalar = string | number | boolean | null
export type MetricKey = 'tpg' | 'cl' | 'pu' | 'itl' | 'ttft'

export interface Plugin {
  name: string
  type: string
  category: 'data' | 'filter' | 'scorer' | 'picker' | 'handler' | 'decider' | 'gate' | 'policy' | 'parser' | 'other'
  params: Record<string, unknown>
}
export interface RouterChain {
  plugins: Plugin[]
  profiles: { name: string; steps: { ref: string; weight: number | null }[] }[]
  flow_control: boolean
  gate: string[]
  feeds: Record<string, string[]>
}
export interface Router {
  path: string
  family: string
  /** "prefix-based-pd-decider nonCachedTokens 256" for a generated arm; "" for a base file. */
  override: string
  summary: string
  yaml: string
  /** null when the shared routing file is no longer in the repo. */
  chain: RouterChain | null
}
export interface Sweep {
  name: string
  narrative: string
  configs: number
  dropped: { count: number; gpus: number; reason: string }[]
  spec: Record<string, unknown>
}
export interface ExploreData {
  run_id: string
  slo: Record<string, string>
  concurrency: number[]
  fixed_engine: Record<string, unknown>
  totals: { configs?: number; runs?: number; servable?: number }
  sweeps: Sweep[]
  keys: string[]
  metrics: MetricKey[]
  rows: Scalar[][]
  /** Distinct variant captions, and each row's index into them. */
  variants?: string[]
  variant?: number[]
  routers: Record<string, Router>
  /** The spec.yaml this report was swept from, as written. */
  spec_yaml: string
}

/** One run, unpacked. Knob values sit under their raw spec keys (tp, prefill_tp, router_config…). */
export interface Run {
  id: number
  sweep: string
  conc: number
  knobs: Record<string, Scalar>
  tpg: number | null
  cl: number | null
  pu: number | null
  itl: number | null
  ttft: number | null
  ok: boolean
  mode: 'aggregated' | 'P/D'
  /** Topology in the dashboard's notation, e.g. "8×TP4xDP2 + EP" or "2P/6D TP8 + EP". */
  topo: string
  /** How the config differs from its sweep's baseline, as the dashboard captions it; "" for none. */
  variant: string
  /** Identity of the config across loads: every knob value. */
  cfg: string
}
