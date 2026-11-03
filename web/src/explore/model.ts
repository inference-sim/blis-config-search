// The explorer's pure logic: unpack explore.json, name fields, filter and rank. No React here.
import type { ExploreData, MetricKey, Run, Scalar } from './types'

export type Layer = 'topology' | 'routing' | 'engine'
export type Pool = 'prefill' | 'decode' | null

/* ---------------- metrics ---------------- */

export interface MetricMeta { label: string; short: string; digits: number; higher: boolean; unit: string }
export const METRIC: Record<MetricKey, MetricMeta> = {
  tpg: { label: 'tok/s per accelerator', short: 'tok/s/acc', digits: 1, higher: true, unit: 'tok/s' },
  cl: { label: 'tok/s fleet', short: 'tok/s fleet', digits: 0, higher: true, unit: 'tok/s' },
  pu: { label: 'tok/s per user', short: 'tok/s/user', digits: 1, higher: true, unit: 'tok/s' },
  itl: { label: 'TPOT p99', short: 'TPOT p99', digits: 1, higher: false, unit: 'ms' },
  ttft: { label: 'TTFT p99', short: 'TTFT p99', digits: 0, higher: false, unit: 'ms' },
}
export const METRIC_KEYS = Object.keys(METRIC) as MetricKey[]
/** The SLO key in the spec that bounds a metric, when there is one. */
export const SLO_OF: Partial<Record<MetricKey, string>> = { itl: 'itl_p99', ttft: 'ttft_p99' }

export function sloMs(v: string | undefined): number | null {
  const m = /^([\d.]+)\s*(us|ms|s)$/.exec(String(v ?? ''))
  return m ? Number(m[1]) * { us: 0.001, ms: 1, s: 1000 }[m[2] as 'us' | 'ms' | 's'] : null
}

/* ---------------- fields ---------------- */

/** Topology fields, per pool. Aggregated runs carry no pool fields and P/D runs no shared ones. */
export const TOPO_FIELDS: { key: string; part: TopoPart; pool: Pool }[] = [
  ...(['instances', 'tp', 'dp', 'pp', 'ep'] as const).flatMap((part) =>
    ([null, 'prefill', 'decode'] as Pool[]).map((pool) => ({ key: topoKey(part, pool), part, pool }))),
]
export type TopoPart = 'instances' | 'tp' | 'dp' | 'pp' | 'ep'
export function topoKey(part: TopoPart, pool: Pool): string {
  const base = part === 'instances' ? (pool ? 'instances' : 'num_instances') : part === 'ep' ? 'enable_expert_parallel' : part
  return pool ? `${pool}_${base}` : base
}
const TOPO_KEYS = new Set(TOPO_FIELDS.map((f) => f.key))
export const TOPO_INFO: Record<TopoPart, { title: string; help: string }> = {
  instances: { title: 'instances', help: 'independent copies of the model, each serving its own requests' },
  tp: { title: 'TP · tensor parallel', help: 'splits each layer across accelerators; faster per token, more all-reduce traffic' },
  dp: { title: 'DP · data parallel', help: 'attention replicas inside one instance; packs more concurrent decodes' },
  pp: { title: 'PP · pipeline parallel', help: 'splits the layers into stages; adds a stage hop, frees memory per accelerator' },
  ep: { title: 'EP · expert parallel', help: 'shards MoE experts across the instance instead of replicating them' },
}
// Keys that always travel with another key and add nothing as a filter of their own.
const SECONDARY = new Set(['speculative_category', 'speculative_thinking'])
const DEFAULTS: Record<string, Scalar> = { dp: 1, pp: 1, enable_expert_parallel: false }

export const poolOf = (k: string): Pool => (/^(prefill|decode)_/.exec(k)?.[1] as Pool) ?? null
export const bareKey = (k: string) => k.replace(/^(prefill|decode)_/, '')
export function layerOf(k: string): Layer {
  if (k === 'mode' || TOPO_KEYS.has(k)) return 'topology'
  if (k === 'router_config') return 'routing'
  return 'engine'
}

export const ENGINE_HELP: Record<string, string> = {
  max_num_batched_tokens: 'token budget per scheduler step',
  max_num_seqs: 'cap on sequences batched together',
  long_prefill_token_threshold: 'chunk prompts longer than this; 0 is off',
  gpu_memory_utilization: 'share of accelerator memory vLLM may use',
  cudagraph_mode: 'how much of the forward pass is captured as CUDA graphs',
  cache_dtype: 'KV cache precision',
  all2all_backend: 'how MoE tokens are dispatched to experts',
  eplb: 'expert-parallel load balancing',
  enable_prefix_caching: 'reuse KV for shared prompt prefixes',
  num_speculative_tokens: 'draft tokens per step; 0 is off',
  speculative_method: 'drafting method',
  max_cudagraph_capture_size: 'largest batch captured as a CUDA graph',
  quantization: 'weight quantization',
  max_model_len: 'longest context accepted',
}

/* ---------------- unpacking ---------------- */

// The dashboard's notation (run_experiment.label_of): "8×TP4xDP2 + EP", "2P/6D TP8 + EP".
const poolLabel = (tp: Scalar, dp: Scalar, pp: Scalar, ep: Scalar) =>
  `TP${tp ?? 1}${Number(dp) > 1 ? `xDP${dp}` : ''}${Number(pp) > 1 ? `xPP${pp}` : ''}${ep ? ' + EP' : ''}`

export function topoLabel(k: Record<string, Scalar>): string {
  if (k.prefill_instances != null) {
    const shape = (pool: string) => poolLabel(k[`${pool}_tp`], k[`${pool}_dp`], k[`${pool}_pp`], k[`${pool}_enable_expert_parallel`])
    const [p, d] = [shape('prefill'), shape('decode')]
    // one shape when the pools match; both when they differ, which label_of would hide
    return p === d ? `${k.prefill_instances}P/${k.decode_instances}D ${p}` : `${k.prefill_instances}P ${p} / ${k.decode_instances}D ${d}`
  }
  return `${k.num_instances}×${poolLabel(k.tp, k.dp, k.pp, k.enable_expert_parallel)}`
}

export function unpack(d: ExploreData): Run[] {
  const nk = d.keys.length
  return d.rows.map((a, id) => {
    const knobs: Record<string, Scalar> = {}
    d.keys.forEach((k, j) => { knobs[k] = a[2 + j] })
    const m = (i: number) => (typeof a[2 + nk + i] === 'number' ? (a[2 + nk + i] as number) : null)
    const mode = knobs.prefill_instances != null ? 'P/D' : 'aggregated'
    return {
      id, sweep: d.sweeps[a[0] as number]?.name ?? '', conc: a[1] as number, knobs,
      tpg: m(0), cl: m(1), pu: m(2), itl: m(3), ttft: m(4), ok: a[2 + nk + 5] === 1,
      mode, defaults: d.defaults ?? {}, topo: topoLabel(knobs), variant: d.variants?.[d.variant?.[id] ?? -1] ?? '', cfg: d.keys.map((k) => JSON.stringify(knobs[k] ?? null)).join('|'),
    }
  })
}

/* ---------------- values ---------------- */

export const NA = 'n/a'
const show = (v: Scalar | object): string => (typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v))
/** A knob's value; an unset one names the value it actually ran at, when the report knows it. */
export function disp(v: Scalar | undefined, def?: Scalar | object): string {
  const known = def !== undefined && def !== null
  if (v === null || v === undefined || v === '') return known ? `${show(def)} (default)` : 'default'
  // stating the default value is the same deployment as leaving it unset
  return known && show(v) === show(def) ? `${show(v)} (default)` : show(v)
}
const isDefault = (v: string) => v === 'default' || v.endsWith(' (default)')
/** A field value without its "(default)" tag. */
export const bareValue = (v: string) => v.replace(/ \(default\)$/, '')

/** A generated router arm is shown as its family plus the overridden parameter. */
export function routerLabel(id: Scalar, routers: ExploreData['routers']): string {
  const r = id == null ? null : routers[String(id)]
  if (!r) return 'default'
  if (!r.override) return r.family
  return `${r.family} ${overrideParts(r.override).map((o) => `${o.param}=${o.value}`).join(', ')}`
}
/** run_experiment.py joins several overrides with ", "; each is "plugin param value". */
export function overrideParts(override: string): { plugin: string; param: string; value: string }[] {
  return override.split(/,\s*/).filter(Boolean).map((o) => {
    const w = o.trim().split(/\s+/)
    return { plugin: w[0], param: w.slice(1, -1).join(' '), value: w[w.length - 1] }
  })
}

/** The value a run shows for a field, as the filter and the chart compare it. */
export function fieldValue(r: Run, k: string, routers: ExploreData['routers']): string {
  if (k === 'mode') return r.mode
  if (k === 'sweep') return r.sweep
  if (k === 'conc') return String(r.conc)
  if (k === 'router_config') return routerLabel(r.knobs[k], routers)
  if (TOPO_KEYS.has(k)) {
    const pooled = poolOf(k) !== null
    if (pooled !== (r.mode === 'P/D')) return NA
    const v = r.knobs[k] ?? DEFAULTS[bareKey(k)] ?? null
    return bareKey(k) === 'enable_expert_parallel' ? (v ? 'on' : 'off') : disp(v)
  }
  if (poolOf(k) && r.mode !== 'P/D') return NA
  return disp(r.knobs[k], r.defaults[k])
}

/** True when every real value (not "n/a" or a bare "default") is a number. */
export function isNumeric(vals: string[]): boolean {
  return vals.filter((v) => v !== NA && v !== 'default').every((v) => !Number.isNaN(Number(bareValue(v))))
}

/** Numbers numerically (a known default in its place), else the default first; "n/a" always last. */
export function sortValues(vals: string[]): string[] {
  const real = vals.filter((v) => v !== NA && v !== 'default')
  real.sort(isNumeric(real) ? (a, b) => Number(bareValue(a)) - Number(bareValue(b))
    : (a, b) => Number(isDefault(b)) - Number(isDefault(a)) || a.localeCompare(b))
  return [...(vals.includes('default') ? ['default'] : []), ...real, ...(vals.includes(NA) ? [NA] : [])]
}

/** Fields worth a filter: at least two real values across the report. */
export function liveFields(runs: Run[], keys: string[], routers: ExploreData['routers']): string[] {
  const out: string[] = []
  if (new Set(runs.map((r) => r.mode)).size > 1) out.push('mode')
  for (const f of TOPO_FIELDS) if (spread(runs, f.key, routers) > 1) out.push(f.key)
  for (const k of keys) {
    if (TOPO_KEYS.has(k) || SECONDARY.has(k)) continue
    if (spread(runs, k, routers) > 1) out.push(k)
  }
  return out
}
function spread(runs: Run[], k: string, routers: ExploreData['routers']) {
  return new Set(runs.map((r) => fieldValue(r, k, routers)).filter((v) => v !== NA)).size
}

/* ---------------- filtering ---------------- */

export type Selection = Record<string, string[]>

export function matches(r: Run, sel: Selection, routers: ExploreData['routers'], skip?: string): boolean {
  for (const [k, vals] of Object.entries(sel)) {
    if (k === skip || vals.length === 0) continue
    if (!vals.includes(fieldValue(r, k, routers))) return false
  }
  return true
}
export function toggle(sel: Selection, k: string, v: string): Selection {
  const cur = sel[k] ?? []
  const next = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]
  const out = { ...sel }
  if (next.length) out[k] = next
  else delete out[k]
  return out
}

/* ---------------- ranking ---------------- */

export const better = (m: MetricKey, a: number, b: number) => (METRIC[m].higher ? a > b : a < b)

/** Each config at its best load that meets every SLO, best first on the target metric. */
export function rankConfigs(runs: Run[], m: MetricKey): Run[] {
  const best = new Map<string, Run>()
  for (const r of runs) {
    const v = r[m]
    if (!r.ok || v === null) continue
    const cur = best.get(r.cfg)
    if (!cur || better(m, v, cur[m] as number)) best.set(r.cfg, r)
  }
  const dir = METRIC[m].higher ? -1 : 1
  return [...best.values()].sort((a, b) => dir * ((a[m] as number) - (b[m] as number)) || (b.tpg ?? 0) - (a.tpg ?? 0) || a.id - b.id)
}

/** Configs whose best runs tie on every metric are one result: the knobs between them made no difference. */
export interface Ranked { lead: Run; ties: Run[] }
export function groupTies(ranked: Run[]): Ranked[] {
  const groups = new Map<string, Ranked>()
  for (const r of ranked) {
    const sig = [r.conc, r.tpg, r.cl, r.pu, r.itl, r.ttft, r.topo].join('|')
    const g = groups.get(sig)
    if (g) g.ties.push(r)
    else groups.set(sig, { lead: r, ties: [] })
  }
  return [...groups.values()]
}

/** The knobs that set two configs apart, as "key: a / b" strings. */
export function knobDiff(a: Run, b: Run, routers: ExploreData['routers']): string[] {
  const keys = new Set([...Object.keys(a.knobs), ...Object.keys(b.knobs)])
  return [...keys].filter((k) => fieldValue(a, k, routers) !== fieldValue(b, k, routers))
    .map((k) => `${bareKey(k) === 'router_config' ? 'router' : k}: ${fieldValue(a, k, routers)} / ${fieldValue(b, k, routers)}`)
}

/** The value each live non-topology knob most often takes in this serving mode: what "usual" means on a tag. */
export function usualValues(runs: Run[], fields: string[], routers: ExploreData['routers']) {
  const out: Record<string, Record<string, string>> = { aggregated: {}, 'P/D': {} }
  for (const mode of ['aggregated', 'P/D'] as const) {
    const rs = runs.filter((r) => r.mode === mode)
    for (const k of fields) {
      const c = new Map<string, number>()
      for (const r of rs) { const v = fieldValue(r, k, routers); c.set(v, (c.get(v) ?? 0) + 1) }
      const top = [...c].sort((x, y) => y[1] - x[1])[0]
      if (top) out[mode][k] = top[0]
    }
  }
  return out
}

/** Knob tags for a config: the non-topology fields where it departs from the usual value. */
export function tagsOf(r: Run, fields: string[], usual: ReturnType<typeof usualValues>, routers: ExploreData['routers']) {
  return fields.filter((k) => layerOf(k) !== 'topology')
    .map((k) => ({ k, v: fieldValue(r, k, routers) }))
    .filter(({ k, v }) => v !== NA && v !== usual[r.mode][k])
    .map(({ k, v }) => ({ layer: layerOf(k), text: `${poolOf(k) ? poolOf(k) + ' ' : ''}${k === 'router_config' ? '' : bareKey(k) + ' '}${v}` }))
}

/** Each config's caption: its sweep's variant when that alone tells it apart, else its knob tags. */
export function captions(runs: Run[], fields: string[], usual: ReturnType<typeof usualValues>, routers: ExploreData['routers']) {
  const cfgs = new Map<string, Set<string>>()
  for (const r of runs) {
    const k = `${r.topo}|${r.variant}`
    cfgs.set(k, (cfgs.get(k) ?? new Set()).add(r.cfg))
  }
  const out = new Map<string, string>()
  for (const r of runs) {
    if (out.has(r.cfg)) continue
    const unique = r.variant && cfgs.get(`${r.topo}|${r.variant}`)!.size === 1
    out.set(r.cfg, unique ? r.variant : tagsOf(r, fields, usual, routers).map((t) => t.text).join(' · ') || r.variant)
  }
  return out
}

/** A config's settings as filters, limited to fields its serving mode has. */
export function configFilters(r: Run, fields: string[], routers: ExploreData['routers']): Selection {
  const sel: Selection = {}
  for (const k of fields) {
    const v = fieldValue(r, k, routers)
    if (v !== NA) sel[k] = [v]
  }
  return sel
}

/** The field whose values move the best result most: a good default to color a chart by. */
export function mostInfluential(runs: Run[], fields: string[], m: MetricKey, routers: ExploreData['routers']): string | null {
  let pick: string | null = null, spreadBest = 1
  for (const k of fields) {
    const best = new Map<string, number>()
    for (const r of runs) {
      const v = r[m]
      if (!r.ok || v === null) continue
      const key = fieldValue(r, k, routers)
      if (key === NA) continue
      const cur = best.get(key)
      if (cur === undefined || better(m, v, cur)) best.set(key, v)
    }
    const vals = [...best.values()].filter((v) => v > 0)
    if (vals.length < 2) continue
    const spread = Math.max(...vals) / Math.min(...vals)
    if (spread > spreadBest) { spreadBest = spread; pick = k }
  }
  return pick
}

/* ---------------- formatting ---------------- */

export function fmt(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–'
  return v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}
export const fmtMetric = (m: MetricKey, v: number | null | undefined) => fmt(v, METRIC[m].digits)
export function quantile(xs: number[], p: number): number | null {
  if (!xs.length) return null
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))]
}
export function niceStep(max: number, n = 5): number {
  const raw = max / n || 1
  const p = 10 ** Math.floor(Math.log10(raw))
  return [1, 2, 2.5, 5, 10].map((f) => f * p).find((s) => s >= raw) ?? raw
}
export function niceMax(v: number): number {
  const s = niceStep(v * 1.05)
  return Math.ceil((v * 1.05) / s) * s || 1
}
