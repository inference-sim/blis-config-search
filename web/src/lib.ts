import { useState } from 'react'
import type { Scenario } from './types'

// ---------- formatting ----------
export const dash = '—'
export function fnum(v: number | null | undefined, digits = 1): string {
  return v == null || !Number.isFinite(v) ? dash : v.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

// ---------- routing ----------
export type Tab = 'reports' | 'models' | 'workloads' | 'gpus' | 'submit'
export interface Route {
  tab: Tab
  runId: string | null
  params: Record<string, string>
}
const TABS: Tab[] = ['reports', 'models', 'workloads', 'gpus', 'submit']

export function parseRoute(hash: string): Route {
  const h = hash.replace(/^#\/?/, '')
  const [path, query = ''] = h.split('?')
  const params: Record<string, string> = {}
  new URLSearchParams(query).forEach((v, k) => {
    params[k] = v
  })
  const parts = path.split('/').filter(Boolean)
  if (parts[0] === 'scenario' && parts[1]) return { tab: 'reports', runId: decodeURIComponent(parts[1]), params }
  const tab = (TABS as string[]).includes(parts[0]) ? (parts[0] as Tab) : 'reports'
  return { tab, runId: null, params }
}

export function reportsHash(params: Record<string, string>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v) q.set(k, v)
  const s = q.toString()
  return `#/reports${s ? `?${s}` : ''}`
}
export const scenarioHash = (runId: string) => `#/scenario/${encodeURIComponent(runId)}`

// ---------- reports filtering ----------
export interface Filters {
  model: string
  hardware: string
  workload: string
  gpu: string // gpu_budget as string
}
export const NO_FILTERS: Filters = { model: '', hardware: '', workload: '', gpu: '' }
export type FilterKey = keyof Filters

const lc = (s: string) => s.toLowerCase()
function matches(s: Scenario, f: Filters, skip?: FilterKey): boolean {
  if (skip !== 'model' && f.model && lc(s.model) !== lc(f.model)) return false
  if (skip !== 'hardware' && f.hardware && lc(s.hardware) !== lc(f.hardware)) return false
  if (skip !== 'workload' && f.workload && lc(s.workload) !== lc(f.workload)) return false
  if (skip !== 'gpu' && f.gpu && String(s.gpu_budget) !== f.gpu) return false
  return true
}
export function filterScenarios(list: Scenario[], f: Filters): Scenario[] {
  return list.filter((s) => matches(s, f))
}
const facetOf = (s: Scenario, k: FilterKey): string => (k === 'gpu' ? String(s.gpu_budget) : s[k])

/** Per-value counts for one facet, applying every OTHER active filter (live counts). */
export function facetCounts(list: Scenario[], f: Filters, key: FilterKey): { value: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const s of list) {
    const v = facetOf(s, key)
    if (!counts.has(v)) counts.set(v, 0)
    if (matches(s, f, key)) counts.set(v, (counts.get(v) ?? 0) + 1)
  }
  const out = [...counts.entries()].map(([value, count]) => ({ value, count }))
  out.sort((a, b) => (key === 'gpu' ? Number(a.value) - Number(b.value) : a.value.localeCompare(b.value)))
  return out
}

/** Recommended configs for the summary: real scenarios only; if none, the filtered mocks (flagged). */
export function topConfigs(filtered: Scenario[]): { items: Scenario[]; mockOnly: boolean } {
  const real = filtered.filter((s) => !s.synthetic)
  if (real.length > 0) return { items: real, mockOnly: false }
  return { items: filtered, mockOnly: filtered.length > 0 }
}

export function countReports(scenarios: Scenario[], key: 'model' | 'hardware' | 'workload', name: string): number {
  return scenarios.filter((s) => lc(s[key]) === lc(name)).length
}

// ---------- sorting ----------
export interface SortState {
  key: string
  dir: 1 | -1
}
export function useSort(initial: SortState | null = null) {
  const [sort, setSort] = useState<SortState | null>(initial)
  const toggle = (key: string) =>
    setSort((cur) => (!cur || cur.key !== key ? { key, dir: 1 } : cur.dir === 1 ? { key, dir: -1 } : null))
  return { sort, toggle }
}
export function sortRows<T>(rows: T[], sort: SortState | null, get: (r: T, key: string) => string | number | null | undefined): T[] {
  if (!sort) return rows
  return [...rows].sort((a, b) => {
    const va = get(a, sort.key)
    const vb = get(b, sort.key)
    if (va == null && vb == null) return 0
    if (va == null) return 1
    if (vb == null) return -1
    const c = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb))
    return c * sort.dir
  })
}

/** Canonicalize filter values against the scenario list (URL params may differ in case). */
export function canonFilters(list: Scenario[], f: Filters): Filters {
  const fix = (k: FilterKey): string => {
    if (!f[k]) return ''
    const hit = list.map((s) => facetOf(s, k)).find((v) => lc(v) === lc(f[k]))
    return hit ?? f[k]
  }
  return { model: fix('model'), hardware: fix('hardware'), workload: fix('workload'), gpu: fix('gpu') }
}

// ---------- record (run) filtering for the Pareto explorer ----------
import type { RunRecord } from './types'
export type Band = '' | 'low' | 'mid' | 'high'
export const BANDS: { value: Band; label: string }[] = [
  { value: '', label: 'any concurrency' },
  { value: 'low', label: 'low (1-8)' },
  { value: 'mid', label: 'mid (9-64)' },
  { value: 'high', label: 'high (65+)' },
]
export function inBand(conc: number | null, band: Band): boolean {
  if (!band) return true
  if (conc == null) return false
  return band === 'low' ? conc <= 8 : band === 'mid' ? conc > 8 && conc <= 64 : conc > 64
}
export interface RecFilters {
  series: string
  band: Band
  servable: '' | 'ok' | 'bad'
}
export function filterRecords(recs: RunRecord[], f: RecFilters): RunRecord[] {
  return recs.filter(
    (r) => (!f.series || r.series === f.series) && inBand(r.conc, f.band) && (!f.servable || (f.servable === 'ok') === r.ok),
  )
}
export const finite = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v)

// ---------- cross-filter selection (brushable histograms + categorical facets) ----------
export type MetricKey = 'cl' | 'pu' | 'itl' | 'ttft' | 'p95' | 'conc'
export const METRICS: { key: MetricKey; label: string; unit: string; digits: number }[] = [
  { key: 'cl', label: 'Fleet throughput', unit: 'tok/s', digits: 0 },
  { key: 'pu', label: 'Per-user rate', unit: 'tok/s', digits: 1 },
  { key: 'itl', label: 'Inter-token latency', unit: 'ms', digits: 1 },
  { key: 'ttft', label: 'TTFT p99', unit: 'ms', digits: 0 },
  { key: 'p95', label: 'E2E p95', unit: 'ms', digits: 0 },
  { key: 'conc', label: 'Concurrency', unit: 'sessions', digits: 0 },
]
export type FacetField = 'series' | 'tp' | 'ni' | 'gpus' | 'epw' | 'ok' | 'src' | 'pd' | 'route' | 'a2a' | 'mtp' | 'mns' | 'mnbt'
export const FACETS: { key: FacetField; label: string }[] = [
  { key: 'series', label: 'Configuration' },
  { key: 'tp', label: 'Tensor parallel' },
  { key: 'ni', label: 'Instances' },
  { key: 'gpus', label: 'GPUs' },
  { key: 'epw', label: 'Expert parallel width' },
  { key: 'ok', label: 'Servable' },
  { key: 'src', label: 'Run source' },
  { key: 'pd', label: 'Prefill/decode' },
  { key: 'route', label: 'Routing' },
  { key: 'a2a', label: 'All-to-all' },
  { key: 'mtp', label: 'MTP depth' },
  { key: 'mns', label: 'Max seqs' },
  { key: 'mnbt', label: 'Max batched tokens' },
]
export type Range = [number, number]
export interface Selection {
  facets: Partial<Record<FacetField, string[]>>
  ranges: Partial<Record<MetricKey, Range>>
}
export const EMPTY_SEL: Selection = { facets: {}, ranges: {} }
export function selectionActive(s: Selection): boolean {
  return Object.values(s.facets).some((v) => v && v.length > 0) || Object.values(s.ranges).some((v) => !!v)
}

/** String level for a categorical field; null/empty/NaN become the dash. */
export function levelOf(r: RunRecord, k: FacetField): string {
  const v = r[k]
  if (k === 'ok') return v ? 'servable' : 'not servable'
  if (v == null || v === '' || (typeof v === 'number' && !Number.isFinite(v))) return dash
  return String(v)
}
const numCmp = (a: string, b: string) => {
  const x = Number(a), y = Number(b)
  return Number.isFinite(x) && Number.isFinite(y) ? x - y : a.localeCompare(b)
}

type Skip = { facet?: FacetField; metric?: MetricKey }
export function matchesSel(r: RunRecord, s: Selection, skip: Skip = {}): boolean {
  for (const [k, lv] of Object.entries(s.facets) as [FacetField, string[] | undefined][]) {
    if (k === skip.facet || !lv || lv.length === 0) continue
    if (!lv.includes(levelOf(r, k))) return false
  }
  for (const [k, rg] of Object.entries(s.ranges) as [MetricKey, Range | undefined][]) {
    if (k === skip.metric || !rg) continue
    const v = r[k]
    if (!finite(v) || v < rg[0] || v > rg[1]) return false
  }
  return true
}
export function selectRecords(recs: RunRecord[], s: Selection): RunRecord[] {
  return recs.filter((r) => matchesSel(r, s))
}

/** Facet fields with >=2 distinct levels across ALL records (single-valued fields are hidden). */
export function visibleFacets(recs: RunRecord[]): FacetField[] {
  return FACETS.map((f) => f.key).filter((k) => new Set(recs.map((r) => levelOf(r, k))).size >= 2)
}
/** Levels of one field with live counts (every OTHER active filter applied). */
export function facetLevels(recs: RunRecord[], s: Selection, k: FacetField): { value: string; count: number }[] {
  const m = new Map<string, number>()
  for (const r of recs) {
    const v = levelOf(r, k)
    m.set(v, (m.get(v) ?? 0) + (matchesSel(r, s, { facet: k }) ? 1 : 0))
  }
  return [...m.entries()].map(([value, count]) => ({ value, count })).sort((a, b) => numCmp(a.value, b.value))
}

export interface Hist {
  min: number
  max: number
  counts: number[]
}
/** Equal-width histogram. Returns null when there are <2 distinct finite values (zero span / empty). */
export function histogram(values: (number | null | undefined)[], bins = 12): Hist | null {
  const xs = values.filter(finite)
  if (xs.length === 0 || bins < 1) return null
  const min = Math.min(...xs)
  const max = Math.max(...xs)
  if (!(max > min)) return null
  const counts = new Array<number>(bins).fill(0)
  const w = (max - min) / bins
  for (const x of xs) counts[Math.min(bins - 1, Math.floor((x - min) / w))]++
  return { min, max, counts }
}
/** Metrics worth a histogram: >=2 distinct finite values across all records. */
export function varyingMetrics(recs: RunRecord[]): MetricKey[] {
  return METRICS.map((m) => m.key).filter((k) => histogram(recs.map((r) => r[k])) !== null)
}
