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
