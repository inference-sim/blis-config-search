import { useState } from 'react'
import type { Catalog, Scenario } from './types'

// ---------- formatting ----------
export const dash = '—'
export function fnum(v: number | null | undefined, digits = 1): string {
  return v == null || !Number.isFinite(v) ? dash : v.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

// ---------- routing ----------
export type Tab = 'reports' | 'coverage' | 'submit'
export interface Route {
  tab: Tab
  runId: string | null
  params: Record<string, string>
}
const TABS: Tab[] = ['reports', 'coverage', 'submit']
export const TAB_LABELS: [Tab, string][] = [
  ['reports', 'Reports'],
  ['coverage', 'Coverage'],
  ['submit', 'Submit'],
]
export const REPO = 'https://github.com/inference-sim/blis-config-search'
export const SPEC_DOC = 'https://github.com/inference-sim/blis-config-search/blob/main/experiments/README.md'

/** blis-catalog's docs for the version we pin; the site is published per minor ("0.5.0" → /0.5/). */
export function catalogBase(catalog: Catalog | null | undefined): string {
  const m = (catalog?.versions['blis-catalog'] ?? '').match(/^v?(\d+)\.(\d+)/)
  return `https://inference-sim.github.io/blis-catalog/${m ? `${m[1]}.${m[2]}` : 'latest'}/reference`
}

/** The catalog's top-level listings, in the catalog's own names. */
export const catalogIndex = (catalog: Catalog | null | undefined) => {
  const base = catalogBase(catalog)
  return [
    { label: 'Models', href: `${base}/models/` },
    { label: 'Chips', href: `${base}/hardware/` },
    { label: 'Workloads', href: `${base}/workloads/` },
  ]
}

/** One entry's spec-sheet page, or null when the catalog does not list it (no dead links). */
export function catalogDoc(catalog: Catalog | null | undefined, kind: 'model' | 'hardware' | 'workload', name: string): string | null {
  const base = catalogBase(catalog)
  const n = name.toLowerCase()
  const is = (v: string | null | undefined) => v?.toLowerCase() === n
  if (kind === 'model') return !catalog || catalog.models.some((e) => is(e.name)) ? `${base}/models/${encodeURIComponent(n)}/` : null
  if (kind === 'hardware') return !catalog || catalog.hardware.some((e) => is(e.name)) ? `${base}/hardware/` : null
  // a replay is reported under its scenario name ("inferencex-agentx-mvp"), not its file name
  const replay = catalog?.workloads.some((w) => w.kind === 'replay' && (is(w.name) || is(w.scenario)))
  return `${base}/workloads/${replay ? '#replay-workloads' : '#shapes'}`
}

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

// ---------- coverage: what has actually been swept ----------

/** One swept combination: the filter tuple plus every report behind it. */
export interface CoverageRow {
  model: string
  hardware: string
  workload: string
  gpu_budget: number
  scenarios: Scenario[]
  /**
   * The report whose config this row recommends, resolved once at build time (the table reads
   * it per cell, per sort). With several reports on one tuple (a re-search, or a narrower
   * sweep) each named its own winner, so they are ranked on tok/s/accelerator: the objective
   * the sweeper picks its own winner on, and the one `operatingPoints` ranks by.
   */
  best: Scenario | null
}

/** Best of a tuple's reports on tok/s/accelerator. See `CoverageRow.best`. */
function pickBest(scenarios: Scenario[]): Scenario | null {
  let best: Scenario | null = null
  for (const s of scenarios) {
    if (best === null || s.winner.tps_per_gpu > best.winner.tps_per_gpu) best = s
  }
  return best
}

/**
 * Instead of asking "how many reports mention this catalog
 * entry?", enumerate the (model, hardware, workload, budget) tuples that HAVE a report and
 * group by them. Only searched combinations get a row, so there is no empty-cell row to read
 * past. Order is the scenario list's own — build_gallery already emits it sorted.
 *
 * Grouping is case-insensitive, matching `canonFilters`. build_gallery
 * normalizes hardware (lowercases it, validates against the known set, emits upper) but passes
 * model and workload straight through from the scenario yaml, so two runs of one combination
 * can disagree on case; they are still one combination. Display uses the first spelling seen.
 */
export function coverageRows(scenarios: Scenario[]): CoverageRow[] {
  const by = new Map<string, CoverageRow>()
  for (const s of scenarios) {
    const key = [lc(s.model), lc(s.hardware), lc(s.workload), s.gpu_budget].join('\u0000')
    let row = by.get(key)
    if (row === undefined) {
      row = { model: s.model, hardware: s.hardware, workload: s.workload, gpu_budget: s.gpu_budget, scenarios: [], best: null }
      by.set(key, row)
    }
    row.scenarios.push(s)
  }
  const rows = [...by.values()]
  for (const r of rows) r.best = pickBest(r.scenarios)
  return rows
}

/**
 * Where a coverage row points: its one report, or the scenario list filtered to its tuple.
 * The multi-report target leans on Reports reading exactly these four filter keys — the same
 * ones `Filters`/`matches` define, compared case-insensitively there via `canonFilters`.
 */
export function coverageHref(r: CoverageRow): string {
  if (r.scenarios.length === 1) return scenarioHash(r.scenarios[0].run_id)
  return reportsHash({ model: r.model, hardware: r.hardware, workload: r.workload, gpu: String(r.gpu_budget) })
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

export const finite = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v)

// ---------- scenario card ----------
/** A colocated config reads as "[TP4 + EP]×8": the topology in brackets, then the instance count.
 *  N (`$1`) is ONLY the leading integer — num_instances. This matters because the topology itself
 *  can contain an "x" (TP4xDP2), so N is anchored to the start of the string and is never a
 *  dimension pulled from inside the brackets; `$2` is everything after the first "<n>x". A single
 *  instance (N=1) drops the brackets and the "×1" entirely — just "TP8". P/D labels ("2P/6D TP8")
 *  have no leading "<n>x" separator, so they do not match and are left untouched. */
export const configLabel = (label: string): string =>
  label.replace(/^(\d+)[x×]\s*(.+)$/, (_, n, topo) => (Number(n) === 1 ? topo : `[${topo}]×${n}`))
