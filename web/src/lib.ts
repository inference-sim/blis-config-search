import { useState } from 'react'
import type { Scenario } from './types'
import * as lex from './lexicon'

// ---------- formatting ----------
export const dash = '—'
export function fnum(v: number | null | undefined, digits = 1): string {
  return v == null || !Number.isFinite(v) ? dash : v.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

// ---------- routing ----------
export type Tab = 'reports' | 'coverage' | 'models' | 'workloads' | 'gpus' | 'submit'
export interface Route {
  tab: Tab
  runId: string | null
  params: Record<string, string>
}
/** Every addressable route. Models/workloads/gpus resolve but are not in the primary nav. */
const TABS: Tab[] = ['reports', 'coverage', 'models', 'workloads', 'gpus', 'submit']
/** The primary nav. The three per-entity spec sheets are demoted to links on Coverage: with
 *  only a handful of scenarios searched, a full spec-sheet per catalog entry is mostly empty
 *  `reports` cells, and the question a reader arrives with — what has been searched? — was
 *  split across all three. Coverage answers it in one table. */
export const TAB_LABELS: [Tab, string][] = [
  ['reports', 'Reports'],
  ['coverage', 'Coverage'],
  ['submit', 'Submit'],
]
/** The spec sheets, as the secondary links Coverage discloses them through. */
export const CATALOG_TABS: [Tab, string][] = [
  ['models', 'Models'],
  ['workloads', 'Workloads'],
  ['gpus', 'GPUs'],
]
/** A demoted spec-sheet route — not a primary tab, so the nav marks Coverage active for it. */
export const isCatalogTab = (t: Tab): boolean => CATALOG_TABS.some(([tab]) => tab === t)

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

// ---------- coverage: what has actually been searched ----------

/** One searched combination: the filter tuple plus every report behind it. */
export interface CoverageRow {
  model: string
  hardware: string
  workload: string
  gpu_budget: number
  scenarios: Scenario[]
  /** Reports that are real BLIS output (the rest are synthetic placeholders). */
  n_real: number
  /**
   * The report whose config this row recommends, resolved once at build time (the table reads
   * it per cell, per sort). With several reports on one tuple — a re-search, or a narrower
   * sweep — each named its own winner, so they are ranked on tok/s/GPU: the dual-Pareto
   * objective the search picks its own winner on, and the one `operatingPoints` ranks by. Real
   * runs beat synthetic placeholders outright, since a mock's numbers are illustrative.
   */
  best: Scenario | null
}

/** Best of a tuple's reports on tok/s/GPU, preferring real runs. See `CoverageRow.best`. */
function pickBest(scenarios: Scenario[], n_real: number): Scenario | null {
  const pool = n_real > 0 ? scenarios.filter((s) => !s.synthetic) : scenarios
  let best: Scenario | null = null
  for (const s of pool) {
    if (best === null || s.winner.tps_per_gpu > best.winner.tps_per_gpu) best = s
  }
  return best
}

/**
 * The inverse of `countReports`: instead of asking "how many reports mention this catalog
 * entry?", enumerate the (model, hardware, workload, budget) tuples that HAVE a report and
 * group by them. Only searched combinations get a row, so there is no empty-cell row to read
 * past. Order is the scenario list's own — build_gallery already emits it sorted.
 *
 * Grouping is case-insensitive, matching `countReports` and `canonFilters`. build_gallery
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
      row = { model: s.model, hardware: s.hardware, workload: s.workload, gpu_budget: s.gpu_budget, scenarios: [], n_real: 0, best: null }
      by.set(key, row)
    }
    row.scenarios.push(s)
    if (!s.synthetic) row.n_real++
  }
  const rows = [...by.values()]
  for (const r of rows) r.best = pickBest(r.scenarios, r.n_real)
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
/* Every label below comes from the shared lexicon (`./lexicon`), not a literal: the standalone
   HTML report labels the same fields from the same file, so one metric cannot be "Per-user rate"
   here and "tok/s per user" there (issue #13, item 7b). */
export type MetricKey = 'cl' | 'pu' | 'itl' | 'ttft' | 'p95' | 'conc'
const METRIC_KEYS: MetricKey[] = ['cl', 'pu', 'itl', 'ttft', 'p95', 'conc']
export const METRICS: { key: MetricKey; label: string; unit: string; digits: number }[] =
  METRIC_KEYS.map((key) => ({ key, ...lex.metric(key) }))
export type FacetField = 'series' | 'tp' | 'ni' | 'gpus' | 'epw' | 'ok' | 'src' | 'pd' | 'route' | 'a2a' | 'mtp' | 'mns' | 'mnbt'
const FACET_KEYS: FacetField[] = [
  'series', 'tp', 'ni', 'gpus', 'epw', 'ok', 'src', 'pd', 'route', 'a2a', 'mtp', 'mns', 'mnbt',
]
export const FACETS: { key: FacetField; label: string }[] =
  FACET_KEYS.map((key) => ({ key, label: lex.label(key) }))
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

// ---- series highlight (emphasis only, never a filter) ----
/** Sticky click highlight wins over transient hover; either is ignored if its series is not in view. */
export function resolveHighlight(sticky: string | null, hover: string | null, inView: readonly string[]): string | null {
  if (sticky !== null && inView.includes(sticky)) return sticky
  if (hover !== null && inView.includes(hover)) return hover
  return null
}
/** Clicking the already-highlighted series clears it. */
export function toggleHighlight(cur: string | null, series: string): string | null {
  return cur === series ? null : series
}
/** 'on' = the highlighted series, 'off' = faded context, 'neutral' = nothing highlighted. */
export function emphasis(active: string | null, series: string): 'on' | 'off' | 'neutral' {
  if (active === null) return 'neutral'
  return active === series ? 'on' : 'off'
}

/* ---------------- operating points: "pick a load, read the flags off it" ----------------
   The same table as the standalone report's "What to run" (report.py `_operating_table`): one row
   per load, pairing the configuration you would deploy with the performance it gave at that load,
   ranked on the one shared objective. The run table below it stays the full per-run log,
   including the runs that broke the floor. */

/** The flags a reader would actually pass, in a stable order, and the values worth no ink (a knob
 *  at its default or an "off" state — printing those buries the settings that actually matter).
 *  Both come from the shared lexicon, which report.py reads as `_FLAG_LABELS` / `_FLAG_SKIP`, so
 *  the two surfaces cannot print different deployments for one run. */
const { FLAGS, FLAG_SKIP } = lex

/** The non-default flags for one run, each as {flag, value} — value is '' for a boolean flag. */
export function runFlags(r: RunRecord): { flag: string; value: string }[] {
  const out: { flag: string; value: string }[] = []
  for (const f of FLAGS) {
    const v = r[f.key] as string | number | null | undefined
    if (v === null || v === undefined || v === '') continue
    if ((FLAG_SKIP[f.key as string] ?? []).includes(v as string | number)) continue
    out.push({ flag: f.flag, value: f.bool ? '' : String(v) })
  }
  return out
}

/** Rank one run on the SHARED objective (lexicon `objective`): tok/s per GPU, then fleet tok/s,
 *  then tok/s per user. Returned as a tuple compared left to right, so the tiebreaks make the
 *  pick deterministic rather than record-order dependent. Mirrors report.py `_obj_rank`. */
function objRank(r: RunRecord): number[] {
  const num = (k: string): number => {
    const v = r[k as keyof RunRecord]
    return typeof v === 'number' && Number.isFinite(v) ? v : -Infinity
  }
  return [num(lex.OBJECTIVE.key), ...lex.OBJECTIVE.tiebreak.filter((k) => k !== 'id').map(num)]
}
/** Lexicographic compare of two rank tuples: >0 when `a` is the better operating point. */
function cmpRank(a: number[], b: number[]): number {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i]
  return 0
}

/**
 * One operating point per load: the best SERVABLE run at that concurrency on the shared objective
 * — tok/s per GPU, the efficiency half of the dual-Pareto objective the search optimises and
 * picks its own winner on (lexicon `objective`, which report.py's `operating_points` reads too).
 *
 * Ranking by per-user rate instead — which the standalone report used to do — made the table
 * recommend the fastest-per-user shape while the card above recommended the most efficient one:
 * on kimi-k2.5 that is TP8+EP at 23 tok/s/GPU versus the winner's TP4xDP2+EP at 67. That was the
 * cross-surface disagreement issue #13 is about; both tables now rank on this one objective.
 */
export function operatingPoints(recs: RunRecord[]): RunRecord[] {
  const best = new Map<number, RunRecord>()
  for (const r of recs) {
    if (!r.ok || !finite(r.conc) || !finite(r.tpg)) continue
    const cur = best.get(r.conc)
    if (cur === undefined) { best.set(r.conc, r); continue }
    const c = cmpRank(objRank(r), objRank(cur))
    if (c > 0 || (c === 0 && String(r.id) < String(cur.id))) best.set(r.conc, r)
  }
  return [...best.values()].sort((a, b) => (a.conc as number) - (b.conc as number))
}
