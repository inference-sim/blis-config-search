// The shared vocabulary of the config-search report surfaces, typed for the dashboard.
//
// config-search renders the same runs twice: this React dashboard and the standalone HTML report
// (.claude/skills/config-search/scripts/report.py). They had grown two vocabularies for one set
// of numbers and, worse, two ranking objectives for "What to run" — the report ranked operating
// points by per-user rate while this app ranked them by tok/s/GPU, so at high load the two tables
// recommended DIFFERENT configurations for one scenario (issue #13).
//
// Both surfaces now read their labels, captions and ranking objective from one JSON file. The
// canonical copy is .claude/skills/config-search/report/lexicon.json; `lexicon.json` beside this
// file is a byte-identical mirror, because Vite cannot import out of the read-only skill dir.
// scripts/test_build.py fails the build if the two drift.
import raw from './lexicon.json'
import type { RunRecord } from './types'

export interface MetricMeta {
  label: string
  unit: string
  digits: number
  help: string
}
export interface FigureMeta {
  surfaces: string[]
  title: string
  xl: string
  yl: string | null
  caption: string
  read: string
}
interface Lexicon {
  objective: {
    key: 'tpg'
    label: string
    tiebreak: string[]
    rule: string
    why: string
    blurb: string
    none: string
  }
  metrics: Record<string, MetricMeta>
  facets: Record<string, { label: string; help: string }>
  axes: Record<string, string>
  flags: { key: string; flag: string; bool?: boolean }[]
  flag_skip: Record<string, (string | number)[]>
  operating_columns: string[]
  figures: Record<string, FigureMeta>
}

/** Expand the JSON's "@a.b" references, so a figure's axis label IS the shared axis string
 *  rather than a copy that can drift from it. Mirrors `_resolve` in lexicon.py. */
function resolve<T>(node: T, root: unknown): T {
  if (typeof node === 'string' && node.startsWith('@')) {
    let cur: unknown = root
    for (const part of node.slice(1).split('.')) cur = (cur as Record<string, unknown>)[part]
    return cur as T
  }
  if (Array.isArray(node)) return node.map((v) => resolve(v, root)) as unknown as T
  if (node !== null && typeof node === 'object') {
    return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, resolve(v, root)])) as T
  }
  return node
}

export const LEX = resolve(raw as unknown as Lexicon, raw)
export const OBJECTIVE = LEX.objective
export const AXES = LEX.axes

/** One metric's label / unit / digits / help, by RunRecord key. */
export const metric = (key: string): MetricMeta => LEX.metrics[key]
/** The display label for a metric OR a facet key — the two namespaces do not overlap. */
export const label = (key: string): string => LEX.metrics[key]?.label ?? LEX.facets[key]?.label ?? key
/** The glossary line for a metric or facet key, for a column tooltip. */
export const help = (key: string): string | undefined => LEX.metrics[key]?.help ?? LEX.facets[key]?.help
/** One figure's title / caption / axis labels / how-to-read text. */
export const figure = (name: string): FigureMeta => LEX.figures[name]

/** The deploy flags a reader would copy, in the fixed shared order. */
export const FLAGS: { key: keyof RunRecord; flag: string; bool?: boolean }[] =
  LEX.flags.map((f) => ({ key: f.key as keyof RunRecord, flag: f.flag, bool: f.bool }))
/** Values worth no ink: a knob at its default, or an "off" state. */
export const FLAG_SKIP: Record<string, (string | number)[]> = LEX.flag_skip
/** The columns the "What to run" table shows, in order. */
export const OPERATING_COLUMNS = LEX.operating_columns
