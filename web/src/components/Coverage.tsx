import type { ReactNode } from 'react'
import type { Catalog, Scenario } from '../types'
import { SPEC_DOC, catalogDoc, catalogIndex, configLabel, coverageHref, coverageRows, dash, fnum, type CoverageRow } from '../lib'
import { CatalogTable, type Col } from './CatalogTable'

/** The winner of the report this row recommends; `coverageRows` already resolved which. */
const bestWinner = (r: CoverageRow) => r.best?.winner ?? null

/**
 * What has been searched, in one table. Replaces the Models / Workloads / Hardware tabs as the
 * primary catalog view: those rendered a spec-sheet row per catalog entry, so with a handful
 * of scenarios searched most rows carried an empty `reports` cell and the reader had to
 * reconstruct coverage across three tables. Here each row IS a searched combination, and its
 * model / hardware / workload link out to their blis-catalog reference pages.
 *
 * Alignment convention: every measured column here is centered (`center` + `numeric`) rather
 * than right-aligned, so the numeric columns read as one set. A new measured column follows
 * suit; mixing the two makes columns of the same kind sit differently.
 */
/** A chip that opens its blis-catalog spec sheet; plain text when the catalog does not list it. */
export function CatRef({ href, children, className = 'spec' }: { href: string | null; children: ReactNode; className?: string }) {
  return href
    ? <a className={`${className} catref`} href={href} target="_blank" rel="noreferrer" title="Open spec sheet in blis-catalog">{children}</a>
    : <span className={className}>{children}</span>
}


export function Coverage({ scenarios, catalog }: { scenarios: Scenario[]; catalog?: Catalog | null }) {
  const rows = coverageRows(scenarios)
  const cols: Col<CoverageRow>[] = [
    { key: 'model', label: 'model', get: (r) => r.model, render: (r) => <CatRef className="mono" href={catalogDoc(catalog, 'model', r.model)}>{r.model}</CatRef> },
    { key: 'hardware', label: 'Hardware', get: (r) => r.hardware, render: (r) => <CatRef href={catalogDoc(catalog, 'hardware', r.hardware)}>{r.hardware}</CatRef> },
    { key: 'budget', label: 'Accelerator budget', get: (r) => r.gpu_budget, center: true, numeric: true, render: (r) => <span className="num">{r.gpu_budget}</span> },
    { key: 'workload', label: 'workload', get: (r) => r.workload, render: (r) => <CatRef href={catalogDoc(catalog, 'workload', r.workload)}>{r.workload}</CatRef> },
    {
      key: 'config',
      label: 'recommended config',
      get: (r) => bestWinner(r)?.name ?? bestWinner(r)?.label ?? dash,
      render: (r) => {
        const w = bestWinner(r)
        if (!w) return <span className="na">{dash}</span>
        return <><span className="covwin">{configLabel(w.label)}</span>{w.variant && <span className="covvar">{w.variant}</span>}</>
      },
    },
    {
      key: 'tpg',
      label: 'tok/s/accelerator',
      get: (r) => bestWinner(r)?.tps_per_gpu,
      center: true,
      numeric: true,
      render: (r) => {
        const w = bestWinner(r)
        return w ? <span className="num">{fnum(w.tps_per_gpu, 0)}</span> : <span className="na">{dash}</span>
      },
    },
    {
      key: 'reports',
      label: 'report',
      get: (r) => r.scenarios.length,
      render: (r) => (
        <a href={coverageHref(r)}>
          {r.scenarios.length === 1 ? 'open report' : `${r.scenarios.length} reports`}
        </a>
      ),
    },
  ]

  return (
    <div>
      <div className="viewtop">
        <h2 className="viewhead">Coverage</h2>
        <nav className="catbtns" aria-label="blis-catalog">
          <span className="catbtns-l">blis-catalog</span>
          {catalogIndex(catalog).map((l) => <a key={l.label} href={l.href} target="_blank" rel="noreferrer">{l.label}</a>)}
        </nav>
      </div>
      <p className="dek">
        The best config for each searched combination. Click a model, hardware or workload for its spec sheet.
      </p>

      {rows.length === 0 ? (
        <div className="empty">No scenarios have been searched yet. Write an <a href={SPEC_DOC} target="_blank" rel="noreferrer">experiment sweep spec</a> and open a pull request.</div>
      ) : (
        <CatalogTable rows={rows} cols={cols} rowKey={(r) => `${r.model}|${r.hardware}|${r.workload}|${r.gpu_budget}`} caption="Searched combinations" />
      )}

    </div>
  )
}
