import type { Scenario } from '../types'
import { CATALOG_TABS, coverageHref, coverageRows, dash, fnum, type CoverageRow } from '../lib'
import { CatalogTable, type Col } from './CatalogTable'
import { MockBadge } from './Reports'

/** The winner of the report this row recommends; `coverageRows` already resolved which. */
const bestWinner = (r: CoverageRow) => r.best?.winner ?? null

/**
 * What has been searched, in one table. Replaces the Models / Workloads / GPUs tabs as the
 * primary catalog view: those rendered a spec-sheet row per catalog entry, so with a handful
 * of scenarios searched most rows carried an empty `reports` cell and the reader had to
 * reconstruct coverage across three tables. Here each row IS a searched combination, and the
 * spec sheets stay one click away below.
 */
export function Coverage({ scenarios }: { scenarios: Scenario[] }) {
  const rows = coverageRows(scenarios)
  const cols: Col<CoverageRow>[] = [
    { key: 'model', label: 'model', get: (r) => r.model, render: (r) => <span className="mono">{r.model}</span> },
    { key: 'hardware', label: 'gpu', get: (r) => r.hardware, render: (r) => <span className="spec">{r.hardware}</span> },
    { key: 'budget', label: 'gpu budget', get: (r) => r.gpu_budget, render: (r) => <span className="num">{r.gpu_budget}×</span>, num: true },
    { key: 'workload', label: 'workload', get: (r) => r.workload, render: (r) => <span className="spec">{r.workload}</span> },
    {
      key: 'config',
      label: 'recommended config',
      get: (r) => bestWinner(r)?.name ?? bestWinner(r)?.label ?? dash,
      render: (r) => {
        const w = bestWinner(r)
        return w ? <span className="covwin">{w.name ?? w.label}</span> : <span className="na">{dash}</span>
      },
    },
    {
      key: 'tpg',
      label: 'tok/s/GPU',
      get: (r) => bestWinner(r)?.tps_per_gpu,
      num: true,
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
          {r.n_real === 0 && <> <MockBadge /></>}
        </a>
      ),
    },
  ]

  return (
    <div>
      <h2 className="viewhead">Coverage</h2>
      <p className="dek">
        Every model × GPU × workload × budget combination that has been searched —{' '}
        <b>{rows.length}</b> {rows.length === 1 ? 'combination' : 'combinations'}, <b>{scenarios.length}</b>{' '}
        {scenarios.length === 1 ? 'report' : 'reports'}. Combinations nobody has searched have no row; to add
        one, build a scenario on the Submit tab.
      </p>

      {rows.length === 0 ? (
        <div className="empty">No scenarios have been searched yet. Build one on the Submit tab and open a PR.</div>
      ) : (
        <CatalogTable rows={rows} cols={cols} rowKey={(r) => `${r.model}|${r.hardware}|${r.workload}|${r.gpu_budget}`} caption="Searched combinations" />
      )}

      <p className="catlinks">
        Full catalog spec-sheets:{' '}
        {CATALOG_TABS.map(([t, label], i) => (
          <span key={t}>
            {i > 0 && ' · '}
            <a href={`#/${t}`}>{label}</a>
          </span>
        ))}
      </p>
    </div>
  )
}
