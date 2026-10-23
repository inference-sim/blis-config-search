import type { Catalog, ModelEntry, Scenario } from '../types'
import { countReports, dash, reportsHash } from '../lib'
import { CatalogTable, type Col } from './CatalogTable'

export function ReportsLink({ n, params }: { n: number; params: Record<string, string> }) {
  return n > 0 ? <a href={reportsHash(params)}>{n} report{n === 1 ? '' : 's'}</a> : <span className="na">{dash}</span>
}

/** The three spec-sheets are reached from Coverage rather than the primary nav, so each one
 *  carries its own way back — nothing in the masthead points at them. */
export function CatalogCrumb() {
  return <p className="crumb"><a href="#/coverage">← coverage</a></p>
}

export function Models({ catalog, scenarios }: { catalog: Catalog; scenarios: Scenario[] }) {
  const cols: Col<ModelEntry>[] = [
    { key: 'name', label: 'model', get: (m) => m.name, render: (m) => <span className="mono">{m.name}</span> },
    { key: 'arch', label: 'arch', get: (m) => m.arch },
    { key: 'type', label: 'type', get: (m) => m.model_type },
    { key: 'params', label: 'params (B)', get: (m) => m.params?.total_b, num: true },
    { key: 'active', label: 'active (B)', get: (m) => m.params?.active_b, num: true },
    { key: 'layers', label: 'layers', get: (m) => m.layers, num: true },
    { key: 'hidden', label: 'hidden', get: (m) => m.hidden, num: true },
    { key: 'hq', label: 'heads q', get: (m) => m.heads_q, num: true },
    { key: 'hkv', label: 'heads kv', get: (m) => m.heads_kv, num: true },
    { key: 'experts', label: 'experts', get: (m) => m.experts, num: true },
    { key: 'topk', label: 'top-k', get: (m) => m.top_k, num: true },
    { key: 'ctx', label: 'context', get: (m) => m.context, num: true },
    { key: 'quant', label: 'quant', get: (m) => m.quant },
    {
      key: 'reports',
      label: 'reports',
      get: (m) => countReports(scenarios, 'model', m.name),
      render: (m) => <ReportsLink n={countReports(scenarios, 'model', m.name)} params={{ model: m.name }} />,
    },
  ]
  return (
    <div>
      <CatalogCrumb />
      <h2 className="viewhead">Models</h2>
      <p className="dek"><b>{catalog.models.length}</b> models in the catalog. Missing values are shown as {dash}.</p>
      <CatalogTable rows={catalog.models} cols={cols} rowKey={(m) => m.name} caption="Model catalog" />
    </div>
  )
}
