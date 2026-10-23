import type { Catalog, Scenario, WorkloadEntry } from '../types'
import { countReports, dash } from '../lib'
import { CatalogTable, type Col } from './CatalogTable'
import { CatalogCrumb, ReportsLink } from './Models'

export function Workloads({ catalog, scenarios }: { catalog: Catalog; scenarios: Scenario[] }) {
  const cols: Col<WorkloadEntry>[] = [
    { key: 'name', label: 'workload', get: (w) => w.name, render: (w) => <span className="mono">{w.name}</span> },
    { key: 'prefix', label: 'prefix tok', get: (w) => w.prefix_tokens, num: true },
    { key: 'im', label: 'ISL mean', get: (w) => w.isl_mean, num: true },
    { key: 'is', label: 'ISL stdev', get: (w) => w.isl_stdev, num: true },
    { key: 'imin', label: 'ISL min', get: (w) => w.isl_min, num: true },
    { key: 'imax', label: 'ISL max', get: (w) => w.isl_max, num: true },
    { key: 'om', label: 'OSL mean', get: (w) => w.osl_mean, num: true },
    { key: 'os', label: 'OSL stdev', get: (w) => w.osl_stdev, num: true },
    { key: 'omin', label: 'OSL min', get: (w) => w.osl_min, num: true },
    { key: 'omax', label: 'OSL max', get: (w) => w.osl_max, num: true },
    {
      key: 'reports',
      label: 'reports',
      get: (w) => countReports(scenarios, 'workload', w.name),
      render: (w) => <ReportsLink n={countReports(scenarios, 'workload', w.name)} params={{ workload: w.name }} />,
    },
  ]
  return (
    <div>
      <CatalogCrumb />
      <h2 className="viewhead">Workloads</h2>
      <p className="dek"><b>{catalog.workloads.length}</b> workload presets (input / output sequence lengths in tokens). Missing values are shown as {dash}.</p>
      <CatalogTable rows={catalog.workloads} cols={cols} rowKey={(w) => w.name} caption="Workload presets" />
    </div>
  )
}
