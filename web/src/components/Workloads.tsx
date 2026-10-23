import type { Catalog, Scenario, WorkloadEntry } from '../types'
import { countReports, dash } from '../lib'
import { CatalogTable, type Col } from './CatalogTable'
import { CatalogCrumb, ReportsLink } from './Models'

/** The catalog's workloads/ namespace holds two kinds of entry, and the difference is the
 *  point of this page rather than a detail to smooth over. A SHAPE (chatbot, sharegpt, …)
 *  states prompt/output token distributions a cost model draws from. A REPLAY (agentx-mvp)
 *  names a load generator's trace-replay scenario over a pinned corpus: its traffic is real
 *  recorded requests, with per-request lengths and a prefix tree the corpus carries, so it
 *  has no ISL/OSL mean to state.
 *
 *  So a replay's distribution cells are not missing data — there is no such number to show.
 *  Rendering them as the same em-dash the page uses for "unknown" would say the opposite, so
 *  a replay row shows its provenance (scenario, corpus, revision) spanning those columns
 *  instead, and the column heads carry the ISL/OSL legend only for the shapes. */
export function Workloads({ catalog, scenarios }: { catalog: Catalog; scenarios: Scenario[] }) {
  const shapes = catalog.workloads.filter((w) => w.kind !== 'replay')
  const replays = catalog.workloads.filter((w) => w.kind === 'replay')

  const reports = (w: WorkloadEntry) => countReports(scenarios, 'workload', w.name)
  const reportsCol: Col<WorkloadEntry> = {
    key: 'reports',
    label: 'reports',
    get: reports,
    render: (w) => <ReportsLink n={reports(w)} params={{ workload: w.name }} />,
  }
  const nameCol: Col<WorkloadEntry> = {
    key: 'name',
    label: 'workload',
    get: (w) => w.name,
    render: (w) => <span className="mono">{w.name}</span>,
  }

  const shapeCols: Col<WorkloadEntry>[] = [
    nameCol,
    { key: 'prefix', label: 'prefix tok', get: (w) => w.prefix_tokens, num: true },
    { key: 'im', label: 'ISL mean', get: (w) => w.isl_mean, num: true },
    { key: 'is', label: 'ISL stdev', get: (w) => w.isl_stdev, num: true },
    { key: 'imin', label: 'ISL min', get: (w) => w.isl_min, num: true },
    { key: 'imax', label: 'ISL max', get: (w) => w.isl_max, num: true },
    { key: 'om', label: 'OSL mean', get: (w) => w.osl_mean, num: true },
    { key: 'os', label: 'OSL stdev', get: (w) => w.osl_stdev, num: true },
    { key: 'omin', label: 'OSL min', get: (w) => w.osl_min, num: true },
    { key: 'omax', label: 'OSL max', get: (w) => w.osl_max, num: true },
    reportsCol,
  ]

  const replayCols: Col<WorkloadEntry>[] = [
    nameCol,
    {
      key: 'scenario',
      label: 'scenario',
      get: (w) => w.scenario,
      render: (w) => <span className="mono">{w.scenario ?? dash}</span>,
    },
    { key: 'harness', label: 'harness', get: (w) => w.harness_version },
    {
      key: 'corpus',
      label: 'corpus',
      get: (w) => w.corpus,
      render: (w) => (
        <span className="mono">
          {w.corpus ?? dash}
          {w.corpus_revision ? <span className="dim">@{w.corpus_revision}</span> : null}
        </span>
      ),
    },
    { key: 'category', label: 'category', get: (w) => w.category },
    { key: 'thinking', label: 'thinking', get: (w) => w.thinking },
    reportsCol,
  ]

  return (
    <div>
      <CatalogCrumb />
      <h2 className="viewhead">Workloads</h2>
      <p className="dek">
        <b>{catalog.workloads.length}</b> workloads: <b>{shapes.length}</b>{' '}
        {shapes.length === 1 ? 'shape' : 'shapes'}, stated as input / output sequence lengths in
        tokens, and <b>{replays.length}</b> trace-driven{' '}
        {replays.length === 1 ? 'replay' : 'replays'} over a pinned corpus. Missing values are
        shown as {dash}.
      </p>

      <h3 className="subhead">Shapes</h3>
      <p className="dek">
        Synthetic traffic: a cost model draws each request's prompt and output length from these
        distributions, sharing one prefix across requests.
      </p>
      <CatalogTable rows={shapes} cols={shapeCols} rowKey={(w) => w.name} caption="Workload shapes" />

      {replays.length > 0 && (
        <>
          <h3 className="subhead">Trace-driven replays</h3>
          <p className="dek">
            Recorded traffic: a harness replays real requests from the corpus below, so the
            per-request lengths and the prefix tree come from the trace and there is no ISL/OSL
            distribution to state. The corpus is fetched and digest-checked, never vendored.
          </p>
          <CatalogTable
            rows={replays}
            cols={replayCols}
            rowKey={(w) => w.name}
            caption="Trace-driven replay workloads"
          />
        </>
      )}
    </div>
  )
}
