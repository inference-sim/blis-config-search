import type { Scenario } from '../types'
import { reportHref } from '../load'
import { Dropdown, type DropdownOption } from './Dropdown'
import { NO_FILTERS, canonFilters, facetCounts, filterScenarios, fnum, reportsHash, scenarioHash, topConfigs, type FilterKey, type Filters } from '../lib'
import { navigate } from '../route'

export function filtersFromParams(p: Record<string, string>): Filters {
  return { model: p.model ?? '', hardware: p.hardware ?? '', workload: p.workload ?? '', gpu: p.gpu ?? '' }
}
const toParams = (f: Filters): Record<string, string> => ({ model: f.model, hardware: f.hardware, workload: f.workload, gpu: f.gpu })

const FACETS: { key: FilterKey; label: string; all: string }[] = [
  { key: 'model', label: 'model', all: 'all models' },
  { key: 'hardware', label: 'hardware', all: 'all hardware' },
  { key: 'workload', label: 'workload', all: 'all workloads' },
  { key: 'gpu', label: 'gpu budget', all: 'any budget' },
]

export function Metrics({ s }: { s: Scenario }) {
  const w = s.winner
  return (
    <dl className="metrics">
      <div><dt>tok/s</dt><dd className="num best">{fnum(w.tps)}</dd></div>
      <div><dt>tok/s/GPU</dt><dd className="num best">{fnum(w.tps_per_gpu)}</dd></div>
      <div><dt>tok/s/user</dt><dd className="num best">{fnum(w.per_user)}</dd></div>
    </dl>
  )
}

export function MockBadge() {
  return <span className="mock" title="Synthetic placeholder scenario, not a real simulation run">mock</span>
}

function ConfigCard({ s }: { s: Scenario }) {
  return (
    <a className={`cfgcard${s.synthetic ? ' is-mock' : ''}`} href={scenarioHash(s.run_id)}>
      <div className="cfglabel">{s.winner.label}{s.synthetic && <MockBadge />}</div>
      <div className="cfgsub mono">{s.model} · {s.hardware} · {s.workload}</div>
      <Metrics s={s} />
    </a>
  )
}

export function Reports({ scenarios, params }: { scenarios: Scenario[]; params: Record<string, string> }) {
  const f = canonFilters(scenarios, filtersFromParams(params))
  const filtered = filterScenarios(scenarios, f)
  const { items: top, mockOnly } = topConfigs(filtered)
  const active = (Object.keys(f) as FilterKey[]).filter((k) => f[k])
  const setFilter = (k: FilterKey, v: string) => navigate(reportsHash(toParams({ ...f, [k]: v })))
  const nReal = filtered.filter((s) => !s.synthetic).length

  return (
    <div>
      <h2 className="viewhead">Reports</h2>
      <section className="summary" aria-label="Summary">
        <p className="dek">
          {scenarios.length === 0 ? (
            'No scenarios have been published yet.'
          ) : (
            <>
              <b>{nReal}</b> real and <b>{filtered.length - nReal}</b> mock scenario{filtered.length === 1 ? '' : 's'} match the current filter. Each report searches parallelism and instance layouts
              for the most output throughput that still holds the per-user interactivity floor. Simulator predictions, not measurements.
            </>
          )}
        </p>
        <div className="seclabel2">{mockOnly ? 'Mock configs (no real scenario matches)' : 'Top configs to try'}</div>
        {top.length === 0 ? (
          <div className="empty">No scenario matches these filters.</div>
        ) : (
          <>
            {mockOnly && <div className="warnnote">Only synthetic mock scenarios match; these configs are illustrative, not simulated results.</div>}
            <div className="cfgcards">{top.map((s) => <ConfigCard key={s.run_id} s={s} />)}</div>
          </>
        )}
      </section>

      <div className="chips filterbar" role="group" aria-label="Filters">
        {FACETS.map(({ key, label, all }) => {
          const opts: DropdownOption[] = [
            { value: '', label: `${all} (${filterScenarios(scenarios, { ...f, [key]: '' }).length})` },
            ...facetCounts(scenarios, f, key).map(({ value, count }) => ({ value, label: `${key === 'gpu' ? `${value} GPUs` : value} (${count})` })),
          ]
          return (
            <span className="flt" key={key}>
              {label}
              <Dropdown value={f[key]} options={opts} onChange={(v) => setFilter(key, v)} ariaLabel={`Filter by ${label}`} />
            </span>
          )
        })}
        <button type="button" className="clearbtn" disabled={active.length === 0} onClick={() => navigate(reportsHash(toParams(NO_FILTERS)))}>
          clear filters{active.length ? ` (${active.length})` : ''}
        </button>
        <span className="facet">{filtered.length} of {scenarios.length} shown</span>
      </div>

      <div className="gallery">
        {filtered.map((s) => (
          <article className={`card${s.synthetic ? ' is-mock' : ''}`} key={s.run_id}>
            <header>
              <h3><a href={scenarioHash(s.run_id)}>{s.model}</a></h3>
              {s.synthetic && <MockBadge />}
            </header>
            <div className="specchips">
              <span className="spec">{s.hardware}</span>
              <span className="spec">{s.gpu_budget} GPUs</span>
              <span className="spec">{s.workload}</span>
              <span className="spec">{s.isl_osl}</span>
            </div>
            <div className="cfglabel">{s.winner.label}</div>
            <Metrics s={s} />
            <div className="cardfoot">
              <span className="servable"><span className="num">{s.n_servable}</span> / <span className="num">{s.n_runs}</span> servable</span>
              <span className="links">
                <a href={scenarioHash(s.run_id)}>details</a>
                <a href={reportHref(s.run_id)} target="_blank" rel="noreferrer">full report ↗</a>
              </span>
            </div>
          </article>
        ))}
        {filtered.length === 0 && <div className="empty">No scenarios match. Clear a filter to see more.</div>}
      </div>
    </div>
  )
}
