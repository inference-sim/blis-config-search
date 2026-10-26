import type { Scenario } from '../types'
import { canonFilters, facetCounts, filterScenarios, fnum, reportsHash, scenarioHash, type FilterKey, type Filters } from '../lib'
import { navigate } from '../route'

export function filtersFromParams(p: Record<string, string>): Filters {
  return { model: p.model ?? '', hardware: p.hardware ?? '', workload: p.workload ?? '', gpu: p.gpu ?? '' }
}
const toParams = (f: Filters): Record<string, string> => ({ model: f.model, hardware: f.hardware, workload: f.workload, gpu: f.gpu })

const FACETS: { key: FilterKey; label: string; fmt?: (v: string) => string }[] = [
  { key: 'model', label: 'Model' },
  { key: 'hardware', label: 'Hardware' },
  { key: 'workload', label: 'Workload' },
  { key: 'gpu', label: 'GPU budget', fmt: (v) => `${v}×` },
]

export function Metrics({ s }: { s: Scenario }) {
  const w = s.winner
  return (
    <div className="perf">
      <div className="m"><b className="num">{fnum(w.tps, 0)}</b><span>tok/s</span></div>
      <div className="m"><b className="num">{fnum(w.tps_per_gpu, 0)}</b><span>tok/s/GPU</span></div>
      <div className="m"><b className="num">{fnum(w.per_user, 0)}</b><span>tok/s/user</span></div>
    </div>
  )
}

export function MockBadge() {
  return <span className="mock" title="Synthetic placeholder scenario, not a real simulation run">mock</span>
}

/** Flags a scenario whose runs are real BLIS output but do not measure what the scenario says
 *  they measure (e.g. a workload label the runs were not actually priced on). Distinct from
 *  `mock`, which marks data that is not a simulation at all. */
export function CaveatBadge({ caveat }: { caveat: string }) {
  return <span className="caveat" title={caveat}>caveat</span>
}

// mock-A facet rail: each facet value is a clickable option with a live count (how many
// scenarios it yields under the OTHER active facets); zero-count options dim and disable.
function FacetRail({ scenarios, f, onToggle, onReset }: {
  scenarios: Scenario[]; f: Filters; onToggle: (k: FilterKey, v: string) => void; onReset: () => void
}) {
  const anyActive = (Object.keys(f) as FilterKey[]).some((k) => f[k])
  return (
    <aside className="facetrail" aria-label="Filters">
      <div className="facethead">
        <h3>Filter</h3>
        {anyActive && <button type="button" className="reset" onClick={onReset}>clear</button>}
      </div>
      {FACETS.map(({ key, label, fmt }) => {
        const counts = facetCounts(scenarios, f, key)
        return (
          <div className="facet" key={key}>
            <p className="flabel">{label}</p>
            {counts.map(({ value, count }) => {
              const on = f[key] === value
              const disabled = count === 0 && !on
              return (
                <button
                  type="button"
                  key={value}
                  className={`opt${on ? ' on' : ''}${disabled ? ' empty' : ''}`}
                  aria-pressed={on}
                  disabled={disabled}
                  onClick={() => onToggle(key, on ? '' : value)}
                >
                  <span>{fmt ? fmt(value) : value}</span>
                  <span className="n">{count}</span>
                </button>
              )
            })}
          </div>
        )
      })}
    </aside>
  )
}

export function Reports({ scenarios, params }: { scenarios: Scenario[]; params: Record<string, string> }) {
  const f = canonFilters(scenarios, filtersFromParams(params))
  const filtered = filterScenarios(scenarios, f)
  const setFilter = (k: FilterKey, v: string) => navigate(reportsHash(toParams({ ...f, [k]: v })))
  const reset = () => navigate(reportsHash(toParams({ model: '', hardware: '', workload: '', gpu: '' })))
  const nReal = filtered.filter((s) => !s.synthetic).length

  return (
    <div className="reportsview">
      <div className="viewtop">
        <h2 className="viewhead">Scenarios</h2>
        <span className="count"><b className="num">{filtered.length}</b> of {scenarios.length} shown · <span className="num">{nReal}</span> real</span>
      </div>

      <div className="specgrid">
        <FacetRail scenarios={scenarios} f={f} onToggle={setFilter} onReset={reset} />
        <main className="cards">
          {filtered.length === 0 ? (
            <div className="empty">No scenarios match this filter. <button type="button" className="reset" onClick={reset}>clear filters</button></div>
          ) : (
            filtered.map((s) => (
              <a className={`card${s.synthetic ? ' is-mock' : ''}`} key={s.run_id} href={scenarioHash(s.run_id)}>
                <div className="cardhead"><h3>{s.model}</h3>{s.synthetic && <MockBadge />}{!s.synthetic && s.caveat && <CaveatBadge caveat={s.caveat} />}</div>
                {/* model / cluster / workload on their own lines — one dimension per row reads
                    more clearly than a single run-on chip row. */}
                <dl className="specrows">
                  <div><dt>Cluster</dt><dd><span className="chip">{s.hardware}</span><span className="chip">{s.gpu_budget}× GPU</span></dd></div>
                  <div><dt>Workload</dt><dd><span className="chip">{s.workload}</span><span className="chip">{s.isl_osl}</span></dd></div>
                </dl>
                <div className="winwrap">
                  <div className="winlbl">Recommended config</div>
                  <div className="win">{s.winner.name ?? s.winner.label}</div>
                  <Metrics s={s} />
                  <div className="foot">
                    <span className="open">details</span>
                  </div>
                </div>
              </a>
            ))
          )}
        </main>
      </div>

      {filtered.some((s) => s.synthetic) && (
        <p className="listnote">Cards marked <MockBadge /> are illustrative placeholder scenarios, not BLIS output. Open a scenario for its recommended configs and interactive Pareto charts.</p>
      )}
      {filtered.some((s) => !s.synthetic && s.caveat) && (
        <p className="listnote">Cards marked <span className="caveat">caveat</span> hold real BLIS runs that do not measure quite what the scenario claims. Open the scenario to read the caveat before deploying from its numbers.</p>
      )}
    </div>
  )
}
