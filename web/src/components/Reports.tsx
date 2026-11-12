import type { Scenario } from '../types'
import { canonFilters, facetCounts, filterScenarios, finite, fnum, reportsHash, scenarioHash, type FilterKey, type Filters } from '../lib'
import { navigate } from '../route'
import { ConfigName } from './ConfigName'

export function filtersFromParams(p: Record<string, string>): Filters {
  return { model: p.model ?? '', hardware: p.hardware ?? '', workload: p.workload ?? '', gpu: p.gpu ?? '' }
}
const toParams = (f: Filters): Record<string, string> => ({ model: f.model, hardware: f.hardware, workload: f.workload, gpu: f.gpu })

const FACETS: { key: FilterKey; label: string; fmt?: (v: string) => string }[] = [
  { key: 'model', label: 'Model' },
  { key: 'workload', label: 'Workload' },
  { key: 'hardware', label: 'Hardware' },
  { key: 'gpu', label: 'Accelerator budget', fmt: (v) => `${v}×` },
]

/** The card's performance strip: per-accelerator throughput (the ranking objective, with the fleet
 *  total under it), per-user speed, and token latency, each its own column. */
export function Metrics({ s }: { s: Scenario }) {
  const w = s.winner
  if (w === null) return null
  return (
    <div className="perf">
      <div className="m key"><span className="l">Per accelerator</span><b className="num">{fnum(w.tps_per_gpu, w.tps_per_gpu >= 100 ? 0 : 1)}<small>tok/s</small></b><span className="sub"><span className="num">{fnum(w.tps, 0)}</span> total</span></div>
      <div className="m"><span className="l">Per user</span><b className="num">{fnum(w.per_user, 1)}<small>tok/s</small></b></div>
      {finite(w.itl_p99) && <div className="m"><span className="l">Latency p99</span><b className="num">{fnum(w.itl_p99, 1)}<small>ms</small></b></div>}
    </div>
  )
}

/** What a report with no winner says: why, and how the runs ended. */
export function NoWinner({ s }: { s: Scenario }) {
  const refused = s.outcomes.refused ?? 0
  const top = s.refusals[0]
  const headline = s.state === 'all_refused'
    ? 'The simulator refused every configuration'
    : 'No configuration held the SLOs'
  return (
    <div className="nowin" role="note">
      <b>{headline}</b>
      <span className="sub">
        {Object.entries(s.outcomes).map(([k, n]) => `${n} ${k.replace(/_/g, ' ')}`).join(', ') || 'no runs recorded'}
      </span>
      {refused > 0 && top && <span className="sub">Most common refusal: {top.reason}</span>}
    </div>
  )
}

function ScenarioCard({ s }: { s: Scenario }) {
  const w = s.winner
  if (w === null) {
    return (
      <a className="card" href={scenarioHash(s.run_id)}>
        <div className="cardhead"><h3>{s.model}</h3>{s.caveat && <CaveatBadge caveat={s.caveat} />}</div>
        <dl className="specrows">
          <div><dt>Cluster</dt><dd><span className="num">{s.gpu_budget}</span> × {s.hardware}</dd></div>
          <div><dt>Workload</dt><dd>{s.workload}</dd></div>
        </dl>
        <div className="cardsec"><NoWinner s={s} /></div>
        <div className="foot"><span /><span className="open">details</span></div>
      </a>
    )
  }
  const front = s.front && s.front.members.length > 0 ? s.front.members.length : null
  return (
    <a className="card" href={scenarioHash(s.run_id)}>
      <div className="cardhead"><h3>{s.model}</h3>{s.caveat && <CaveatBadge caveat={s.caveat} />}</div>
      <dl className="specrows">
        <div><dt>Cluster</dt><dd><span className="num">{s.gpu_budget}</span> × {s.hardware}</dd></div>
        <div><dt>Workload</dt><dd>{s.workload}</dd></div>
      </dl>
      <div className="cardsec">
        <div className="winlbl">#1 config</div>
        <ConfigName label={w.label} variant={w.variant} reserve />
      </div>
      <div className="cardsec">
        <Metrics s={s} />
      </div>
      <div className="foot">
        <span>{w.c_use != null && <>At <span className="num">{fnum(w.c_use, 0)}</span> concurrent users</>}
          {front !== null && s.ranking.used === 'pareto' && <> · front of <span className="num">{front}</span></>}</span>
        <span className="open">details</span>
      </div>
    </a>
  )
}

/** Flags a scenario whose run ledger mixes seeds or stacks. */
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
                  className={`opt${on ? ' on' : ''}${disabled ? ' off' : ''}`}
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

  return (
    <div className="reportsview">
      <div className="viewtop">
        <h2 className="viewhead">Scenarios</h2>
        <span className="count"><b className="num">{filtered.length}</b> of {scenarios.length} shown</span>
      </div>

      <div className="specgrid">
        <FacetRail scenarios={scenarios} f={f} onToggle={setFilter} onReset={reset} />
        <main className="cards">
          {filtered.length === 0 ? (
            <div className="empty">No scenarios match this filter. <button type="button" className="reset" onClick={reset}>clear filters</button></div>
          ) : (
            filtered.map((s) => (
              <ScenarioCard s={s} key={s.run_id} />
            ))
          )}
        </main>
      </div>

      {filtered.some((s) => s.caveat) && (
        <p className="listnote">Cards marked <span className="caveat">caveat</span> mix runs from different seeds or stacks. Open the scenario to read why before comparing its numbers.</p>
      )}
    </div>
  )
}
