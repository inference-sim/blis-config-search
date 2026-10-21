import type { Scenario } from '../types'
import { reportHref } from '../load'
import { Dropdown, type DropdownOption } from './Dropdown'
import { NO_FILTERS, canonFilters, facetCounts, filterScenarios, fnum, reportsHash, scenarioHash, topConfigs, type FilterKey, type Filters } from '../lib'
import { navigate } from '../route'

// Model palette for the cross-scenario overview (teal reserved for "the number", so model
// hues start after it). Mirrors the detail explorer's ramp.
const MODEL_PALETTE = ['#1f6e5c', '#34366e', '#b5651d', '#a3301f', '#6b7d1f', '#7a4a8c', '#2a7f9e', '#8a6d00']
const OW = 560, OH = 290, OM = { l: 52, r: 16, t: 14, b: 40 }

function oticks(max: number, n = 5): number[] {
  const step = max / n
  return Array.from({ length: n + 1 }, (_, i) => i * step)
}

// Deck-style cross-scenario Pareto: one marker per filtered scenario's winner, model = colour,
// hollow = mock, click = open that scenario. Updates live with the Reports filters.
function ParetoOverview({ scenarios, colors }: { scenarios: Scenario[]; colors: Map<string, string> }) {
  const pts = scenarios
    .filter((s) => Number.isFinite(s.winner.tps) && Number.isFinite(s.winner.per_user))
    .map((s) => ({ s, x: s.winner.tps, y: s.winner.per_user }))
  if (pts.length < 2) return null // a frontier needs ≥2 points
  const xmax = Math.max(...pts.map((p) => p.x), 1) * 1.08
  const ymax = Math.max(...pts.map((p) => p.y), 1) * 1.08
  const pw = OW - OM.l - OM.r, ph = OH - OM.t - OM.b
  const px = (x: number) => OM.l + (x / xmax) * pw
  const py = (y: number) => OM.t + ph - (y / ymax) * ph
  const models = [...new Set(pts.map((p) => p.s.model))]
  return (
    <figure className="vc overview">
      <div className="vc-h"><h4 className="vc-t">Deployment frontier</h4><span className="vc-r">{pts.length} in view</span></div>
      <p className="vc-s">each scenario's recommended config · cluster tok/s vs per-user tok/s</p>
      <svg viewBox={`0 0 ${OW} ${OH}`} className="chart" role="img" aria-label="Cross-scenario Pareto frontier of recommended configs">
        <g className="axes">
          {oticks(ymax).map((t) => {
            const y = OM.t + ph - (t / ymax) * ph
            return <g key={`y${t}`}><line x1={OM.l} x2={OW - OM.r} y1={y} y2={y} className="grid" /><text x={OM.l - 6} y={y + 3} textAnchor="end" className="tick">{fnum(t, 0)}</text></g>
          })}
          {oticks(xmax).map((t) => <text key={`x${t}`} x={OM.l + (t / xmax) * pw} y={OH - OM.b + 15} textAnchor="middle" className="tick">{fnum(t, 0)}</text>)}
          <line x1={OM.l} x2={OW - OM.r} y1={OM.t + ph} y2={OM.t + ph} className="axis" />
          <line x1={OM.l} x2={OM.l} y1={OM.t} y2={OM.t + ph} className="axis" />
          <text x={OM.l + pw / 2} y={OH - 5} textAnchor="middle" className="axlabel">cluster tok/s</text>
          <text transform={`translate(12 ${OM.t + ph / 2}) rotate(-90)`} textAnchor="middle" className="axlabel">tok/s per user</text>
        </g>
        {pts.map((p) => {
          const c = colors.get(p.s.model) ?? MODEL_PALETTE[0]
          return (
            <a key={p.s.run_id} href={scenarioHash(p.s.run_id)} aria-label={`Open ${p.s.model} ${p.s.hardware}`}>
              <circle cx={px(p.x)} cy={py(p.y)} r={5} fill={p.s.synthetic ? 'var(--panel)' : c} stroke={c} strokeWidth={1.6} className="opt">
                <title>{`${p.s.model} · ${p.s.hardware} · ${p.s.workload}${p.s.synthetic ? ' (mock)' : ''} — ${p.s.winner.label}: ${fnum(p.x)} tok/s, ${fnum(p.y)} tok/s/user`}</title>
              </circle>
            </a>
          )
        })}
      </svg>
      <p className="vc-k">up and to the right serves more users faster; hollow = mock · click a point to open it</p>
      <div className="legend">{models.map((m) => <span key={m}><i className="sw2" style={{ background: colors.get(m) }} />{m}</span>)}</div>
    </figure>
  )
}

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
  // stable model→colour map across all scenarios so a model keeps its hue as filters change
  const modelColors = new Map([...new Set(scenarios.map((s) => s.model))].sort().map((m, i) => [m, MODEL_PALETTE[i % MODEL_PALETTE.length]]))

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

      {filtered.length >= 2 && (
        <section className="overview-wrap" aria-label="Deployment frontier">
          <ParetoOverview scenarios={filtered} colors={modelColors} />
        </section>
      )}

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
