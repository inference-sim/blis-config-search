// The top configs at the head of the page, ranked on the target. Tied configs fold into one card.
import { useState } from 'react'
import type { Explorer } from './useExplorer'
import type { MetricKey } from './types'
import { METRIC, METRIC_KEYS, fmtMetric, knobDiff } from './model'
import type { BarMeasure } from './useExplorer'
import { ConfigName } from '../components/ConfigName'

export function Controls({ x }: { x: Explorer }) {
  const bars: [BarMeasure, string][] = [['best', 'best of the target'], ['pass', '% meeting SLO'], ['count', 'run count']]
  return (
    <div className="xf-controls">
      <div className="xf-seg" role="group" aria-label="Rank by">
        <span className="xf-seg-l">rank by</span>
        {METRIC_KEYS.map((m) => <button type="button" key={m} aria-pressed={x.target === m} onClick={() => x.setTarget(m)}>{METRIC[m].label}</button>)}
      </div>
      <div className="xf-seg" role="group" aria-label="Bars show">
        <span className="xf-seg-l">bars show</span>
        {bars.map(([b, l]) => <button type="button" key={b} aria-pressed={x.bar === b} onClick={() => x.setBar(b)}>{l}</button>)}
      </div>
    </div>
  )
}

export function TopConfigs({ x }: { x: Explorer }) {
  const [more, setMore] = useState(false)
  const m = x.target
  const shown = x.ranked.slice(0, more ? 6 : 3)
  // every metric shows: the ranked one large, the other four in equal columns
  const rest: MetricKey[] = METRIC_KEYS.filter((k) => k !== m)
  const n = x.ranked.reduce((a, g) => a + 1 + g.ties.length, 0)
  return (
    <section className="xf-top" aria-label="Top configs">
      <div className="xf-top-h">
        <p className="eyebrow">Top configs</p>
        <span className="xf-sub">{METRIC[m].higher ? 'highest' : 'lowest'} {METRIC[m].label} at any load that meets every SLO · {n.toLocaleString('en-US')} configs met it</span>
      </div>
      {shown.length ? (
        <div className="xf-cards">
          {shown.map((g, i) => {
            const r = g.lead
            // name only the knobs that varied among the tied configs; the hover lists the values
            const noDiff = (ties: typeof g.ties) => {
              const keys = [...new Set(ties.flatMap((t) => knobDiff(r, t, x.data.routers).map((d) => d.split(':')[0])))]
              return keys.length > 2 ? `${keys.slice(0, 2).join(', ')} and ${keys.length - 2} more` : keys.join(' and ')
            }
            return (
              <button type="button" key={r.cfg} className="xf-cfg" aria-pressed={x.picked === r.cfg} onClick={() => x.pick(r.cfg)}>
                <span className="xf-cfg-rank"><b>{i + 1}</b><span className={`xf-mode${r.mode === 'P/D' ? ' pd' : ''}`}>{r.mode}</span><span className="xf-sub">{r.conc} sessions · from sweep <span className="mono">{r.sweep}</span></span></span>
                <ConfigName label={r.topo} variant={x.caption(r)} reserve />
                <span className="xf-metrics">
                  <span className="xf-hero">
                    <span className="xf-hero-v">{fmtMetric(m, r[m])}<small>{METRIC[m].unit}</small></span>
                    <span className="xf-hero-l">{METRIC[m].label}{METRIC[m].higher ? '' : ' (lower is better)'}</span>
                  </span>
                  <span className="xf-kpis">{rest.map((k) => (
                    <span key={k}><span className="xf-kpi-l">{METRIC[k].short}{METRIC[k].unit === 'ms' ? ' ms' : ''}</span><span className="xf-kpi-v">{fmtMetric(k, r[k])}</span></span>
                  ))}</span>
                </span>
                {g.ties.length > 0 && (
                  <span className="xf-ties" title={g.ties.map((t) => knobDiff(r, t, x.data.routers).join(', ')).join('\n')}>
                    same result with {g.ties.length} other config{g.ties.length === 1 ? '' : 's'}: {noDiff(g.ties)} made no difference
                  </span>
                )}
                <span className="xf-cfg-act" aria-hidden="true">{x.picked === r.cfg ? 'Showing this config · click to clear' : 'Filter the page to this config →'}</span>
              </button>
            )
          })}
        </div>
      ) : <p className="xf-empty">No run in the current filter meets every SLO.</p>}
      <div className="xf-top-f">
        {x.ranked.length > 3 && <button type="button" className="linkbtn" onClick={() => setMore((v) => !v)}>{more ? 'show top 3' : 'show top 6'}</button>}
      </div>
    </section>
  )
}
