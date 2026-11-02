// One crossfilter field as bars: one bar per value, sized by the page's bar measure. Click to filter.
import type { Explorer } from './useExplorer'
import { METRIC, better, fmt, fmtMetric, poolOf, rankConfigs, sortValues, NA } from './model'

interface Bar { v: string; n: number; ok: number; best: number | null; bestTopo: string; bestConc: number | null }

export function FieldChart({ x, k, title, help }: { x: Explorer; k: string; title: string; help?: string }) {
  const sub = x.shown(k)
  const vals = sortValues([...new Set(x.runs.map((r) => x.fv(r, k)))]).filter((v) => v !== NA)
  const sel = x.sel[k] ?? []
  const m = x.target
  const bars: Bar[] = vals.map((v) => {
    const rs = sub.filter((r) => x.fv(r, k) === v)
    const b = rankConfigs(rs, m)[0]
    return { v, n: rs.length, ok: rs.filter((r) => r.ok).length, best: b ? (b[m] as number) : null, bestTopo: b?.topo ?? '', bestConc: b?.conc ?? null }
  }).filter((b) => b.n > 0 || sel.includes(b.v))  // empty bars hide; a selected one stays so it can be cleared
  if (!bars.length) return null
  const measure = (b: Bar) => x.bar === 'pass' ? (b.n ? b.ok / b.n : null) : x.bar === 'count' ? b.n : b.best
  const text = (b: Bar) => x.bar === 'pass' ? (b.n ? `${fmt((b.ok / b.n) * 100, 0)}%` : '–') : x.bar === 'count' ? b.n.toLocaleString('en-US') : fmtMetric(m, b.best)
  // For a latency target the lowest value is the tallest bar.
  const inverted = x.bar === 'best' && !METRIC[m].higher
  const got = bars.map(measure).filter((v): v is number => v !== null && v > 0)
  const hi = Math.max(...got, 0), lo = Math.min(...got, Infinity)
  const size = (b: Bar) => { const v = measure(b); return v === null || !(v > 0) ? 0 : Math.max(3, (inverted ? lo / v : v / hi) * 100) }
  const win = bars.reduce<Bar | null>((a, b) => {
    const v = measure(b)
    if (v === null || (x.bar === 'best' && b.best === null)) return a
    if (!a) return b
    const av = measure(a) as number
    return (x.bar === 'best' ? better(m, v, av) : v > av) ? b : a
  }, null)
  const horizontal = bars.length > 5 || bars.some((b) => b.v.length > 7)
  const tip = (b: Bar) => `${title} ${b.v}: ${b.n} runs in filter, ${b.ok} meet every SLO`
    + (b.best !== null ? `; best ${fmtMetric(m, b.best)} ${METRIC[m].label} (${b.bestTopo}, ${b.bestConc} sessions)` : '')
  const cls = (b: Bar, base: string) => [base, sel.includes(b.v) && 'on', sel.length && !sel.includes(b.v) && 'dim', b === win && 'win', !b.n && 'zero'].filter(Boolean).join(' ')
  const pool = poolOf(k)

  return (
    <div className={`xf-field${horizontal ? ' wide' : ''}`} data-field={k}>
      <div className="xf-field-h">
        {pool && <span className="xf-pool">{pool}</span>}
        <b>{title}</b>
        {sel.length > 0 && <button type="button" className="linkbtn" onClick={() => x.clearField(k)}>clear</button>}
      </div>
      {help && <p className="xf-help">{help}</p>}
      {horizontal ? (
        <div className="xf-hbars">
          {bars.map((b) => (
            <button type="button" key={b.v} className={cls(b, 'xf-hbar')} aria-pressed={sel.includes(b.v)} title={tip(b)} aria-label={tip(b)} onClick={() => x.toggleValue(k, b.v)}>
              <span className="xf-v">{b.v}</span>
              <span className="xf-track"><i style={{ width: `${size(b)}%` }} /></span>
              <span className="xf-n">{text(b)}</span>
            </button>
          ))}
        </div>
      ) : (
        <div className="xf-vbars">
          {bars.map((b) => (
            <button type="button" key={b.v} className={cls(b, 'xf-vbar')} aria-pressed={sel.includes(b.v)} title={tip(b)} aria-label={tip(b)} onClick={() => x.toggleValue(k, b.v)}>
              <span className="xf-n">{text(b)}</span>
              <span className="xf-col"><i style={{ height: `${size(b)}%` }} /></span>
              <span className="xf-v">{b.v}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function barKey(x: Explorer): string {
  if (x.bar === 'pass') return 'Bars show the share of runs meeting every SLO.'
  if (x.bar === 'count') return 'Bars show how many runs are in the filter.'
  return `Bars show the best ${METRIC[x.target].label} among runs meeting every SLO${METRIC[x.target].higher ? '' : '; lower is better, so lower values draw taller'}.`
}
