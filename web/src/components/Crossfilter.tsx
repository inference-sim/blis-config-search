import { useRef, type KeyboardEvent, type PointerEvent } from 'react'
import type { RunRecord } from '../types'
import {
  FACETS, METRICS, facetLevels, fnum, histogram, matchesSel, selectionActive, varyingMetrics, visibleFacets,
  type FacetField, type MetricKey, type Range, type Selection,
} from '../lib'

const HW = 220
const HH = 84
const HM = { l: 4, r: 4, t: 6, b: 18 }
const BINS = 12

function BrushHistogram({ recs, sel, metric, onRange }: {
  recs: RunRecord[]; sel: Selection; metric: MetricKey; onRange: (r: Range | null) => void
}) {
  const meta = METRICS.find((m) => m.key === metric)!
  // Domain from ALL records so the axis does not jump while brushing; counts honour OTHER filters.
  const domain = histogram(recs.map((r) => r[metric]), BINS)
  const rows = recs.filter((r) => matchesSel(r, sel, { metric })).map((r) => r[metric])
  const anchor = useRef<{ v: number; x: number; moved: boolean } | null>(null)
  const ref = useRef<SVGSVGElement>(null)
  const range = sel.ranges[metric] ?? null
  if (!domain) return <div className="hist-note">{meta.label}: no spread to brush.</div>
  const { min, max } = domain
  const w = (max - min) / BINS
  const counts = new Array<number>(BINS).fill(0)
  for (const v of rows) if (typeof v === 'number' && Number.isFinite(v) && v >= min && v <= max) counts[Math.min(BINS - 1, Math.floor((v - min) / w))]++
  const peak = Math.max(1, ...counts)
  const pw = HW - HM.l - HM.r
  const ph = HH - HM.t - HM.b
  const xOf = (v: number) => HM.l + ((v - min) / (max - min)) * pw
  const valAt = (clientX: number): number => {
    const box = ref.current?.getBoundingClientRect()
    if (!box || box.width <= 0) return min
    const px = ((clientX - box.left) / box.width) * HW
    const t = Math.min(1, Math.max(0, (px - HM.l) / pw))
    return min + t * (max - min)
  }
  const down = (e: PointerEvent<SVGSVGElement>) => {
    e.currentTarget.setPointerCapture?.(e.pointerId)
    anchor.current = { v: valAt(e.clientX), x: e.clientX, moved: false }
  }
  const move = (e: PointerEvent<SVGSVGElement>) => {
    const a = anchor.current
    if (!a) return
    if (!a.moved && Math.abs(e.clientX - a.x) < 3) return
    a.moved = true
    const v = valAt(e.clientX)
    onRange(v === a.v ? null : [Math.min(a.v, v), Math.max(a.v, v)])
  }
  const up = () => {
    const a = anchor.current
    anchor.current = null
    if (a && !a.moved) onRange(null) // plain click clears
  }
  const key = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key === 'Escape' && range) { e.stopPropagation(); onRange(null) }
  }
  const label = `${meta.label} (${meta.unit}) histogram, ${fnum(min, meta.digits)} to ${fnum(max, meta.digits)}.`
    + (range ? ` Brushed ${fnum(range[0], meta.digits)} to ${fnum(range[1], meta.digits)}.` : ' Drag to brush a range; Escape clears.')
  return (
    <div className="hist">
      <div className="hist-h">
        <span className="hist-t">{meta.label} <span className="hist-u">{meta.unit}</span></span>
        {range && <button type="button" className="hist-clear" onClick={() => onRange(null)} aria-label={`Clear ${meta.label} brush`}>clear</button>}
      </div>
      <svg ref={ref} viewBox={`0 0 ${HW} ${HH}`} className="chart hist-svg" role="img" aria-label={label} tabIndex={0}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key}>
        {counts.map((c, i) => {
          const lo = min + i * w
          const hi = lo + w
          const inside = !range || (hi >= range[0] && lo <= range[1])
          const h = (c / peak) * ph
          return (
            <rect key={i} x={xOf(lo) + 0.5} y={HM.t + ph - h} width={Math.max(0, pw / BINS - 1)} height={h}
              className={inside ? 'hbar' : 'hbar off'}>
              <title>{`${fnum(lo, meta.digits)}–${fnum(hi, meta.digits)} ${meta.unit}: ${c} run${c === 1 ? '' : 's'}`}</title>
            </rect>
          )
        })}
        <line x1={HM.l} x2={HW - HM.r} y1={HM.t + ph} y2={HM.t + ph} className="axis" />
        {range && (
          <rect x={xOf(Math.max(min, range[0]))} y={HM.t} width={Math.max(1, xOf(Math.min(max, range[1])) - xOf(Math.max(min, range[0])))}
            height={ph} className="brushbox" />
        )}
        <text x={HM.l} y={HH - 4} className="tick">{fnum(min, meta.digits)}</text>
        <text x={HW - HM.r} y={HH - 4} textAnchor="end" className="tick">{fnum(max, meta.digits)}</text>
      </svg>
      {range && <div className="hist-r num">{fnum(range[0], meta.digits)} – {fnum(range[1], meta.digits)} {meta.unit}</div>}
    </div>
  )
}

export function Crossfilter({ recs, sel, setSel, shownCount, onPickSeries }: {
  recs: RunRecord[]; sel: Selection; setSel: (s: Selection) => void; shownCount: number
  onPickSeries?: (value: string, on: boolean) => void
}) {
  const metrics = varyingMetrics(recs)
  const fields = visibleFacets(recs)
  const active = selectionActive(sel)
  const setRange = (k: MetricKey, r: Range | null) => {
    const ranges = { ...sel.ranges }
    if (r) ranges[k] = r
    else delete ranges[k]
    setSel({ ...sel, ranges })
  }
  const toggle = (k: FacetField, v: string) => {
    const cur = sel.facets[k] ?? []
    const next = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]
    setSel({ ...sel, facets: { ...sel.facets, [k]: next } })
    if (k === 'series') onPickSeries?.(v, !cur.includes(v))
  }
  return (
    <div className="xf" onKeyDown={(e) => { if (e.key === 'Escape' && active) setSel({ facets: {}, ranges: {} }) }}>
      <div className="xf-head">
        <span className="xf-count"><b className="num">{shownCount}</b> of <span className="num">{recs.length}</span> runs selected</span>
        <button type="button" className="xf-clear" disabled={!active} onClick={() => setSel({ facets: {}, ranges: {} })}>clear all filters</button>
      </div>
      {metrics.length > 0 && (
        <>
          <div className="xf-sec">Filter by outcome — brush a range</div>
          <div className="hists">
            {metrics.map((m) => <BrushHistogram key={m} recs={recs} sel={sel} metric={m} onRange={(r) => setRange(m, r)} />)}
          </div>
        </>
      )}
      {fields.length > 0 && (
        <>
          <div className="xf-sec">Filter by setting — click a level</div>
          <div className="facetgrid">
            {fields.map((k) => {
              const picked = sel.facets[k] ?? []
              return (
                <div className="fgroup" key={k} role="group" aria-label={FACETS.find((f) => f.key === k)!.label}>
                  <div className="fgroup-l">{FACETS.find((f) => f.key === k)!.label}</div>
                  {facetLevels(recs, sel, k).map((lv) => {
                    const on = picked.includes(lv.value)
                    return (
                      <button type="button" key={lv.value} className={`lvl${on ? ' on' : ''}`} aria-pressed={on}
                        disabled={lv.count === 0 && !on} onClick={() => toggle(k, lv.value)}>
                        <span className="lvl-n">{lv.value}</span><span className="lvl-c num">{lv.count}</span>
                      </button>
                    )
                  })}
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
