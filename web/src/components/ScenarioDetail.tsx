import { useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import type { RunRecord, Scenario } from '../types'
import { reportHref } from '../load'
import { EMPTY_SEL, dash, emphasis, finite, fnum, operatingPoints, resolveHighlight, runFlags, selectRecords, sortRows, toggleHighlight, useSort, type Selection, type SortState } from '../lib'
import { Crossfilter } from './Crossfilter'
import { MockBadge } from './Reports'

const PALETTE = ['#1f6e5c', '#34366e', '#b5651d', '#a3301f', '#6b7d1f', '#7a4a8c', '#2a7f9e', '#8a6d00']

/** Highlight wiring shared by every chart: `active` is the already-resolved emphasised series. */
export interface HL {
  active: string | null
  onHover: (s: string | null) => void
  onToggle: (s: string) => void
  onClear: () => void
}
const GREY = '#9aa1a6'
const NO_HL: HL = { active: null, onHover: () => {}, onToggle: () => {}, onClear: () => {} }
/** Props that make an SVG group a focusable, clickable, hoverable per-series button. */
function seriesProps(hl: HL, name: string) {
  return {
    tabIndex: 0, role: 'button' as const, className: 'hl-target',
    'aria-label': `${name}${hl.active === name ? ' (highlighted)' : ''} — press Enter to toggle highlight`,
    'aria-pressed': hl.active === name,
    onPointerEnter: () => hl.onHover(name), onPointerLeave: () => hl.onHover(null),
    onFocus: () => hl.onHover(name), onBlur: () => hl.onHover(null),
    onClick: (e: { stopPropagation: () => void }) => { e.stopPropagation(); hl.onToggle(name) },
    onKeyDown: (e: KeyboardEvent<SVGGElement>) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); hl.onToggle(name) }
    },
  }
}

interface Pt {
  series: string
  x: number
  y: number
  ok: boolean
  color: string
  title: string
  /** RunRecord.id — lets a click resolve back to the run whose deployment to show. */
  id?: string
  /** Concurrency, used only to order a series' points into a trajectory polyline in the scatter. */
  order?: number
  /** Config identity (series + variant): one trajectory per config, as the Lines charts group. */
  cfg?: string
}
const W = 560
const H = 300
const M = { l: 48, r: 14, t: 12, b: 38 }

function ticks(max: number, n = 5): number[] {
  const step = max / n
  return Array.from({ length: n + 1 }, (_, i) => i * step)
}

function Axes({ xmax, ymax, xl, yl }: { xmax: number; ymax: number; xl: string; yl: string }) {
  const pw = W - M.l - M.r
  const ph = H - M.t - M.b
  return (
    <g className="axes">
      {ticks(ymax).map((t) => {
        const y = M.t + ph - (t / ymax) * ph
        return (
          <g key={`y${t}`}>
            <line x1={M.l} x2={W - M.r} y1={y} y2={y} className="grid" />
            <text x={M.l - 6} y={y + 3} textAnchor="end" className="tick">{fnum(t, 0)}</text>
          </g>
        )
      })}
      {ticks(xmax).map((t) => {
        const x = M.l + (t / xmax) * pw
        return <text key={`x${t}`} x={x} y={H - M.b + 14} textAnchor="middle" className="tick">{fnum(t, 0)}</text>
      })}
      <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} className="axis" />
      <line x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} className="axis" />
      <text x={M.l + pw / 2} y={H - 4} textAnchor="middle" className="axlabel">{xl}</text>
      <text transform={`translate(11 ${M.t + ph / 2}) rotate(-90)`} textAnchor="middle" className="axlabel">{yl}</text>
    </g>
  )
}

function Empty({ msg }: { msg: string }) {
  return <div className="chartempty">{msg}</div>
}

/** One labelled row in the detail panel. Rows whose value is absent are dropped by `defList`, so
 *  a run that predates a field shows a shorter panel rather than "null" or a blank cell. */
type Row = [string, string | number | null | undefined]

function defList(rows: Row[]) {
  const shown = rows.filter(([, v]) => v !== null && v !== undefined && v !== '')
  if (shown.length === 0) return null
  return (
    <dl className="dd-list2">
      {shown.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

const ms = (v: number | null | undefined) =>
  !finite(v) ? null : (v as number) >= 1000 ? `${fnum(v, 0)} ms` : `${fnum(v, 1)} ms`

/**
 * The clicked run's full deployment: what BLIS actually consumed, read from the run's own record
 * rather than reconstructed, so it cannot drift from the simulation. Sits inline under the chart
 * (the standalone report's panel does the same, so the two surfaces behave alike).
 */
export function PointDetail({ r, onClose, pinned = true, scenarioHasCmd = true }: {
  r: RunRecord; onClose: () => void; pinned?: boolean
  /** Whether ANY run in this scenario recorded a command. When false the whole fixture lacks
   *  `cmd` (a data gap, not a per-run one), so we drop the Reproduce panel rather than show the
   *  "not recorded" message on every point of the report. */
  scenarioHasCmd?: boolean
}) {
  const name = r.variant ? `${r.series} (${r.variant})` : r.series
  const box = useRef<HTMLDivElement>(null)
  // The panel renders below the chart grid, so on a laptop viewport it opens BELOW the fold —
  // measured ~70px past it, which makes a click look like it did nothing. Bring it into view
  // whenever it opens or switches to another run. `nearest` scrolls the minimum needed, so a
  // panel already on screen does not jump.
  useEffect(() => {
    // Only pull the panel into view on a deliberate click. On hover it follows the mouse across
    // series, and auto-scrolling on each one would make the page lurch under the cursor.
    if (!pinned) return
    // guarded: jsdom (and older engines) do not implement scrollIntoView
    box.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' })
  }, [r.id, pinned])
  return (
    <div ref={box} className="ptd" data-testid="pt-detail" role="region" aria-label={`Deployment for ${name}`}>
      <div className="ptd-h">
        <div>
          <div className="ptd-t">{r.series}{!pinned && <span className="ptd-peek">operating point · click to pin</span>}</div>
          {r.variant ? <div className="ptd-v">{r.variant}</div> : <div className="ptd-v dim">all knobs at defaults</div>}
        </div>
        {pinned && <button type="button" className="ptd-x" onClick={onClose}>close</button>}
      </div>
      {!r.ok && r.viol.length > 0 && (
        <p className="ptd-bad">Not servable: {r.viol.join('; ')}</p>
      )}
      <div className="ptd-g">
        <section>
          <h5>Parallelism</h5>
          {defList([
            ['Tensor parallel', r.tp], ['Data parallel', r.dp], ['Pipeline parallel', r.pp],
            ['Expert parallel', r.epw ? `width ${r.epw}` : 'off'],
            ['Instances', r.ni], ['GPUs total', finite(r.gpus) ? fnum(r.gpus, 0) : null],
            ['Nodes', r.nodes], ['Fabric', r.fabric],
          ])}
        </section>
        <section>
          <h5>Engine</h5>
          {defList([
            ['block_size', r.bs], ['max_num_seqs', r.mns],
            ['max_num_batched_tokens', finite(r.mnbt) ? fnum(r.mnbt, 0) : null],
            ['max_model_len', finite(r.mml) ? fnum(r.mml, 0) : null],
            ['gpu_memory_utilization', r.gmu], ['cudagraph_mode', r.cg],
            ['quantization', r.quant ?? 'checkpoint'], ['cache_dtype', r.cdt ?? 'checkpoint'],
            ['all2all_backend', r.a2a === 'n/a' ? null : r.a2a],
          ])}
        </section>
        <section>
          <h5>Routing &amp; policy</h5>
          {defList([
            ['routing_policy', r.route || 'round-robin'],
            ['routing_scorers', r.scorers],
            ['scheduler', r.schedp ?? 'default'],
            ['admission_policy', r.admit ?? 'always-admit'],
            ['prefill/decode', r.pd],
            ['draft tokens (MTP)', r.mtp || 'off'],
          ])}
        </section>
        <section>
          <h5>Measured at {finite(r.conc) ? `${fnum(r.conc, 0)} sessions` : 'this load'}</h5>
          {defList([
            ['Speed per user', finite(r.pu) ? `${fnum(r.pu, 1)} tok/s` : null],
            ['Fleet output', finite(r.cl) ? `${fnum(r.cl, 0)} tok/s` : null],
            ['Per GPU', finite(r.tpg) ? `${fnum(r.tpg, 1)} tok/s` : null],
            ['ITL / TPOT p99', ms(r.itl)], ['TTFT p99', ms(r.ttft)],
            ['E2E p99', ms(r.e2e)], ['Queue delay p99', ms(r.sched)],
            ['Servable', r.ok ? 'yes' : 'no'],
          ])}
        </section>
      </div>
      {/* Only show the Reproduce panel when the scenario records commands at all. If NO run in the
          fixture has one, the message would fire on every point — a data gap, so hide it entirely;
          if only this run lacks one (others have it), keep the honest per-run note. */}
      {scenarioHasCmd && (
        <div className="ptd-cmd">
          <h5>Reproduce this run</h5>
          {r.cmd
            ? <pre>{r.cmd}</pre>
            : <p className="ptd-none">
                The command was not recorded for this run{r.scen ? ` (scenario ${r.scen})` : ''}.
              </p>}
        </div>
      )}
    </div>
  )
}

/** One-line glossary for the throughput/latency column heads, surfaced as native `title`
 *  tooltips (same affordance as the mock badge). Keyed by the exact header text. */
const COL_HELP: Record<string, string> = {
  sessions: 'Concurrent sessions — simultaneous users/requests — at this operating point.',
  configuration: 'The serving topology and the engine/routing flags to deploy it with.',
  'tok/s per user': 'Output tokens per second a single session sees — the interactivity each caller feels.',
  'tok/s fleet': 'Total output tokens per second across every instance and GPU in the deployment — the whole config’s serving capacity.',
  'tok/s GPU': 'Fleet throughput divided by total GPUs — how efficiently each GPU is used.',
  'ITL ms': 'Inter-token latency (p99): the gap between output tokens. Lower feels snappier.',
  'TTFT ms': 'Time to first token (p99): how long a caller waits before the reply starts streaming.',
  // full per-run table below uses slightly different labels for the same quantities
  series: 'The serving topology — GPUs × tensor-parallel × expert-parallel × instances.',
  conc: 'Concurrent sessions — simultaneous users/requests — this run was swept at.',
  GPUs: 'Total GPUs the deployment consumes (tensor × pipeline parallel × instances).',
  'cluster tok/s': 'Total output tokens per second across the whole deployment — its serving capacity.',
  'tok/s/user': 'Output tokens per second a single session sees — the interactivity each caller feels.',
  status: 'Whether the run held the interactivity floor, or which constraint it broke.',
}
/** A header term with its glossary tooltip. The whole label (term + the "?" icon) opens a real
 *  hover/focus tooltip that shows the term's description. It is portaled to <body> and fixed-
 *  positioned so the surrounding tables' `overflow:auto`/`hidden` can't clip it (the reason the
 *  native `title` felt dead), and it opens on keyboard focus too, not hover alone. */
function Hcol({ term }: { term: string }) {
  const help = COL_HELP[term]
  const ref = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)
  // Position the bubble under the label each time it opens, from the label's viewport rect.
  useLayoutEffect(() => {
    if (!open || !ref.current) return
    const r = ref.current.getBoundingClientRect()
    setPos({ left: r.left, top: r.bottom + 6 })
  }, [open])
  if (!help) return <>{term}</>
  return (
    <span
      ref={ref}
      className="hcol"
      tabIndex={0}
      role="note"
      aria-label={`${term}: ${help}`}
      onPointerEnter={() => setOpen(true)}
      onPointerLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {term}<span className="hicon" aria-hidden="true">?</span>
      {open && pos && createPortal(
        <span className="hcol-pop" role="tooltip" style={{ left: pos.left, top: pos.top }}>{help}</span>,
        document.body,
      )}
    </span>
  )
}

/** A sortable column header: the glossary label wrapped in the shared `.sortbtn`, with the
 *  ascending/descending arrow and `aria-sort` matching CatalogTable's pattern. `sortKey` is the
 *  field this column sorts on; clicking cycles asc → desc → unsorted. */
function SortTh({ term, sortKey, sort, onToggle, num = false }: {
  term: string; sortKey: string; sort: SortState | null; onToggle: (k: string) => void; num?: boolean
}) {
  const on = sort?.key === sortKey
  return (
    <th className={num ? 'num' : undefined}
      aria-sort={on ? (sort!.dir === 1 ? 'ascending' : 'descending') : 'none'}>
      <button type="button" className="sortbtn" onClick={() => onToggle(sortKey)}>
        <Hcol term={term} />
        <span className="arrow">{on ? (sort!.dir === 1 ? '▲' : '▼') : ''}</span>
      </button>
    </th>
  )
}

/** The configuration cell shared by both tables: topology name, its knob delta, and the deploy
 *  flags as chips. Keeping one renderer means the "What to run" table and the full run log read
 *  identically (the richer operating-table format the review preferred). */
function ConfigCell({ r }: { r: RunRecord }) {
  return (
    <td>
      <div className="op-name mono">{r.series}</div>
      {r.variant ? <div className="op-var">{r.variant}</div> : null}
      <div className="op-flags">
        {runFlags(r).map((f) => (
          <span className="op-fl" key={f.flag}>
            <i>{f.flag}</i>{f.value ? ` ${f.value}` : ''}
          </span>
        ))}
      </div>
    </td>
  )
}

/**
 * "What to run": one row per load, pairing the configuration you would deploy with the
 * performance it gave there — the standalone report's section 9, which reads as "pick an
 * operating point and read the flags off it". The run table below stays the full per-run log,
 * including the runs that broke the floor.
 */
export function OperatingTable({ recs, onPick, picked = null, active = null }: {
  recs: RunRecord[]
  onPick?: (id: string) => void
  picked?: string | null
  /** The emphasised series (from a chart hover/click); non-matching rows dim while it is set. */
  active?: string | null
}) {
  // Default order is by load (operatingPoints returns it concurrency-ascending); a click on any
  // header sorts by that column instead. `configuration` sorts on the series+variant name.
  const { sort, toggle } = useSort()
  const opGet = (r: RunRecord, k: string): string | number | null | undefined => {
    switch (k) {
      case 'sessions': return r.conc
      case 'configuration': return `${r.series}${r.variant ? ` ${r.variant}` : ''}`
      case 'tok/s per user': return r.pu
      case 'tok/s fleet': return r.cl
      case 'tok/s GPU': return r.tpg
      case 'ITL ms': return r.itl
      case 'TTFT ms': return r.ttft
      default: return undefined
    }
  }
  const base = operatingPoints(recs)
  const rows = sortRows(base, sort, opGet)
  if (base.length === 0) {
    return (
      <p className="op-none">
        No run held the interactivity floor, so there is no operating point to recommend. The run
        log below lists every run and why each one missed.
      </p>
    )
  }
  return (
    <div className="tscroll hscroll">
      <table className="readout op-table" data-testid="operating-table">
        <thead>
          <tr className="col">
            <SortTh term="sessions" sortKey="sessions" sort={sort} onToggle={toggle} />
            <SortTh term="configuration" sortKey="configuration" sort={sort} onToggle={toggle} />
            <SortTh term="tok/s per user" sortKey="tok/s per user" sort={sort} onToggle={toggle} num />
            <SortTh term="tok/s fleet" sortKey="tok/s fleet" sort={sort} onToggle={toggle} num />
            <SortTh term="tok/s GPU" sortKey="tok/s GPU" sort={sort} onToggle={toggle} num />
            <SortTh term="ITL ms" sortKey="ITL ms" sort={sort} onToggle={toggle} num />
            <SortTh term="TTFT ms" sortKey="TTFT ms" sort={sort} onToggle={toggle} num />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className={`op-row${r.id === picked ? ' hl' : ''}${active !== null && r.series !== active ? ' dim' : ''}`}
              onClick={onPick ? () => onPick(r.id) : undefined}
              title={onPick ? 'Show the full deployment for this run' : undefined}>
              <td className="num hero"><span className="num">{fnum(r.conc, 0)}</span></td>
              <ConfigCell r={r} />
              <td className="num"><span className="num">{fnum(r.pu)}</span></td>
              <td className="num"><span className="num">{fnum(r.cl, 0)}</span></td>
              <td className="num"><span className="num">{fnum(r.tpg)}</span></td>
              <td className="num"><span className="num">{fnum(r.itl, 2)}</span></td>
              <td className="num"><span className="num">{fnum(r.ttft, 0)}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Scatter({ pts, xl, yl, label, hl = NO_HL, onPick, picked = null }: {
  pts: Pt[]; xl: string; yl: string; label: string; hl?: HL
  /** Called with a point's record id when it is clicked — opens its deployment panel. */
  onPick?: (id: string) => void
  /** The currently inspected record id, ringed so you can see which point the panel describes. */
  picked?: string | null
}) {
  if (pts.length === 0) return <Empty msg="No points to plot for the current filters." />
  const xmax = Math.max(...pts.map((p) => p.x), 1) * 1.05
  const ymax = Math.max(...pts.map((p) => p.y), 1) * 1.05
  const px = (x: number) => M.l + (x / xmax) * (W - M.l - M.r)
  const py = (y: number) => H - M.b - (y / ymax) * (H - M.t - M.b)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={`chart${hl.active ? ' hl-on' : ''}`} role="group" aria-label={label}
      data-highlight={hl.active ?? ''} onClick={hl.onClear}>
      <Axes xmax={xmax} ymax={ymax} xl={xl} yl={yl} />
      {/* Per-config trajectory: connect a configuration's points in load order so the frontier and
          each config's path read as a line, matching the two Lines charts rather than a point
          cloud. Drawn BEFORE the markers so points sit on top. Needs order (concurrency) + cfg;
          a dataset without them simply draws no connectors. */}
      {(() => {
        const byCfg = new Map<string, Pt[]>()
        for (const p of pts) {
          if (p.cfg === undefined || p.order === undefined) continue
          const g = byCfg.get(p.cfg) ?? []
          g.push(p); byCfg.set(p.cfg, g)
        }
        return [...byCfg.entries()].map(([cfg, ps]) => {
          if (ps.length < 2) return null
          const sorted = [...ps].sort((a, b) => (a.order as number) - (b.order as number))
          const em = emphasis(hl.active, sorted[0].series)
          const col = em === 'off' ? GREY : sorted[0].color
          return (
            <polyline key={`traj-${cfg}`} fill="none" stroke={col} strokeWidth={em === 'on' ? 2 : 1.3}
              strokeLinejoin="round" strokeLinecap="round" opacity={em === 'off' ? 0.18 : 0.5}
              className="traj" points={sorted.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')} />
          )
        })
      })()}
      {/* deck marker: series-filled when servable, panel-filled (hollow) when not, always
          stroked in the series colour — the instrument-panel look from the BLIS deck.
          Highlighted series stay coloured and draw last; the rest fade to grey context. */}
      {[...pts.keys()].sort((i, j) => Number(emphasis(hl.active, pts[i].series) === 'on') - Number(emphasis(hl.active, pts[j].series) === 'on')).map((i) => {
        const p = pts[i]
        const em = emphasis(hl.active, p.series)
        const col = em === 'off' ? GREY : p.color
        return (
          // A click on the POINT inspects that run; a click on empty chart space still clears the
          // highlight (the <svg> handler), and the legend keeps series highlighting. So the new
          // behaviour does not take over a gesture that already meant something else.
          // keyed by record id for the same reason as the line points above
          <g key={p.id ?? `i${i}`} {...seriesProps(hl, p.series)} data-series={p.series} data-em={em}>
            <circle cx={px(p.x)} cy={py(p.y)} r={em === 'on' ? 5.5 : 4} fill={p.ok ? col : 'var(--panel)'} stroke={col}
              strokeWidth={em === 'on' ? 2 : 1.5} opacity={em === 'off' ? 0.25 : 1}
              className={`pt${p.id && p.id === picked ? ' picked' : ''}`}
              onClick={p.id && onPick
                ? (e) => { e.stopPropagation(); onPick(p.id as string) }
                : undefined}>
              <title>{p.title}</title>
            </circle>
          </g>
        )
      })}
    </svg>
  )
}

/** Series legend shared by all three charts: colour swatch → configuration, clickable to toggle
 *  the highlight, hoverable to emphasise. Hoisted out of the scatter so the two Lines charts get
 *  the same key (they previously relied on colour alone, with no mapping to a config). `showState`
 *  adds the servable/not swatches and the recommended marker — on for the scatter, off for the
 *  line charts to keep them compact. */
function SeriesLegend({ series, colors, hl, setHover, sticky, winner, winnerName, showState = false }: {
  series: string[]; colors: Map<string, string>; hl: HL; setHover: (s: string | null) => void
  sticky: string | null; winner?: string | null; winnerName?: string | null; showState?: boolean
}) {
  return (
    <div className="legend">
      {showState && <><span><i className="dot filled" /> servable</span><span><i className="dot hollow" /> not servable</span></>}
      {series.map((s) => (
        <button type="button" key={s} className={`lg${hl.active === s ? ' on' : ''}`} aria-pressed={sticky === s}
          onClick={() => hl.onToggle(s)} onPointerEnter={() => setHover(s)} onPointerLeave={() => setHover(null)}>
          <i className="sw2" style={{ background: colors.get(s) }} />{s}
          {showState && s === winner && (
            <em className="recmark" title={`Recommended: ${winnerName ?? s}`}>recommended</em>
          )}
        </button>
      ))}
    </div>
  )
}

// The "knee": the first concurrency at which a series stops being servable — the interactivity
// cliff the deck marks with a violet vertical line. Computed per series from the run records.
/** First load at which this CONFIG breaks the floor. Keyed by series + variant, because knob
 *  variants of one topology break at different loads (on kimi-k2.5, least-loaded routing holds to
 *  1024 while the default breaks at 512) — a per-topology knee would draw one marker for all. */
function kneeConc(recs: RunRecord[], series: string, variant = ''): number | null {
  const rows = recs.filter((r) => r.series === series && (r.variant ?? '') === variant && finite(r.conc))
    .sort((a, b) => (a.conc as number) - (b.conc as number))
  const firstBad = rows.find((r) => !r.ok)
  return firstBad ? (firstBad.conc as number) : null
}

export function Lines({ recs, shownRecs, get, xl, yl, label, colors, showKnee = false, hl = NO_HL,
  onPick, picked = null }: {
  recs: RunRecord[]; shownRecs: RunRecord[]; get: (r: RunRecord) => number | null
  xl: string; yl: string; label: string; colors: Map<string, string>; showKnee?: boolean; hl?: HL
  /** Called with a point's record id when clicked — same contract as Scatter's. */
  onPick?: (id: string) => void
  /** The inspected record id, ringed here too so you can see the run across all three charts. */
  picked?: string | null
}) {
  // Group by CONFIG identity (series + variant), not by series alone. A line is meant to walk one
  // configuration up the load: grouping by topology put every knob variant on one polyline, so a
  // refined shape drew six different configs as a single zigzag — on kimi-k2.5 one "line" spanned
  // 27.9 to 68.0 tok/s at the SAME concurrency. The id rides along so a point can open its
  // deployment, as in the scatter.
  const byConfig = new Map<string, { name: string; variant: string; pts: { x: number; y: number; ok: boolean; id?: string }[] }>()
  for (const r of shownRecs) {
    const y = get(r)
    if (!finite(r.conc) || !finite(y)) continue
    const variant = r.variant ?? ''
    const key = `${r.series}\u0000${variant}`
    const g = byConfig.get(key) ?? { name: r.series, variant, pts: [] }
    g.pts.push({ x: r.conc, y: y as number, ok: r.ok, id: r.id })
    byConfig.set(key, g)
  }
  // Colour and highlight still key off the SERIES — the legend lists topologies, and a variant
  // should read as a sibling of its shape rather than an unrelated colour.
  const bySeries = new Map([...byConfig.entries()].map(([k, g]) => [k, g.pts]))
  const nameOf = (key: string) => byConfig.get(key)!.name
  const variantOf = (key: string) => byConfig.get(key)!.variant
  const all = [...bySeries.values()].flat()
  if (all.length === 0) return <Empty msg="No concurrency data to plot for the current filters." />
  const xmax = Math.max(...all.map((p) => p.x), 1) * 1.05
  const ymax = Math.max(...all.map((p) => p.y), 1) * 1.05
  const pw = W - M.l - M.r
  const px = (x: number) => M.l + (x / xmax) * pw
  const py = (y: number) => H - M.b - (y / ymax) * (H - M.t - M.b)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={`chart${hl.active ? ' hl-on' : ''}`} role="group" aria-label={label}
      data-highlight={hl.active ?? ''} onClick={hl.onClear}>
      <Axes xmax={xmax} ymax={ymax} xl={xl} yl={yl} />
      {/* violet knee markers: where each series first breaks the floor (deck convention) */}
      {showKnee && [...bySeries.keys()].map((key) => {
        const name = nameOf(key)
        const k = kneeConc(recs, name, variantOf(key))
        if (k == null || k > xmax) return null
        const x = px(k)
        return (
          <g key={`knee-${key}`} className="knee" opacity={emphasis(hl.active, name) === 'off' ? 0.25 : 1}>
            <line x1={x} x2={x} y1={M.t} y2={H - M.b} />
            <text x={x} y={M.t - 2} textAnchor="middle" className="kneelab">knee</text>
          </g>
        )
      })}
      {[...bySeries.entries()].sort(([a], [b]) => Number(emphasis(hl.active, nameOf(a)) === 'on') - Number(emphasis(hl.active, nameOf(b)) === 'on')).map(([key, pts]) => {
        const name = nameOf(key)
        const variant = variantOf(key)
        const sorted = [...pts].sort((a, b) => a.x - b.x)
        const em = emphasis(hl.active, name)
        const c = em === 'off' ? GREY : (colors.get(name) ?? PALETTE[0])
        const op = em === 'off' ? 0.25 : 1
        return (
          <g key={key} {...seriesProps(hl, name)} data-series={name} data-variant={variant} data-em={em}>
            {sorted.length > 1 && (
              <polyline fill="none" stroke={c} strokeWidth={em === 'on' ? 3 : 2} strokeLinejoin="round" strokeLinecap="round"
                opacity={op} className="ln" points={sorted.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')} />
            )}
            {/* Key by the record id, not the index: the groups are re-sorted by highlight state
                every render, so index keys let React reuse a circle across DIFFERENT configs and
                its click handler then opened the wrong run's deployment. */}
            {sorted.map((p, i) => (
              <circle key={p.id ?? `i${i}`} cx={px(p.x)} cy={py(p.y)} r={em === 'on' ? 4.2 : 2.8} fill={p.ok ? c : 'var(--panel)'} stroke={c}
                strokeWidth={em === 'on' ? 1.8 : 1.3} opacity={op}
                className={`pt${p.id && p.id === picked ? ' picked' : ''}`}
                onClick={p.id && onPick
                  ? (e) => { e.stopPropagation(); onPick(p.id as string) }
                  : undefined}>
                <title>{`${name}${variant ? ` (${variant})` : ''} · conc ${fnum(p.x, 0)} · ${fnum(p.y)}`
                  + `${p.ok ? '' : ' · not servable'}`
                  + (p.id && onPick ? ' — click for the deployment' : '')}</title>
              </circle>
            ))}
          </g>
        )
      })}
    </svg>
  )
}

/* ---------------- per-chart explanations ----------------
   Each figure gets a static "how to read it" line plus a finding DERIVED from the runs
   currently in view, so the prose cites this scenario's own numbers instead of generic text.
   Every helper tolerates empty/!finite data and returns null rather than inventing a claim. */

function fmtList(items: string[]): string {
  if (items.length <= 1) return items[0] ?? ''
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

/** Which config axes the search actually VARIED, derived from the runs: a dimension is "searched"
 *  when it took more than one distinct value across the scenario's records. This removes the
 *  ambiguity the review flagged — a reader can tell whether a knob was tried-and-lost or never
 *  explored. Returns [] for a scenario where nothing varied (e.g. one run). */
function axesSearched(recs: RunRecord[]): string[] {
  const dims: { key: keyof RunRecord; label: string }[] = [
    { key: 'tp', label: 'tensor-parallel' },
    { key: 'dp', label: 'data-parallel' },
    { key: 'epw', label: 'expert-parallel' },
    { key: 'ni', label: 'instances' },
    { key: 'mns', label: 'max-num-seqs' },
    { key: 'mnbt', label: 'max-num-batched-tokens' },
    { key: 'bs', label: 'block-size' },
    { key: 'gmu', label: 'gpu-memory-utilization' },
    { key: 'route', label: 'routing policy' },
    { key: 'pd', label: 'prefill/decode' },
    { key: 'mtp', label: 'speculative tokens' },
  ]
  const out: string[] = []
  for (const d of dims) {
    const vals = new Set(recs.map((r) => {
      const v = r[d.key]
      return v === null || v === undefined || v === '' ? null : String(v)
    }).filter((v) => v !== null))
    if (vals.size > 1) out.push(d.label)
  }
  return out
}

/** Scatter: the throughput/interactivity trade-off — name the best fleet point that still holds
 *  the floor, and what pushing past it costs. */
function explainScatter(recs: RunRecord[]): string | null {
  const ok = recs.filter((r) => r.ok && finite(r.cl) && finite(r.pu))
  const bad = recs.filter((r) => !r.ok && finite(r.cl) && finite(r.pu))
  if (ok.length === 0) return bad.length ? 'No run in view holds the interactivity floor — every point here breaks it.' : null
  const best = ok.reduce((a, b) => ((b.cl as number) > (a.cl as number) ? b : a))
  let s = `The furthest-right filled point is ${best.series} at ${fnum(best.conc, 0)} concurrent `
    + `sessions: ${fnum(best.cl)} tok/s across the fleet while each user still sees ${fnum(best.pu)} tok/s.`
  if (bad.length) {
    const over = bad.reduce((a, b) => ((b.cl as number) > (a.cl as number) ? b : a))
    const drop = finite(best.pu) && finite(over.pu) ? Math.round((1 - (over.pu as number) / (best.pu as number)) * 100) : null
    s += ` Hollow points push further on fleet throughput — up to ${fnum(over.cl)} tok/s — but`
      + (drop !== null && drop > 0 ? ` per-user speed falls ${drop}% to ${fnum(over.pu)} tok/s,` : ' per-user speed collapses,')
      + ' which is why peak throughput alone is the wrong thing to rank on.'
  } else {
    s += ' No run in view breaks the floor, so the frontier is still rising — the limit is past the loads swept here.'
  }
  return s
}

/** Per-user-vs-load: where each config's cliff is, i.e. how far it scales before it breaks. */
function explainPerUser(recs: RunRecord[], seriesList: string[]): string | null {
  const limits: { s: string; last: number; knee: number | null }[] = []
  for (const s of seriesList) {
    const rows = recs.filter((r) => r.series === s && finite(r.conc)).sort((a, b) => (a.conc as number) - (b.conc as number))
    if (!rows.length) continue
    const okRows = rows.filter((r) => r.ok)
    const firstBad = rows.find((r) => !r.ok)
    limits.push({
      s,
      last: okRows.length ? (okRows[okRows.length - 1].conc as number) : 0,
      knee: firstBad ? (firstBad.conc as number) : null,
    })
  }
  if (!limits.length) return null
  const parts = limits
    .sort((a, b) => b.last - a.last)
    .map((l) => `${l.s} holds to ${l.last} session${l.last === 1 ? '' : 's'}${l.knee !== null ? ` (breaks at ${l.knee})` : ''}`)
  const spread = limits.length > 1 && limits[0].last > 0
    ? ` That is a ${(limits[0].last / Math.max(1, limits[limits.length - 1].last)).toFixed(1)}× spread in how many sessions a configuration carries — from configuration alone, on the same hardware and workload.`
    : ''
  const anyKnee = limits.some((l) => l.knee !== null)
  return `${fmtList(parts)}.${spread}`
    + (anyKnee
      ? ' The violet knee marks the first load that breaks the floor: per-user speed holds roughly flat, then falls away within a session or two.'
      : ' No configuration in view reached its knee, so each one still has headroom above the loads swept here.')
}

/** ITL-vs-load: latency is the mechanism behind the cliff — quantify flat vs post-knee. */
function explainItl(recs: RunRecord[]): string | null {
  const withItl = recs.filter((r) => finite(r.itl) && finite(r.conc))
  if (!withItl.length) return null
  const ok = withItl.filter((r) => r.ok)
  const bad = withItl.filter((r) => !r.ok)
  if (!ok.length) return 'Every run in view is already over the latency ceiling.'
  const itls = ok.map((r) => r.itl as number)
  const lo = Math.min(...itls)
  const hi = Math.max(...itls)
  let s = `While a configuration is healthy its inter-token latency barely moves — ${fnum(lo, 1)}`
    + `${hi > lo ? `–${fnum(hi, 1)}` : ''} ms across the servable loads in view.`
  if (bad.length) {
    const worst = Math.max(...bad.map((r) => r.itl as number))
    s += ` Past the knee it climbs to ${fnum(worst, 1)} ms — roughly ${Math.max(2, Math.round(worst / Math.max(lo, 0.001)))}× the healthy value.`
      + ' Adding load beyond that point buys little throughput and costs every user noticeably.'
  } else {
    s += ' Nothing in view crosses the ceiling, so the decode path is not yet the binding constraint.'
  }
  return s
}

export function Explorer({ recs, runId, winner = null, winnerName = null }: {
  recs: RunRecord[]; runId: string
  /** The winning TOPOLOGY (matches RunRecord.series, which is topology-only). */
  winner?: string | null
  /** The winning config in full, incl. its knob delta — the legend groups by topology, so the
   *  badge marks a whole series and must say which variant of it was recommended. */
  winnerName?: string | null
}) {
  const [sel, setSel] = useState<Selection>(EMPTY_SEL)
  const [sticky, setSticky] = useState<string | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  // The run whose deployment the inline panel is showing, by RunRecord.id.
  const [picked, setPicked] = useState<string | null>(null)
  // Sort for the full per-run log. Unsorted = the order build_gallery emitted (series, then load).
  const { sort: runSort, toggle: toggleRunSort } = useSort()
  const series = useMemo(() => [...new Set(recs.map((r) => r.series))].sort(), [recs])
  const colors = useMemo(() => new Map(series.map((s, i) => [s, PALETTE[i % PALETTE.length]])), [series])
  const shown = useMemo(() => selectRecords(recs, sel), [recs, sel])
  // Regime tag (deck convention): a run that broke the floor with KV preemptions is KV-bound;
  // otherwise the cliff is a decode-compute ceiling. Records don't carry preemptions, so label
  // honestly from what we have — "compute-bound" when the floor break is latency-only.
  const shownSeries = useMemo(() => [...new Set(shown.map((r) => r.series))], [shown])
  const active = resolveHighlight(sticky, hover, shownSeries)
  const hl: HL = {
    active, onHover: setHover, onClear: () => setSticky(null),
    onToggle: (s) => setSticky((cur) => toggleHighlight(resolveHighlight(cur, null, shownSeries), s)),
  }
  // Clicking a point/row pins its deployment AND highlights that configuration (greying the rest);
  // clicking the same one again (or closing the panel) unpins and clears the highlight, so the
  // chart returns to showing everything. Highlight keys off the series (the legend's unit), so a
  // click selects the whole configuration the point belongs to.
  const pickRun = (id: string) => {
    if (id === picked) { setPicked(null); setSticky(null); return }
    setPicked(id)
    const s = recs.find((r) => r.id === id)?.series ?? null
    setSticky(s)
  }
  const clearPick = () => { setPicked(null); setSticky(null) }
  const regime = useMemo(() => (recs.some((r) => !r.ok && (r.viol ?? []).some((v) => /kv|preempt|memory/i.test(v))) ? 'KV-bound' : 'compute-bound'), [recs])
  // Hovering a chart line emphasises a whole series; the panel previews THAT series' operating
  // point — its most efficient servable run (max tok/s/GPU, the objective the search and the
  // "What to run" table rank on). Resolve within the series: operatingPoints(shown) keeps only
  // the single best series per load, so a series that never wins a load would resolve to nothing
  // and the box would stay blank for most lines. Declared BEFORE the early return below so the
  // hook order is stable when navigating from a scenario with runs to one without (React throws
  // "rendered fewer hooks than expected" otherwise).
  const hoverRec = useMemo(() => {
    if (active === null) return null
    const mine = shown.filter((r) => r.series === active && r.ok && finite(r.tpg))
    if (mine.length === 0) return null
    return mine.reduce((a, b) => ((b.tpg as number) > (a.tpg as number) ? b : a))
  }, [active, shown])

  if (recs.length === 0) {
    return (
      <div className="empty">
        No per-run records are available for this scenario, so the Pareto explorer is unavailable.{' '}
        <a href={reportHref(runId)} target="_blank" rel="noreferrer">Open full report ↗</a>
      </div>
    )
  }
  const pts: Pt[] = shown.filter((r) => finite(r.cl) && finite(r.pu)).map((r) => ({
    x: r.cl as number,
    y: r.pu as number,
    ok: r.ok,
    color: colors.get(r.series) ?? PALETTE[0],
    series: r.series,
    id: r.id,
    // order + cfg let the scatter connect a configuration's points into a load-ordered trajectory
    // (same config identity the Lines charts group by: series + variant).
    order: finite(r.conc) ? (r.conc as number) : undefined,
    cfg: `${r.series}\u0000${r.variant ?? ''}`,
    title: `${r.series}${r.variant ? ` (${r.variant})` : ''} · conc ${r.conc ?? dash} · ${fnum(r.cl)} tok/s · ${fnum(r.pu)} tok/s/user · ${r.ok ? 'servable' : r.status} — click for the deployment`,
  }))
  const hidden = shown.length - pts.length
  // Whether ANY run in the scenario recorded a command — drives whether the Reproduce panel shows
  // at all (see PointDetail). Checked over the full recs, not the filtered view, so a filter that
  // happens to hide the command-bearing runs does not make the panel vanish.
  const scenarioHasCmd = recs.some((r) => !!r.cmd)
  // Resolve against the FILTERED set: if a filter or brush removes the inspected run, the panel
  // closes with it rather than describing a point no longer on the chart.
  const pickedRec = picked === null ? null : shown.find((r) => r.id === picked) ?? null
  // Hover wins while the cursor is on a DIFFERENT series than the pinned run — that is the whole
  // point of the preview: the box tracks the line under the cursor. A click still pins one exact
  // run, so hovering that same series (e.g. resting on the clicked point) keeps showing it rather
  // than swapping it for the series' operating point. With nothing pinned, hover is all there is.
  const panelRec = hoverRec && (!pickedRec || hoverRec.series !== pickedRec.series)
    ? hoverRec
    : pickedRec ?? hoverRec
  // Column → sortable value for the full run log. Same columns as the operating table (shared
  // format); `status` sorts servable-first ('' ranks before any violation text) so like failures
  // group together.
  const runLogGet = (r: RunRecord, k: string): string | number | null | undefined => {
    switch (k) {
      case 'sessions': return r.conc
      case 'configuration': return `${r.series}${r.variant ? ` ${r.variant}` : ''}`
      case 'tok/s per user': return r.pu
      case 'tok/s fleet': return r.cl
      case 'tok/s GPU': return r.tpg
      case 'ITL ms': return r.itl
      case 'TTFT ms': return r.ttft
      case 'status': return r.ok ? '' : r.viol.length ? r.viol.join(', ') : r.status
      default: return undefined
    }
  }

  return (
    <div>
      <Crossfilter recs={recs} sel={sel} setSel={setSel} shownCount={shown.length}
        onPickSeries={(v, on) => setSticky((cur) => (on ? v : cur === v ? null : cur))} />

      <div className="charts">
        <figure className="vc">
          <div className="vc-h"><h4 className="vc-t">Throughput ↔ interactivity</h4><span className="vc-r">{regime}</span></div>
          <p className="vc-s">cluster tok/s vs per-user tok/s · one point per run</p>
          <Scatter pts={pts} xl="cluster tok/s" yl="tok/s per user" label="Scatter of cluster throughput against per-user throughput" hl={hl}
            onPick={pickRun} picked={picked} />
          <p className="vc-k"><b>How to read it.</b> Each point is one run: x is what the whole fleet delivers, y is what one
            caller feels. Up and to the right is better for everyone; filled points hold the interactivity floor, hollow
            ones break it. A faint line connects a configuration's runs in load order, so you can trace where adding
            sessions takes it.</p>
          {explainScatter(shown) && <p className="vc-x">{explainScatter(shown)}</p>}
          {hidden > 0 && <p className="pts-note">{hidden} run{hidden === 1 ? ' has' : 's have'} no throughput values and {hidden === 1 ? 'is' : 'are'} not plotted.</p>}
          {pts.length === 1 && <p className="pts-note">Only one point matches; a trade-off frontier needs two or more.</p>}
        </figure>
        <figure className="vc">
          <div className="vc-h"><h4 className="vc-t">Per-user rate vs load</h4><span className="vc-r">interactivity</span></div>
          <p className="vc-s">tok/s per user as concurrency climbs</p>
          <Lines recs={recs} shownRecs={shown} get={(r) => r.pu} xl="concurrency" yl="tok/s per user" label="Per-user throughput by concurrency" colors={colors} showKnee hl={hl}
            onPick={pickRun} picked={picked} />
          <p className="vc-k"><b>How to read it.</b> One line per configuration, walking the load up. Flat is good — it means
            adding sessions has not yet slowed anyone down. The violet line marks that configuration's knee, the first load
            that breaks the floor.</p>
          {explainPerUser(shown, series) && <p className="vc-x">{explainPerUser(shown, series)}</p>}
        </figure>
        <figure className="vc">
          <div className="vc-h"><h4 className="vc-t">Inter-token latency vs load</h4><span className="vc-r">latency</span></div>
          <p className="vc-s">ITL (ms) as concurrency climbs</p>
          <Lines recs={recs} shownRecs={shown} get={(r) => r.itl} xl="concurrency" yl="ITL (ms)" label="Inter-token latency by concurrency" colors={colors} showKnee hl={hl}
            onPick={pickRun} picked={picked} />
          <p className="vc-k"><b>How to read it.</b> Inter-token latency is the gap between output tokens — the mechanism behind
            the cliff above. Lower is better, and the floor is a ceiling on this number, so a line crossing its knee is what
            makes the per-user rate fall.</p>
          {explainItl(shown) && <p className="vc-x">{explainItl(shown)}</p>}
        </figure>
      </div>

      {/* One legend for all three charts — the colour key is shared, so it lives once under the
          grid rather than being duplicated in each figure. The highlight-status bar sits with it,
          since highlighting is global across the charts too. */}
      <div className="charts-legend">
        <SeriesLegend series={series} colors={colors} hl={hl} setHover={setHover} sticky={sticky}
          winner={winner} winnerName={winnerName} showState />
        {active !== null && (
          <div className="hl-bar" role="status">
            <span>Highlighting <b className="mono">{active}</b>{sticky === null && ' (hover)'} — emphasis only, filters unchanged.</span>
            {sticky !== null && <button type="button" className="hl-clear" onClick={hl.onClear}>clear highlight</button>}
          </div>
        )}
      </div>

      {/* The deployment box sits BELOW the chart grid, not inside the scatter's figure: nested in
          one column of a 3-up grid it was ~970px tall with the command wrapping every few words.
          Full width keeps the four sections side by side. It shows the clicked (pinned) run, or
          failing that previews the operating point of whatever series the cursor is on. */}
      {panelRec && <PointDetail r={panelRec} pinned={panelRec === pickedRec}
        scenarioHasCmd={scenarioHasCmd} onClose={clearPick} />}

      <div className="op-sec">
        <h4 className="op-h">What to run</h4>
        <p className="op-s">
          One row per load: the most efficient configuration that still held the floor there, and
          the flags to deploy it with. Pick the load you expect, then read the flags off it —
          click a row for the full deployment.
        </p>
        <OperatingTable recs={shown} onPick={pickRun} picked={picked} active={active} />
      </div>

      <div className="op-sec">
        <h4 className="op-h">All runs</h4>
        <p className="op-s">
          Every run in the sweep — including the ones that broke the floor, with the reason in the
          status column. Same format as above; click a row for the full deployment.
        </p>
      </div>
      <div className="tscroll hscroll runtable">
        <table className="readout op-table" data-testid="run-log">
          <thead>
            <tr className="col">
              <SortTh term="sessions" sortKey="sessions" sort={runSort} onToggle={toggleRunSort} />
              <SortTh term="configuration" sortKey="configuration" sort={runSort} onToggle={toggleRunSort} />
              <SortTh term="tok/s per user" sortKey="tok/s per user" sort={runSort} onToggle={toggleRunSort} num />
              <SortTh term="tok/s fleet" sortKey="tok/s fleet" sort={runSort} onToggle={toggleRunSort} num />
              <SortTh term="tok/s GPU" sortKey="tok/s GPU" sort={runSort} onToggle={toggleRunSort} num />
              <SortTh term="ITL ms" sortKey="ITL ms" sort={runSort} onToggle={toggleRunSort} num />
              <SortTh term="TTFT ms" sortKey="TTFT ms" sort={runSort} onToggle={toggleRunSort} num />
              <SortTh term="status" sortKey="status" sort={runSort} onToggle={toggleRunSort} />
            </tr>
          </thead>
          <tbody>
            {sortRows(shown, runSort, runLogGet).map((r) => (
              <tr key={r.id} className={`op-row${r.ok ? '' : ' bad'}${r.id === picked ? ' hl' : ''}${active !== null && r.series !== active ? ' dim' : ''}`}
                onPointerEnter={() => setHover(r.series)} onPointerLeave={() => setHover(null)}
                onClick={() => pickRun(r.id)} title="Show the full deployment for this run">
                <td className="num hero"><span className="num">{fnum(r.conc, 0)}</span></td>
                <ConfigCell r={r} />
                <td className="num"><span className="num">{fnum(r.pu)}</span></td>
                <td className="num"><span className="num">{fnum(r.cl, 0)}</span></td>
                <td className="num"><span className="num">{fnum(r.tpg)}</span></td>
                <td className="num"><span className="num">{fnum(r.itl, 2)}</span></td>
                <td className="num"><span className="num">{fnum(r.ttft, 0)}</span></td>
                <td>{r.ok ? 'servable' : r.viol.length ? r.viol.join(', ') : r.status}</td>
              </tr>
            ))}
            {shown.length === 0 && <tr><td colSpan={8} className="pts-empty">No runs match these filters.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function ScenarioDetail({ runId, scenarios }: { runId: string; scenarios: Scenario[] }) {
  const s = scenarios.find((x) => x.run_id === runId)
  // Records are inlined on the scenario (stage-data.mjs), so the explorer renders from the
  // one scenarios.json load — no per-scenario fetch, no loading state, works everywhere.
  const recs: RunRecord[] = s?.records ?? []

  if (!s) {
    return (
      <div>
        <p><a href="#/reports">← all reports</a></p>
        <div className="empty">Unknown scenario <span className="mono">{runId}</span>.</div>
      </div>
    )
  }
  const w = s.winner
  // The two best alternatives to the recommended config, ranked by efficiency (tok/s/GPU
  // then total tok/s), shown side-by-side under the recommendation.
  //
  // A config is identified by label + variant, not label alone: once the search refines engine
  // and routing knobs, several finalists share a topology, and filtering on label dropped every
  // one of them from the runners-up (often the most interesting comparison — the same shape with
  // and without a knob change).
  const key = (f: { label: string; variant?: string }) => `${f.label}\u0000${f.variant ?? ''}`
  const runnersUp = s.finalists
    .filter((f) => key(f) !== key(w))
    .sort((a, b) => b.tps_per_gpu - a.tps_per_gpu || b.tps - a.tps)
    .slice(0, 2)
  return (
    <div>
      <p className="crumb"><a href="#/reports">← all reports</a></p>
      <h2 className="viewhead">{s.model}{s.synthetic && <MockBadge />}</h2>
      <div className="specchips">
        <span className="spec">{s.hardware}</span><span className="spec">{s.gpu_budget} GPUs</span>
        <span className="spec">{s.workload}</span><span className="spec">{s.isl_osl}</span>
        {/* The interactivity floor is the premise of every servable/not call below, so state it
            here with the other constraints rather than only in the dek far down the briefing. */}
        {s.floor && <span className="spec floor" title="Interactivity floor: a run is servable only if it holds this.">floor: {s.floor}</span>}
        <a className="fullreport" href={reportHref(s.run_id)} target="_blank" rel="noreferrer">Open full report ↗</a>
      </div>
      {s.synthetic && <div className="warnnote">Mock scenario: synthetic placeholder data, not a real simulation run.</div>}

      <section className="briefing" aria-label="Top configs to try">
        <div className="seclabel2">Top configs to try</div>
        {/* Row 1: the recommended config, prominent. Row 2: the next two runners-up,
            ranked by tok/s/GPU (efficiency), side by side. */}
        <div className="reccard">
          <div className="recrank">RECOMMENDED</div>
          <div className="rechead">
            <div className="reclabel">{w.name ?? w.label}</div>
            <div className="recnote">{s.reason}</div>
          </div>
          <div className="recstats">
            <div><dt>cluster tok/s</dt><dd className="num best">{fnum(w.tps)}</dd></div>
            <div><dt>tok/s/GPU</dt><dd className="num best">{fnum(w.tps_per_gpu)}</dd></div>
            <div><dt>tok/s/user</dt><dd className="num best">{fnum(w.per_user)}</dd></div>
            <div><dt>ITL p99 (ms)</dt><dd className="num best">{fnum(w.itl_p99, 2)}</dd></div>
            <div><dt>GPUs · load</dt><dd className="num"><span className="num">{fnum(w.gpus, 0)}</span> · c<span className="num">{fnum(w.c_use, 0)}</span></dd></div>
          </div>
        </div>
        {runnersUp.length > 0 && (
          <div className="runnerups">
            {runnersUp.map((f, i) => (
              <div className="runcard" key={key(f)}>
                <div className="runrank">#{i + 2} alternative</div>
                <div className="runlabel">{f.name ?? f.label}</div>
                <div className="runstats">
                  <div><dt>tok/s</dt><dd className="num">{fnum(f.tps)}</dd></div>
                  <div><dt>tok/s/GPU</dt><dd className="num">{fnum(f.tps_per_gpu)}</dd></div>
                  <div><dt>tok/s/user</dt><dd className="num">{fnum(f.per_user)}</dd></div>
                  <div><dt>ITL p99</dt><dd className="num">{fnum(f.itl_p99, 2)}</dd></div>
                  <div><dt>GPUs · load</dt><dd className="num"><span className="num">{fnum(f.gpus, 0)}</span> · c<span className="num">{fnum(f.c_use, 0)}</span></dd></div>
                </div>
              </div>
            ))}
          </div>
        )}
        <p className="dek"><b>Floor held:</b> <span className="mono">{s.floor}</span>{s.headroom != null && <> (headroom <span className="num">{fnum(s.headroom, 2)}</span>)</>}.{' '}
          <span className="num">{s.n_servable}</span> of <span className="num">{s.n_runs}</span> simulated runs are servable.
          {s.finalists.length > runnersUp.length + 1 && <> All <span className="num">{s.finalists.length}</span> finalists are in the run table below.</>}</p>
        {/* State the method so the ranking is legible: a reader should know WHY these are the
            finalists and how the order was chosen. */}
        <p className="dek method"><b>How these were chosen:</b> every candidate topology that held
          the floor at some load becomes a finalist, recorded at its usable operating point
          (headroom × the highest servable load). Finalists — and the runners-up above — are ranked
          by tok/s per GPU, the search objective; the recommendation is the top of that ranking.</p>
      </section>

      <section aria-label="Pareto explorer">
        <div className="seclabel2">Pareto explorer</div>
        {/* Axes searched: which config dimensions actually varied across the runs. Removes the
            ambiguity of whether a knob was tried-and-lost vs never explored (review #4). */}
        {(() => {
          const axes = axesSearched(recs)
          return axes.length > 0 ? (
            <p className="axes-searched"><b>Axes searched:</b> {fmtList(axes)} (over a concurrency ramp).
              Any knob not listed was held at its default, not explored.</p>
          ) : null
        })()}
        <Explorer recs={recs} runId={s.run_id} winner={w.label} winnerName={w.name ?? w.label} />
      </section>
    </div>
  )
}
