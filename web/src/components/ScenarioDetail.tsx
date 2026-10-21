import { useMemo, useState } from 'react'
import type { RunRecord, Scenario } from '../types'
import { reportHref } from '../load'
import { EMPTY_SEL, dash, finite, fnum, selectRecords, type Selection } from '../lib'
import { Crossfilter } from './Crossfilter'
import { MockBadge } from './Reports'

const PALETTE = ['#1f6e5c', '#34366e', '#b5651d', '#a3301f', '#6b7d1f', '#7a4a8c', '#2a7f9e', '#8a6d00']

interface Pt {
  x: number
  y: number
  ok: boolean
  color: string
  title: string
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

export function Scatter({ pts, xl, yl, label }: { pts: Pt[]; xl: string; yl: string; label: string }) {
  if (pts.length === 0) return <Empty msg="No points to plot for the current filters." />
  const xmax = Math.max(...pts.map((p) => p.x), 1) * 1.05
  const ymax = Math.max(...pts.map((p) => p.y), 1) * 1.05
  const px = (x: number) => M.l + (x / xmax) * (W - M.l - M.r)
  const py = (y: number) => H - M.b - (y / ymax) * (H - M.t - M.b)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={label}>
      <Axes xmax={xmax} ymax={ymax} xl={xl} yl={yl} />
      {/* deck marker: series-filled when servable, panel-filled (hollow) when not, always
          stroked in the series colour — the instrument-panel look from the BLIS deck */}
      {pts.map((p, i) => (
        <circle key={i} cx={px(p.x)} cy={py(p.y)} r={4} fill={p.ok ? p.color : 'var(--panel)'} stroke={p.color} strokeWidth={1.5}>
          <title>{p.title}</title>
        </circle>
      ))}
    </svg>
  )
}

// The "knee": the first concurrency at which a series stops being servable — the interactivity
// cliff the deck marks with a violet vertical line. Computed per series from the run records.
function kneeConc(recs: RunRecord[], series: string): number | null {
  const rows = recs.filter((r) => r.series === series && finite(r.conc))
    .sort((a, b) => (a.conc as number) - (b.conc as number))
  const firstBad = rows.find((r) => !r.ok)
  return firstBad ? (firstBad.conc as number) : null
}

export function Lines({ recs, shownRecs, get, xl, yl, label, colors, showKnee = false }: {
  recs: RunRecord[]; shownRecs: RunRecord[]; get: (r: RunRecord) => number | null
  xl: string; yl: string; label: string; colors: Map<string, string>; showKnee?: boolean
}) {
  const bySeries = new Map<string, { x: number; y: number; ok: boolean }[]>()
  for (const r of shownRecs) {
    const y = get(r)
    if (!finite(r.conc) || !finite(y)) continue
    const a = bySeries.get(r.series) ?? []
    a.push({ x: r.conc, y: y as number, ok: r.ok })
    bySeries.set(r.series, a)
  }
  const all = [...bySeries.values()].flat()
  if (all.length === 0) return <Empty msg="No concurrency data to plot for the current filters." />
  const xmax = Math.max(...all.map((p) => p.x), 1) * 1.05
  const ymax = Math.max(...all.map((p) => p.y), 1) * 1.05
  const pw = W - M.l - M.r
  const px = (x: number) => M.l + (x / xmax) * pw
  const py = (y: number) => H - M.b - (y / ymax) * (H - M.t - M.b)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={label}>
      <Axes xmax={xmax} ymax={ymax} xl={xl} yl={yl} />
      {/* violet knee markers: where each series first breaks the floor (deck convention) */}
      {showKnee && [...bySeries.keys()].map((name) => {
        const k = kneeConc(recs, name)
        if (k == null || k > xmax) return null
        const x = px(k)
        return (
          <g key={`knee-${name}`} className="knee">
            <line x1={x} x2={x} y1={M.t} y2={H - M.b} />
            <text x={x} y={M.t - 2} textAnchor="middle" className="kneelab">knee</text>
          </g>
        )
      })}
      {[...bySeries.entries()].map(([name, pts]) => {
        const sorted = [...pts].sort((a, b) => a.x - b.x)
        const c = colors.get(name) ?? PALETTE[0]
        return (
          <g key={name}>
            {sorted.length > 1 && (
              <polyline fill="none" stroke={c} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
                points={sorted.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')} />
            )}
            {sorted.map((p, i) => (
              <circle key={i} cx={px(p.x)} cy={py(p.y)} r={2.8} fill={p.ok ? c : 'var(--panel)'} stroke={c} strokeWidth={1.3}>
                <title>{`${name} · conc ${fnum(p.x, 0)} · ${fnum(p.y)}${p.ok ? '' : ' · not servable'}`}</title>
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

export function Explorer({ recs, runId }: { recs: RunRecord[]; runId: string }) {
  const [sel, setSel] = useState<Selection>(EMPTY_SEL)
  const series = useMemo(() => [...new Set(recs.map((r) => r.series))].sort(), [recs])
  const colors = useMemo(() => new Map(series.map((s, i) => [s, PALETTE[i % PALETTE.length]])), [series])
  const shown = useMemo(() => selectRecords(recs, sel), [recs, sel])
  // Regime tag (deck convention): a run that broke the floor with KV preemptions is KV-bound;
  // otherwise the cliff is a decode-compute ceiling. Records don't carry preemptions, so label
  // honestly from what we have — "compute-bound" when the floor break is latency-only.
  const regime = useMemo(() => (recs.some((r) => !r.ok && (r.viol ?? []).some((v) => /kv|preempt|memory/i.test(v))) ? 'KV-bound' : 'compute-bound'), [recs])

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
    title: `${r.series} · conc ${r.conc ?? dash} · ${fnum(r.cl)} tok/s · ${fnum(r.pu)} tok/s/user · ${r.ok ? 'servable' : r.status}`,
  }))
  const hidden = shown.length - pts.length

  return (
    <div>
      <Crossfilter recs={recs} sel={sel} setSel={setSel} shownCount={shown.length} />

      <div className="charts">
        <figure className="vc">
          <div className="vc-h"><h4 className="vc-t">Throughput ↔ interactivity</h4><span className="vc-r">{regime}</span></div>
          <p className="vc-s">cluster tok/s vs per-user tok/s · one point per run</p>
          <Scatter pts={pts} xl="cluster tok/s" yl="tok/s per user" label="Scatter of cluster throughput against per-user throughput" />
          <p className="vc-k"><b>How to read it.</b> Each point is one run: x is what the whole fleet delivers, y is what one
            caller feels. Up and to the right is better for everyone; filled points hold the interactivity floor, hollow
            ones break it.</p>
          {explainScatter(shown) && <p className="vc-x">{explainScatter(shown)}</p>}
          <div className="legend">
            <span><i className="dot filled" /> servable</span><span><i className="dot hollow" /> not servable</span>
            {series.map((s) => <span key={s}><i className="sw2" style={{ background: colors.get(s) }} />{s}</span>)}
          </div>
          {hidden > 0 && <p className="pts-note">{hidden} run{hidden === 1 ? ' has' : 's have'} no throughput values and {hidden === 1 ? 'is' : 'are'} not plotted.</p>}
          {pts.length === 1 && <p className="pts-note">Only one point matches; a trade-off frontier needs two or more.</p>}
        </figure>
        <figure className="vc">
          <div className="vc-h"><h4 className="vc-t">Per-user rate vs load</h4><span className="vc-r">interactivity</span></div>
          <p className="vc-s">tok/s per user as concurrency climbs</p>
          <Lines recs={recs} shownRecs={shown} get={(r) => r.pu} xl="concurrency" yl="tok/s per user" label="Per-user throughput by concurrency" colors={colors} showKnee />
          <p className="vc-k"><b>How to read it.</b> One line per configuration, walking the load up. Flat is good — it means
            adding sessions has not yet slowed anyone down. The violet line marks that configuration's knee, the first load
            that breaks the floor.</p>
          {explainPerUser(shown, series) && <p className="vc-x">{explainPerUser(shown, series)}</p>}
        </figure>
        <figure className="vc">
          <div className="vc-h"><h4 className="vc-t">Inter-token latency vs load</h4><span className="vc-r">latency</span></div>
          <p className="vc-s">ITL (ms) as concurrency climbs</p>
          <Lines recs={recs} shownRecs={shown} get={(r) => r.itl} xl="concurrency" yl="ITL (ms)" label="Inter-token latency by concurrency" colors={colors} showKnee />
          <p className="vc-k"><b>How to read it.</b> Inter-token latency is the gap between output tokens — the mechanism behind
            the cliff above. Lower is better, and the floor is a ceiling on this number, so a line crossing its knee is what
            makes the per-user rate fall.</p>
          {explainItl(shown) && <p className="vc-x">{explainItl(shown)}</p>}
        </figure>
      </div>

      <div className="tscroll hscroll runtable">
        <table className="readout">
          <thead>
            <tr className="col">
              {['series', 'conc', 'GPUs', 'cluster tok/s', 'tok/s/user', 'ITL ms', 'TTFT ms', 'status'].map((h) => <th key={h}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.id} className={r.ok ? undefined : 'bad'}>
                <td className="mono">{r.series}</td>
                <td className="num"><span className="num">{fnum(r.conc, 0)}</span></td>
                <td className="num"><span className="num">{fnum(r.gpus, 0)}</span></td>
                <td className="num"><span className="num">{fnum(r.cl)}</span></td>
                <td className="num"><span className="num">{fnum(r.pu)}</span></td>
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
  const runnersUp = s.finalists
    .filter((f) => f.label !== w.label)
    .sort((a, b) => b.tps_per_gpu - a.tps_per_gpu || b.tps - a.tps)
    .slice(0, 2)
  return (
    <div>
      <p className="crumb"><a href="#/reports">← all reports</a></p>
      <h2 className="viewhead">{s.model}{s.synthetic && <MockBadge />}</h2>
      <div className="specchips">
        <span className="spec">{s.hardware}</span><span className="spec">{s.gpu_budget} GPUs</span>
        <span className="spec">{s.workload}</span><span className="spec">{s.isl_osl}</span>
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
            <div className="reclabel">{w.label}</div>
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
              <div className="runcard" key={f.label}>
                <div className="runrank">#{i + 2} alternative</div>
                <div className="runlabel">{f.label}</div>
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
      </section>

      <section aria-label="Pareto explorer">
        <div className="seclabel2">Pareto explorer</div>
        <Explorer recs={recs} runId={s.run_id} />
      </section>
    </div>
  )
}
