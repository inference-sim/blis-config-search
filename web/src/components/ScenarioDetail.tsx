import { useEffect, useMemo, useState } from 'react'
import type { RunRecord, Scenario } from '../types'
import { loadRecords, reportHref } from '../load'
import { BANDS, dash, filterRecords, finite, fnum, type Band, type RecFilters } from '../lib'
import { Dropdown } from './Dropdown'
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
      {pts.map((p, i) => (
        <circle key={i} cx={px(p.x)} cy={py(p.y)} r={4.5} fill={p.ok ? p.color : 'var(--panel)'} stroke={p.color} strokeWidth={1.6}>
          <title>{p.title}</title>
        </circle>
      ))}
    </svg>
  )
}

export function Lines({ recs, get, xl, yl, label, colors }: { recs: RunRecord[]; get: (r: RunRecord) => number | null; xl: string; yl: string; label: string; colors: Map<string, string> }) {
  const bySeries = new Map<string, { x: number; y: number }[]>()
  for (const r of recs) {
    const y = get(r)
    if (!finite(r.conc) || !finite(y)) continue
    const a = bySeries.get(r.series) ?? []
    a.push({ x: r.conc, y })
    bySeries.set(r.series, a)
  }
  const all = [...bySeries.values()].flat()
  if (all.length === 0) return <Empty msg="No concurrency data to plot for the current filters." />
  const xmax = Math.max(...all.map((p) => p.x), 1) * 1.05
  const ymax = Math.max(...all.map((p) => p.y), 1) * 1.05
  const px = (x: number) => M.l + (x / xmax) * (W - M.l - M.r)
  const py = (y: number) => H - M.b - (y / ymax) * (H - M.t - M.b)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={label}>
      <Axes xmax={xmax} ymax={ymax} xl={xl} yl={yl} />
      {[...bySeries.entries()].map(([name, pts]) => {
        const sorted = [...pts].sort((a, b) => a.x - b.x)
        const c = colors.get(name) ?? PALETTE[0]
        return (
          <g key={name}>
            {sorted.length > 1 && <polyline fill="none" stroke={c} strokeWidth={1.6} points={sorted.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')} />}
            {sorted.map((p, i) => (
              <circle key={i} cx={px(p.x)} cy={py(p.y)} r={2.8} fill={c}>
                <title>{`${name} · conc ${fnum(p.x, 0)} · ${fnum(p.y)}`}</title>
              </circle>
            ))}
          </g>
        )
      })}
    </svg>
  )
}

export function Explorer({ recs, runId }: { recs: RunRecord[]; runId: string }) {
  const [f, setF] = useState<RecFilters>({ series: '', band: '', servable: '' })
  const series = useMemo(() => [...new Set(recs.map((r) => r.series))].sort(), [recs])
  const colors = useMemo(() => new Map(series.map((s, i) => [s, PALETTE[i % PALETTE.length]])), [series])
  const shown = useMemo(() => filterRecords(recs, f), [recs, f])

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
      <div className="chips filterbar" role="group" aria-label="Run filters">
        <span className="flt">series
          <Dropdown ariaLabel="Filter by series / topology" value={f.series} onChange={(v) => setF({ ...f, series: v })}
            options={[{ value: '', label: `all series (${recs.length})` }, ...series.map((s) => ({ value: s, label: `${s} (${recs.filter((r) => r.series === s).length})` }))]} />
        </span>
        <span className="flt">concurrency
          <Dropdown ariaLabel="Filter by concurrency band" value={f.band} onChange={(v) => setF({ ...f, band: v as Band })}
            options={BANDS.map((b) => ({ value: b.value, label: b.label }))} />
        </span>
        <div className="tierseg" role="group" aria-label="Servable">
          {([['', 'all'], ['ok', 'servable'], ['bad', 'not servable']] as const).map(([v, l]) => (
            <button type="button" key={v} className={`segbtn${f.servable === v ? ' on' : ''}`} aria-pressed={f.servable === v} onClick={() => setF({ ...f, servable: v })}>{l}</button>
          ))}
        </div>
        <span className="facet">{shown.length} of {recs.length} runs</span>
      </div>

      <div className="charts">
        <figure className="chartbox">
          <figcaption>Pareto: cluster tok/s vs per-user tok/s</figcaption>
          <Scatter pts={pts} xl="cluster tok/s" yl="tok/s per user" label="Scatter of cluster throughput against per-user throughput" />
          <div className="legend">
            <span><i className="dot filled" /> servable</span><span><i className="dot hollow" /> not servable</span>
            {series.map((s) => <span key={s}><i className="sw2" style={{ background: colors.get(s) }} />{s}</span>)}
          </div>
          {hidden > 0 && <p className="pts-note">{hidden} run{hidden === 1 ? ' has' : 's have'} no throughput values and {hidden === 1 ? 'is' : 'are'} not plotted.</p>}
          {pts.length === 1 && <p className="pts-note">Only one point matches; a trade-off frontier needs two or more.</p>}
        </figure>
        <figure className="chartbox">
          <figcaption>Per-user tok/s vs concurrency</figcaption>
          <Lines recs={shown} get={(r) => r.pu} xl="concurrency" yl="tok/s per user" label="Per-user throughput by concurrency" colors={colors} />
        </figure>
        <figure className="chartbox">
          <figcaption>ITL (ms) vs concurrency</figcaption>
          <Lines recs={shown} get={(r) => r.itl} xl="concurrency" yl="ITL (ms)" label="Inter-token latency by concurrency" colors={colors} />
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
  const [recs, setRecs] = useState<RunRecord[] | null>(null)
  useEffect(() => {
    let live = true
    setRecs(null)
    loadRecords(runId).then((r) => live && setRecs(Array.isArray(r) ? r : [])).catch(() => live && setRecs([]))
    return () => {
      live = false
    }
  }, [runId])

  if (!s) {
    return (
      <div>
        <p><a href="#/reports">← all reports</a></p>
        <div className="empty">Unknown scenario <span className="mono">{runId}</span>.</div>
      </div>
    )
  }
  const w = s.winner
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

      <section className="briefing" aria-label="Briefing">
        <div className="seclabel2">Recommended config</div>
        <div className="reclabel">{w.label}</div>
        <div className="stats">
          <div><dt>cluster tok/s</dt><dd className="num best">{fnum(w.tps)}</dd></div>
          <div><dt>tok/s/GPU</dt><dd className="num best">{fnum(w.tps_per_gpu)}</dd></div>
          <div><dt>tok/s/user</dt><dd className="num best">{fnum(w.per_user)}</dd></div>
          <div><dt>ITL p99 (ms)</dt><dd className="num best">{fnum(w.itl_p99, 2)}</dd></div>
        </div>
        <p className="dek"><b>Floor held:</b> <span className="mono">{s.floor}</span>{s.headroom != null && <> (headroom <span className="num">{fnum(s.headroom, 2)}</span>)</>}. {s.reason}{' '}
          <span className="num">{s.n_servable}</span> of <span className="num">{s.n_runs}</span> simulated runs are servable.</p>
        <div className="tbl-title">Finalists</div>
        <div className="tscroll hscroll">
          <table className="readout">
            <thead><tr className="col">{['config', 'GPUs', 'conc', 'tok/s', 'tok/s/GPU', 'tok/s/user', 'ITL p99 ms'].map((h) => <th key={h}>{h}</th>)}</tr></thead>
            <tbody>
              {s.finalists.map((f) => (
                <tr key={f.label} className={f.label === w.label ? 'sel' : undefined}>
                  <td className="mono">{f.label}{f.label === w.label && <span className="ver">recommended</span>}</td>
                  <td className="num"><span className="num">{fnum(f.gpus, 0)}</span></td>
                  <td className="num"><span className="num">{fnum(f.c_use, 0)}</span></td>
                  <td className="num"><span className="num">{fnum(f.tps)}</span></td>
                  <td className="num"><span className="num">{fnum(f.tps_per_gpu)}</span></td>
                  <td className="num"><span className="num">{fnum(f.per_user)}</span></td>
                  <td className="num"><span className="num">{fnum(f.itl_p99, 2)}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-label="Pareto explorer">
        <div className="seclabel2">Pareto explorer</div>
        {recs === null ? <div className="loading">Loading runs…</div> : <Explorer recs={recs} runId={s.run_id} />}
      </section>
    </div>
  )
}
