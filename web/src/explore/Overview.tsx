// The two overview charts. Both follow the ranking target, share one hover (pointing at a config
// on either chart lights it up on both), and reset when you click empty chart space.
import { useMemo } from 'react'
import type { Explorer } from './useExplorer'
import type { MetricKey, Run } from './types'
import { METRIC, NA, SLO_OF, TOPO_FIELDS, TOPO_INFO, bareKey, fmt, fmtMetric, loadLabel, niceMax, poolOf, quantile, sloMs, sortValues, tagsOf } from './model'
import { Backdrop, H, LegendBtn, M, W, YAxis } from './chart'

const PAL = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c7)', 'var(--c5)']
// categorical order for coloring by a field; past seven values the rest share one "other" hue
const CAT = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)', 'var(--c6)', 'var(--c7)']
const OTHER = 'var(--c8)'

/** Each value of the color-by field mapped to a hue, in the field's own value order. */
export function colorScale(x: Explorer): { values: string[]; of: (r: Run) => string; field: string | null } {
  const k = x.colorBy
  if (!k) return { values: [], of: () => CAT[0], field: null }
  const values = sortValues([...new Set(x.runs.map((r) => x.fv(r, k)))]).filter((v) => v !== NA)
  const hue = new Map(values.map((v, i) => [v, i < CAT.length ? CAT[i] : OTHER]))
  return { values, of: (r) => hue.get(x.fv(r, k)) ?? OTHER, field: k }
}
export const fieldTitle = (k: string) => {
  if (k === 'mode') return 'serving mode'
  if (k === 'router_config') return 'router'
  const t = TOPO_FIELDS.find((f) => f.key === k)
  const base = t ? TOPO_INFO[t.part].title.split(' · ')[0] : bareKey(k)
  return `${poolOf(k) ? poolOf(k) + ' ' : ''}${base}`
}

/** Latency targets cap the y range near the bulk of the runs, so a few timeouts do not flatten the rest. */
function yRange(x: Explorer, m: MetricKey, vals: number[]) {
  const slo = SLO_OF[m] ? sloMs(x.data.slo[SLO_OF[m] as string]) : null
  const ym = niceMax(METRIC[m].higher ? Math.max(...vals, 0) : Math.max(quantile(vals, 0.95) ?? 0, (slo ?? 0) * 1.5))
  return { ym, slo }
}
/** For latency, lower plots higher, so "better" is up on every chart. */
const yScale = (m: MetricKey, ym: number) => (v: number) => {
  const t = Math.min(v, ym) / ym
  return METRIC[m].higher ? H - M.b - t * (H - M.t - M.b) : M.t + t * (H - M.t - M.b)
}

export function runTip(x: Explorer, r: Run) {
  const tags = tagsOf(r, x.fields, x.usual, x.data.routers).map((t) => t.text).join(' · ')
  return `${r.topo} (${r.mode})${tags ? `\n${tags}` : ''}\n${loadLabel(r.conc)} · sweep ${r.sweep}\n`
    + `${fmt(r.tpg, 1)} tok/s/acc · ${fmt(r.pu, 1)} tok/s/user\nTPOT p99 ${fmt(r.itl, 1)} ms · TTFT p99 ${fmt(r.ttft, 0)} ms\n${r.ok ? 'meets every SLO' : 'misses an SLO'}`
}

export function Scatter({ x }: { x: Explorer }) {
  const tips = useMemo(() => new Map(x.runs.map((r) => [r.id, runTip(x, r)])), [x.runs, x.fields, x.usual]) // eslint-disable-line react-hooks/exhaustive-deps
  const y = x.target
  const xk: MetricKey = y === 'pu' ? 'tpg' : 'pu'
  const rs = x.shown().filter((r) => r[xk] !== null && r[y] !== null)
  if (!rs.length) return <p className="xf-empty">No runs match these filters. <button type="button" className="linkbtn" onClick={x.resetCharts}>reset the charts</button></p>
  const hi = METRIC[y].higher
  const xm = niceMax(Math.max(...rs.map((r) => r[xk] as number)))
  const { ym, slo } = yRange(x, y, rs.map((r) => r[y] as number))
  const px = (v: number) => M.l + (v / xm) * (W - M.l - M.r)
  const py = yScale(y, ym)
  const focus = x.hover ?? x.picked
  const on = (r: Run) => !focus || r.cfg === focus
  const over = rs.filter((r) => (r[y] as number) > ym).length
  const xticks = Array.from({ length: 6 }, (_, i) => (xm / 5) * i)
  const small = rs.length > 1500 ? 2.4 : 3.4
  const color = colorScale(x)
  return (
    <svg className={`chart xf-chart${focus ? ' focused' : ''}`} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${METRIC[y].label} against ${METRIC[xk].label}`}
      onPointerLeave={() => x.setHover(null)}>
      <YAxis ym={ym} py={py} label={`${METRIC[y].label}${hi ? ' →' : ' (lower is up)'}`} />
      <Backdrop onClick={x.resetCharts} />
      {xticks.map((v) => <text key={v} x={px(v)} y={H - M.b + 13} textAnchor="middle" className="tick">{fmt(v, 0)}</text>)}
      <text x={(M.l + W - M.r) / 2} y={H - 3} textAnchor="middle" className="axlabel">{METRIC[xk].label} →</text>
      {slo !== null && slo <= ym && <g className="xf-slo"><line x1={M.l} x2={W - M.r} y1={py(slo)} y2={py(slo)} /><text x={W - M.r} y={py(slo) - 4} textAnchor="end">SLO</text></g>}
      {[...rs].sort((a, b) => Number(on(a)) - Number(on(b))).map((r) => (
        <circle key={r.id} cx={px(r[xk] as number)} cy={py(r[y] as number)} r={focus && on(r) ? 5.2 : small}
          style={on(r) ? { color: color.of(r) } : undefined}
          className={`xf-dot${on(r) ? '' : ' off'}${r.ok ? '' : ' miss'}${focus && on(r) ? ' hot' : ''}`}
          onPointerEnter={() => x.setHover(r.cfg)} onClick={() => x.pick(r.cfg)}>
          <title>{tips.get(r.id)}</title>
        </circle>
      ))}
      {x.ranked.slice(0, 3).map((g, i) => rs.some((r) => r.id === g.lead.id) && <text key={i} className="xf-rank" x={px(g.lead[xk] as number) + 7} y={py(g.lead[y] as number) - 6}>{i + 1}</text>)}
      {over > 0 && <text className="axlabel" x={W - M.r} y={hi ? M.t + 8 : H - M.b - 6} textAnchor="end">{over} run{over === 1 ? '' : 's'} beyond {fmt(ym, 0)} pinned to the edge</text>}
    </svg>
  )
}

/** The configs the load ramp draws as lines: the top three, then whatever is picked or hovered. */
function followed(x: Explorer): string[] {
  return [...new Set([...x.ranked.slice(0, 3).map((g) => g.lead.cfg), x.picked, x.hover].filter((c): c is string => !!c))]
}

export function LoadRamp({ x }: { x: Explorer }) {
  const m = x.target
  const loads = x.data.concurrency
  const rs = x.shown().filter((r) => r[m] !== null && loads.includes(r.conc))
  if (!rs.length) return <p className="xf-empty">No runs match these filters.</p>
  const { ym, slo } = yRange(x, m, rs.map((r) => r[m] as number))
  const px = (c: number) => M.l + (loads.length === 1 ? 0.5 : loads.indexOf(c) / (loads.length - 1)) * (W - M.l - M.r)
  const py = yScale(m, ym)
  const hi = METRIC[m].higher
  const span = ((W - M.l - M.r) / Math.max(1, loads.length - 1)) * 0.35
  const jit = (id: number) => (((id * 2654435761) % 1000) / 1000 - 0.5) * span
  const follow = followed(x)
  const focus = x.hover ?? x.picked
  return (
    <svg className={`chart xf-chart${focus ? ' focused' : ''}`} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${METRIC[m].label} by concurrency`}
      onPointerLeave={() => x.setHover(null)}>
      <YAxis ym={ym} py={py} label={`${METRIC[m].label}${hi ? '' : ' ms (lower is up)'}`} />
      <Backdrop onClick={x.resetCharts} />
      {loads.map((c) => <text key={c} x={px(c)} y={H - M.b + 13} textAnchor="middle" className="tick">{c}</text>)}
      <text x={(M.l + W - M.r) / 2} y={H - 3} textAnchor="middle" className="axlabel">concurrency (the spec’s loads)</text>
      {slo !== null && <g className="xf-slo"><line x1={M.l} x2={W - M.r} y1={py(slo)} y2={py(slo)} /><text x={W - M.r} y={py(slo) - 4} textAnchor="end">SLO {x.data.slo[SLO_OF[m] as string]}</text></g>}
      {rs.map((r) => (
        <circle key={r.id} cx={px(r.conc) + jit(r.id)} cy={py(r[m] as number)} r={r.cfg === focus ? 3.2 : 1.8}
          className={`xf-bg${r.cfg === focus ? ' hot' : ''}`} onPointerEnter={() => x.setHover(r.cfg)} onClick={() => x.pick(r.cfg)} />
      ))}
      {follow.map((cfg, i) => {
        const path = rs.filter((r) => r.cfg === cfg).sort((a, b) => a.conc - b.conc)
        if (!path.length) return null
        return (
          <g key={cfg} style={{ color: PAL[i % PAL.length] }} className={`xf-path${focus && cfg !== focus ? ' faded' : ''}${cfg === focus ? ' hot' : ''}`}
            onPointerEnter={() => x.setHover(cfg)}>
            <polyline points={path.map((r) => `${px(r.conc)},${py(r[m] as number)}`).join(' ')} />
            {path.map((r) => <circle key={r.id} cx={px(r.conc)} cy={py(r[m] as number)} r="3.8" className={r.ok ? '' : 'miss'} onClick={() => x.pick(r.cfg)}><title>{runTip(x, r)}</title></circle>)}
          </g>
        )
      })}
    </svg>
  )
}

export function OverviewSection({ x }: { x: Explorer }) {
  const y = x.target, xk: MetricKey = y === 'pu' ? 'tpg' : 'pu'
  const follow = followed(x)
  const color = colorScale(x)
  const cSel = color.field ? x.sel[color.field] ?? [] : []
  const anyChartFilter = cSel.length > 0 || x.okOnly || !!x.picked
  const shownVals = color.values.slice(0, 7)
  return (
    <section className="xf-sec" id="sec-overview" aria-labelledby="h-overview">
      <div className="xf-sec-h"><h2 id="h-overview">Overview</h2>
        <p>Point at a run to light up its config on both charts; click it to filter the page. Click empty chart space to reset.</p>
        {anyChartFilter && <button type="button" className="linkbtn" onClick={x.resetCharts}>reset charts</button>}
      </div>
      <div className="xf-grid2">
        <div className="xf-card">
          <div className="xf-card-h">
            <div><h3>The trade-off</h3><p className="xf-sub">{METRIC[y].label} against {METRIC[xk].label}, one point per run</p></div>
            {x.fields.length > 0 && (
              <label className="xf-colorby" htmlFor="xf-colorby">color by
                <select id="xf-colorby" value={color.field ?? ''} onChange={(e) => x.setColorBy(e.target.value)}>
                  {x.fields.map((k) => <option key={k} value={k}>{fieldTitle(k)}</option>)}
                </select>
              </label>
            )}
          </div>
          <Scatter x={x} />
          <div className="legend xf-legend">
            {color.field && shownVals.map((v, i) => (
              <LegendBtn key={v} on={cSel.includes(v)} off={cSel.length > 0 && !cSel.includes(v)} onClick={() => x.toggleValue(color.field as string, v)}
                label={`${cSel.includes(v) ? 'Stop filtering to' : 'Show only'} ${fieldTitle(color.field as string)} ${v}`}>
                <i className="sw2" style={{ background: CAT[i] }} />{v}
              </LegendBtn>
            ))}
            {color.values.length > 7 && <span className="xf-lg-static"><i className="sw2" style={{ background: OTHER }} />{color.values.length - 7} more</span>}
            {color.field && x.shown().some((r) => x.fv(r, color.field as string) === NA) && (
              <span className="xf-lg-static" title={`runs that have no ${fieldTitle(color.field)} (the other serving mode)`}><i className="sw2" style={{ background: OTHER }} />n/a</span>
            )}
            <span className="xf-lg-sep" aria-hidden="true" />
            <LegendBtn on={x.okOnly} onClick={() => x.setOkOnly(!x.okOnly)} label={x.okOnly ? 'Show runs that miss an SLO too' : 'Hide runs that miss an SLO'}>
              <i className="dot filled" />meets SLO
            </LegendBtn>
            <LegendBtn off={x.okOnly} onClick={() => x.setOkOnly(!x.okOnly)} label={x.okOnly ? 'Show runs that miss an SLO' : 'Hide runs that miss an SLO'}>
              <i className="dot hollow" />misses
            </LegendBtn>
            <span className="xf-lg-static">· · · best trade-off</span>
          </div>
        </div>
        <div className="xf-card">
          <div className="xf-card-h">
            <div><h3>How it scales with load</h3><p className="xf-sub">{METRIC[y].label} at each of the spec’s loads; lines follow the top configs</p></div>
          </div>
          <LoadRamp x={x} />
          <div className="legend xf-legend">
            {follow.map((cfg, i) => {
              const r = x.runs.find((q) => q.cfg === cfg)
              const rank = x.ranked.findIndex((g) => g.lead.cfg === cfg)
              if (!r) return null
              return (
                <LegendBtn key={cfg} on={x.picked === cfg} off={!!x.hover && x.hover !== cfg} onClick={() => x.pick(cfg)}
                  onHover={(h) => x.setHover(h ? cfg : null)} label={`${x.picked === cfg ? 'Clear' : 'Filter the page to'} ${r.topo}`}>
                  <i className="sw2" style={{ background: PAL[i % PAL.length] }} />{rank >= 0 && rank < 3 ? `#${rank + 1} ` : ''}{r.topo}
                </LegendBtn>
              )
            })}
            <span className="xf-lg-static"><i className="sw2 bgdot" />every other run</span>
          </div>
        </div>
      </div>
      <HoverLine x={x} idle="Point at a run on either chart to compare it across both." />
    </section>
  )
}

/** What the shared hover points at: the hovered config's best run, read out under the charts. */
export function HoverLine({ x, idle }: { x: Explorer; idle: string }) {
  const y = x.target
  const hovered = x.hover ? x.runs.find((r) => r.cfg === x.hover) : null
  const best = hovered ? x.ranked.find((g) => g.lead.cfg === x.hover)?.lead ?? hovered : null
  return (
    <p className={`xf-hoverline${best ? '' : ' idle'}`} aria-live="polite">
      {best
        ? <><b className="mono">{best.topo}</b> {tagsOf(best, x.fields, x.usual, x.data.routers).map((t) => t.text).join(' · ')} · best {fmtMetric(y, best[y])} {METRIC[y].label} at {loadLabel(best.conc)} · click to filter</>
        : idle}
    </p>
  )
}
