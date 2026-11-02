// The three layer sections. Each is one crossfilter over every run in the report.
import { useState } from 'react'
import type { Explorer } from './useExplorer'
import type { Run } from './types'
import { FieldChart, barKey } from './FieldChart'
import { RouterPanel } from './RouterView'
import { H, M, W, YAxis } from './chart'
import { ENGINE_HELP, METRIC, TOPO_FIELDS, TOPO_INFO, bareKey, fmt, fmtMetric, layerOf, matches, niceMax, poolOf, rankConfigs, type Pool } from './model'

/* ---------------- topology ---------------- */

export function TopologySection({ x }: { x: Explorer }) {
  const modes = new Set(x.runs.map((r) => r.mode))
  const inView = new Set(x.shown().map((r) => r.mode))
  const groups: { pool: Pool; label: string | null }[] = modes.has('P/D')
    ? [{ pool: null, label: 'Aggregated' }, { pool: 'prefill', label: 'Prefill pool' }, { pool: 'decode', label: 'Decode pool' }]
        .filter((g) => inView.has(g.pool ? 'P/D' : 'aggregated')) as { pool: Pool; label: string | null }[]
    : [{ pool: null, label: null }]
  const best = (x.picked && rankConfigs(x.shown().filter((r) => r.cfg === x.picked), x.target)[0]) || rankConfigs(x.shown(), x.target)[0]
  const pdInView = inView.has('P/D')
  return (
    <section className="xf-sec" id="sec-topology" aria-labelledby="h-topology">
      <SectionHead id="h-topology" layer="topology" title="Topology" sub="How the accelerators are split: parallelism, instance count, and aggregated or prefill/decode pools." />
      <div className="xf-card">
        <div className="xf-card-h"><h3>Parallelism</h3><span className="xf-sub">{barKey(x)} Click a bar to cross-filter.</span></div>
        <div className="xf-topo">
          <div className="xf-topo-charts">
            {x.fields.includes('mode') && <div className="xf-grid"><FieldChart x={x} k="mode" title="serving mode" help="one shared pool (aggregated) or separate prefill and decode pools (P/D)" /></div>}
            {groups.map((g) => {
              const ks = TOPO_FIELDS.filter((f) => f.pool === g.pool && x.fields.includes(f.key))
              if (!ks.length) return null
              return (
                <div className="xf-group" key={g.pool ?? 'shared'}>
                  {g.label && <p className="eyebrow">{g.label}</p>}
                  <div className="xf-grid">{ks.map((f) => <FieldChart key={f.key} x={x} k={f.key} title={TOPO_INFO[f.part].title} help={TOPO_INFO[f.part].help} />)}</div>
                </div>
              )
            })}
          </div>
          {best && <GpuLayout r={best} x={x} />}
        </div>
      </div>
      {modes.has('P/D') && pdInView && (
        <div className="xf-grid2">
          {modes.size > 1 && <ModeChart x={x} />}
          <RatioHeatmap x={x} />
        </div>
      )}
    </section>
  )
}

export function SectionHead({ id, layer, title, sub }: { id: string; layer: string; title: string; sub: string }) {
  return (
    <div className="xf-sec-h">
      <h2 id={id}><i className={`xf-swatch l-${layer}`} />{title}</h2>
      <p>{sub}</p>
    </div>
  )
}

/** Accelerators drawn as squares: one block per instance, columns TP, rows DP × PP. */
function GpuLayout({ r, x }: { r: Run; x: Explorer }) {
  const pools: Pool[] = r.mode === 'P/D' ? ['prefill', 'decode'] : [null]
  const pp = pools.some((p) => Number(r.knobs[p ? `${p}_pp` : 'pp'] ?? 1) > 1)
  return (
    <aside className="xf-gpu" aria-label="Accelerator layout">
      <p className="eyebrow">{x.picked ? 'Selected config' : 'Best in the filter'}</p>
      <p className="xf-gpu-name mono">{r.topo}</p>
      <p className="xf-sub">{fmtMetric(x.target, r[x.target])} {METRIC[x.target].label} at {r.conc} sessions</p>
      <div className="xf-gpu-row">
        {pools.map((p, i) => (
          <div key={p ?? 'shared'} className="xf-gpu-pool">
            {i > 0 && <span className="xf-gpu-kv">KV cache hands off ↓</span>}
            <PoolBlocks r={r} pool={p} />
          </div>
        ))}
      </div>
      <dl className="xf-gpu-key">
        <div><dt><i className="k1" /></dt><dd>one accelerator</dd></div>
        <div><dt><i className="k3" /></dt><dd>one instance: columns are TP, rows are DP{pp ? ' × PP stages (darker is later)' : ''}</dd></div>
        <div><dt><i className="k2" /></dt><dd>solid frame: experts sharded across the instance (EP)</dd></div>
      </dl>
    </aside>
  )
}
function PoolBlocks({ r, pool }: { r: Run; pool: Pool }) {
  const g = (k: string) => Number(r.knobs[pool ? `${pool}_${k}` : k] ?? (k === 'tp' || k === 'dp' || k === 'pp' ? 1 : 0))
  const n = Number(pool ? r.knobs[`${pool}_instances`] : r.knobs.num_instances) || 1
  const tp = g('tp') || 1, dp = g('dp') || 1, pp = g('pp') || 1
  const ep = !!r.knobs[pool ? `${pool}_enable_expert_parallel` : 'enable_expert_parallel']
  const per = tp * dp * pp, cell = per > 16 ? 5 : per > 8 ? 7 : 9, gap = 1
  const rows = dp * pp, w = tp * (cell + gap) + 6, h = rows * (cell + gap) + 6
  const shown = Math.min(n, 12)
  return (
    <span className="xf-gpu-blocks">
      <span className="xf-gpu-h">{pool && <span className="xf-pool">{pool}</span>}<span className="mono">{n} × (TP{tp}{dp > 1 ? ` × DP${dp}` : ''}{pp > 1 ? ` × PP${pp}` : ''}){ep ? ' · EP' : ''}</span><span className="xf-sub">{n * per} accelerators</span></span>
      <span className="xf-gpu-inst">
        {Array.from({ length: shown }, (_, i) => (
          <svg key={i} width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
            <rect x=".5" y=".5" width={w - 1} height={h - 1} rx="3" className={ep ? 'xf-frame ep' : 'xf-frame'} />
            {Array.from({ length: rows * tp }, (_, j) => {
              const row = Math.floor(j / tp), col = j % tp
              return <rect key={j} x={3 + col * (cell + gap)} y={3 + row * (cell + gap)} width={cell} height={cell} rx="1.5"
                className={pool === 'prefill' ? 'xf-acc pre' : 'xf-acc'} style={pp > 1 ? { opacity: 0.45 + 0.55 * ((row % pp) / (pp - 1)) } : undefined} />
            })}
          </svg>
        ))}
        {n > shown && <span className="xf-sub">+{n - shown}</span>}
      </span>
    </span>
  )
}


function ModeChart({ x }: { x: Explorer }) {
  const m = x.target, loads = x.data.concurrency
  const rs = x.shown()
  const best = (mode: string, c: number) => rankConfigs(rs.filter((r) => r.mode === mode && r.conc === c), m)[0]
  const series = (['aggregated', 'P/D'] as const).map((mode) => ({ mode, pts: loads.map((c) => best(mode, c)).filter((r): r is Run => !!r) }))
  const all = series.flatMap((s) => s.pts.map((r) => r[m] as number))
  if (!all.length) return <div className="xf-card"><h3>Aggregated vs disaggregated</h3><p className="xf-empty">No run in this filter meets every SLO.</p></div>
  const ym = niceMax(Math.max(...all))
  const px = (c: number) => M.l + (loads.indexOf(c) / Math.max(1, loads.length - 1)) * (W - M.l - M.r)
  const py = (v: number) => H - M.b - (v / ym) * (H - M.t - M.b)
  return (
    <div className="xf-card">
      <div className="xf-card-h"><h3>Aggregated vs disaggregated</h3></div>
      <p className="xf-sub">Best {METRIC[m].label} at each load among runs that meet every SLO.</p>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`best ${METRIC[m].label} by serving mode`}>
        <YAxis ym={ym} py={py} label={`best ${METRIC[m].label}`} />
        {loads.map((c) => <text key={c} x={px(c)} y={H - M.b + 13} textAnchor="middle" className="tick">{c}</text>)}
        <text x={(M.l + W - M.r) / 2} y={H - 3} textAnchor="middle" className="axlabel">sessions</text>
        {series.map((s) => (
          <g key={s.mode} className={s.mode === 'P/D' ? 'xf-ser pd' : 'xf-ser'}>
            {s.pts.length > 1 && <polyline points={s.pts.map((r) => `${px(r.conc)},${py(r[m] as number)}`).join(' ')} />}
            {s.pts.map((r) => <circle key={r.id} cx={px(r.conc)} cy={py(r[m] as number)} r="4"><title>{`${s.mode} · ${r.topo} · ${r.conc} sessions · ${fmtMetric(m, r[m])} ${METRIC[m].label}`}</title></circle>)}
          </g>
        ))}
      </svg>
      <div className="legend"><span><i className="sw2 c-colo" />aggregated</span><span><i className="sw2 c-pd" />P/D</span></div>
    </div>
  )
}


function RatioHeatmap({ x }: { x: Explorer }) {
  const [measure, setMeasure] = useState<'best' | 'pass'>('best')
  const m = x.target
  // Every cell counts the runs the other filters keep; the ratio filters themselves only outline a cell.
  const others = { ...x.sel }
  delete others.prefill_instances
  delete others.decode_instances
  const pd = x.runs.filter((r) => r.mode === 'P/D' && (!x.okOnly || r.ok) && matches(r, others, x.data.routers))
  const all = x.runs.filter((r) => r.mode === 'P/D')
  const P = [...new Set(all.map((r) => Number(r.knobs.prefill_instances)))].sort((a, b) => a - b)
  const D = [...new Set(all.map((r) => Number(r.knobs.decode_instances)))].sort((a, b) => a - b)
  const cell = (p: number, d: number) => pd.filter((r) => Number(r.knobs.prefill_instances) === p && Number(r.knobs.decode_instances) === d)
  const val = (rs: Run[]) => !rs.length ? null : measure === 'pass' ? rs.filter((r) => r.ok).length / rs.length : (rankConfigs(rs, m)[0]?.[m] ?? 0)
  const vals = P.flatMap((p) => D.map((d) => val(cell(p, d)))).filter((v): v is number => !!v)
  const hi = Math.max(...vals, 1e-9), lo = Math.min(...vals, hi)
  const shade = (v: number) => measure === 'pass' || METRIC[m].higher ? v / hi : lo / v
  const selP = x.sel.prefill_instances ?? [], selD = x.sel.decode_instances ?? []
  // a cell is pressed only when it is exactly the ratio filter, so pressed and click agree
  const isOn = (p: number, d: number) => selP.length === 1 && selD.length === 1 && selP[0] === String(p) && selD[0] === String(d)
  const click = (p: number, d: number) => {
    const on = isOn(p, d)
    x.setField('prefill_instances', on ? [] : [String(p)])
    x.setField('decode_instances', on ? [] : [String(d)])
  }
  return (
    <div className="xf-card">
      <div className="xf-card-h"><h3>Prefill : decode ratio</h3>
        <div className="xf-seg" role="group" aria-label="Cells show">
          <button type="button" aria-pressed={measure === 'best'} onClick={() => setMeasure('best')}>best {METRIC[m].short}</button>
          <button type="button" aria-pressed={measure === 'pass'} onClick={() => setMeasure('pass')}>% meeting SLO</button>
        </div>
      </div>
      <p className="xf-sub">One cell per ratio, across every pool shape and load in the filter. Click a cell to filter to it; empty cells have no runs in the filter.</p>
      <div className="xf-heat" style={{ gridTemplateColumns: `auto repeat(${D.length}, minmax(48px, 1fr))` }}>
        <span />
        {D.map((d) => <span key={d} className="xf-heat-h">{d}D</span>)}
        {P.map((p) => [
          <span key={`l${p}`} className="xf-heat-h">{p}P</span>,
          ...D.map((d) => {
            const rs = cell(p, d), v = val(rs), b = rankConfigs(rs, m)[0]
            const on = isOn(p, d)
            if (v === null) return <span key={`${p}-${d}`} className="xf-heat-c empty" />
            const t = v ? shade(v) : 0
            return (
              <button type="button" key={`${p}-${d}`} className={`xf-heat-c${on ? ' on' : ''}`} aria-pressed={on} onClick={() => click(p, d)}
                aria-label={`${p}P / ${d}D: ${measure === 'pass' ? `${fmt(v * 100, 0)}% of runs meet every SLO` : v ? `best ${fmtMetric(m, v)} ${METRIC[m].label}` : 'no run meets every SLO'}`}
                style={{ ['--t' as string]: String(0.08 + t * 0.82) }}
                title={`${p}P / ${d}D: ${rs.length} runs, ${rs.filter((r) => r.ok).length} meet every SLO${b ? `; best ${b.topo}, ${fmtMetric(m, b[m])} ${METRIC[m].label} at ${b.conc} sessions` : ''}`}>
                <span className={t > 0.55 ? 'light' : ''}>{measure === 'pass' ? `${fmt(v * 100, 0)}%` : v ? fmtMetric(m, v) : 'none'}</span>
              </button>
            )
          }),
        ])}
      </div>
    </div>
  )
}

/* ---------------- routing ---------------- */

export function RoutingSection({ x }: { x: Explorer }) {
  return (
    <section className="xf-sec" id="sec-routing" aria-labelledby="h-routing">
      <SectionHead id="h-routing" layer="routing" title="Routing" sub="The llm-d router config that picks an instance for each request." />
      <div className="xf-card"><RouterPanel x={x} /></div>
    </section>
  )
}

/* ---------------- engine ---------------- */

export function EngineSection({ x }: { x: Explorer }) {
  const ek = x.fields.filter((k) => layerOf(k) === 'engine')
  const pd = x.runs.some((r) => r.mode === 'P/D')
  const groups: { pool: Pool; label: string | null }[] = [
    { pool: null, label: pd ? 'Aggregated, and shared by both pools' : null }, { pool: 'prefill', label: 'Prefill pool' }, { pool: 'decode', label: 'Decode pool' },
  ]
  return (
    <section className="xf-sec" id="sec-engine" aria-labelledby="h-engine">
      <SectionHead id="h-engine" layer="engine" title="vLLM engine" sub="Engine flags, per pool when the deployment is disaggregated." />
      {ek.length ? (
        <div className="xf-card">
          <div className="xf-card-h"><h3>Engine flags</h3><span className="xf-sub">{barKey(x)} Click a bar to cross-filter.</span></div>
          {groups.map((g) => {
            const ks = ek.filter((k) => poolOf(k) === g.pool)
            if (!ks.length) return null
            return (
              <div className="xf-group" key={g.pool ?? 'shared'}>
                {g.label && <p className="eyebrow">{g.label}</p>}
                <div className="xf-grid">{ks.map((k) => <FieldChart key={k} x={x} k={k} title={bareKey(k)} help={ENGINE_HELP[bareKey(k)]} />)}</div>
              </div>
            )
          })}
        </div>
      ) : <p className="xf-empty">No engine flag varies in this report.</p>}
    </section>
  )
}
