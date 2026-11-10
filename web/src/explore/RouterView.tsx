// One router family: its plugin chain as a diagram, as numbered steps, or as the YAML itself.
import { useState } from 'react'
import type { Explorer } from './useExplorer'
import type { Plugin, RouterChain } from './types'
import { METRIC, fmtMetric, loadLabel, overrideParts, rankConfigs, routerLabel } from './model'

const GLOSS: Record<string, string> = {
  'random-picker': 'picks any remaining instance at random',
  'max-score-picker': 'picks the instance with the highest combined score',
  'weighted-random-picker': 'picks at random, weighted by score',
  'queue-scorer': 'prefers instances with fewer waiting requests',
  'kv-cache-utilization-scorer': 'prefers instances with more free KV cache',
  'load-aware-scorer': 'prefers less-loaded instances',
  'token-load-scorer': 'prefers instances with fewer tokens in flight',
  'prefix-cache-scorer': 'prefers instances that already hold the prompt prefix',
  'prefix-cache-affinity-filter': 'keeps instances with a strong prefix match unless sticking costs too much TTFT',
  'session-affinity-filter': 'keeps a session on the instance that served it before',
  'prefill-filter': 'keeps only prefill instances',
  'decode-filter': 'keeps only decode instances',
  'approx-prefix-cache-producer': 'estimates which instance holds each prefix, from routing history',
  'precise-prefix-cache-producer': 'tracks which instance holds each prefix, from engine KV events',
  'inflight-load-producer': 'counts requests and tokens in flight per instance',
  'always-disagg-pd-decider': 'sends every request through a prefill instance first',
  'prefix-based-pd-decider': 'sends a request to prefill only when enough of its prompt is uncached',
  'concurrency-detector': 'marks an instance saturated past an in-flight limit',
  'utilization-detector': 'marks an instance saturated when a queue builds or KV fills',
  'max-saturation-detector': 'saturated if any of its detectors says so',
  'disagg-profile-handler': 'runs the prefill profile, then the decode profile',
  'metrics-data-source': 'scrapes each engine’s metrics endpoint',
  'core-metrics-extractor': 'reads queue depth and KV cache use from those metrics',
  'endpoint-notification-source': 'listens for the engines’ KV cache events',
  'token-producer': 'tokenizes the prompt so prefixes can be matched',
}
export const gloss = (t: string) => GLOSS[t] ?? t.replace(/-/g, ' ')
const isScalar = (v: unknown) => v === null || typeof v !== 'object'
/** The scalar parameters, for the one-line "in plain words" view. */
const scalarParams = (p: Plugin) => Object.entries(p.params).filter(([, v]) => isScalar(v)).map(([k, v]) => [k, String(v)] as const)

/** A parameter value of any shape: scalars inline, dicts and lists as a wrapping nested list. */
function ParamValue({ v }: { v: unknown }) {
  if (isScalar(v)) return <span className="xf-pv">{String(v)}</span>
  const entries: [string, unknown][] = Array.isArray(v) ? v.map((x, i) => [`[${i}]`, x]) : Object.entries(v as Record<string, unknown>)
  return (
    <dl className="xf-pdict">
      {entries.map(([k, x]) => <div key={k}><dt>{k}</dt><dd><ParamValue v={x} /></dd></div>)}
    </dl>
  )
}
function Params({ p, hot }: { p: Plugin; hot: string[] }) {
  const all = Object.entries(p.params)
  const [open, setOpen] = useState(false)
  if (!all.length) return null
  const flat = all.filter(([, v]) => isScalar(v)), nested = all.filter(([, v]) => !isScalar(v))
  return (
    <span className="xf-pl-p">
      {flat.map(([k, v], i) => <span key={k}>{i > 0 && ' · '}{hot.includes(k) ? <mark>{k} {String(v)}</mark> : `${k} ${v}`}</span>)}
      {nested.length > 0 && (
        <>
          <button type="button" className="linkbtn xf-pl-more" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            {open ? 'hide' : `${nested.map(([k]) => k).join(', ')} ▸`}
          </button>
          {open && nested.map(([k, v]) => <span key={k} className="xf-pl-nested"><b>{k}</b><ParamValue v={v} /></span>)}
        </>
      )}
    </span>
  )
}

type View = 'diagram' | 'steps' | 'yaml'

/** The router section as list and detail: every config on the left, the one you open on the right. */
export function RouterPanel({ x }: { x: Explorer }) {
  const routers = x.data.routers
  const ids = [...new Set(x.runs.map((r) => (r.knobs.router_config == null ? 'default' : String(r.knobs.router_config))))]
  const label = (id: string) => routerLabel(id === 'default' ? null : id, routers)
  ids.sort((a, b) => (a === 'default' ? -1 : b === 'default' ? 1 : label(a).localeCompare(label(b), undefined, { numeric: true })))
  const sel = x.sel.router_config ?? []
  // the selected configs open side by side; with none selected, the first one previews
  const chosen = ids.filter((id) => sel.includes(label(id)))
  const [view, setView] = useState<View>('diagram')
  const shownIds = chosen.length ? chosen : ids.slice(0, 1)
  const m = x.target
  const sub = x.shown('router_config')
  const rows = ids.map((id) => {
    const rs = sub.filter((r) => routerLabel(r.knobs.router_config, routers) === label(id))
    const b = rankConfigs(rs, m)[0]
    return { id, n: rs.length, ok: rs.filter((r) => r.ok).length, best: b ? (b[m] as number) : null }
  })
  const vals = rows.map((r) => r.best).filter((v): v is number => v !== null && v > 0)
  const hi = Math.max(...vals, 0), lo = Math.min(...vals, Infinity)
  const size = (v: number | null) => (v === null || !(v > 0) ? 0 : Math.max(3, (METRIC[m].higher ? v / hi : lo / v) * 100))
  const live = x.fields.includes('router_config')
  const click = (id: string) => { if (live) x.toggleValue('router_config', label(id)) }
  return (
    <div className="xf-rt">
      <div className="xf-rt-list" role="list" aria-label="Router configs">
        <p className="xf-sub">{live ? `Best ${METRIC[m].label} per config. Click configs to compare them; each one you pick filters the page and opens beside the list.` : 'Every run in this report used one router.'}</p>
        {rows.map((row, i) => {
          const on = sel.includes(label(row.id))
          const fam = routers[row.id]?.family ?? 'default'
          const arm = routers[row.id]?.override ? overrideParts(routers[row.id].override).map((o) => `${o.param}=${o.value}`).join(', ') : ''
          // generated arms of one family sit under its name, each showing only what differs
          const head = arm && (i === 0 || (routers[rows[i - 1].id]?.family ?? 'default') !== fam)
          return [
            head ? <p key={`h-${fam}`} className="xf-rt-fam mono" role="presentation">{fam}</p> : null,
            <button type="button" role="listitem" key={row.id} className={`xf-rt-row${arm ? ' arm' : ''}${shownIds.includes(row.id) ? ' open' : ''}${on ? ' on' : ''}${sel.length && !on ? ' dim' : ''}`}
              aria-pressed={on} aria-label={`${label(row.id)}: ${row.n} runs, ${row.ok} meet every SLO${row.best !== null ? `, best ${fmtMetric(m, row.best)} ${METRIC[m].label}` : ''}`}
              onClick={() => click(row.id)}>
              <span className="xf-rt-name mono" title={label(row.id)}>{arm || fam}</span>
              <span className="xf-track"><i style={{ width: `${size(row.best)}%` }} /></span>
              <span className="xf-n">{row.best !== null ? fmtMetric(m, row.best) : '–'}</span>
            </button>,
          ]
        })}
        {sel.length > 0 && <button type="button" className="linkbtn" onClick={() => x.clearField('router_config')}>clear {sel.length > 1 ? `all ${sel.length}` : 'router filter'}</button>}
      </div>
      <div className="xf-rt-details">
        <div className="xf-rt-bar">
          <span className="xf-sub">{chosen.length > 1 ? `Comparing ${chosen.length} router configs` : chosen.length ? 'Selected router config' : 'Previewing the first config; pick one or more on the left to compare them'}</span>
          {shownIds.some((id) => routers[id]) && (
            <div className="xf-seg" role="group" aria-label="Show each config as">
              {(['diagram', 'steps', 'yaml'] as View[]).map((v) => (
                <button type="button" key={v} aria-pressed={view === v} onClick={() => setView(v)}>{v === 'steps' ? 'in plain words' : v}</button>
              ))}
            </div>
          )}
        </div>
        {shownIds.map((id, i) => <RouterDetail key={id} x={x} id={id} view={view} index={chosen.length > 1 ? i + 1 : null}
          onClose={chosen.length ? () => x.toggleValue('router_config', label(id)) : undefined} />)}
      </div>
    </div>
  )
}

function RouterDetail({ x, id, view, index, onClose }: { x: Explorer; id: string; view: View; index: number | null; onClose?: () => void }) {
  const routers = x.data.routers
  const r = routers[id]
  const lbl = routerLabel(id === 'default' ? null : id, routers)
  const rs = x.shown().filter((run) => routerLabel(run.knobs.router_config, routers) === lbl)
  const best = rankConfigs(rs, x.target)[0]
  // same serving mode as `best`, so the delta measures the router and not aggregated vs P/D
  const defBest = best && id !== 'default' && rankConfigs(x.shown('router_config').filter((run) => run.knobs.router_config == null && run.mode === best.mode), x.target)[0]
  const m = x.target
  const delta = best && defBest ? (best[m] as number) / (defBest[m] as number) - 1 : null
  const good = delta !== null && (METRIC[m].higher ? delta >= 0 : delta <= 0)
  const hot = r?.override ? overrideParts(r.override).map((o) => o.param) : []
  return (
    <article className="xf-rt-detail" aria-label={`Router config ${lbl}`}>
      <div className="xf-router-h">
        {index !== null && <b className="xf-rt-idx">{index}</b>}
        <span className="xf-router-n">{lbl}</span>
        {onClose && <button type="button" className="xf-rt-close" aria-label={`Remove ${lbl} from the comparison`} onClick={onClose}>✕</button>}
      </div>
      <p className="xf-router-sum">{r ? r.summary : 'No EndpointPickerConfig was recorded for these runs.'}</p>
      {r?.override && <p className="xf-sub">Generated from <span className="mono">{r.family}</span> with <mark className="xf-hot">{r.override}</mark>; the highlighted parameter is all that differs from its siblings.</p>}
      <div className="xf-router-stats">
        {best ? <span>best <b>{fmtMetric(m, best[m])}</b> {METRIC[m].label} · {best.topo} · {loadLabel(best.conc)}</span> : <span>no run in this filter meets every SLO</span>}
        {delta !== null && <span>vs the default router, same serving mode: <b className={good ? 'good' : 'bad'}>{delta >= 0 ? '+' : '−'}{Math.abs(delta * 100).toFixed(0)}%</b></span>}
      </div>
      {r?.chain ? (view === 'yaml' ? <pre className="xf-yaml">{r.yaml}</pre> : view === 'steps' ? <Steps c={r.chain} hot={hot} /> : <Diagram c={r.chain} hot={hot} />)
        : r && <p className="xf-sub">This routing file is no longer in the repo, so its plugin chain cannot be drawn.</p>}
      {r?.chain && <p className="xf-router-path mono">{r.path} · {r.chain.plugins.length} plugins{r.chain.flow_control ? ' · flow control on' : ''}</p>}
    </article>
  )
}

function useChain(c: RouterChain) {
  const by = Object.fromEntries(c.plugins.map((p) => [p.name, p]))
  const profiles = c.profiles.map((pr) => {
    const steps = pr.steps.map((s) => ({ p: by[s.ref], w: s.weight })).filter((s) => s.p)
    return { name: pr.name, filters: steps.filter((s) => s.p.category === 'filter'), scorers: steps.filter((s) => s.p.category === 'scorer'), pickers: steps.filter((s) => s.p.category === 'picker') }
  })
  return {
    profiles, deciders: c.plugins.filter((p) => p.category === 'decider'),
    gate: c.gate.map((n) => by[n]).filter(Boolean), policies: c.plugins.filter((p) => p.category === 'policy'),
    data: c.plugins.filter((p) => p.category === 'data'),
  }
}

function PluginLine({ p, w, hot }: { p: Plugin; w?: number | null; hot: string[] }) {
  return (
    <span className="xf-pl">
      {w != null ? <span className="xf-pl-w"><i style={{ width: Math.min(40, w * 12) }} />{p.name} ×{w}</span> : p.name}
      <span className="xf-pl-g">{gloss(p.type)}</span>
      <Params p={p} hot={hot} />
    </span>
  )
}

function Diagram({ c, hot }: { c: RouterChain; hot: string[] }) {
  const ch = useChain(c)
  const Stage = ({ label, gate, children }: { label: string; gate?: boolean; children: React.ReactNode }) => (
    <div className={`xf-stage${gate ? ' gate' : ''}`}><span className="xf-stage-l">{label}</span>{children}</div>
  )
  const Arrow = () => <span className="xf-arrow" aria-hidden="true">→</span>
  return (
    <>
      <div className="xf-flow">
        <span className="xf-end">request</span><Arrow />
        {c.flow_control && <><Stage label="admission · flow control" gate>{ch.gate.map((g) => <PluginLine key={g.name} p={g} hot={hot} />)}{ch.policies.map((p) => <span key={p.name} className="xf-pl def">{p.name}</span>)}</Stage><Arrow /></>}
        {ch.deciders.length > 0 && <><Stage label="prefill hop?">{ch.deciders.map((d) => <PluginLine key={d.name} p={d} hot={hot} />)}</Stage><Arrow /></>}
        <div className="xf-profiles">
          {ch.profiles.map((pr) => (
            <div className="xf-profile" key={pr.name}>
              {ch.profiles.length > 1 && <span className="xf-sub mono">{pr.name} pool</span>}
              <div className="xf-profile-row">
                {pr.filters.length > 0 && <><Stage label="filter">{pr.filters.map((s) => <PluginLine key={s.p.name} p={s.p} hot={hot} />)}</Stage><Arrow /></>}
                {pr.scorers.length > 0 && <><Stage label="score">{pr.scorers.map((s) => <PluginLine key={s.p.name} p={s.p} w={s.w ?? 1} hot={hot} />)}</Stage><Arrow /></>}
                <Stage label="pick">{pr.pickers.length ? pr.pickers.map((s) => <PluginLine key={s.p.name} p={s.p} hot={hot} />) : <span className="xf-pl def">max-score (default)</span>}</Stage>
              </div>
            </div>
          ))}
        </div>
        <Arrow /><span className="xf-end">instance</span>
      </div>
      {ch.data.length > 0 && (
        <div className="xf-feeds"><span className="xf-sub">fed by</span> {ch.data.map((d) => <span key={d.name} className="xf-src"><b>{d.name}</b> {gloss(d.type)}{c.feeds[d.name]?.length ? ` → ${c.feeds[d.name].join(', ')}` : ''}<Params p={d} hot={hot} /></span>)}</div>
      )}
    </>
  )
}

function Steps({ c, hot }: { c: RouterChain; hot: string[] }) {
  const ch = useChain(c)
  const params = (p: Plugin) => { const kp = scalarParams(p); return kp.length ? <span className="xf-k"> ({kp.map(([k, v]) => hot.includes(k) ? `${k} ${v} (swept)` : `${k} ${v}`).join(', ')})</span> : null }
  const items: React.ReactNode[] = []
  if (c.flow_control) items.push(<><b>Hold</b> new requests while an instance is saturated: {ch.gate.slice(1).map((g, i) => <span key={g.name}>{i > 0 && '; '}<b className="mono">{g.name}</b> {gloss(g.type)}{params(g)}</span>)}. Release them {ch.policies.map((p) => gloss(p.type)).join(', ') || 'in order'}.</>)
  if (ch.deciders.length) items.push(<><b>Decide</b> whether to split: {ch.deciders.map((d) => <span key={d.name}>{gloss(d.type)}{params(d)}</span>)}.</>)
  for (const pr of ch.profiles) {
    const tw = pr.scorers.reduce((a, s) => a + (s.w ?? 1), 0)
    items.push(<>{ch.profiles.length > 1 && <>In the <b>{pr.name}</b> pool, </>}
      {pr.filters.length > 0 && <>filter: {pr.filters.map((s) => gloss(s.p.type)).join('; ')}; then </>}
      {pr.scorers.length > 0 && <>score: {pr.scorers.map((s, i) => <span key={s.p.name}>{i > 0 && '; '}{gloss(s.p.type)} <span className="xf-k">({Math.round(((s.w ?? 1) / tw) * 100)}%)</span></span>)}; then </>}
      pick: {pr.pickers.length ? gloss(pr.pickers[0].p.type) : 'the highest score'}.</>)
  }
  if (ch.data.length) items.push(<><b>Inputs</b>: {ch.data.map((d) => `${d.name} ${gloss(d.type)}`).join('; ')}.</>)
  return <ol className="xf-steps">{items.map((it, i) => <li key={i}>{it}</li>)}</ol>
}
