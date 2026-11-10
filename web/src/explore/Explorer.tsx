// The report page: header, top configs, then a rail beside the overview, Topology, Routing,
// vLLM engine and Runs sections. One crossfilter drives all of it.
import { useEffect, useState } from 'react'
import type { Catalog, Scenario } from '../types'
import type { ExploreData } from './types'
import { useExplorer, type Explorer as X } from './useExplorer'
import { Controls, TopConfigs } from './TopConfigs'
import { OverviewSection } from './Overview'
import { EngineSection, RoutingSection, TopologySection } from './Sections'
import { RunsTable } from './RunsTable'
import { OpenSetup, SetupButton, useSetupDialog } from './SetupDialog'
import { CONCURRENCY_NOTE, METRIC, SLO_OF, TOPO_FIELDS, bareKey, layerOf, loadLabel, poolOf, sortValues } from './model'
import { catalogDoc } from '../lib'
import { CatRef } from '../components/Coverage'
import { CaveatBadge } from '../components/Reports'

const REPO = 'https://github.com/inference-sim/blis-config-search'
/** The spec's SLO key in the site's metric vocabulary ("itl_p99" -> "TPOT p99"). */
const sloLabel = (k: string) => {
  const m = (Object.keys(SLO_OF) as (keyof typeof SLO_OF)[]).find((x) => SLO_OF[x] === k)
  return m ? METRIC[m].label : k.replace('_', ' ')
}
const BASE = import.meta.env.BASE_URL.replace(/\/$/, '')

export async function loadExplore(runId: string): Promise<ExploreData> {
  const res = await fetch(`${BASE}/data/reports/${encodeURIComponent(runId)}/explore.json`)
  if (!res.ok) throw new Error(`explore.json for ${runId}: HTTP ${res.status}`)
  return res.json() as Promise<ExploreData>
}

export function ScenarioPage({ runId, scenarios, catalog, load = loadExplore }: {
  runId: string; scenarios: Scenario[]; catalog?: Catalog | null; load?: (id: string) => Promise<ExploreData>
}) {
  const s = scenarios.find((x) => x.run_id === runId)
  const [data, setData] = useState<ExploreData | null>(null)
  const setup = useSetupDialog(data?.setup)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    setData(null); setError(null)
    load(runId).then((d) => live && setData(d)).catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)))
    return () => { live = false }
  }, [runId, load])

  if (!s) return <div><p className="crumb"><a href="#/reports">← all reports</a></p><div className="empty">Unknown scenario <span className="mono">{runId}</span>.</div></div>
  const modelDoc = catalogDoc(catalog, 'model', s.model)
  return (
    <OpenSetup.Provider value={setup.open}>
    <div className="xf">
      {setup.dialog}
      <p className="crumb"><a href="#/reports">← all reports</a></p>
      <h2 className="viewhead"><span>{s.model} on {s.gpu_budget} × {s.hardware}, {s.workload}</span>
        {s.caveat && <CaveatBadge caveat={s.caveat} />}
        <span className="xf-repro"><SetupButton>Reproduce</SetupButton></span></h2>
      <div className="specchips">
        <CatRef href={modelDoc}>{s.model}</CatRef>
        <CatRef href={catalogDoc(catalog, 'hardware', s.hardware)}>{s.gpu_budget} {s.hardware}</CatRef>
        <CatRef href={catalogDoc(catalog, 'workload', s.workload)}>{s.workload}</CatRef>
        {data && Object.entries(data.slo).map(([k, v]) => <span key={k} className="spec floor" title={`From the spec's scenario.slo (${k}); a run is servable only if it meets every SLO.`}>SLO {sloLabel(k)} ≤ {v}</span>)}
        <a className="xf-speclink" href={`${REPO}/blob/main/experiments/${encodeURIComponent(runId)}.yaml`} target="_blank" rel="noreferrer">experiment spec ↗</a>
      </div>
      {s.caveat && <div className="warnnote">{s.caveat}</div>}
      {error && <div className="viewerror"><h3>Could not load this report’s runs</h3><pre>{error}</pre></div>}
      {!data && !error && <div className="loading">Loading runs…</div>}
      {data && <Body data={data} />}
    </div>
    </OpenSetup.Provider>
  )
}

function Body({ data }: { data: ExploreData }) {
  const x = useExplorer(data)
  return (
    <>
      <Controls x={x} />
      <TopConfigs x={x} />
      <div className="xf-layout">
        <Rail x={x} />
        <div className="xf-main">
          <Sweeps x={x} />
          <OverviewSection x={x} />
          <TopologySection x={x} />
          <RoutingSection x={x} />
          <EngineSection x={x} />
          <RunsTable x={x} />
        </div>
      </div>
    </>
  )
}

function fieldName(k: string): string {
  if (k === 'mode') return 'mode'
  if (k === 'sweep') return 'sweep'
  if (k === 'conc') return 'concurrency'
  if (k === 'router_config') return 'router'
  const t = TOPO_FIELDS.find((f) => f.key === k)
  const base = t ? t.part : bareKey(k)
  return `${poolOf(k) ? poolOf(k) + ' ' : ''}${base}`
}

function Rail({ x }: { x: X }) {
  const count = (layer: string) => x.fields.filter((k) => layerOf(k) === layer).length
  const active = Object.entries(x.sel).flatMap(([k, vs]) => vs.map((v) => [k, v] as const))
  const loads = sortValues([...new Set(x.runs.map((r) => String(r.conc)))])
  const nav: [string, string, string | null, number | null][] = [
    ['sec-overview', 'Overview', null, null], ['sec-topology', 'Topology', 'topology', count('topology')],
    ['sec-routing', 'Routing', 'routing', count('routing')], ['sec-engine', 'vLLM engine', 'engine', count('engine')], ['sec-runs', 'Runs', null, null],
  ]
  return (
    <aside className="xf-rail" aria-label="Sections and filters">
      <nav className="xf-nav" aria-label="Sections">
        {nav.map(([id, label, layer, n]) => (
          <a key={id} href={`#${id}`} onClick={(e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView?.({ behavior: 'smooth' }) }}>
            <i className={layer ? `xf-swatch l-${layer}` : 'xf-swatch'} />{label}{n !== null && <small>{n} field{n === 1 ? '' : 's'}</small>}
          </a>
        ))}
      </nav>
      <div className="xf-rail-h"><p className="eyebrow">Filters</p><button type="button" className="linkbtn" onClick={x.reset}>reset all</button></div>
      <div className="xf-active">
        {x.picked && <button type="button" className="xf-achip pick" onClick={() => x.pick(x.picked as string)}>showing a top config · clear ✕</button>}
        {active.length ? active.map(([k, v]) => (
          <button type="button" key={`${k}=${v}`} className="xf-achip" onClick={() => x.toggleValue(k, v)}>{fieldName(k)} <b>{k === 'conc' ? loadLabel(v) : v}</b> ✕</button>
        )) : <span className="xf-sub">No filters yet. Click bars in any section, or a top config.</span>}
      </div>
      <span className="xf-sub">{x.shown().length.toLocaleString('en-US')} of {x.runs.length.toLocaleString('en-US')} runs shown</span>
      <label className="xf-toggle"><input type="checkbox" checked={x.okOnly} onChange={(e) => x.setOkOnly(e.target.checked)} /> only runs that meet every SLO</label>
      <div className="xf-facet">
        <span className="xf-facet-l" title={CONCURRENCY_NOTE}>concurrency</span>
        <div className="xf-opts">{loads.map((v) => (
          <button type="button" key={v} className="xf-chip" aria-pressed={(x.sel.conc ?? []).includes(v)} onClick={() => x.toggleValue('conc', v)}>{loadLabel(v)}</button>
        ))}</div>
      </div>
      <p className="xf-sub xf-rail-note">Ranked on {METRIC[x.target].label}. Topology, routing and engine filters live in their sections.</p>
    </aside>
  )
}

function Sweeps({ x }: { x: X }) {
  const sel = x.sel.sweep ?? []
  const pool = x.shown('sweep')
  const picked = x.data.sweeps.filter((s) => sel.includes(s.name))
  const [spec, setSpec] = useState(false)
  return (
    <div className="xf-card xf-sweeps">
      <div className="xf-card-h"><h3>Sweeps</h3><span className="xf-sub">All {x.data.sweeps.length} sweeps feed every section. Pick sweeps to scope the page and read what each one set out to test.</span></div>
      <div className="xf-opts">
        {x.data.sweeps.map((s) => (
          <button type="button" key={s.name} className="xf-chip" aria-pressed={sel.includes(s.name)} onClick={() => x.toggleValue('sweep', s.name)}>
            {s.name}<small>{pool.filter((r) => r.sweep === s.name).length}</small>
          </button>
        ))}
        {x.data.spec_yaml && <button type="button" className="linkbtn" onClick={() => setSpec((v) => !v)}>{spec ? 'hide spec' : 'show the spec this ran'}</button>}
      </div>
      {picked.map((s) => {
        const skipped = s.dropped.reduce((a, d) => a + d.count, 0)
        return <p key={s.name} className="xf-hyp"><b className="mono">{s.name}</b> {s.narrative}{skipped ? <span className="xf-sub"> · {skipped} combinations skipped (accelerator count off budget)</span> : null}</p>
      })}
      {spec && <pre className="xf-yaml">{x.data.spec_yaml}</pre>}
    </div>
  )
}
