import { useEffect, useState } from 'react'
import type { Catalog, Mode, Scenario } from './types'
import { loadCatalog, loadMode, loadScenarios } from './load'
import { useHashRoute } from './route'
import { Reports } from './components/Reports'
import { ScenarioDetail } from './components/ScenarioDetail'
import { Coverage } from './components/Coverage'
import { Models } from './components/Models'
import { Workloads } from './components/Workloads'
import { Gpus } from './components/Gpus'
import { Submit } from './components/Submit'
import { Footer } from './components/Footer'
import { TAB_LABELS, type Tab } from './lib'

interface Data {
  scenarios: Scenario[]
  catalog: Catalog
  mode: Mode
}

export function App() {
  const route = useHashRoute()
  const [data, setData] = useState<Data | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    Promise.all([loadScenarios(), loadCatalog(), loadMode()])
      .then(([scenarios, catalog, mode]) => live && setData({ scenarios, catalog, mode }))
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)))
    return () => {
      live = false
    }
  }, [])

  let body
  if (error) {
    body = (
      <div className="viewerror">
        <h3>Could not load data</h3>
        <pre>{error}</pre>
      </div>
    )
  } else if (!data) {
    body = <div className="loading">Loading…</div>
  } else if (route.runId) {
    body = <ScenarioDetail runId={route.runId} scenarios={data.scenarios} />
  } else {
    switch (route.tab) {
      case 'coverage':
        body = <Coverage scenarios={data.scenarios} />
        break
      case 'models':
        body = <Models catalog={data.catalog} scenarios={data.scenarios} />
        break
      case 'workloads':
        body = <Workloads catalog={data.catalog} scenarios={data.scenarios} />
        break
      case 'gpus':
        body = <Gpus catalog={data.catalog} scenarios={data.scenarios} />
        break
      case 'submit':
        body = <Submit catalog={data.catalog} mode={data.mode} />
        break
      default:
        body = <Reports scenarios={data.scenarios} params={route.params} />
    }
  }

  // Two routes are not themselves primary tabs: the scenario detail (addressed by runId) sits
  // under Reports, and the three catalog spec-sheets are reached from Coverage. Mark their
  // parent tab so the nav never shows nothing selected.
  const CATALOG: Tab[] = ['models', 'workloads', 'gpus']
  const activeTab: Tab = route.runId ? 'reports' : CATALOG.includes(route.tab) ? 'coverage' : route.tab

  return (
    <div className="page">
      <header className="masthead">
        <div className="mastinner">
          <div className="masttitle">
            <h1><b>config-search</b> · deployment reports</h1>
            <p className="dek">Pareto-optimal BLIS serving configurations for a model, GPU budget and workload — each a full interactive report. Simulator predictions; validate before deploying.</p>
          </div>
          <nav className="tabs" aria-label="Sections">
            {TAB_LABELS.map(([t, label]) => (
              <a key={t} href={`#/${t}`} className={`tab${activeTab === t ? ' on' : ''}`} aria-current={activeTab === t ? 'page' : undefined}>
                {label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className="main">{body}</main>
      <Footer catalog={data?.catalog ?? null} mode={data?.mode ?? null} hasMock={!!data?.scenarios.some((s) => s.synthetic)} />
    </div>
  )
}
