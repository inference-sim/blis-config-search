import { useEffect, useState } from 'react'
import type { Catalog, Mode, Scenario } from './types'
import { loadCatalog, loadMode, loadScenarios } from './load'
import { useHashRoute } from './route'
import { Reports } from './components/Reports'
import { ScenarioDetail } from './components/ScenarioDetail'
import { Coverage } from './components/Coverage'
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
    body = <ScenarioDetail runId={route.runId} scenarios={data.scenarios} catalog={data.catalog} />
  } else {
    switch (route.tab) {
      case 'coverage':
        body = <Coverage scenarios={data.scenarios} catalog={data.catalog} />
        break
      case 'submit':
        body = <Submit catalog={data.catalog} mode={data.mode} />
        break
      default:
        body = <Reports scenarios={data.scenarios} params={route.params} />
    }
  }

  // The scenario detail sits under Reports, so the nav marks that tab for it.
  const activeTab: Tab = route.runId ? 'reports' : route.tab

  return (
    <div className="page">
      <header className="masthead">
        <div className="mastinner">
          <a className="ghlink masthub" aria-label="inference-sim/blis-config-search on GitHub" href="https://github.com/inference-sim/blis-config-search" target="_blank" rel="noreferrer">
            <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.33c-2.22.48-2.69-1.07-2.69-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.71 1.22 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
            </svg>
            <span className="masthub-name">inference-sim/blis-config-search</span>
          </a>
          <div className="masttitle">
            <h1><b>config-search</b> · deployment reports</h1>
            <p className="dek">Pareto-optimal BLIS serving configurations for a model, accelerator budget and workload, each a full interactive report. Simulator predictions; validate before deploying.</p>
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
