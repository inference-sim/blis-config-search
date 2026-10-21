// Same-origin relative fetches of the staged data. With Vite `base: './'` these resolve
// against the page, so they work at any path on a static host (GitHub Pages) with no server.
import type { Scenario, Catalog, Mode } from './types'

// Vite rewrites import.meta.env.BASE_URL to the configured base ('./'); join cleanly.
const BASE = import.meta.env.BASE_URL.replace(/\/$/, '')
const url = (p: string) => `${BASE}/data/${p}`

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(url(path))
  if (!res.ok) throw new Error(`fetch ${path}: HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export async function loadScenarios(): Promise<Scenario[]> {
  return getJson<Scenario[]>('scenarios.json')
}
export async function loadCatalog(): Promise<Catalog> {
  return getJson<Catalog>('catalog.json')
}
export async function loadMode(): Promise<Mode> {
  try {
    const m = await getJson<{ mode: Mode }>('mode.json')
    return m.mode === 'local' ? 'local' : 'pages'
  } catch {
    return 'pages'
  }
}
// (Per-run records are now inlined into scenarios.json by stage-data.mjs, so no separate
// records fetch is needed — the detail explorer reads scenario.records directly.)

// The self-contained report lives beside its data; link target (new tab).
export const reportHref = (runId: string) =>
  `${BASE}/data/reports/${encodeURIComponent(runId)}/index.html`
