import type { Catalog, Mode } from '../types'

export function Rail({ catalog, mode }: { catalog: Catalog | null; mode: Mode | null }) {
  const versions = catalog ? Object.entries(catalog.versions) : []
  return (
    <aside className="rail">
      <h1>config-search</h1>
      <p className="sub">Simulated BLIS deployment recommendations: which parallelism and instance layout to run for a model, GPU budget and workload.</p>

      <div className="seclabel">Catalog</div>
      {versions.length === 0 && <div className="kv"><span className="k">versions</span><span className="v">{'—'}</span></div>}
      {versions.map(([k, v]) => (
        <div className="kv" key={k}>
          <span className="k">{k}</span>
          <span className="pill">{v}</span>
        </div>
      ))}

      <div className="seclabel">Source</div>
      <a
        className="ghlink"
        href="https://github.com/inference-sim/blis-config-search"
        target="_blank"
        rel="noreferrer"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.33c-2.22.48-2.69-1.07-2.69-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.71 1.22 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
        </svg>
        inference-sim/blis-config-search
      </a>

      <div className="seclabel">Note</div>
      <p className="railnote">
        Numbers are <b>simulator predictions, not measured</b> results. Scenarios badged <span className="mock">mock</span> are synthetic placeholders.
      </p>
      {mode && <div className="kv"><span className="k">mode</span><span className="v">{mode}</span></div>}
    </aside>
  )
}
