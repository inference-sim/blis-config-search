import { useState } from 'react'
import type { Catalog, Mode, Stack } from '../types'

// The provenance/trust strip, moved out of the old sidebar rail (the left column on the
// Reports page is now the mock-A facet-filter rail). The pinned-stack disclosure + the
// simulator-predictions note + the GitHub link live here.

// The stack is ONE dependency, not a flat row of peers: the BLIS simulator binary prices every
// run, and the kernel/catalog/registry are what that binary resolves to. So BLIS is the only
// pill shown at rest, and expanding it reveals the transitives (see build_catalog._stack).
function StackPills({ stack }: { stack: Stack }) {
  const [open, setOpen] = useState(false)
  return (
    <span className="footstack">
      <button
        type="button"
        className="pill stackroot"
        aria-expanded={open}
        title={stack.root.note}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="caret" aria-hidden="true">{open ? '▾' : '▸'}</span>
        <span className="pk">{stack.root.name}</span> {stack.root.version}
        {!open && stack.deps.length > 0 && (
          <span className="depcount">+{stack.deps.length}</span>
        )}
      </button>
      {open && (
        <span className="footdeps">
          {stack.deps.map((d) => (
            <span className="pill dep" key={d.name} title={d.note}>
              <span className="pk">{d.name}</span> {d.version}
            </span>
          ))}
        </span>
      )}
    </span>
  )
}

export function Footer({ catalog, mode, hasMock }: { catalog: Catalog | null; mode: Mode | null; hasMock: boolean }) {
  // Fall back to the old flat pills for a catalog.js generated before `stack` existed.
  const versions = catalog && !catalog.stack ? Object.entries(catalog.versions) : []
  return (
    <footer className="sitefoot">
      <div className="footinner">
        <p className="footnote">
          <b>Simulator predictions, not measured results.</b> Each report is a pure
          {' '}<span className="mono">run-dir → index.html</span> transform; no server runs at view time. Add one by
          running config-search locally and opening a PR — CI regenerates the site after a maintainer merges.
          {hasMock && <> Scenarios badged <span className="mock">mock</span> are synthetic placeholders, not BLIS output.</>}
        </p>
        <div className="footmeta">
          {catalog?.stack && <StackPills stack={catalog.stack} />}
          {versions.length > 0 && (
            <span className="footvers">
              {versions.map(([k, v]) => (
                <span className="pill" key={k} title={k}><span className="pk">{k}</span> {v}</span>
              ))}
            </span>
          )}
          {mode && <span className="footmode mono">mode: {mode}</span>}
          <a className="ghlink" href="https://github.com/inference-sim/blis-config-search" target="_blank" rel="noreferrer">
            <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.33c-2.22.48-2.69-1.07-2.69-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.71 1.22 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
            </svg>
            inference-sim/blis-config-search
          </a>
        </div>
      </div>
    </footer>
  )
}
