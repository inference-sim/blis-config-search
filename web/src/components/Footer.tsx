import { useState } from 'react'
import type { Catalog, Stack } from '../types'
import { SPEC_DOC } from '../lib'

// The provenance/trust strip, moved out of the old sidebar rail (the left column on the
// Reports page is now the mock-A facet-filter rail). The pinned-stack disclosure + the
// simulator-predictions note live here; the GitHub link is in the masthead.

// The stack is ONE dependency, not a flat row of peers: the BLIS simulator binary prices every
// run, and the kernel/catalog/registry are what that binary resolves to. So BLIS is the only
// pill shown at rest; the +N toggle reveals the transitives (see build_catalog._stack).
// Every pinned repo lives under the inference-sim org (llm-d-router is its fork), so link the tag.
const GH = 'https://github.com/inference-sim'
const repoUrl = (name: string, version: string) =>
  /^v?\d/.test(version) ? `${GH}/${name}/tree/${version}` : `${GH}/${name}`

function Pill({ name, version, title }: { name: string; version: string; title: string }) {
  return (
    <a className="pill" href={repoUrl(name, version)} target="_blank" rel="noreferrer" title={title}>
      <span className="pk">{name}</span> {version}
    </a>
  )
}

function StackPills({ stack }: { stack: Stack }) {
  const [open, setOpen] = useState(false)
  // The root lists every version that priced a committed report, pin first; show only the pin.
  const [pinned, ...older] = stack.root.version.split(', ')
  const title = older.length ? `${stack.root.note} (some reports: ${older.join(', ')})` : stack.root.note
  const n = stack.deps.length
  return (
    <span className="footstack">
      <Pill name={stack.root.name} version={pinned} title={title} />
      {open && stack.deps.map((d) => <Pill key={d.name} name={d.name} version={d.version} title={d.note} />)}
      {n > 0 && (
        <button
          type="button"
          className="depcount"
          aria-expanded={open}
          aria-label={open ? 'hide dependencies' : `show ${n} dependencies`}
          onClick={() => setOpen((o) => !o)}
        >
          {open ? 'show less' : `+${n} more`}
        </button>
      )}
    </span>
  )
}

export function Footer({ catalog }: { catalog: Catalog | null }) {
  return (
    <footer className="sitefoot">
      <div className="footinner">
        <p className="footnote">
          <b>Simulator predictions, not measured results.</b> To add a report, write an{' '}
          <a href={SPEC_DOC} target="_blank" rel="noreferrer">experiment sweep spec</a> and open a pull request.
        </p>
        <div className="footmeta">
          {catalog?.stack && <StackPills stack={catalog.stack} />}
        </div>
      </div>
    </footer>
  )
}
