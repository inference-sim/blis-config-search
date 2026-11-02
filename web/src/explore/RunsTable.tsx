// Every run in the filter: what it was, what it measured, and how to rerun it.
import { Fragment, useEffect, useState } from 'react'
import type { Explorer } from './useExplorer'
import type { Run } from './types'
import { METRIC, SLO_OF, fmt, sloMs } from './model'
import { ConfigName } from '../components/ConfigName'
import { CopyButton } from './CopyButton'
import { SetupButton } from './SetupDialog'

/** commands.json: each run's reproducible command, and the scenario files those commands read. */
export interface Commands {
  scenarios: Record<string, string>
  /** [command, scenario file, deployment when its scenario could not be regenerated] */
  runs: ([string, string | null, unknown] | null)[]
}
const BASE = import.meta.env.BASE_URL.replace(/\/$/, '')
// One fetch per report, on the first command a reader opens.
const cache = new Map<string, Promise<Commands>>()
export function loadCommands(runId: string): Promise<Commands> {
  if (!cache.has(runId)) {
    cache.set(runId, fetch(`${BASE}/data/reports/${encodeURIComponent(runId)}/commands.json`)
      .then((r) => { if (!r.ok) throw new Error(`commands.json: HTTP ${r.status}`); return r.json() as Promise<Commands> })
      .catch((e) => { cache.delete(runId); throw e }))  // a failed fetch can be retried
  }
  return cache.get(runId) as Promise<Commands>
}

const PAGE = 50
const COLS: { k: keyof Run; label: string; digits: number }[] = [
  { k: 'conc', label: 'sessions', digits: 0 },
  { k: 'tpg', label: METRIC.tpg.short, digits: 1 },
  { k: 'cl', label: METRIC.cl.short, digits: 0 },
  { k: 'pu', label: METRIC.pu.short, digits: 1 },
  { k: 'itl', label: `${METRIC.itl.label} ms`, digits: 1 },
  { k: 'ttft', label: `${METRIC.ttft.label} ms`, digits: 0 },
]

/** The deployment block as indented YAML-like text, for reading rather than parsing. */
function yamlish(v: unknown, indent = ''): string {
  if (v === null || typeof v !== 'object') return String(v)
  return Object.entries(v as Record<string, unknown>)
    .filter(([, x]) => x !== null && x !== undefined)
    .map(([k, x]) => (typeof x === 'object' ? `${indent}${k}:\n${yamlish(x, indent + '  ')}` : `${indent}${k}: ${x}`))
    .join('\n')
}

function CommandRow({ runId, r, span, load }: { runId: string; r: Run; span: number; load: (id: string) => Promise<Commands> }) {
  const [all, setAll] = useState<Commands | undefined>(undefined)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    load(runId).then((c) => live && setAll(c)).catch((e: unknown) => live && setErr(String(e)))
    return () => { live = false }
  }, [runId, load])
  const run = all?.runs[r.id] ?? null
  const scen = run?.[1] ?? null
  const yaml = scen ? all?.scenarios[scen] : undefined
  let body
  if (err) body = <p className="xf-empty">Could not load the command: {err}</p>
  else if (all === undefined) body = <p className="xf-sub">Loading…</p>
  else if (run === null) body = <p className="xf-empty">This run has no recorded command.</p>
  else body = (
    <div className="xf-cmd">
      <div className="xf-cmd-h"><b>0. Install this report's pinned BLIS, catalog and registry once</b><SetupButton>Show setup</SetupButton></div>
      {yaml && (
        <>
          <div className="xf-cmd-h"><b>1. Save the scenario as <span className="mono">scenarios/{scen}</span></b><CopyButton text={yaml} /></div>
          <pre className="xf-cmd-line xf-yaml">{yaml}</pre>
        </>
      )}
      <div className="xf-cmd-h"><b>{yaml ? '2. Run' : 'Command'}</b><CopyButton text={run[0]} /></div>
      <pre className="xf-cmd-line">{run[0]}</pre>
      {!yaml && run[2] != null && (
        <details>
          <summary>the deployment it priced (scenario <span className="mono">{scen}</span>, not regenerable from this report)</summary>
          <pre className="xf-yaml">{yamlish(run[2])}</pre>
        </details>
      )}
    </div>
  )
  return <tr className="xf-cmdrow"><td colSpan={span}>{body}</td></tr>
}

export function RunsTable({ x, load = loadCommands }: { x: Explorer; load?: (id: string) => Promise<Commands> }) {
  const [sort, setSort] = useState<{ k: keyof Run; dir: 1 | -1 } | null>(null)
  const [limit, setLimit] = useState(PAGE)
  const [open, setOpen] = useState<Set<number>>(new Set())
  const m = x.target
  const val = (r: Run, k: keyof Run) => (typeof r[k] === 'number' ? (r[k] as number) : null)
  const list = [...x.shown()].sort(sort
    ? (a, b) => sort.dir * ((val(a, sort.k) ?? -Infinity) - (val(b, sort.k) ?? -Infinity))
    : (a, b) => Number(b.ok) - Number(a.ok) || (METRIC[m].higher ? (b[m] ?? -Infinity) - (a[m] ?? -Infinity) : (a[m] ?? Infinity) - (b[m] ?? Infinity)))
  const miss = (r: Run) => {
    const out: string[] = []
    for (const k of ['itl', 'ttft'] as const) {
      const key = SLO_OF[k] as string, lim = sloMs(x.data.slo[key])
      if (lim !== null && (r[k] ?? 0) > lim) out.push(`${METRIC[k].label} over ${x.data.slo[key]}`)
    }
    return out.join(', ') || 'missed an SLO'
  }
  const click = (k: keyof Run) => setSort((s) => (s?.k === k ? (s.dir === 1 ? { k, dir: -1 } : null) : { k, dir: 1 }))
  const toggle = (id: number) => setOpen((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n })
  const span = COLS.length + 3
  return (
    <section className="xf-sec" id="sec-runs" aria-labelledby="h-runs">
      <div className="xf-sec-h"><h2 id="h-runs">Runs</h2><p>{list.length.toLocaleString('en-US')} runs, best {METRIC[m].label} first. Tags show where a run differs from this report’s usual setting; open a run for the command that reproduces it.</p></div>
      <div className="xf-card"><div className="tscroll hscroll">
        <table className="readout xf-runs">
          <thead><tr className="col">
            <th>config</th>
            {COLS.map((c) => (
              <th key={c.k} className="num" aria-sort={sort?.k === c.k ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}>
                <button type="button" className="sortbtn" onClick={() => click(c.k)}>{c.label}<span className="arrow">{sort?.k === c.k ? (sort.dir === 1 ? '▲' : '▼') : ''}</span></button>
              </th>
            ))}
            <th>SLO</th>
            <th><span className="sr">command</span></th>
          </tr></thead>
          <tbody>
            {list.slice(0, limit).map((r) => {
              const isOpen = open.has(r.id)
              return (
                <Fragment key={r.id}>
                  <tr className={`${r.ok ? '' : 'bad'}${r.cfg === x.picked ? ' sel' : ''}`}>
                    <td className="xf-cfgcol">
                      <button type="button" className="xf-cfglink" onClick={() => x.pick(r.cfg)} title="Filter the page to this config"><ConfigName label={r.topo} variant={x.caption(r)} /></button>
                    </td>
                    {COLS.map((c) => <td key={c.k} className={`num${c.k === m ? ' xf-on' : ''}`}>{fmt(val(r, c.k), c.digits)}</td>)}
                    <td className="xf-status">{r.ok ? 'met' : miss(r)}</td>
                    <td><button type="button" className="xf-cmdbtn" aria-expanded={isOpen} onClick={() => toggle(r.id)}>{isOpen ? 'hide' : 'command'}</button></td>
                  </tr>
                  {isOpen && <CommandRow runId={x.data.run_id} r={r} span={span} load={load} />}
                </Fragment>
              )
            })}
            {list.length === 0 && <tr><td colSpan={span} className="pts-empty">No runs match these filters.</td></tr>}
          </tbody>
        </table>
      </div>
      {list.length > limit && <button type="button" className="linkbtn xf-more" onClick={() => setLimit((l) => l + PAGE * 4)}>show {Math.min(PAGE * 4, list.length - limit)} more of {(list.length - limit).toLocaleString('en-US')}</button>}
      </div>
    </section>
  )
}
