import { useState } from 'react'
import type { Catalog, Mode } from '../types'
import { Dropdown } from './Dropdown'
import { fnum } from '../lib'

const REPO = 'https://github.com/inference-sim/blis-config-search'

export interface Draft {
  model: string
  hardware: string
  workload: string
  gpu_budget: number
  floor: number
}
const HEADROOM = 0.8

export function buildYaml(d: Draft): string {
  return [
    `model: ${d.model}`,
    `hardware: ${d.hardware}`,
    `gpu_budget: ${d.gpu_budget}`,
    `workload:`,
    `  preset: ${d.workload}`,
    `objective: dual-pareto`,
    // interactivity_floor is a mapping everywhere it is read (run_search, report.py, serve_local
    // all call .get on it). A bare `interactivity_floor: 50` crashes the pipeline with
    // AttributeError, so emit the nested shape.
    `interactivity_floor:`,
    `  tok_per_s_per_user: ${d.floor}   # min tok/s/user`,
    `headroom: ${HEADROOM}`,
    '',
  ].join('\n')
}
export function prLink(d: Draft, yaml: string): string {
  // CI (generate-report.yml) only accepts submissions/*.yaml; a scenarios/ path is rejected.
  const q = new URLSearchParams({ filename: `submissions/${d.model}-${d.hardware}-${d.workload}.yaml`, value: yaml })
  return `${REPO}/new/main?${q.toString()}`
}

/**
 * A number field you can actually edit: the text being typed lives in local state, so clearing
 * the box (or passing through "" / "1." on the way to another value) leaves it cleared instead of
 * snapping back to a fallback. `onCommit` fires only for values that parse to a positive number,
 * so the draft — and the YAML below — hold the last good value while the box is mid-edit. On blur
 * an empty or invalid box is restored to that last good value.
 */
function NumField({ value, integer, onCommit, ariaLabel }: { value: number; integer?: boolean; onCommit: (n: number) => void; ariaLabel: string }) {
  const [text, setText] = useState(String(value))
  // Track the committed value so an outside change (not our own typing) refreshes the box.
  const [seen, setSeen] = useState(value)
  if (seen !== value) {
    setSeen(value)
    setText(String(value))
  }
  return (
    <input
      className="numin"
      type="number"
      min={1}
      step={1}
      aria-label={ariaLabel}
      value={text}
      onChange={(e) => {
        setText(e.target.value)
        const n = Number(e.target.value)
        if (e.target.value.trim() !== '' && Number.isFinite(n) && n > 0) {
          const v = integer ? Math.round(n) : n
          setSeen(v)
          onCommit(v)
        }
      }}
      onBlur={() => setText(String(value))}
    />
  )
}

export function Submit({ catalog, mode }: { catalog: Catalog; mode: Mode }) {
  const [d, setD] = useState<Draft>({
    model: catalog.models[0]?.name ?? '',
    hardware: catalog.hardware.find((h) => h.name === 'h100')?.name ?? catalog.hardware[0]?.name ?? '',
    workload: catalog.workloads[0]?.name ?? '',
    gpu_budget: 16,
    floor: 50,
  })
  const [copied, setCopied] = useState(false)
  const [log, setLog] = useState('')
  const [running, setRunning] = useState(false)
  const yaml = buildYaml(d)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(yaml)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }
  const run = async () => {
    setRunning(true)
    setLog('')
    try {
      // serve_local.validate_scenario reads interactivity_floor.tok_per_s_per_user (nested) but
      // workload as a flat string. The flat Draft sent `floor`, which the server never read, so
      // every local run fell back to the default floor (posting 120 came back as 50). Send the
      // nested floor; keep workload flat to match the server's reader.
      const body = {
        model: d.model, hardware: d.hardware, gpu_budget: d.gpu_budget,
        workload: d.workload,
        interactivity_floor: { tok_per_s_per_user: d.floor },
      }
      const res = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      if (!res.ok || !res.body) {
        setLog(`Request failed: HTTP ${res.status}. Is the local server running (serve_local)?`)
      } else {
        const reader = res.body.getReader()
        const dec = new TextDecoder()
        for (;;) {
          const { value, done } = await reader.read()
          if (done) break
          setLog((l) => l + dec.decode(value, { stream: true }))
        }
      }
    } catch (e) {
      setLog(`Could not reach /api/search: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setRunning(false)
    }
  }
  const opts = (xs: { name: string }[]) => xs.map((x) => ({ value: x.name, label: x.name }))

  return (
    <div>
      <h2 className="viewhead">Submit a scenario</h2>
      <p className="dek">
        Build a scenario and {mode === 'local' ? 'run it on this machine, or ' : ''}open a pull request with the generated <span className="mono">scenario.yaml</span>.{' '}
        {mode === 'pages' && <>This hosted page <b>does not run simulations</b>; CI runs the search after your PR is opened.</>}
      </p>
      <div className="formgrid">
        <label className="field"><span>model</span>
          <Dropdown ariaLabel="Model" value={d.model} options={opts(catalog.models)} onChange={(v) => setD({ ...d, model: v })} />
        </label>
        <label className="field"><span>hardware</span>
          <Dropdown ariaLabel="Hardware" value={d.hardware} options={opts(catalog.hardware)} onChange={(v) => setD({ ...d, hardware: v })} />
        </label>
        <label className="field"><span>workload</span>
          <Dropdown ariaLabel="Workload" value={d.workload} options={opts(catalog.workloads)} onChange={(v) => setD({ ...d, workload: v })} />
        </label>
        <label className="field"><span>gpu_budget</span>
          <NumField ariaLabel="GPU budget" value={d.gpu_budget} integer onCommit={(n) => setD({ ...d, gpu_budget: n })} />
        </label>
        <label className="field"><span>floor (tok/s/user)</span>
          <NumField ariaLabel="Interactivity floor" value={d.floor} onCommit={(n) => setD({ ...d, floor: n })} />
        </label>
      </div>

      <div className="tbl-title">scenario.yaml</div>
      <pre className="yaml" aria-label="Generated scenario.yaml">{yaml}</pre>
      <div className="actions">
        <button type="button" className="btn" onClick={copy}>{copied ? 'Copied' : 'Copy YAML'}</button>
        <a className="btn" href={prLink(d, yaml)} target="_blank" rel="noreferrer">Open prefilled GitHub PR ↗</a>
        {mode === 'local' && (
          <button type="button" className="btn primary" disabled={running} onClick={run}>{running ? 'Running…' : 'Run on your machine'}</button>
        )}
      </div>
      <p className="pts-note">Budget <span className="num">{fnum(d.gpu_budget, 0)}</span> GPUs, floor <span className="num">{fnum(d.floor, 0)}</span> tok/s/user, headroom <span className="num">{HEADROOM}</span>.</p>

      {mode === 'local' && log !== '' && <pre className="log" aria-label="Run output">{log}</pre>}

      <div className="tbl-title steps">How to submit</div>
      <ol className="steps">
        <li>Choose the model, hardware, workload, GPU budget and interactivity floor above.</li>
        <li>Click <b>Open prefilled GitHub PR</b> (or copy the YAML into a new file under <span className="mono">scenarios/</span>).</li>
        <li>Commit the file to a new branch and open a pull request.</li>
        <li>CI runs the BLIS search and publishes the report; the scenario then appears under Reports.</li>
      </ol>
    </div>
  )
}
