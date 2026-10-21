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
    `interactivity_floor: ${d.floor}   # min tok/s/user`,
    `headroom: ${HEADROOM}`,
    '',
  ].join('\n')
}
export function prLink(d: Draft, yaml: string): string {
  const q = new URLSearchParams({ filename: `scenarios/${d.model}-${d.hardware}-${d.workload}.yaml`, value: yaml })
  return `${REPO}/new/main?${q.toString()}`
}

const num = (v: string, fallback: number) => {
  const n = Number(v)
  return Number.isFinite(n) && n > 0 ? n : fallback
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
      const res = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(d) })
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
          <input className="numin" type="number" min={1} step={1} value={d.gpu_budget} onChange={(e) => setD({ ...d, gpu_budget: Math.round(num(e.target.value, 1)) })} />
        </label>
        <label className="field"><span>floor (tok/s/user)</span>
          <input className="numin" type="number" min={1} step={1} value={d.floor} onChange={(e) => setD({ ...d, floor: num(e.target.value, 1) })} />
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
