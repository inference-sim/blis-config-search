import { useState, type ReactNode } from 'react'
import type { Catalog, HardwareEntry, Mode, ModelEntry, WorkloadEntry } from '../types'
import { Dropdown } from './Dropdown'
import { SPEC_DOC, fnum } from '../lib'

const REPO = 'https://github.com/inference-sim/blis-config-search'
const SKILL_DOC = `${REPO}/blob/main/.claude/skills/blis-experiment/SKILL.md`
// Per-instance footprints stay within one 8-GPU node, so no shape straddles a node boundary.
const TP_CHOICES = [8, 4, 2, 1]
const ROUTERS = ['random.yaml', 'queue.yaml', 'load-aware.yaml', 'queue-kv.yaml', 'prefix-affinity.yaml']

export type Unit = 'us' | 'ms' | 's'
export interface Target { on: boolean; value: number; unit: Unit }
export type TargetKey = 'itl_p99' | 'ttft_p99' | 'e2e_p99'
export interface Draft {
  /** null = derived from the scenario, so it follows the pickers until someone types one. */
  name: string | null
  question: string
  model: string
  hardware: string
  workload: string
  /** Replay workloads only: the arm of the catalog this workload belongs to. */
  replay: boolean
  moe: boolean
  /** The model's catalog context window; a replay sets max_model_len to it. */
  context: number | null
  gpus: number
  concurrency: number[]
  slo: Record<TargetKey, Target>
  /** Floor on output tok/s per accelerator. */
  throughput: { on: boolean; value: number }
  sweeps: { batch: boolean; routing: boolean }
}

export const TARGETS: { key: TargetKey; label: string; hint: string }[] = [
  { key: 'itl_p99', label: 'Inter-token latency', hint: 'p99 gap between streamed tokens' },
  { key: 'ttft_p99', label: 'Time to first token', hint: 'p99 wait before the first token' },
  { key: 'e2e_p99', label: 'End-to-end latency', hint: 'p99 time for a whole request' },
]

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
export const autoName = (d: Pick<Draft, 'model' | 'hardware' | 'workload' | 'gpus'>) =>
  slug(`${d.model}-${d.hardware}-${d.workload}-${d.gpus}gpu`).slice(0, 64).replace(/-+$/, '')
export const specName = (d: Draft) => d.name ?? autoName(d)
export const NAME_RE = /^[a-z0-9][a-z0-9-]{0,63}$/

const toMs = (t: Target) => t.value * (t.unit === 's' ? 1000 : t.unit === 'us' ? 0.001 : 1)
/** An ITL cap is a per-user streaming rate: 20 ms/token == 50 tok/s/user. */
export const perUserRate = (t: Target) => 1000 / toMs(t)

/** The (tp, instances) pairs that use exactly `gpus`, widest instance first. */
export function shapes(gpus: number): { tp: number; ni: number }[] {
  return TP_CHOICES.filter((tp) => gpus % tp === 0).map((tp) => ({ tp, ni: gpus / tp }))
}

interface Sweep { name: string; narrative: string; shapes: { tp: number; ni: number }[]; engine?: string; router?: string; values: number }

export function plannedSweeps(d: Draft): Sweep[] {
  const all = shapes(d.gpus)
  const ends = all.length > 1 ? [all[0], all[all.length - 1]] : all
  const multi = all.filter((s) => s.ni > 1)
  const out: Sweep[] = [{
    name: 'topology',
    narrative: 'Wider instances should win at the SLO until the extra replicas of narrow ones buy back more than their per-step latency costs.',
    shapes: all, values: 1,
  }]
  if (d.sweeps.batch) out.push({
    name: 'batch-cap-vs-shape',
    narrative: 'The sequence cap should bind hard on one wide instance and barely on many narrow ones.',
    shapes: ends, engine: 'max_num_seqs: [32, 128, 512]', values: 3,
  })
  if (d.sweeps.routing && multi.length > 0) out.push({
    name: 'routing',
    narrative: 'Load-aware routing should beat random most where instances are many and queues form.',
    shapes: multi.length > 1 ? [multi[0], multi[multi.length - 1]] : multi,
    router: `endpoint_picker_config: [${ROUTERS.join(', ')}]`, values: ROUTERS.length,
  })
  return out
}

export const runCount = (d: Draft) =>
  plannedSweeps(d).reduce((n, s) => n + s.shapes.length * s.values, 0) * d.concurrency.length

const list = (xs: (number | string | boolean)[]) => `[${xs.join(', ')}]`
const duration = (t: Target) => `${+t.value.toFixed(3)}${t.unit}`

export function buildYaml(d: Draft): string {
  const slo = [
    ...TARGETS.filter((t) => d.slo[t.key].on).map((t) => `${t.key}: ${duration(d.slo[t.key])}`),
    ...(d.throughput.on ? [`throughput_per_gpu: ${+d.throughput.value.toFixed(3)}`] : []),
  ]
  const lines = [
    `# ${d.question.trim() || 'What question does this experiment answer? (one line, as a # comment)'}`,
    `name: ${specName(d)}`,
    '',
    'scenario:',
    `  model: ${d.model}`,
    `  hardware: ${d.hardware}`,
    `  workload: ${d.workload}`,
    `  gpus: ${d.gpus}`,
    ...(slo.length ? [`  slo: {${slo.join(', ')}}`] : []),
    `  concurrency: ${list(d.concurrency)}`,
    ...(d.replay ? [
      ...(d.context ? [`  engine: {max_model_len: ${d.context}}   # at the 32k default every replay trace is filtered out`] : []),
      '  replay:',
      '    corpus: .corpus/traces.jsonl   # fetched by scripts/fetch_corpus.py',
      '    grace_period: inf              # the 30s default misreads saturated loads as incomplete',
    ] : []),
    '',
    'sweeps:',
  ]
  for (const s of plannedSweeps(d)) {
    const par = [`tp: ${list(s.shapes.map((x) => x.tp))}`, ...(d.moe ? ['enable_expert_parallel: [true]'] : [])]
    lines.push(
      `  - name: ${s.name}`,
      `    narrative: ${s.narrative}`,
      `    num-instances: ${list(s.shapes.map((x) => x.ni))}`,
      `    parallel: {${par.join(', ')}}`,
      ...(s.engine ? [`    engine: {${s.engine}}`] : []),
      ...(s.router ? [`    router: {${s.router}}`] : []),
      '',
    )
  }
  return lines.join('\n')
}

export function prLink(d: Draft, yaml: string): string {
  // GitHub's new-file page forks the repo for anyone without write access, then offers the PR.
  const q = new URLSearchParams({ filename: `experiments/${specName(d)}.yaml`, value: yaml })
  return `${REPO}/new/main?${q.toString()}`
}

/**
 * A number field you can actually edit: the typed text lives in local state, so clearing the box
 * leaves it cleared. `onCommit` fires only for positive numbers; blur restores the last good value.
 */
function NumField({ value, integer, onCommit, ariaLabel }: { value: number; integer?: boolean; onCommit: (n: number) => void; ariaLabel: string }) {
  const [text, setText] = useState(String(value))
  const [seen, setSeen] = useState(value)
  if (seen !== value) {
    setSeen(value)
    setText(String(value))
  }
  return (
    <input
      className="numin"
      type="number"
      min={integer ? 1 : 0}
      step={integer ? 1 : 'any'}
      aria-label={ariaLabel}
      value={text}
      onChange={(e) => {
        setText(e.target.value)
        const n = Number(e.target.value)
        if (e.target.value.trim() !== '' && Number.isFinite(n) && n > 0) {
          const v = integer ? Math.max(1, Math.round(n)) : n
          setSeen(v)
          onCommit(v)
        }
      }}
      onBlur={() => setText(String(value))}
    />
  )
}

/** Comma-separated load ladder; commits a sorted, de-duplicated list of positive integers. */
export function parseLadder(text: string): number[] | null {
  const parts = text.split(/[\s,]+/).filter(Boolean)
  const ns = parts.map(Number)
  if (!parts.length || ns.some((n) => !Number.isInteger(n) || n < 1)) return null
  return [...new Set(ns)].sort((a, b) => a - b)
}

function LadderField({ value, onCommit }: { value: number[]; onCommit: (v: number[]) => void }) {
  const [text, setText] = useState(value.join(', '))
  const bad = parseLadder(text) === null
  return (
    <input
      className={`numin${bad ? ' bad' : ''}`}
      aria-label="Concurrency ladder"
      aria-invalid={bad}
      value={text}
      onChange={(e) => {
        setText(e.target.value)
        const v = parseLadder(e.target.value)
        if (v) onCommit(v)
      }}
      onBlur={() => setText(value.join(', '))}
    />
  )
}

function Section({ title, sub, children }: { title: string; sub: string; children: ReactNode }) {
  return (
    <section className="sb-sec">
      <div className="sb-sechead"><h3>{title}</h3><p>{sub}</p></div>
      <div className="sb-secbody">{children}</div>
    </section>
  )
}

/** One muted line of catalog facts under a picker, e.g. "MoE · 128 experts · 16k context". */
const facts = (xs: (string | number | null | false | undefined)[]) => xs.filter(Boolean).join(' · ')
const modelFacts = (m?: ModelEntry) => !m ? '' : facts([
  m.arch === 'MoE' && m.experts ? `MoE, ${m.experts} experts` : m.arch,
  m.params && `${fnum(m.params.total_b, 0)}B params`,
  m.context && `${fnum(m.context / 1024, 0)}k context`,
  m.quant,
])
const hwFacts = (h?: HardwareEntry) => !h ? '' : facts([
  h.memory_gib && `${fnum(h.memory_gib, 0)} GiB`,
  h.bw_tbs && `${fnum(h.bw_tbs, 2)} TB/s`,
])
const wlFacts = (w?: WorkloadEntry) => !w ? '' : w.kind === 'replay'
  ? facts(['trace replay', w.corpus])
  : facts(['synthetic', w.isl_mean && `~${fnum(w.isl_mean, 0)} tok in`, w.osl_mean && `~${fnum(w.osl_mean, 0)} tok out`])

export function Submit({ catalog, mode }: { catalog: Catalog; mode: Mode }) {
  const isReplay = (w?: WorkloadEntry) => w?.kind === 'replay'
  const modelOf = (name: string) => catalog.models.find((m) => m.name === name)
  const isMoe = (name: string) => modelOf(name)?.arch === 'MoE'
  // A replay sorts ahead of the shapes, so default to the first shape: it needs no corpus.
  const firstWl = catalog.workloads.find((w) => !isReplay(w)) ?? catalog.workloads[0]
  const firstModel = catalog.models[0]?.name ?? ''
  const [d, setD] = useState<Draft>({
    name: null,
    question: '',
    model: firstModel,
    hardware: catalog.hardware.find((h) => h.name === 'h100')?.name ?? catalog.hardware[0]?.name ?? '',
    workload: firstWl?.name ?? '',
    replay: isReplay(firstWl),
    moe: isMoe(firstModel),
    context: modelOf(firstModel)?.context ?? null,
    gpus: 16,
    concurrency: [4, 16, 64, 256],
    slo: {
      itl_p99: { on: true, value: 20, unit: 'ms' },
      ttft_p99: { on: false, value: 2, unit: 's' },
      e2e_p99: { on: false, value: 30, unit: 's' },
    },
    throughput: { on: false, value: 500 },
    sweeps: { batch: true, routing: true },
  })
  const [copied, setCopied] = useState(false)
  const set = (patch: Partial<Draft>) => setD((p) => ({ ...p, ...patch }))
  const setTarget = (k: TargetKey, patch: Partial<Target>) => setD((p) => ({ ...p, slo: { ...p.slo, [k]: { ...p.slo[k], ...patch } } }))

  const yaml = buildYaml(d)
  const name = specName(d)
  const nameOk = NAME_RE.test(name)
  const file = `experiments/${name}.yaml`
  const runs = runCount(d)
  const multi = shapes(d.gpus).some((s) => s.ni > 1)
  const model = catalog.models.find((m) => m.name === d.model)
  const hw = catalog.hardware.find((h) => h.name === d.hardware)
  const wl = catalog.workloads.find((w) => w.name === d.workload)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(yaml)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }
  const opts = (xs: { name: string }[]) => xs.map((x) => ({ value: x.name, label: x.name }))
  const wlOpts = catalog.workloads.map((w) => ({ value: w.name, label: isReplay(w) ? `${w.name} (replay)` : w.name }))

  return (
    <div className="sb">
      <h2 className="viewhead">Design an experiment</h2>
      <p className="dek">An experiment is a sweep spec in <span className="mono">experiments/</span>. Merged specs are run by CI and published as reports.</p>

      <ol className="sb-how">
        <li><b>Fill in the form.</b> The spec preview updates as you go.</li>
        <li><b>Open a pull request.</b> GitHub opens the file prefilled.</li>
        <li><b>Merge it.</b> CI runs it and a bot PR adds the report.</li>
      </ol>

      <div className="sb-layout">
        <div className="sb-form">
          <Section title="Model" sub="The catalog model to serve.">
            <Dropdown ariaLabel="Model" value={d.model} options={opts(catalog.models)} onChange={(v) => set({ model: v, moe: isMoe(v), context: modelOf(v)?.context ?? null })} />
            <p className="sb-note">{modelFacts(model)}</p>
          </Section>

          <Section title="Hardware" sub="Every run uses exactly this many accelerators.">
            <div className="sb-pair">
              <Dropdown ariaLabel="Hardware" value={d.hardware} options={opts(catalog.hardware)} onChange={(v) => set({ hardware: v })} />
              <span className="sb-times">×</span>
              <NumField ariaLabel="Accelerator budget" value={d.gpus} integer onCommit={(n) => set({ gpus: n })} />
            </div>
            <p className="sb-note">{facts([hwFacts(hw), `shapes ${shapes(d.gpus).map((x) => `${x.ni}×TP${x.tp}`).join(', ')}`])}</p>
          </Section>

          <Section title="Workload" sub="The traffic and the loads to measure at.">
            <Dropdown ariaLabel="Workload" value={d.workload} options={wlOpts}
              onChange={(v) => set({ workload: v, replay: isReplay(catalog.workloads.find((w) => w.name === v)) })} />
            <p className="sb-note">{wlFacts(wl)}</p>
            <label className="sb-inline"><span>concurrency</span>
              <LadderField value={d.concurrency} onCommit={(v) => set({ concurrency: v })} />
            </label>
          </Section>

          <Section title="Targets" sub="Servable configs must meet every checked target.">
            {TARGETS.map((t) => {
              const v = d.slo[t.key]
              return (
                <div key={t.key} className={`sb-target${v.on ? '' : ' off'}`}>
                  <label className="sb-check">
                    <input type="checkbox" checked={v.on} aria-label={t.label} onChange={(e) => setTarget(t.key, { on: e.target.checked })} />
                    {t.label} <small>p99</small>
                  </label>
                  <NumField ariaLabel={`${t.label} cap`} value={v.value} onCommit={(n) => setTarget(t.key, { value: n })} />
                  <Dropdown ariaLabel={`${t.label} unit`} value={v.unit} onChange={(u) => setTarget(t.key, { unit: u as Unit })}
                    options={(['us', 'ms', 's'] as Unit[]).map((u) => ({ value: u, label: u }))} />
                </div>
              )
            })}
            <div className={`sb-target${d.throughput.on ? '' : ' off'}`}>
              <label className="sb-check">
                <input type="checkbox" checked={d.throughput.on} aria-label="Throughput" onChange={(e) => set({ throughput: { ...d.throughput, on: e.target.checked } })} />
                Throughput <small>min</small>
              </label>
              <NumField ariaLabel="Throughput floor" value={d.throughput.value} onCommit={(n) => set({ throughput: { ...d.throughput, value: n } })} />
              <span className="sb-unit">tok/s/accel</span>
            </div>
            {d.slo.itl_p99.on && <p className="sb-note">An inter-token cap of {duration(d.slo.itl_p99)} is about <span className="num">{fnum(perUserRate(d.slo.itl_p99), 1)}</span> tok/s per user.</p>}
          </Section>

          <Section title="Sweeps" sub="Starter arms; edit the YAML to add more.">
            <label className="sb-check"><input type="checkbox" checked disabled aria-label="Topology sweep" />topology <small>always on, the baseline</small></label>
            <label className="sb-check"><input type="checkbox" checked={d.sweeps.batch} aria-label="Batch cap sweep" onChange={(e) => set({ sweeps: { ...d.sweeps, batch: e.target.checked } })} />batch cap vs. shape</label>
            <label className="sb-check"><input type="checkbox" checked={d.sweeps.routing && multi} disabled={!multi} aria-label="Routing sweep" onChange={(e) => set({ sweeps: { ...d.sweeps, routing: e.target.checked } })} />routing policies{!multi && <small>needs more than one instance</small>}</label>
          </Section>

          <Section title="Name" sub="Also the report's directory.">
            <input className={`numin${nameOk ? '' : ' bad'}`} aria-label="Spec name" aria-invalid={!nameOk} value={name}
              onChange={(e) => set({ name: e.target.value === '' ? null : e.target.value })} />
            {!nameOk && <p className="sb-note warn">Use lowercase letters, digits and dashes (64 at most).</p>}
            <input className="numin" aria-label="Question" placeholder="The question it answers (optional)" value={d.question} onChange={(e) => set({ question: e.target.value })} />
          </Section>
        </div>

        <aside className="sb-preview">
          <div className="sb-specbar">
            <span className="mono">{file}</span>
            <span className="sb-runs"><span className="num">{fnum(runs, 0)}</span> runs</span>
          </div>
          <pre className="yaml" aria-label="Generated experiment spec">{yaml}</pre>
          <div className="actions">
            <a className={`btn primary${nameOk ? '' : ' off'}`} href={nameOk ? prLink(d, yaml) : undefined} aria-disabled={!nameOk} target="_blank" rel="noreferrer">Open a pull request ↗</a>
            <button type="button" className="btn" onClick={copy}>{copied ? 'Copied' : 'Copy YAML'}</button>
          </div>
          <details className="sb-more">
            <summary>Check it before you submit</summary>
            <p>CI runs these on every PR. Aim for under about 3 hours of serial runtime.</p>
            <pre className="yaml cmd">{`python3 -I scripts/validate_experiment.py ${file}\npython3 -I scripts/run_experiment.py ${file} --dry-run`}</pre>
            {mode === 'local' && <pre className="yaml cmd">{`python3 -I scripts/run_experiment.py ${file}`}</pre>}
            <p>
              Need help with knobs? The <a href={SKILL_DOC} target="_blank" rel="noreferrer"><span className="mono">blis-experiment</span></a> skill drafts a spec from a one-line question; the <a href={SPEC_DOC} target="_blank" rel="noreferrer">spec guide</a> lists every rule.
            </p>
          </details>
        </aside>
      </div>
    </div>
  )
}
