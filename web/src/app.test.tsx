import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import type { Catalog, Scenario, WorkloadEntry } from './types'
import { facetCounts, filterScenarios, NO_FILTERS, parseRoute, coverageRows, TAB_LABELS } from './lib'
import { ScenarioPage } from './explore/Explorer'
import { FIXTURE } from './explore/model.test'
import { RunsTable, type Command } from './explore/RunsTable'
import { useExplorer } from './explore/useExplorer'
import type { Scalar } from './explore/types'
import { Reports } from './components/Reports'
import { Submit } from './components/Submit'
import { Coverage } from './components/Coverage'
import { Footer } from './components/Footer'

const w = { label: '1x TP8', tp: 8, ni: 1, gpus: 8, c_use: 1, tps: 1, tps_per_gpu: 1, per_user: 1, itl_p99: 1 }
const mk = (id: string, model: string, hw: string, wl: string, gpu: number): Scenario => ({
  run_id: id, model, hardware: hw, gpu_budget: gpu, workload: wl, caveat: '', floor: 'f', headroom: null,
  n_runs: 1, n_servable: 1, winner: w, finalists: [], reason: '',
})
const S = [mk('a', 'm1', 'H100', 'chatbot', 16), mk('b', 'm1', 'H200', 'chatbot', 16), mk('c', 'm2', 'H100', 'multidoc', 8)]

describe('filters', () => {
  it('filters and gives live facet counts that ignore the facet itself', () => {
    expect(filterScenarios(S, { ...NO_FILTERS, model: 'm1' }).map((s) => s.run_id)).toEqual(['a', 'b'])
    const hw = facetCounts(S, { ...NO_FILTERS, model: 'm2' }, 'hardware')
    expect(hw).toEqual([{ value: 'H100', count: 1 }, { value: 'H200', count: 0 }])
    const gpu = facetCounts(S, NO_FILTERS, 'gpu')
    expect(gpu.map((g) => g.value)).toEqual(['8', '16'])
  })
})

describe('#11 coverage view', () => {
  // Two scenarios share one (model, hardware, workload, budget) tuple; a third differs by budget.
  const C = [
    mk('a', 'm1', 'H100', 'chatbot', 16),
    mk('a2', 'm1', 'H100', 'chatbot', 16),
    mk('c', 'm2', 'H200', 'multidoc', 512),
  ]

  it('enumerates one row per searched combination, never an unsearched one', () => {
    const rows = coverageRows(C)
    expect(rows).toHaveLength(2)
    expect(rows.map((r) => [r.model, r.hardware, r.workload, r.gpu_budget])).toEqual([
      ['m1', 'H100', 'chatbot', 16],
      ['m2', 'H200', 'multidoc', 512],
    ])
    // the row carries its scenarios, so the count is the number of reports behind it
    expect(rows[0].scenarios.map((s) => s.run_id)).toEqual(['a', 'a2'])
    expect(rows[1].scenarios).toHaveLength(1)
    // a combination with no report never appears — there is no row to have an empty count
    expect(rows.every((r) => r.scenarios.length > 0)).toBe(true)
  })

  it('groups one combination written in different cases into a single row', () => {
    // build_gallery normalizes hardware (lower + validate, emit upper) but passes model and
    // workload through from the scenario yaml, so two runs of one combination can disagree on
    // case. They are the same searched combination and must not split into two rows.
    const rows = coverageRows([
      mk('a', 'glm-5.3-flash', 'H100', 'chatbot', 16),
      mk('b', 'GLM-5.3-Flash', 'h100', 'Chatbot', 16),
    ])
    expect(rows).toHaveLength(1)
    expect(rows[0].scenarios.map((s) => s.run_id)).toEqual(['a', 'b'])
  })

  it('links a single-report row straight to its scenario, a multi-report row to the filtered list', () => {
    render(<Coverage scenarios={C} />)
    const hrefs = [...document.querySelectorAll('.readout tbody a')].map((a) => a.getAttribute('href') ?? '')
    // the m2/H200/multidoc/512 row has exactly one report -> deep-link to the report itself
    expect(hrefs).toContain('#/scenario/c')
    // the m1 row has two -> link to Reports pre-filtered on the whole tuple
    const multi = hrefs.find((h) => h.startsWith('#/reports?')) ?? ''
    const q = new URLSearchParams(multi.split('?')[1])
    expect([q.get('model'), q.get('hardware'), q.get('workload'), q.get('gpu')]).toEqual(['m1', 'H100', 'chatbot', '16'])
  })

  it('shows the recommended config inline, so the table answers "what should I run?"', () => {
    render(<Coverage scenarios={C} />)
    // the winner is the payload a reader came for; every row states one
    expect(document.querySelector('.readout tbody')?.textContent).toContain('1×TP8')
    expect(document.querySelectorAll('.readout tbody .covwin')).toHaveLength(2)
  })

  it('names the best config by tok/s/accelerator when a tuple has several reports', () => {
    // two searches of the same tuple: the row must still answer with ONE config, picked on the
    // objective the search itself optimises, not print "2 reports" in the config column too.
    const win = (label: string, tpg: number) => ({ ...w, label, name: label, tps_per_gpu: tpg })
    const rows = [
      { ...mk('old', 'm1', 'H100', 'chatbot', 16), winner: win('4x TP4', 40) },
      { ...mk('new', 'm1', 'H100', 'chatbot', 16), winner: win('8x TP2 + EP', 71) },
    ]
    render(<Coverage scenarios={rows} />)
    const tr = document.querySelector('.readout tbody tr') as Element
    expect(tr.querySelector('.covwin')?.textContent).toBe('8×TP2 + EP')
    expect(tr.textContent).toContain('71')
    // the count still lives in the report column, where it belongs
    expect(tr.querySelector('td:last-child a')?.textContent).toContain('2 reports')
  })

  it('centers tok/s/accelerator without riding on num, which means right-aligned', () => {
    render(<Coverage scenarios={C} />)
    const tbl = document.querySelector('table.readout') as Element
    const head = [...tbl.querySelectorAll('thead th')]
    const i = head.findIndex((t) => (t.textContent ?? '').includes('tok/s/accelerator'))
    const cell = tbl.querySelectorAll('tbody tr')[0].querySelectorAll('td')[i]
    // centered, and NOT carrying the right-align class
    expect(cell.classList.contains('ctr')).toBe(true)
    expect(cell.classList.contains('num')).toBe(false)
    // the header follows its column
    expect(head[i].classList.contains('ctr')).toBe(true)
    // still tabular numerals, so the digits line up column-wise
    expect(cell.querySelector('.num')).toBeTruthy()
  })

  it('aligns every measured column the same way, so they read as a set', () => {
    // One convention for the table: measured quantities are centered. A mix (accelerator budget
    // right-aligned beside a centered tok/s/accelerator) makes two numeric columns of the same kind
    // sit differently. Asserted over all numeric columns, so a new one cannot quietly diverge.
    render(<Coverage scenarios={C} />)
    const tbl = document.querySelector('table.readout') as Element
    const head = [...tbl.querySelectorAll('thead th')].map((t) => (t.textContent ?? '').replace(/[▲▼]/g, '').trim())
    const cells = [...tbl.querySelectorAll('tbody tr')[0].querySelectorAll('td')]
    const measured = ['Accelerator budget', 'tok/s/accelerator']
    for (const label of measured) {
      const i = head.indexOf(label)
      expect(i, `${label} column present`).toBeGreaterThan(-1)
      expect(cells[i].classList.contains('ctr'), `${label} centered`).toBe(true)
      expect(cells[i].classList.contains('num'), `${label} not right-aligned`).toBe(false)
      expect(head[i], `${label} header centered`).toBeTruthy()
      expect(cells[i].querySelector('.num'), `${label} tabular figures`).toBeTruthy()
    }
    // no column in this table is right-aligned any more — the convention is centered
    expect(cells.some((c) => c.classList.contains('num'))).toBe(false)
  })

  it('capitalizes the hardware and budget headers and drops the × from the budget cell', () => {
    render(<Coverage scenarios={C} />)
    const labels = [...document.querySelectorAll('table.readout thead th')]
      .map((t) => (t.textContent ?? '').replace(/[▲▼]/g, '').trim())
    expect(labels).toContain('Hardware')
    expect(labels).toContain('Accelerator budget')
    expect(labels).not.toContain('hardware')
    expect(labels).not.toContain('accelerator budget')
    // the budget reads as a bare count; the Hardware column already names the device
    const cells = [...document.querySelectorAll('table.readout tbody tr')]
      .map((tr) => tr.querySelectorAll('td')[2].textContent ?? '')
    expect(cells).toContain('16')
    expect(cells.every((c) => !c.includes('×'))).toBe(true)
  })

  it('reads the intro as one clean sentence flow, with no doubled or missing spaces', () => {
    render(<Coverage scenarios={C} />)
    const text = document.querySelector('.dek')?.textContent ?? ''
    // the exact flow a reader sees: JSX whitespace juggling must not double or drop a space
    expect(text).toBe('The best config for each searched combination. Click a model, hardware or workload for its spec sheet.')
    expect(text).not.toMatch(/\s{2,}/)
    expect(text).not.toMatch(/\s+[.,]/)
  })

  it('lets the intro use the full width instead of balancing it into early breaks', () => {
    render(<Coverage scenarios={C} />)
    expect(document.querySelector('.dek')?.classList.contains('balance')).toBe(false)
  })

  it('says so when nothing has been searched yet', () => {
    render(<Coverage scenarios={[]} />)
    expect(screen.getByText(/no scenarios have been searched/i)).toBeTruthy()
    expect(document.querySelector('.readout')).toBeNull()
  })

  it('points to blis-catalog for spec sheets instead of hosting them', () => {
    expect(TAB_LABELS.map(([t]) => t)).toEqual(['reports', 'coverage', 'submit'])
    expect(parseRoute('#/models').tab).toBe('reports')
    const catalog = {
      models: [{ name: 'm1' }], hardware: [{ name: 'h100' }], fabrics: [], storage: [], registry: [],
      workloads: [{ name: 'agentx-mvp', kind: 'replay', scenario: 'inferencex-agentx-mvp' }], versions: { 'blis-catalog': '0.5.0' },
    } as unknown as Catalog
    const rows = [mk('a', 'm1', 'H100', 'chatbot', 16), mk('b', 'm9', 'H100', 'inferencex-agentx-mvp', 8)]
    render(<Coverage scenarios={rows} catalog={catalog} />)
    const [a, b] = [...document.querySelectorAll('.readout tbody tr')].map((tr) => [...tr.querySelectorAll('a.catref')].map((x) => x.getAttribute('href')))
    const base = 'https://inference-sim.github.io/blis-catalog/0.5/reference'
    expect(a).toEqual([`${base}/models/m1/`, `${base}/hardware/`, `${base}/workloads/#shapes`])
    // a model the catalog does not list gets no dead link; a replay jumps to the replay section
    expect(b).toEqual([`${base}/hardware/`, `${base}/workloads/#replay-workloads`])
    const btns = [...document.querySelectorAll('.catbtns a')]
    expect(btns.map((x) => [x.textContent, x.getAttribute('href')])).toEqual([['Models', `${base}/models/`], ['Chips', `${base}/hardware/`], ['Workloads', `${base}/workloads/`]])
  })

  it('links the scenario page to its model, hardware and workload spec sheets', () => {
    const catalog = { models: [{ name: 'm1' }], hardware: [{ name: 'h100' }], workloads: [], fabrics: [], storage: [], registry: [], versions: { 'blis-catalog': '0.5.0' } } as unknown as Catalog
    render(<ScenarioPage runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16)]} catalog={catalog} load={() => new Promise(() => {})} />)
    const base = 'https://inference-sim.github.io/blis-catalog/0.5/reference'
    expect(document.querySelector('.viewhead .specsheet')?.getAttribute('href')).toBe(`${base}/models/m1/`)
    expect([...document.querySelectorAll('.specchips a.catref')].map((x) => x.getAttribute('href'))).toEqual([`${base}/hardware/`, `${base}/workloads/#shapes`])
  })

  it('joins the replica count and captions the config with its variant', () => {
    const s = mk('v', 'm1', 'H100', 'chatbot', 16)
    render(<Coverage scenarios={[{ ...s, winner: { ...s.winner, label: '8x TP4 + EP', variant: 'max seqs 256' } }]} />)
    expect(document.querySelector('.covwin')?.textContent).toBe('8×TP4 + EP')
    expect(document.querySelector('.covvar')?.textContent).toBe('max seqs 256')
  })
})

describe('report explorer page', () => {
  const page = () => render(<ScenarioPage runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16)]} load={() => Promise.resolve(FIXTURE)} />)
  const cards = () => [...document.querySelectorAll('.xf-cfg')]

  it('loads the report and leads with the top configs, ties folded into one card', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    expect(cards()[0].querySelector('.win')?.textContent).toBe('8×TP4xDP2')
    expect(cards()[0].textContent).toMatch(/same result with 1 other config/)
    expect(document.querySelector('.xf-speclink')?.getAttribute('href')).toMatch(/experiments\/a\.yaml$/)
  })

  it('shows every metric on a card, the ranked one large and the rest in equal columns', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const card = cards()[0]
    expect(card.querySelector('.xf-hero-l')?.textContent).toBe('tok/s per accelerator')
    expect([...card.querySelectorAll('.xf-kpi-l')].map((e) => e.textContent)).toEqual(['tok/s fleet', 'tok/s/user', 'TPOT p99 ms', 'TTFT p99 ms'])
    fireEvent.click(screen.getByRole('button', { name: 'TTFT p99' }))
    expect(cards()[0].querySelector('.xf-hero-l')?.textContent).toBe('TTFT p99 (lower is better)')
    expect(cards()[0].querySelectorAll('.xf-kpi-l')).toHaveLength(4)
  })

  it('re-ranks every view when the target changes', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    fireEvent.click(screen.getByRole('button', { name: 'TTFT p99' }))
    expect(cards()[0].querySelector('.win')?.textContent).toBe('8×TP8')
    expect(screen.getByText(/TTFT p99 at each of the spec’s loads/)).toBeTruthy()
  })

  it('picking a top config filters the page to its settings, and picking it again clears them', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    fireEvent.click(cards()[0])
    expect(screen.getByText(/2 of 7 runs shown/)).toBeTruthy()
    expect([...document.querySelectorAll('.xf-achip')].map((b) => b.textContent)).toContain('tp 4 ✕')
    fireEvent.click(cards()[0])
    expect(screen.getByText(/7 of 7 runs shown/)).toBeTruthy()
  })

  it('unpicking restores the filters the reader had set, and the cards hold still meanwhile', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const bar = (field: string, v: string) => [...document.querySelectorAll(`[data-field="${field}"] button`)].find((b) => b.querySelector('.xf-v')?.textContent === v) as Element
    fireEvent.click(bar('tp', '4')); fireEvent.click(bar('tp', '8'))
    const before = cards().map((c) => c.querySelector('.win')?.textContent)
    fireEvent.click(cards()[0])
    expect(cards().map((c) => c.querySelector('.win')?.textContent)).toEqual(before)
    fireEvent.click(cards()[0])
    const chips = [...document.querySelectorAll('.xf-achip')].map((b) => b.textContent)
    expect(chips).toEqual(['tp 4 ✕', 'tp 8 ✕'])
  })

  it('labels only the top configs the scatter still plots', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const ranks = () => [...document.querySelectorAll('#sec-overview .xf-rank')].map((t) => t.textContent)
    expect(ranks()).toEqual(['1', '2', '3'])
    fireEvent.click(cards()[0])
    expect(ranks()).toEqual(['1'])
  })

  it('pointing at a run on the load ramp lights its config up on the scatter too', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const [scatter, ramp] = [...document.querySelectorAll('#sec-overview svg.chart')]
    fireEvent.pointerEnter(ramp.querySelector('.xf-bg') as Element)
    expect(scatter.classList.contains('focused')).toBe(true)
    expect(scatter.querySelectorAll('.xf-dot.hot').length).toBeGreaterThan(0)
    expect(document.querySelector('.xf-hoverline')?.textContent).toMatch(/click to filter/)
  })

  it('the legend filters, and clicking empty chart space puts everything back', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const legend = (t: string) => [...document.querySelectorAll('.xf-lg')].find((b) => b.textContent === t) as Element
    fireEvent.change(document.getElementById('xf-colorby') as Element, { target: { value: 'mode' } })
    fireEvent.click(legend('aggregated'))
    expect(screen.getByText(/5 of 7 runs shown/)).toBeTruthy()
    fireEvent.click(legend('misses'))  // the fixture has one aggregated run that misses
    expect(screen.getByText(/4 of 7 runs shown/)).toBeTruthy()
    fireEvent.click(document.querySelector('#sec-overview .xf-backdrop') as Element)
    expect(screen.getByText(/7 of 7 runs shown/)).toBeTruthy()
  })

  it('lists router configs beside the one you open, and opening one filters the page', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const rows = [...document.querySelectorAll('.xf-rt-row')]
    // a generated arm sits under its family and shows only the parameter that differs
    expect(rows.map((r) => r.querySelector('.xf-rt-name')?.textContent)).toEqual(['default', 'nonCachedTokens=256'])
    expect(document.querySelector('.xf-rt-fam')?.textContent).toBe('pd-prefix')
    fireEvent.click(rows[1])
    expect(document.querySelector('.xf-rt-detail .xf-router-n')?.textContent).toBe('pd-prefix nonCachedTokens=256')
    expect(screen.getByText(/2 of 7 runs shown/)).toBeTruthy()
  })

  it('compares several router configs at once, one diagram each', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const rows = () => [...document.querySelectorAll('.xf-rt-row')]
    fireEvent.click(rows()[0]); fireEvent.click(rows()[1])
    const titles = () => [...document.querySelectorAll('.xf-rt-detail .xf-router-n')].map((e) => e.textContent)
    expect(titles()).toEqual(['default', 'pd-prefix nonCachedTokens=256'])
    expect(screen.getByText(/7 of 7 runs shown/)).toBeTruthy()
    fireEvent.click(document.querySelector('.xf-rt-detail .xf-rt-close') as Element)
    expect(titles()).toEqual(['pd-prefix nonCachedTokens=256'])
  })

  it('colors the trade-off chart by the field that moves the result most, and lets you change it', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const sel = document.getElementById('xf-colorby') as HTMLSelectElement
    expect(sel.value).not.toBe('')
    fireEvent.change(sel, { target: { value: 'tp' } })
    expect([...document.querySelectorAll('#sec-overview .xf-lg')].slice(0, 3).map((b) => b.textContent)).toEqual(['4', '8', '16'])
    const dots = [...document.querySelectorAll('#sec-overview .xf-dot')] as SVGElement[]
    expect(new Set(dots.map((d) => d.style.color)).size).toBeGreaterThan(1)
  })

  it('draws a bar only for values with runs in the filter, keeping a selected one so it can be cleared', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const vals = (field: string) => [...document.querySelectorAll(`[data-field="${field}"] button .xf-v`)].map((e) => e.textContent)
    const bar = (field: string, v: string) => [...document.querySelectorAll(`[data-field="${field}"] button`)].find((b) => b.querySelector('.xf-v')?.textContent === v) as Element
    expect(vals('dp')).toEqual(['1', '2'])
    fireEvent.click(bar('tp', '8'))
    expect(vals('dp')).toEqual(['1'])
    expect(vals('num_instances')).toEqual(['8'])
    expect(vals('max_num_seqs')).toEqual(['64'])
    fireEvent.click(bar('tp', '4')); fireEvent.click(bar('dp', '2'))
    expect(vals('tp')).toEqual(['4', '8'])
    expect(bar('tp', '8').getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(bar('tp', '8'))
    expect(vals('tp')).toEqual(['4'])
  })

  it('names each bar for a screen reader by its field, value and result', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const b = document.querySelector('[data-field="tp"] button') as Element
    expect(b.getAttribute('aria-label')).toMatch(/^TP · tensor parallel 4: \d+ runs in filter/)
  })

  it('cross-filters from a section bar and names both pools of a P/D config', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const pd = [...document.querySelectorAll('[data-field="mode"] button')].find((b) => b.textContent?.includes('P/D')) as Element
    fireEvent.click(pd)
    expect(screen.getByText(/2 of 7 runs shown/)).toBeTruthy()
    expect(document.querySelector('.xf-runs tbody .xf-cfglink .win')?.textContent).toBe('2P TP8 / 6D TP16')
  })

  it('opens a run\'s exact blis run command, fetched once for the report', async () => {
    let fetches = 0
    const cmds = FIXTURE.rows.map((_, i) => [`blis run --scenario s${i}.yaml --concurrency 8`, { scenario: `s${i}.yaml`, gpus: 8 }] as [string, unknown])
    render(<RunsTableHost load={() => { fetches++; return Promise.resolve(cmds) }} />)
    await waitFor(() => expect(document.querySelector('.xf-cmdbtn')).toBeTruthy())
    const btns = () => [...document.querySelectorAll('.xf-cmdbtn')] as HTMLElement[]
    fireEvent.click(btns()[0])
    await waitFor(() => expect(document.querySelector('.xf-cmd-line')?.textContent).toMatch(/^blis run --scenario s\d+\.yaml/))
    expect(document.querySelector('.xf-cmd summary')?.textContent).toMatch(/regenerated from the spec/)
    fireEvent.click(btns()[1])
    await waitFor(() => expect(document.querySelectorAll('.xf-cmd-line')).toHaveLength(2))
    expect(fetches).toBeLessThanOrEqual(2)
  })

  it('tells apart router arms whose sweep caption is empty, in the dashboard caption style', async () => {
    const arm = { ...FIXTURE.routers['pd-prefix-0123456789'], override: 'prefix-based-pd-decider nonCachedTokens 1024' }
    const pd = FIXTURE.rows.filter((r) => r[0] === 1)
    const data = {
      ...FIXTURE, routers: { ...FIXTURE.routers, 'pd-prefix-1024': arm },
      rows: [...FIXTURE.rows, ...pd.map((r) => r.map((v) => (v === 'pd-prefix-0123456789' ? 'pd-prefix-1024' : v)))],
      variants: [''], variant: FIXTURE.rows.concat(pd).map(() => 0),
    }
    render(<ScenarioPage runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16)]} load={() => Promise.resolve(data)} />)
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    fireEvent.click([...document.querySelectorAll('[data-field="mode"] button')].find((b) => b.textContent?.includes('P/D')) as Element)
    const names = [...document.querySelectorAll('.xf-runs tbody .xf-cfglink .cfgname')]
    expect(names.length).toBe(4)
    expect(new Set(names.map((n) => n.querySelector('.win')?.textContent)).size).toBe(1)
    expect(new Set(names.map((n) => n.querySelector('.winsub')?.textContent)).size).toBe(2)
    expect(names.map((n) => n.querySelector('.winsub')?.textContent)).toContain('pd-prefix nonCachedTokens=1024')
  })

  it('reserves the caption line on an uncaptioned top config card so the numbers line up', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    for (const c of cards()) expect(c.querySelectorAll('.cfgname .winsub, .cfgname .winsub-blank')).toHaveLength(1)
    expect(cards().some((c) => c.querySelector('.winsub-blank'))).toBe(true)
  })

  it('drops a pool heading in the engine section once none of its flags has runs in the filter', async () => {
    const at = 2 + FIXTURE.keys.length
    const add = (r: Scalar[], v: Scalar) => [...r.slice(0, at), v, ...r.slice(at)]
    const pd = FIXTURE.rows.filter((r) => r[0] === 1)
    const data = {
      ...FIXTURE, keys: [...FIXTURE.keys, 'prefill_max_num_seqs'],
      rows: [...FIXTURE.rows.map((r) => add(r, r[0] === 1 ? 32 : null)), ...pd.map((r) => add(r, 128))],
    }
    render(<ScenarioPage runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16)]} load={() => Promise.resolve(data)} />)
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    const heads = () => [...document.querySelectorAll('#sec-engine .eyebrow')].map((e) => e.textContent)
    expect(heads()).toContain('Prefill pool')
    fireEvent.click([...document.querySelectorAll('[data-field="mode"] button')].find((b) => b.textContent?.includes('aggregated')) as Element)
    expect(heads()).not.toContain('Prefill pool')
    expect(document.querySelector('#sec-engine [data-field="max_num_seqs"]')).toBeTruthy()
  })

  it('calls a single-pool deployment aggregated, not colocated', async () => {
    page()
    await waitFor(() => expect(cards().length).toBeGreaterThan(0))
    expect(cards()[0].querySelector('.xf-mode')?.textContent).toBe('aggregated')
    expect(document.body.textContent).not.toMatch(/colocated/i)
  })
})

function RunsTableHost({ load }: { load: (id: string) => Promise<Command[]> }) {
  const x = useExplorer(FIXTURE)
  return <RunsTable x={x} load={load} />
}

describe('routing + records', () => {
  it('parses routes', () => {
    expect(parseRoute('#/scenario/a%20b').runId).toBe('a b')
    expect(parseRoute('#/reports?model=x').params.model).toBe('x')
    expect(parseRoute('').tab).toBe('reports')
  })
})

describe('Submit experiment spec', () => {
  const wl = (name: string, kind: 'shape' | 'replay'): WorkloadEntry => ({ name, kind } as WorkloadEntry)
  const cat = (workloads: WorkloadEntry[] = [wl('chatbot', 'shape')]): Catalog => ({
    models: [{ name: 'm1', arch: 'MoE', context: 262144 } as Catalog['models'][number]],
    hardware: [{ name: 'h100' } as Catalog['hardware'][number]],
    workloads, fabrics: [], storage: [], registry: [], versions: {},
  })
  const spec = () => screen.getByLabelText('Generated experiment spec').textContent ?? ''
  // Dropdown is a custom listbox, so its options only exist once it is opened.
  const offered = () => {
    fireEvent.click(screen.getByLabelText('Workload'))
    return [...document.querySelectorAll('[role="option"]')].map((o) => o.textContent)
  }

  it('emits an experiment spec, not a legacy scenario', () => {
    render(<Submit catalog={cat()} />)
    const y = spec()
    expect(y).toContain('name: m1-h100-chatbot-16gpu')
    expect(y).toContain('  gpus: 16')
    expect(y).toContain('slo: {itl_p99: 20ms}')
    expect(y).toContain('sweeps:')
    expect(y).not.toContain('gpu_budget')
    expect(y).not.toContain('interactivity_floor')
    const pr = screen.getByRole('link', { name: /Open a pull request/ }).getAttribute('href') ?? ''
    expect(new URL(pr).searchParams.get('filename')).toBe('experiments/m1-h100-chatbot-16gpu.yaml')
  })

  it('keeps every sweep on the GPU budget', () => {
    render(<Submit catalog={cat()} />)
    // 16 GPUs: tp [8, 4, 2, 1] x num-instances [2, 4, 8, 16] pair up to exactly 16
    expect(spec()).toContain('num-instances: [2, 4, 8, 16]')
    expect(spec()).toContain('parallel: {tp: [8, 4, 2, 1], enable_expert_parallel: [true]}')
  })

  it('lets targets other than inter-token latency be set', () => {
    render(<Submit catalog={cat()} />)
    fireEvent.click(screen.getByLabelText('Inter-token latency'))
    fireEvent.click(screen.getByLabelText('Time to first token'))
    fireEvent.change(screen.getByLabelText('Time to first token cap'), { target: { value: '1.5' } })
    expect(spec()).toContain('slo: {ttft_p99: 1.5s}')
    fireEvent.click(screen.getByLabelText('Time to first token'))
    expect(spec()).not.toContain('slo:')
  })

  it('adds a throughput floor per accelerator', () => {
    render(<Submit catalog={cat()} />)
    fireEvent.click(screen.getByLabelText('Throughput'))
    fireEvent.change(screen.getByLabelText('Throughput floor'), { target: { value: '800' } })
    expect(spec()).toContain('slo: {itl_p99: 20ms, throughput_per_gpu: 800}')
  })

  it('offers replays and adds the replay block for one', () => {
    render(<Submit catalog={cat([wl('chatbot', 'shape'), wl('agentx-mvp', 'replay')])} />)
    expect(offered()).toEqual(['chatbot', 'agentx-mvp (replay)'])
    fireEvent.mouseDown(screen.getByText('agentx-mvp (replay)'))
    expect(spec()).toContain('  workload: agentx-mvp')
    expect(spec()).toContain('corpus: .corpus/traces.jsonl')
    expect(spec()).toContain('  engine: {max_model_len: 262144}')
  })

  it('defaults to a shape even when a replay sorts first', () => {
    render(<Submit catalog={cat([wl('agentx-mvp', 'replay'), wl('chatbot', 'shape')])} />)
    expect(spec()).toContain('  workload: chatbot')
    expect(spec()).not.toContain('replay:')
    expect(spec()).not.toContain('max_model_len')
  })

  it('drops the routing sweep when one instance takes the whole budget', () => {
    render(<Submit catalog={cat()} />)
    fireEvent.change(screen.getByLabelText('Accelerator budget'), { target: { value: '8' } })
    // 8 GPUs still allows 2x TP4 etc.; 1 GPU leaves only 1x TP1
    expect(spec()).toContain('name: routing')
    fireEvent.change(screen.getByLabelText('Accelerator budget'), { target: { value: '1' } })
    expect(spec()).not.toContain('name: routing')
  })

  it('refuses a name the validator would reject', () => {
    render(<Submit catalog={cat()} />)
    fireEvent.change(screen.getByLabelText('Spec name'), { target: { value: 'Bad Name' } })
    expect(screen.queryByRole('link', { name: /Open a pull request/ })).toBeNull()
    expect(document.body.textContent).toContain('lowercase letters, digits and dashes')
  })
})

describe('Submit numeric inputs', () => {
  const catalog: Catalog = {
    models: [{ name: 'm1' } as Catalog['models'][number]],
    hardware: [{ name: 'h100' } as Catalog['hardware'][number]],
    workloads: [{ name: 'chatbot' } as Catalog['workloads'][number]],
    fabrics: [], storage: [], registry: [], versions: {},
  }
  const fields = () => {
    render(<Submit catalog={catalog} />)
    return { budget: screen.getByLabelText('Accelerator budget') as HTMLInputElement, itl: screen.getByLabelText('Inter-token latency cap') as HTMLInputElement }
  }
  const spec = () => screen.getByLabelText('Generated experiment spec').textContent ?? ''

  it('lets you clear the field instead of snapping back to 1', () => {
    const { budget } = fields()
    fireEvent.change(budget, { target: { value: '' } })
    expect(budget.value).toBe('')
  })

  it('lets you retype a value digit by digit', () => {
    const { itl } = fields()
    fireEvent.change(itl, { target: { value: '' } })
    fireEvent.change(itl, { target: { value: '2' } })
    fireEvent.change(itl, { target: { value: '25' } })
    expect(itl.value).toBe('25')
    expect(spec()).toContain('itl_p99: 25ms')
    expect(document.body.textContent).toContain('40 tok/s per user')
  })

  it('keeps the YAML on the last valid value while the field is empty', () => {
    const { budget } = fields()
    fireEvent.change(budget, { target: { value: '' } })
    expect(spec()).toContain('gpus: 16')
  })

  it('restores the last valid value when an emptied field loses focus', () => {
    const { budget } = fields()
    fireEvent.change(budget, { target: { value: '' } })
    fireEvent.blur(budget)
    expect(budget.value).toBe('16')
  })
})

describe('scenario cards', () => {
  it('lays out cluster and workload on their own labelled rows', () => {
    render(<Reports scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16)]} params={{}} />)
    const card = document.querySelector('.card') as Element
    const rows = [...card.querySelectorAll('.specrows > div')].map((d) => d.textContent ?? '')
    expect(rows.some((t) => t.startsWith('Cluster'))).toBe(true)
    expect(rows.some((t) => t.startsWith('Workload'))).toBe(true)
    // the model is the card heading, a line of its own
    expect(card.querySelector('.cardhead h3')?.textContent).toBe('m1')
  })

  it('shows the cluster as count × hardware', () => {
    render(<Reports scenarios={[mk('a', 'm1', 'B200', 'chatbot', 32)]} params={{}} />)
    const dd = document.querySelector('.card .specrows > div:first-child dd')
    expect(dd?.textContent).toBe('32 × B200')
  })

  it('joins the replica count to the shape and leaves node count off the cluster', () => {
    const s = mk('a', 'm1', 'B200', 'chatbot', 32)
    render(<Reports scenarios={[{ ...s, winner: { ...s.winner, label: '8x TP4 + EP' } }]} params={{}} />)
    expect(document.querySelector('.card .win')?.textContent).toBe('8×TP4 + EP')
    expect(document.querySelector('.card .specrows dd')?.textContent).not.toMatch(/nodes/)
  })

  it('captions the config with its knob variant only, never the replica shape', () => {
    const s = mk('a', 'm1', 'B200', 'chatbot', 32)
    const withVariant = { ...s, run_id: 'v', winner: { ...s.winner, ni: 8, tp: 4, variant: 'prefix-affinity routing' } }
    const bare = { ...s, run_id: 'b', winner: { ...s.winner, ni: 8, tp: 4, variant: '' } }
    render(<Reports scenarios={[withVariant, bare]} params={{}} />)
    const [a, b] = [...document.querySelectorAll('.card')]
    expect(a.querySelector('.winsub')?.textContent).toBe('prefix-affinity routing')
    expect(b.querySelector('.winsub')).toBeNull()
    // the caption line stays reserved so both cards' numbers start at the same height
    expect(b.querySelector('.cfgname .winsub-blank')?.getAttribute('aria-hidden')).toBe('true')
    expect(a.querySelector('.winsub-blank')).toBeNull()
  })

  it('labels each performance number and gives latency its own column', () => {
    render(<Reports scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16)]} params={{}} />)
    const labels = [...document.querySelectorAll('.card .perf .l')].map((l) => l.textContent)
    expect(labels).toEqual(['Per accelerator', 'Per user', 'Latency p99'])
  })

  it('drops the latency column when the winner has no latency', () => {
    const s = mk('a', 'm1', 'H100', 'chatbot', 16)
    render(<Reports scenarios={[{ ...s, winner: { ...s.winner, itl_p99: null as unknown as number } }]} params={{}} />)
    expect([...document.querySelectorAll('.card .perf .l')].map((l) => l.textContent)).toEqual(['Per accelerator', 'Per user'])
  })

  it('orders the filter rail model, workload, hardware, then budget', () => {
    render(<Reports scenarios={[mk('a', 'm1', 'B200', 'chatbot', 32)]} params={{}} />)
    const labels = [...document.querySelectorAll('.facetrail .flabel')].map((p) => p.textContent)
    expect(labels).toEqual(['Model', 'Workload', 'Hardware', 'Accelerator budget'])
  })

  it('does not show the sweep-size "N/M servable" line on the card', () => {
    render(<Reports scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16)]} params={{}} />)
    const card = document.querySelector('.card') as Element
    expect(card.textContent).not.toMatch(/servable/i)
  })
})

describe('footer pinned-stack disclosure', () => {
  // The footer used to list blis-catalog / blis-registry / inference-sim as three flat pills,
  // which reads as three dependencies of equal standing. There is one: the BLIS simulator.
  const stack: Catalog['stack'] = {
    root: { name: 'inference-sim', version: 'v0.12.0', note: 'prices every run' },
    deps: [
      { name: 'blis-latency-kernel', version: 'v0.3.0', note: 'cost model' },
      { name: 'blis-schemas', version: 'v0.5.0', note: 'structs the scenario decodes against' },
      { name: 'blis-catalog', version: '0.4.0', note: 'models, hardware, workloads' },
      { name: 'blis-registry', version: 'v0.2.0', note: 'coefficients' },
    ],
  }
  const cat = (over: Partial<Catalog> = {}): Catalog => ({
    models: [], hardware: [], workloads: [], fabrics: [], storage: [], registry: [],
    versions: { 'blis-catalog': '0.4.0', 'blis-registry': 'v0.2.0', 'inference-sim': 'v0.12.0' },
    stack, ...over,
  })

  const pills = () => [...document.querySelectorAll('.footstack .pill')].map((p) => p.textContent?.trim())
  const toggle = () => screen.getByRole('button')

  it('shows BLIS alone at rest as a plain pill, with a +N toggle beside it', () => {
    render(<Footer catalog={cat()} />)
    expect(pills()).toEqual(['inference-sim v0.12.0'])
    // only the count is clickable, not the BLIS pill
    expect(screen.getAllByRole('button')).toHaveLength(1)
    expect(toggle().textContent).toBe('+4 more')
    expect(toggle().getAttribute('aria-expanded')).toBe('false')
    expect(document.querySelector('.footmeta')?.textContent).not.toContain('blis-catalog')
  })

  it('reveals the deps as the same pills when expanded, and collapses again', () => {
    render(<Footer catalog={cat()} />)
    fireEvent.click(toggle())
    expect(pills()).toEqual([
      'inference-sim v0.12.0', 'blis-latency-kernel v0.3.0', 'blis-schemas v0.5.0',
      'blis-catalog 0.4.0', 'blis-registry v0.2.0',
    ])
    // every dep carries the same pill class as the root, so they render identically
    expect(new Set([...document.querySelectorAll('.footstack .pill')].map((p) => p.className))).toEqual(new Set(['pill']))
    expect(toggle().textContent).toBe('show less')
    expect(toggle().getAttribute('aria-expanded')).toBe('true')
    fireEvent.click(toggle())
    expect(pills()).toEqual(['inference-sim v0.12.0'])
    expect(toggle().textContent).toBe('+4 more')
  })

  it('surfaces the latency kernel, which the old flat version pills dropped entirely', () => {
    // KERNEL_REF prices every run but was never in `versions`, so it was invisible.
    render(<Footer catalog={cat()} />)
    fireEvent.click(toggle())
    expect(document.querySelector('.footstack')?.textContent).toContain('blis-latency-kernel')
  })

  it('renders however many deps the pin states, not a fixed four', () => {
    // blis-vllm is a dependency only from BLIS v0.14.0; build_catalog omits it at older pins.
    const withVllm: Catalog['stack'] = {
      root: { name: 'inference-sim', version: 'v0.15.0', note: 'prices every run' },
      deps: [
        ...stack!.deps.slice(0, 1),
        { name: 'blis-vllm', version: 'v0.32.0-blis.1', note: 'engine config surface' },
        ...stack!.deps.slice(1),
      ],
    }
    render(<Footer catalog={cat({ stack: withVllm })} />)
    expect(toggle().textContent).toBe('+5 more')
    fireEvent.click(toggle())
    const names = [...document.querySelectorAll('.footstack .pill .pk')].map((p) => p.textContent)
    expect(names).toEqual([
      'inference-sim', 'blis-latency-kernel', 'blis-vllm', 'blis-schemas', 'blis-catalog', 'blis-registry',
    ])
  })

  it('shows only the pinned simulator version, with older ones in the tooltip', () => {
    const multi: Catalog['stack'] = { root: { name: 'inference-sim', version: 'v0.15.0, v0.12.0', note: 'prices every run' }, deps: stack!.deps }
    render(<Footer catalog={cat({ stack: multi })} />)
    const root = document.querySelector('.footstack .pill') as Element
    expect(root.textContent).toContain('v0.15.0')
    expect(root.textContent).not.toContain('v0.12.0')
    expect(root.getAttribute('title')).toContain('v0.12.0')
  })

  it('shows no toggle when the stack somehow has none', () => {
    render(<Footer catalog={cat({ stack: { root: stack!.root, deps: [] } })} />)
    expect(screen.queryByRole('button')).toBeNull()
    expect(pills()).toEqual(['inference-sim v0.12.0'])
  })

})
