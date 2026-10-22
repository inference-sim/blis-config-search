import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import type { Catalog, RunRecord, Scenario } from './types'
import { facetCounts, filterScenarios, topConfigs, NO_FILTERS, parseRoute, filterRecords, countReports } from './lib'
import { ScenarioDetail } from './components/ScenarioDetail'
import { Submit } from './components/Submit'

const w = { label: '1x TP8', tp: 8, ni: 1, gpus: 8, c_use: 1, tps: 1, tps_per_gpu: 1, per_user: 1, itl_p99: 1 }
const mk = (id: string, model: string, hw: string, wl: string, gpu: number, synthetic: boolean, records: RunRecord[] = []): Scenario => ({
  run_id: id, synthetic, model, hardware: hw, gpu_budget: gpu, workload: wl, isl_osl: 'x', floor: 'f', headroom: null,
  n_runs: 1, n_servable: 1, generated: '', winner: w, finalists: [], reason: '', records,
})
const S = [mk('a', 'm1', 'H100', 'chatbot', 16, false), mk('b', 'm1', 'H200', 'chatbot', 16, true), mk('c', 'm2', 'H100', 'multidoc', 8, true)]

describe('filters', () => {
  it('filters and gives live facet counts that ignore the facet itself', () => {
    expect(filterScenarios(S, { ...NO_FILTERS, model: 'm1' }).map((s) => s.run_id)).toEqual(['a', 'b'])
    const hw = facetCounts(S, { ...NO_FILTERS, model: 'm2' }, 'hardware')
    expect(hw).toEqual([{ value: 'H100', count: 1 }, { value: 'H200', count: 0 }])
    const gpu = facetCounts(S, NO_FILTERS, 'gpu')
    expect(gpu.map((g) => g.value)).toEqual(['8', '16'])
    expect(countReports(S, 'model', 'M1')).toBe(2)
  })
})

describe('topConfigs', () => {
  it('uses real scenarios only when any exist', () => {
    const r = topConfigs(S)
    expect(r.mockOnly).toBe(false)
    expect(r.items.map((s) => s.run_id)).toEqual(['a'])
  })
  it('falls back to flagged mocks, and to empty', () => {
    expect(topConfigs(S.slice(1)).mockOnly).toBe(true)
    expect(topConfigs([])).toEqual({ items: [], mockOnly: false })
  })
})

describe('routing + records', () => {
  it('parses routes', () => {
    expect(parseRoute('#/scenario/a%20b').runId).toBe('a b')
    expect(parseRoute('#/reports?model=x').params.model).toBe('x')
    expect(parseRoute('').tab).toBe('reports')
  })
  it('filters records by band/servable', () => {
    const base = { id: '', src: '', conc: 4, tp: null, dp: null, ni: null, gpus: null, series: 's', route: '', pd: '', mtp: 0, mnbt: null, mns: null, pu: 1, cl: 1, itl: 1, ttft: 1, hit: null, p95: null, ok: true, status: '', viol: [] }
    const recs = [base, { ...base, conc: 100, ok: false }]
    expect(filterRecords(recs, { series: '', band: 'low', servable: '' })).toHaveLength(1)
    expect(filterRecords(recs, { series: '', band: '', servable: 'bad' })).toHaveLength(1)
  })
})

describe('winner naming with knob variants', () => {
  // Several finalists share a topology once the search refines engine/routing knobs, so the
  // winner is identified by label + variant. Matching on label alone excluded EVERY same-shape
  // variant from the runners-up, and showed a name that did not say which config won.
  const wv = (label: string, variant: string, tpg: number) => ({
    ...w, label, variant, name: variant ? `${label} (${variant})` : label, tps_per_gpu: tpg,
  })
  const scen = (): Scenario => ({
    ...mk('a', 'm1', 'H200', 'multidoc', 64, false),
    winner: wv('64x TP4xDP2 + EP', 'least-loaded routing', 67),
    finalists: [
      wv('64x TP4xDP2 + EP', 'least-loaded routing', 67),
      wv('64x TP4xDP2 + EP', '', 36),
      wv('64x TP4xDP2 + EP', 'max seqs 64', 35),
      wv('64x TP2xDP4 + EP', '', 30),
    ],
  })

  it('shows the knob delta in the recommended config name', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen()]} />)
    expect(screen.getByText(/least-loaded routing/)).toBeTruthy()
  })

  it('keeps same-topology variants as runners-up', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen()]} />)
    // the baseline and the max-seqs variant share the winner's topology but are different
    // configs — excluding them by label left the runners-up empty or wrong
    expect(screen.getByText(/max seqs 64/)).toBeTruthy()
  })

  it('marks exactly one series as recommended', () => {
    // The explorer's "recommended" badge keys off the series name. Since series is topology-only,
    // a winner that is one VARIANT of that topology must not badge its losing siblings too.
    const rec = (id: string, series: string, variant: string): RunRecord => ({
      id, src: '', conc: 8, tp: 4, dp: 2, ni: 64, gpus: 512, series, variant,
      route: '', pd: '', mtp: 0, mnbt: null, mns: null, pu: 1, cl: 1, itl: 1, ttft: 1,
      hit: null, p95: null, ok: true, status: 'complete', viol: [],
    } as unknown as RunRecord)
    const s: Scenario = {
      ...scen(),
      records: [rec('1', '64x TP4xDP2 + EP', 'least-loaded routing'),
                rec('2', '64x TP4xDP2 + EP', 'max seqs 64'),
                rec('3', '64x TP2xDP4 + EP', '')],
    }
    render(<ScenarioDetail runId="a" scenarios={[s]} />)
    expect(screen.getAllByText('recommended')).toHaveLength(1)
  })
})

describe('what to run: operating points', () => {
  const rec = (over: Partial<RunRecord> = {}): RunRecord => ({
    id: 'r1', src: 'probe', conc: 128, tp: 4, dp: 2, ni: 64, gpus: 512,
    series: '64x TP4xDP2 + EP', variant: '', route: 'round-robin', pd: 'colocated',
    mtp: 0, mnbt: 8192, mns: 256, epw: 8, pu: 83.7, cl: 34307.6, tpg: 67,
    itl: 11.95, ttft: 455.8, hit: null, p95: 11098.9, ok: true, status: 'complete', viol: [],
    ...over,
  })
  const scen = (records: RunRecord[]): Scenario =>
    ({ ...mk('a', 'kimi-k2.5', 'H200', 'multidoc', 512, false, records) })

  it('shows one row per load, pairing the flags with what they delivered', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([
      rec({ id: 'a', conc: 128, tpg: 67 }),
      rec({ id: 'b', conc: 256, tpg: 55, pu: 70 }),
    ])]} />)
    const tbl = screen.getByTestId('operating-table')
    const rows = tbl.querySelectorAll('tbody tr')
    expect(rows).toHaveLength(2)
    // each row states the deployment, not just a series name
    expect(rows[0].textContent).toContain('--tensor-parallel-size')
    expect(rows[0].textContent).toContain('--enable-expert-parallel')
    // ... and the performance at that load
    expect(rows[0].textContent).toContain('83.7')
  })

  it('names the config and its knob delta per row', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([
      rec({ id: 'a', conc: 128, variant: 'least-loaded routing', route: 'least-loaded' }),
    ])]} />)
    const tbl = screen.getByTestId('operating-table')
    expect(tbl.textContent).toContain('64x TP4xDP2 + EP')
    expect(tbl.textContent).toContain('least-loaded routing')
  })

  it('sorts by any column when its header is clicked (default: by load)', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([
      rec({ id: 'lo', conc: 128, pu: 90, tpg: 67 }),
      rec({ id: 'hi', conc: 256, pu: 40, tpg: 55 }),
    ])]} />)
    const tbl = screen.getByTestId('operating-table')
    const sessions = (i: number) => tbl.querySelectorAll('tbody tr')[i].querySelector('td')?.textContent
    // default order is by load, ascending
    expect(sessions(0)).toBe('128')
    // click "tok/s per user" -> ascending by per-user rate, so the conc=256 (pu 40) row leads
    const puBtn = [...tbl.querySelectorAll('thead .sortbtn')].find((b) => b.textContent?.includes('tok/s per user')) as Element
    fireEvent.click(puBtn)
    expect(sessions(0)).toBe('256')
    // click again -> descending, so the pu=90 (conc 128) row leads
    fireEvent.click(puBtn)
    expect(sessions(0)).toBe('128')
    // aria-sort reflects the state
    const th = puBtn.closest('th') as Element
    expect(th.getAttribute('aria-sort')).toBe('descending')
  })

  it('picks the best-per-GPU run at each load, matching the objective', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([
      rec({ id: 'fast-user', conc: 512, pu: 131, tpg: 23, series: '64x TP8 + EP' }),
      rec({ id: 'best-gpu', conc: 512, pu: 83, tpg: 67, series: '64x TP4xDP2 + EP' }),
    ])]} />)
    const tbl = screen.getByTestId('operating-table')
    expect(tbl.querySelectorAll('tbody tr')).toHaveLength(1)
    expect(tbl.textContent).toContain('64x TP4xDP2 + EP')
    expect(tbl.textContent).not.toContain('64x TP8 + EP')
  })

  it('says so instead of rendering an empty table when nothing is servable', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([
      rec({ ok: false, viol: ['itl_p99_ms 45 > 20'] }),
    ])]} />)
    expect(screen.queryByTestId('operating-table')).toBeNull()
    expect(screen.getByText(/no run held the interactivity floor/i)).toBeTruthy()
  })

  it('clicking a row opens that run\'s deployment panel', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([rec({ id: 'a', conc: 128 })])]} />)
    fireEvent.click(screen.getByTestId('operating-table').querySelector('tbody tr') as Element)
    expect(screen.getByTestId('pt-detail')).toBeTruthy()
  })
})

describe('click a point for its deployment', () => {
  const rec = (over: Partial<RunRecord> = {}): RunRecord => ({
    id: 'r1', src: 'probe', conc: 512, tp: 4, dp: 2, ni: 64, gpus: 512,
    series: '64x TP4xDP2 + EP', variant: 'least-loaded routing',
    route: 'least-loaded', pd: 'colocated', mtp: 0, mnbt: 8192, mns: 256,
    epw: 8, a2a: 'ep', pu: 83.7, cl: 34307.6, itl: 11.95, ttft: 455.8,
    hit: null, p95: 11098.9, ok: true, status: 'complete', viol: [],
    pp: 1, nodes: 64, fabric: 'ib-400g', bs: 16, mml: 32768, gmu: 0.9,
    cg: 'PIECEWISE', tpg: 67, e2e: 12836, sched: 40.4,
    scen: 'kimi-k2.5-h200-abc.yaml', cmd: '/p/blis run --scenario kimi.yaml --concurrency 512',
    ...over,
  })
  const scen = (records: RunRecord[]): Scenario => ({
    ...mk('a', 'kimi-k2.5', 'H200', 'multidoc', 512, false, records),
  })
  const clickFirstPoint = () => {
    const pt = document.querySelector('svg.chart circle.pt')
    expect(pt).toBeTruthy()
    fireEvent.click(pt as Element)
  }

  it('shows no panel until a point is clicked', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([rec()])]} />)
    expect(screen.queryByTestId('pt-detail')).toBeNull()
  })

  it('previews a series operating point while hovering its line', () => {
    const recs = [
      rec({ id: 'X', series: '64x TP8 + EP', variant: '', tpg: 23 }),
      rec({ id: 'Y', series: '64x TP4xDP2 + EP', variant: 'least-loaded routing', tpg: 67 }),
    ]
    render(<ScenarioDetail runId="a" scenarios={[scen(recs)]} />)
    expect(screen.queryByTestId('pt-detail')).toBeNull()
    // hover the group for series X on the scatter
    const scatter = [...document.querySelectorAll('svg.chart')]
      .find((s) => s.querySelectorAll('circle.pt').length > 0) as Element
    const grpX = [...scatter.querySelectorAll('[data-series]')]
      .find((g) => g.getAttribute('data-series') === '64x TP8 + EP') as Element
    fireEvent.pointerEnter(grpX)
    const panel = screen.getByTestId('pt-detail')
    expect(panel.textContent).toContain('64x TP8 + EP')
  })

  it('does not crash navigating from a scenario with runs to one without', () => {
    // The hoverRec useMemo must sit ABOVE the early return for the empty-records case, or React
    // throws "rendered fewer hooks than expected" when the same component instance re-renders
    // from a populated scenario to an empty one (back/forward, or editing the URL).
    const withRuns = mk('a', 'kimi-k2.5', 'H200', 'multidoc', 512, false, [rec()])
    const noRuns = mk('a', 'kimi-k2.5', 'H200', 'multidoc', 512, false, [])
    const { rerender } = render(<ScenarioDetail runId="a" scenarios={[withRuns]} />)
    expect(() => rerender(<ScenarioDetail runId="a" scenarios={[noRuns]} />)).not.toThrow()
    expect(screen.getByText(/no per-run records are available/i)).toBeTruthy()
  })

  it('after clicking a point, hovering a different line still updates the box', () => {
    const recs = [
      rec({ id: 'X', series: '64x TP8 + EP', variant: '', tpg: 23 }),
      rec({ id: 'Y', series: '64x TP4xDP2 + EP', variant: 'least-loaded routing', tpg: 67 }),
    ]
    render(<ScenarioDetail runId="a" scenarios={[scen(recs)]} />)
    const scatter = [...document.querySelectorAll('svg.chart')]
      .find((s) => s.querySelectorAll('circle.pt').length > 0) as Element
    const grp = (series: string) => [...scatter.querySelectorAll('[data-series]')]
      .find((g) => g.getAttribute('data-series') === series) as Element
    // click a point for series Y -> box pinned to Y
    fireEvent.click(grp('64x TP4xDP2 + EP').querySelector('circle.pt') as Element)
    expect(screen.getByTestId('pt-detail').textContent).toContain('64x TP4xDP2 + EP')
    // now hover series X's line -> box should follow the hover to X
    fireEvent.pointerEnter(grp('64x TP8 + EP'))
    expect(screen.getByTestId('pt-detail').textContent).toContain('64x TP8 + EP')
  })

  it('opens the deployment panel for the clicked run', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([rec()])]} />)
    clickFirstPoint()
    const panel = screen.getByTestId('pt-detail')
    expect(panel.textContent).toContain('64x TP4xDP2 + EP')
    expect(panel.textContent).toContain('least-loaded routing')
    // parallelism, engine and policy must all be stated — this panel is what a reader deploys
    expect(panel.textContent).toContain('ib-400g')
    expect(panel.textContent).toContain('PIECEWISE')
    expect(panel.textContent).toMatch(/blis run/)
  })

  it('closes the panel again', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([rec()])]} />)
    clickFirstPoint()
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(screen.queryByTestId('pt-detail')).toBeNull()
  })

  it('shows the Reproduce note on a cmd-less run when others in the scenario have one', () => {
    // a per-run gap: some runs recorded cmd, this one did not. The panel shows (scenario HAS cmds)
    // and the clicked cmd-less run carries the honest note.
    render(<ScenarioDetail runId="a" scenarios={[scen([
      rec({ id: 'has', conc: 128, cmd: '/p/blis run --concurrency 128' }),
      rec({ id: 'missing', conc: 512, cmd: null }),
    ])]} />)
    // click the cmd-less run via its operating-table row (deterministic, unlike point order)
    const row = [...screen.getByTestId('operating-table').querySelectorAll('tbody tr')]
      .find((tr) => tr.textContent?.includes('512')) as Element
    fireEvent.click(row)
    const panel = screen.getByTestId('pt-detail')
    expect(panel.textContent).toMatch(/reproduce this run/i)
    expect(panel.textContent).toMatch(/not recorded/i)
  })

  it('hides the Reproduce panel entirely when NO run in the scenario has a command', () => {
    // a scenario-wide data gap (the committed glm fixture): showing "not recorded" on every point
    // would undermine the report, so the panel is suppressed rather than shown empty.
    render(<ScenarioDetail runId="a" scenarios={[scen([rec({ cmd: null })])]} />)
    clickFirstPoint()
    const panel = screen.getByTestId('pt-detail')
    expect(panel.textContent).not.toMatch(/reproduce this run/i)
    expect(panel.textContent).not.toMatch(/not recorded/i)
  })

  it('opens the panel from a point on a line chart too', () => {
    // The two "vs load" charts plot the same runs as the scatter, so a point there must inspect
    // its run as well. Lines built its points as bare {x,y,ok}, dropping the record — so there
    // was no id to click and only the scatter was clickable.
    render(<ScenarioDetail runId="a" scenarios={[scen([
      rec({ id: 'A', conc: 128, itl: 9 }), rec({ id: 'B', conc: 256, itl: 12 }),
    ])]} />)
    const charts = document.querySelectorAll('svg.chart')
    // chart 0 is the scatter; the per-user and ITL line charts follow
    const lineChart = charts[charts.length - 1]
    const pt = lineChart.querySelector('circle.pt')
    expect(pt).toBeTruthy()
    fireEvent.click(pt as Element)
    expect(screen.getByTestId('pt-detail')).toBeTruthy()
  })

  it('scrolls the panel into view when it opens off-screen', () => {
    // The panel renders below the chart grid, so on a laptop viewport it opened ~70px BELOW the
    // fold: the click appeared to do nothing. Opening it must bring it into view.
    const seen: unknown[] = []
    const orig = Element.prototype.scrollIntoView
    Element.prototype.scrollIntoView = function (this: Element) { seen.push(this) }
    try {
      render(<ScenarioDetail runId="a" scenarios={[scen([rec({ id: 'A' })])]} />)
      const pt = document.querySelector('svg.chart circle.pt')
      fireEvent.click(pt as Element)
      const panel = screen.getByTestId('pt-detail')
      expect(seen).toContain(panel)
    } finally {
      Element.prototype.scrollIntoView = orig
    }
  })

  it('every plotted chart opens the panel, not just the first', () => {
    // "applicable to all charts": assert per-chart, since a test that clicks only one chart
    // would pass while another stayed inert — which is exactly how the Lines charts were missed.
    const recs = [rec({ id: 'A', conc: 128, itl: 9 }), rec({ id: 'B', conc: 256, itl: 12 })]
    const charts = () => [...document.querySelectorAll('svg.chart')]
      .filter((s) => s.querySelectorAll('circle.pt').length > 0)
    const { unmount } = render(<ScenarioDetail runId="a" scenarios={[scen(recs)]} />)
    const n = charts().length
    expect(n).toBe(3)        // scatter + per-user-vs-load + ITL-vs-load
    unmount()
    for (let i = 0; i < n; i++) {
      const v = render(<ScenarioDetail runId="a" scenarios={[scen(recs)]} />)
      fireEvent.click(charts()[i].querySelector('circle.pt') as Element)
      expect(screen.getByTestId('pt-detail'), `chart ${i} did not open the panel`).toBeTruthy()
      v.unmount()
    }
  })

  it('rings the inspected point on every chart that plots it', () => {
    render(<ScenarioDetail runId="a" scenarios={[scen([
      rec({ id: 'A', conc: 128, itl: 9 }), rec({ id: 'B', conc: 256, itl: 12 }),
    ])]} />)
    clickFirstPoint()
    // scatter + per-user-vs-load + itl-vs-load all plot run A, so all three ring it
    expect(document.querySelectorAll('circle.pt.picked').length).toBeGreaterThan(1)
  })

  it('draws one line per config, not per topology', () => {
    // A refined topology has one run per knob variant at EVERY load, so grouping the lines by
    // series (topology only) drew six different configs as a single zigzag curve — on kimi-k2.5
    // one "line" spanned 27.9 to 68.0 tok/s at the same concurrency. Lines must group by the
    // config identity (series + variant).
    const two = [
      rec({ id: 'a1', conc: 128, pu: 100, variant: '' }),
      rec({ id: 'a2', conc: 256, pu: 90, variant: '' }),
      rec({ id: 'b1', conc: 128, pu: 60, variant: 'max seqs 64' }),
      rec({ id: 'b2', conc: 256, pu: 50, variant: 'max seqs 64' }),
    ]
    render(<ScenarioDetail runId="a" scenarios={[scen(two)]} />)
    // the line charts are whichever SVGs contain polylines (chart order is not the contract)
    const lines = [...document.querySelectorAll('svg.chart')]
      .filter((s) => s.querySelectorAll('polyline.ln').length > 0)
    expect(lines.length).toBeGreaterThan(0)
    // two configs over two loads each => two polylines per line chart, not one 4-point zigzag
    for (const c of lines) expect(c.querySelectorAll('polyline.ln')).toHaveLength(2)
  })

  it('opens the run that was actually clicked, after a re-sort', () => {
    // Points were keyed by index within their group, and the groups re-sort by highlight state on
    // every render — so React reused a circle across DIFFERENT configs and its click handler
    // opened someone else's deployment (clicking "TP2xDP4 + EP [batch tokens 2,048]" showed
    // "TP4xDP2 [gpu mem util 0.7]"). Hovering forces that re-sort, then the click must still land
    // on the right run.
    const recs = [
      rec({ id: 'X', series: '64x TP8 + EP', variant: '', conc: 128, pu: 120 }),
      rec({ id: 'Y', series: '64x TP4xDP2 + EP', variant: 'max seqs 64', conc: 128, pu: 60 }),
    ]
    render(<ScenarioDetail runId="a" scenarios={[scen(recs)]} />)
    // the scatter is the first chart WITH points (the brush histograms are svg.chart too)
    const scatter = [...document.querySelectorAll('svg.chart')]
      .find((s) => s.querySelectorAll('circle.pt').length > 0) as Element
    const circles = [...scatter.querySelectorAll('circle.pt')]
    expect(circles.length).toBeGreaterThan(1)
    const target = circles[circles.length - 1]
    const expected = (target.parentElement as Element).getAttribute('data-series')
    // provoke the emphasis re-sort before clicking
    fireEvent.pointerEnter(circles[0].parentElement as Element)
    fireEvent.pointerLeave(circles[0].parentElement as Element)
    fireEvent.click(target)
    expect(screen.getByTestId('pt-detail').textContent).toContain(expected as string)
  })

  it('renders a run with missing metrics without crashing', () => {
    // build_records emits nulls for rejected / missing-metric runs; the panel must not throw
    // and must not print "NaN" or "null" at the reader.
    render(<ScenarioDetail runId="a" scenarios={[scen([rec({
      pu: 5, cl: 10, itl: null, ttft: null, e2e: null, sched: null, tpg: null,
      gmu: null, bs: null, mml: null, cg: null, nodes: null, fabric: null,
      ok: false, viol: ['itl_p99_ms 45.0 > 20.0'],
    })])]} />)
    clickFirstPoint()
    const panel = screen.getByTestId('pt-detail')
    expect(panel.textContent).not.toContain('NaN')
    expect(panel.textContent).not.toContain('null')
    expect(panel.textContent).toContain('itl_p99_ms 45.0 > 20.0')   // why it is not servable
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
    render(<Submit catalog={catalog} mode="pages" />)
    const [budget, floor] = screen.getAllByRole('spinbutton') as HTMLInputElement[]
    return { budget, floor }
  }

  it('lets you clear the field instead of snapping back to 1', () => {
    const { budget } = fields()
    fireEvent.change(budget, { target: { value: '' } })
    expect(budget.value).toBe('')
  })

  it('lets you retype a value digit by digit', () => {
    const { floor } = fields()
    fireEvent.change(floor, { target: { value: '' } })
    fireEvent.change(floor, { target: { value: '8' } })
    fireEvent.change(floor, { target: { value: '80' } })
    expect(floor.value).toBe('80')
    // interactivity_floor is a mapping (the pipeline calls .get on it), so the value lands on the
    // nested tok_per_s_per_user key, not a bare `interactivity_floor: 80`.
    expect(screen.getByLabelText('Generated scenario.yaml').textContent).toContain('tok_per_s_per_user: 80')
  })

  it('keeps the YAML on the last valid value while the field is empty', () => {
    const { budget } = fields()
    fireEvent.change(budget, { target: { value: '' } })
    expect(screen.getByLabelText('Generated scenario.yaml').textContent).toContain('gpu_budget: 16')
  })

  it('restores the last valid value when an emptied field loses focus', () => {
    const { budget } = fields()
    fireEvent.change(budget, { target: { value: '' } })
    fireEvent.blur(budget)
    expect(budget.value).toBe('16')
  })
})

describe('ScenarioDetail', () => {
  it('handles empty records without crashing', () => {
    render(<ScenarioDetail runId="a" scenarios={S} />)
    expect(screen.getByText(/No per-run records/)).toBeTruthy()
    expect(document.body.textContent).not.toContain('NaN')
  })
  it('renders a single-point scatter without NaN', () => {
    const rec: RunRecord = { id: '1', src: '', conc: 1, tp: 1, dp: 1, ni: 1, gpus: 1, series: 's', route: '', pd: '', mtp: 0, mnbt: null, mns: null, pu: 10, cl: 20, itl: null, ttft: null, hit: null, p95: null, ok: true, status: 'complete', viol: [] }
    const S1 = [mk('a', 'm1', 'H100', 'chatbot', 16, false, [rec]), ...S.slice(1)]
    render(<ScenarioDetail runId="a" scenarios={S1} />)
    expect(screen.getByText(/Only one point/)).toBeTruthy()
    expect(document.body.innerHTML).not.toContain('NaN')
  })

  const twoSeries = (): RunRecord[] => {
    const base = (over: Partial<RunRecord>): RunRecord => ({
      id: '', src: '', conc: 128, tp: 4, dp: 2, ni: 64, gpus: 512, series: 'A', variant: '',
      route: '', pd: '', mtp: 0, mnbt: null, mns: null, pu: 80, cl: 1000, itl: 10, ttft: 100,
      hit: null, p95: null, ok: true, status: 'complete', viol: [], tpg: 50, ...over,
    })
    return [
      base({ id: 'a1', series: 'A', conc: 128 }), base({ id: 'a2', series: 'A', conc: 256, pu: 60 }),
      base({ id: 'b1', series: 'B', conc: 128, pu: 70 }), base({ id: 'b2', series: 'B', conc: 256, pu: 55 }),
    ]
  }

  it('#8 renders a series legend under every chart, not just the scatter', () => {
    render(<ScenarioDetail runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16, false, twoSeries())]} />)
    // three charts (scatter + 2 line charts) => three legends, each listing both series
    const legends = document.querySelectorAll('.legend')
    expect(legends.length).toBe(3)
    for (const lg of legends) {
      expect([...lg.querySelectorAll('.lg')].map((b) => b.textContent)).toEqual(
        expect.arrayContaining([expect.stringContaining('A'), expect.stringContaining('B')]))
    }
  })

  it('#3 shows the interactivity floor as a spec-chip in the header', () => {
    const s = { ...mk('a', 'm1', 'H100', 'chatbot', 16, false, twoSeries()), floor: '50 tok/s/user' }
    render(<ScenarioDetail runId="a" scenarios={[s]} />)
    const chip = document.querySelector('.specchips .spec.floor')
    expect(chip?.textContent).toContain('50 tok/s/user')
  })

  it('#5 states how finalists were chosen in the briefing', () => {
    render(<ScenarioDetail runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16, false, twoSeries())]} />)
    expect(screen.getByText(/how these were chosen/i)).toBeTruthy()
    expect(document.querySelector('.dek.method')?.textContent).toMatch(/tok\/s per GPU/i)
  })

  it('#9 header terms open a hover/focus tooltip showing the description', () => {
    render(<ScenarioDetail runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16, false, twoSeries())]} />)
    const labels = [...document.querySelectorAll('th .hcol')]
    // every column with help is reachable (incl. series/configuration via their aria-label)
    const aria = labels.map((e) => e.getAttribute('aria-label') ?? '')
    expect(aria.some((l) => l.startsWith('configuration:'))).toBe(true)
    // no dead native title — the help is a real tooltip, not a hover-only attribute
    expect(labels.every((e) => e.getAttribute('title') === null)).toBe(true)
    // hovering a header renders a tooltip bubble carrying that term's description
    const fleet = labels.find((e) => (e.getAttribute('aria-label') ?? '').startsWith('tok/s fleet:')) as Element
    expect(document.querySelector('.hcol-pop')).toBeNull()
    fireEvent.pointerEnter(fleet)
    const pop = document.querySelector('.hcol-pop')
    expect(pop?.textContent).toMatch(/serving capacity/i)
    fireEvent.pointerLeave(fleet)
    expect(document.querySelector('.hcol-pop')).toBeNull()
  })

  it('makes the full run-log table sortable by any column', () => {
    const recs = twoSeries()  // series A (pu 80/60) and B (pu 70/55), conc 128/256
    render(<ScenarioDetail runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16, false, recs)]} />)
    const tbl = screen.getByTestId('run-log')
    // configuration column (2nd cell) sorts by series+variant name
    const cfgText = () => [...tbl.querySelectorAll('tbody tr')].map((tr) => tr.querySelectorAll('td')[1].textContent ?? '')
    const cfgBtn = [...tbl.querySelectorAll('thead .sortbtn')].find((b) => b.textContent?.includes('configuration')) as Element
    fireEvent.click(cfgBtn)
    const col = cfgText()
    expect(col).toEqual([...col].sort())  // ascending by configuration name
    // tok/s per user descending puts the highest per-user run first (A @ conc128, pu 80); per-user
    // is the 3rd cell in the shared format (sessions, configuration, tok/s per user, ...)
    const puBtn = [...tbl.querySelectorAll('thead .sortbtn')].find((b) => (b.getAttribute('aria-label') ?? b.textContent ?? '').includes('tok/s per user') || b.textContent?.includes('tok/s per user')) as Element
    fireEvent.click(puBtn); fireEvent.click(puBtn)
    const puCells = [...tbl.querySelectorAll('tbody tr')].map((tr) => Number(tr.querySelectorAll('td')[2].textContent))
    expect(puCells[0]).toBe(Math.max(...puCells))
  })

  it('the run log uses the same rich format as the operating table (config cell + all runs)', () => {
    const recs = [
      ...twoSeries(),
      // a failed run must still appear in the run log, with its violation in the status column
      { ...twoSeries()[0], id: 'bad', series: 'C', conc: 512, ok: false, status: 'over', viol: ['itl_p99_ms 45 > 20'] },
    ]
    render(<ScenarioDetail runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16, false, recs)]} />)
    const tbl = screen.getByTestId('run-log')
    // rich format: the config cell renders deploy flags, not a bare series string
    expect(tbl.querySelector('tbody .op-flags')).toBeTruthy()
    // the failed run is listed with its reason
    expect(tbl.textContent).toContain('itl_p99_ms 45 > 20')
  })

  it('#4 lists only the axes that actually varied across runs', () => {
    // tp is constant (4) across both series; series A vs B differ by dp (2 vs 4) and the ramp
    // varies conc. So "data-parallel" is a searched axis; "tensor-parallel" is not.
    const recs = twoSeries().map((r) => r.series === 'B' ? { ...r, dp: 4 } : r)
    render(<ScenarioDetail runId="a" scenarios={[mk('a', 'm1', 'H100', 'chatbot', 16, false, recs)]} />)
    const line = document.querySelector('.axes-searched')?.textContent ?? ''
    expect(line).toMatch(/axes searched/i)
    expect(line).toContain('data-parallel')
    expect(line).not.toContain('tensor-parallel')
  })
})
