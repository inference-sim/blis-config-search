import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import type { Scenario } from './types'
import { facetCounts, filterScenarios, topConfigs, NO_FILTERS, parseRoute, filterRecords, countReports } from './lib'

const w = { label: '1x TP8', tp: 8, ni: 1, gpus: 8, c_use: 1, tps: 1, tps_per_gpu: 1, per_user: 1, itl_p99: 1 }
const mk = (id: string, model: string, hw: string, wl: string, gpu: number, synthetic: boolean): Scenario => ({
  run_id: id, synthetic, model, hardware: hw, gpu_budget: gpu, workload: wl, isl_osl: 'x', floor: 'f', headroom: null,
  n_runs: 1, n_servable: 1, generated: '', winner: w, finalists: [], reason: '',
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

describe('ScenarioDetail', () => {
  it('handles empty records without crashing', async () => {
    vi.resetModules()
    vi.doMock('./load', () => ({ loadRecords: async () => [], reportHref: (id: string) => `r/${id}` }))
    const { ScenarioDetail } = await import('./components/ScenarioDetail')
    render(<ScenarioDetail runId="a" scenarios={S} />)
    await waitFor(() => expect(screen.getByText(/No per-run records/)).toBeTruthy())
    expect(document.body.textContent).not.toContain('NaN')
  })
  it('renders a single-point scatter without NaN', async () => {
    vi.resetModules()
    const rec = { id: '1', src: '', conc: 1, tp: 1, dp: 1, ni: 1, gpus: 1, series: 's', route: '', pd: '', mtp: 0, mnbt: null, mns: null, pu: 10, cl: 20, itl: null, ttft: null, hit: null, p95: null, ok: true, status: 'complete', viol: [] }
    vi.doMock('./load', () => ({ loadRecords: async () => [rec], reportHref: (id: string) => `r/${id}` }))
    const { ScenarioDetail } = await import('./components/ScenarioDetail')
    render(<ScenarioDetail runId="a" scenarios={S} />)
    await waitFor(() => expect(screen.getByText(/Only one point/)).toBeTruthy())
    expect(document.body.innerHTML).not.toContain('NaN')
  })
})
