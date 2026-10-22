import { describe, it, expect } from 'vitest'
import type { RunRecord } from './types'
import { EMPTY_SEL, facetLevels, histogram, operatingPoints, runFlags, selectRecords, varyingMetrics, visibleFacets } from './lib'

const r = (id: string, o: Partial<RunRecord>): RunRecord => ({
  id, src: 'probe', conc: 1, tp: 8, dp: 1, ni: 1, gpus: 8, series: 'A', route: 'default', pd: 'colocated', mtp: 0,
  mnbt: 8192, mns: 256, pu: 10, cl: 100, itl: 5, ttft: 50, hit: null, p95: 900, ok: true, status: 'ok', viol: [], ...o,
})
const recs = [r('1', { cl: 100, series: 'A' }), r('2', { cl: 200, series: 'B', ok: false }), r('3', { cl: 300, series: 'B' })]

describe('histogram', () => {
  it('bins values, including the max in the last bin', () => {
    const h = histogram([0, 5, 10], 10)!
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(3)
    expect(h.counts[9]).toBe(1)
  })
  it('returns null for empty, all-null, single and zero-span input', () => {
    expect(histogram([])).toBeNull()
    expect(histogram([null, NaN])).toBeNull()
    expect(histogram([4])).toBeNull()
    expect(histogram([4, 4, 4])).toBeNull()
  })
  it('varyingMetrics skips constant metrics', () => {
    expect(varyingMetrics(recs)).toEqual(['cl'])
  })
})

describe('selection', () => {
  it('brush range filters inclusively and drops non-finite values', () => {
    const withNull = [...recs, r('4', { cl: null })]
    expect(selectRecords(withNull, { facets: {}, ranges: { cl: [150, 300] } }).map((x) => x.id)).toEqual(['2', '3'])
    expect(selectRecords(withNull, EMPTY_SEL)).toHaveLength(4)
  })
  it('facets OR within a field, AND across, with counts ignoring own field', () => {
    const sel = { facets: { series: ['B'] }, ranges: { cl: [250, 300] as [number, number] } }
    expect(selectRecords(recs, sel).map((x) => x.id)).toEqual(['3'])
    expect(facetLevels(recs, sel, 'series')).toEqual([{ value: 'A', count: 0 }, { value: 'B', count: 1 }])
    expect(selectRecords(recs, { facets: { series: ['A', 'B'] }, ranges: {} })).toHaveLength(3)
  })
  it('hides single-valued categorical fields', () => {
    expect(visibleFacets(recs).sort()).toEqual(['ok', 'series'])
    expect(visibleFacets([])).toEqual([])
  })
})

import { emphasis, resolveHighlight, toggleHighlight } from './lib'
describe('series highlight', () => {
  it('emphasis picks on/off/neutral', () => {
    expect(emphasis('A', 'A')).toBe('on')
    expect(emphasis('A', 'B')).toBe('off')
    expect(emphasis(null, 'A')).toBe('neutral')
  })
  it('toggling the same series twice clears it; another series switches', () => {
    const once = toggleHighlight(null, 'A')
    expect(once).toBe('A')
    expect(toggleHighlight(once, 'B')).toBe('B')
    expect(toggleHighlight(once, 'A')).toBeNull()
  })
  it('sticky beats hover, and a series absent from view is ignored', () => {
    expect(resolveHighlight('A', 'B', ['A', 'B'])).toBe('A')
    expect(resolveHighlight(null, 'B', ['A', 'B'])).toBe('B')
    expect(resolveHighlight('Z', null, ['A', 'B'])).toBeNull()
    expect(resolveHighlight('Z', 'B', ['A', 'B'])).toBe('B')
    expect(resolveHighlight('A', null, [])).toBeNull()
  })
})

describe('operatingPoints', () => {
  const p = (id: string, conc: number, o: Partial<RunRecord>): RunRecord =>
    r(id, { conc, tpg: 10, ...o })

  it('keeps one row per load — the best servable run there', () => {
    const rows = operatingPoints([
      p('a', 128, { tpg: 50, series: 'X' }),
      p('b', 128, { tpg: 67, series: 'Y' }),
      p('c', 256, { tpg: 40, series: 'X' }),
    ])
    expect(rows.map((x) => x.conc)).toEqual([128, 256])
    expect(rows[0].id).toBe('b')          // 67 > 50 tok/s/GPU
  })

  it('ranks by tok/s per GPU, the objective the winner is chosen on', () => {
    // On kimi-k2.5 the per-USER best at every load is TP8+EP, while the search's winner is
    // TP4xDP2+EP (best per GPU). Ranking per-user made the table contradict the recommendation.
    const rows = operatingPoints([
      p('fast-user', 512, { pu: 131, tpg: 23, series: '64x TP8 + EP' }),
      p('best-gpu', 512, { pu: 83, tpg: 67, series: '64x TP4xDP2 + EP' }),
    ])
    expect(rows[0].id).toBe('best-gpu')
  })

  it('excludes runs that break the floor', () => {
    const rows = operatingPoints([
      p('bad', 128, { tpg: 99, ok: false, viol: ['itl'] }),
      p('ok', 128, { tpg: 10 }),
    ])
    expect(rows.map((x) => x.id)).toEqual(['ok'])
  })

  it('skips loads with no servable run, and tolerates missing values', () => {
    expect(operatingPoints([p('x', 128, { ok: false })])).toEqual([])
    expect(operatingPoints([p('y', 128, { tpg: null })])).toEqual([])
    expect(operatingPoints([r('z', { conc: null })])).toEqual([])
    expect(operatingPoints([])).toEqual([])
  })

  it('orders by load ascending', () => {
    const rows = operatingPoints([p('c', 512, {}), p('a', 1, {}), p('b', 128, {})])
    expect(rows.map((x) => x.conc)).toEqual([1, 128, 512])
  })
})

describe('runFlags', () => {
  it('names the flags a reader would deploy with', () => {
    const f = runFlags(r('1', { tp: 4, dp: 2, epw: 8, ni: 64, mns: 256, mnbt: 8192 }))
    const keys = f.map((x) => x.flag)
    expect(keys).toContain('--tensor-parallel-size')
    expect(keys).toContain('--data-parallel-size')
    expect(keys).toContain('--enable-expert-parallel')
    expect(keys).toContain('--num-instances')
  })

  it('prints a boolean flag bare', () => {
    const ep = runFlags(r('1', { epw: 8 })).find((x) => x.flag === '--enable-expert-parallel')
    expect(ep!.value).toBe('')
  })

  it('omits knobs sitting at their default, so the real settings stay visible', () => {
    const f = runFlags(r('1', { dp: 1, epw: null, mtp: 0, route: 'round-robin' }))
    const keys = f.map((x) => x.flag)
    expect(keys).not.toContain('--data-parallel-size')        // dp=1 is the default
    expect(keys).not.toContain('--enable-expert-parallel')    // EP off
    expect(keys).not.toContain('--num-speculative-tokens')    // MTP off
    expect(keys).not.toContain('routing')                     // round-robin is the default
  })

  it('shows routing once it is not the default', () => {
    const f = runFlags(r('1', { route: 'least-loaded' }))
    expect(f.find((x) => x.flag === 'routing')!.value).toBe('least-loaded')
  })
})
