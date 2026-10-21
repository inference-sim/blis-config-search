import { describe, it, expect } from 'vitest'
import type { RunRecord } from './types'
import { EMPTY_SEL, facetLevels, histogram, selectRecords, varyingMetrics, visibleFacets } from './lib'

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
