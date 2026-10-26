import { describe, it, expect } from 'vitest'
import type { RunRecord } from './types'
import { EMPTY_SEL, dash, facetLevels, fnum, histogram, operatingPoints, runFlags, selectRecords, varyingMetrics, visibleFacets } from './lib'
import { FACETS, METRICS } from './lib'
import { AXES, OBJECTIVE, OPERATING_COLUMNS, figure, help as lexHelp, label as lexLabel } from './lexicon'

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

// ---------------------------------------------------------------------------
// Issue #13 — cross-surface parity. The dashboard and the standalone HTML report render the same
// runs; they had drifted into two vocabularies and two ranking objectives for "What to run", so
// at high load they recommended DIFFERENT configurations for one scenario. These pin the seam
// from the dashboard side (the report side is pinned in the skill's test_report.py `Parity`).
describe('report/dashboard parity (issue #13)', () => {
  const rec = (over: Partial<RunRecord> = {}): RunRecord => ({
    id: 'r', src: 'probe', conc: 512, tp: 8, dp: 1, ni: 1, gpus: 100,
    series: '1x TP8 + EP', variant: '', route: 'round-robin', pd: 'colocated',
    mtp: 0, mnbt: 8192, mns: 256, epw: 8, pu: 100, cl: 5000, tpg: 50,
    itl: 10, ttft: 400, hit: null, p95: 9000, ok: true, status: 'complete', viol: [],
    ...over,
  })

  it('ranks operating points on tok/s per GPU, the stated shared objective', () => {
    expect(OBJECTIVE.key).toBe('tpg')
    // The exact disagreement from the issue: the fastest-per-user run is not the most efficient.
    const pts = operatingPoints([
      rec({ id: 'fast-user', pu: 131, tpg: 23, series: '64x TP8 + EP' }),
      rec({ id: 'best-gpu', pu: 83, tpg: 67, series: '64x TP4xDP2 + EP' }),
    ])
    expect(pts).toHaveLength(1)
    expect(pts[0].id).toBe('best-gpu')
  })

  it('breaks ties deterministically, so neither surface depends on record order', () => {
    const a = rec({ id: 'zzz' })
    const b = rec({ id: 'aaa' })
    expect(operatingPoints([a, b])[0].id).toBe('aaa')
    expect(operatingPoints([b, a])[0].id).toBe('aaa')
  })

  it('falls through the declared tiebreak chain when the objective ties', () => {
    // equal tok/s/GPU -> fleet tok/s decides (the first tiebreak the lexicon declares)
    const pts = operatingPoints([
      rec({ id: 'small', tpg: 50, cl: 400, gpus: 8 }),
      rec({ id: 'big', tpg: 50, cl: 5000, gpus: 100 }),
    ])
    expect(pts[0].id).toBe('big')
    expect(OBJECTIVE.tiebreak[0]).toBe('cl')
  })

  it('skips runs with no per-GPU figure rather than ranking them as infinitely inefficient', () => {
    const pts = operatingPoints([rec({ id: 'no-tpg', tpg: null }), rec({ id: 'ok', tpg: 50 })])
    expect(pts.map((p) => p.id)).toEqual(['ok'])
  })

  it('takes every metric and facet label from the shared lexicon', () => {
    // Each surface looks its labels up by key, so these ARE the report's words.
    for (const m of METRICS) expect(m.label).toBe(lexLabel(m.key))
    for (const f of FACETS) expect(f.label).toBe(lexLabel(f.key))
  })

  it('prints the same deploy flags the report does, from the shared flag list', () => {
    const flags = runFlags(rec({ dp: 2, route: 'least-loaded' }))
    const names = flags.map((f) => f.flag)
    expect(names).toContain('--tensor-parallel-size')
    expect(names).toContain('--data-parallel-size')      // shown when it is not 1
    expect(names).toContain('--enable-expert-parallel')
    expect(flags.find((f) => f.flag === '--enable-expert-parallel')?.value).toBe('')  // bare
    expect(names).toContain('routing')                   // shown when not round-robin
    // ... and omits the knobs sitting at their defaults
    const plain = runFlags(rec()).map((f) => f.flag)
    expect(plain).not.toContain('--data-parallel-size')
    expect(plain).not.toContain('routing')
    expect(plain).not.toContain('--num-speculative-tokens')
    expect(plain).not.toContain('prefill/decode')
  })

  it('gives every shared column a label and a definition for its tooltip', () => {
    for (const key of OPERATING_COLUMNS) {
      expect(lexLabel(key)).toBeTruthy()
      expect(lexHelp(key)).toBeTruthy()
    }
  })

  it('resolves the figures\' "@axes.*" references to the shared axis names', () => {
    // A figure's axis label must BE the shared string, not a copy that can drift from it.
    expect(figure('efficiency_interactivity').yl).toBe(AXES.tpg)
    expect(figure('per_user_vs_load').yl).toBe(AXES.pu)
    expect(figure('operating_curve').xl).toBe(AXES.cl_log)
    expect(figure('tail_scatter').xl).toBe(AXES.p95_log)
  })

  it('states the objective in the "What to run" blurb both surfaces print', () => {
    expect(OBJECTIVE.blurb).toContain(OBJECTIVE.label)
  })

  it('formats every shared column to the lexicon digits (the report does too)', () => {
    // report.py `_fnum` reproduces this exact formatter so one number is not printed 143.9 on one
    // surface and 144 on the other (PR #16 review). These cases are mirrored in
    // test_report.py `test_fnum_matches_the_dashboard_formatter` -- keep the two lists together.
    const cases: [number | null, number, string][] = [
      [143.9, 0, '144'], [18.0, 1, '18'], [6.72, 2, '6.72'], [7476.19, 0, '7,476'],
      [0.5, 0, '1'], [1.5, 0, '2'], [2.5, 0, '3'], [0.15, 1, '0.2'],
      [1e6, 0, '1,000,000'], [0.0, 2, '0'], [null, 1, dash],
    ]
    for (const [v, digits, want] of cases) expect(fnum(v, digits)).toBe(want)
  })

  it('names the no-servable-runs case without saying which way the run log sits', () => {
    // The dashboard's run log is BELOW this table; the report's explorer table is ABOVE it, so a
    // shared sentence cannot say "below".
    expect(OBJECTIVE.none).toBeTruthy()
    expect(OBJECTIVE.none.toLowerCase()).not.toContain('below')
  })
})
