import { describe, it, expect } from 'vitest'
import { visibleFacets, facetLevels, levelOf, EMPTY_SEL, FACETS } from '../src/lib'
import type { RunRecord } from '../src/types'

// A drafting record set shaped exactly like what report.py now emits.
const mk = (o: Partial<RunRecord>): RunRecord => ({
  id: String(Math.random()), src: 'probe', conc: 102, tp: 4, dp: 1, ni: 8, gpus: 32,
  series: '8x TP4 + EP', route: 'default', pd: 'colocated', mtp: 0,
  mnbt: 8192, mns: 256, pu: 100, cl: 1000, itl: 10, ttft: 1000, hit: null, p95: 2000,
  ok: true, viol: [], ...o,
} as RunRecord)

const recs: RunRecord[] = [
  mk({ mtp: 0, spec: null, al: null }),
  mk({ mtp: 1, spec: 'dspark', al: 1.81, pu: 302.88, pu_basis: 'derived-from-acceptance' }),
  mk({ mtp: 3, spec: 'dspark', al: 2.9, pu: 400, pu_basis: 'derived-from-acceptance' }),
  mk({ mtp: 5, spec: 'dspark', al: 3.51, pu: 450, pu_basis: 'derived-from-acceptance' }),
]

describe('dashboard surfaces the speculative-decoding knobs', () => {
  it('shows the draft-depth facet once drafting is swept', () => {
    expect(visibleFacets(recs)).toContain('mtp')
  })
  it('shows the drafter facet once drafting is swept', () => {
    expect(visibleFacets(recs)).toContain('spec')
  })
  it('hides both when nothing drafted', () => {
    const none = [mk({}), mk({ conc: 64 })]
    expect(visibleFacets(none)).not.toContain('mtp')
    expect(visibleFacets(none)).not.toContain('spec')
  })
  it('lists every draft depth as a filterable level', () => {
    const lv = facetLevels(recs, EMPTY_SEL, 'mtp').map((l) => l.value)
    expect(lv).toEqual(['0', '1', '3', '5'])
  })
  it('renders a null drafter as the dash, not the string "null"', () => {
    expect(levelOf(recs[0], 'spec')).not.toMatch(/null|None|undefined/)
    expect(levelOf(recs[1], 'spec')).toBe('dspark')
  })
  it('both facets carry a human label from the lexicon', () => {
    const byKey = Object.fromEntries(FACETS.map((f) => [f.key, f.label]))
    expect(byKey.mtp).toBe('draft tokens')
    expect(byKey.spec).toBe('drafter')
  })
  it('filtering to one depth selects only those runs', () => {
    const sel = { ...EMPTY_SEL, facets: { mtp: ['5'] } }
    const lv = facetLevels(recs, sel, 'spec')
    const dspark = lv.find((l) => l.value === 'dspark')
    expect(dspark?.count).toBe(1)
  })
})
