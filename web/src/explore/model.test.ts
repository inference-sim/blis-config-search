import { describe, expect, it } from 'vitest'
import type { ExploreData, Scalar } from './types'
import {
  configFilters, overrideParts, sloMs, fieldValue, groupTies, liveFields, matches, rankConfigs, routerLabel, sortValues, toggle, topoLabel, unpack,
} from './model'

const KEYS = ['decode_instances', 'decode_tp', 'dp', 'max_num_seqs', 'num_instances', 'prefill_instances', 'prefill_tp', 'router_config', 'tp']
// [sweep, conc, ...KEYS, tpg, cl, pu, itl, ttft, ok]
const row = (s: number, conc: number, k: Scalar[], m: number[], ok = 1): Scalar[] => [s, conc, ...k, ...m, ok]
export const FIXTURE: ExploreData = {
  run_id: 'x', slo: { itl_p99: '20ms', ttft_p99: '5s' }, concurrency: [8, 64], fixed_engine: {},
  totals: { runs: 7 }, keys: KEYS, metrics: ['tpg', 'cl', 'pu', 'itl', 'ttft'],
  sweeps: [{ name: 'colo', narrative: 'n', configs: 3, dropped: [], spec: {} }, { name: 'pd', narrative: 'n', configs: 1, dropped: [], spec: {} }],
  routers: {
    'pd-prefix-0123456789': { path: 'p', family: 'pd-prefix', override: 'prefix-based-pd-decider nonCachedTokens 256', summary: '', yaml: '', chain: { plugins: [], profiles: [], flow_control: false, gate: [], feeds: {} } },
  },
  spec_yaml: '',
  rows: [
    row(0, 8, [null, null, 2, 64, 8, null, null, null, 4], [30, 240, 90, 10, 500]),
    row(0, 64, [null, null, 2, 64, 8, null, null, null, 4], [50, 400, 60, 15, 900]),
    row(0, 64, [null, null, 2, 256, 8, null, null, null, 4], [50, 400, 60, 15, 900]), // ties the row above
    row(0, 64, [null, null, 1, 64, 8, null, null, null, 8], [44, 352, 92, 11, 300]),
    row(0, 8, [null, null, 1, 64, 4, null, null, null, 16], [20, 160, 100, 8, 200], 0),
    row(1, 8, [6, 16, null, 64, null, 2, 8, 'pd-prefix-0123456789', null], [40, 320, 97, 10, 2900]),
    row(1, 64, [6, 16, null, 64, null, 2, 8, 'pd-prefix-0123456789', null], [42, 336, 70, 19, 4800]),
  ],
}
const runs = unpack(FIXTURE)
const R = FIXTURE.routers

describe('unpack and labels', () => {
  it('names both pools of a disaggregated config, not just the prefill one', () => {
    expect(runs[5].mode).toBe('P/D')
    expect(runs[5].topo).toBe('2P TP8 / 6D TP16')
    expect(topoLabel({ num_instances: 8, tp: 4, dp: 2, enable_expert_parallel: true })).toBe('8× TP4×DP2 EP')
  })
  it('shows a generated router arm as its family and the parameter it overrides', () => {
    expect(routerLabel('pd-prefix-0123456789', R)).toBe('pd-prefix nonCachedTokens=256')
    expect(routerLabel(null, R)).toBe('default')
    const two = { ...R['pd-prefix-0123456789'], override: 'prefix-based-pd-decider nonCachedTokens 256, queue-scorer weight 2' }
    expect(routerLabel('a', { a: two })).toBe('pd-prefix nonCachedTokens=256, weight=2')
    expect(overrideParts(two.override).map((o) => o.plugin)).toEqual(['prefix-based-pd-decider', 'queue-scorer'])
  })
})

describe('fields', () => {
  it('splits topology into per-pool fields and marks the other serving mode n/a', () => {
    expect(fieldValue(runs[0], 'tp', R)).toBe('4')
    expect(fieldValue(runs[0], 'prefill_tp', R)).toBe('n/a')
    expect(fieldValue(runs[5], 'tp', R)).toBe('n/a')
    expect(fieldValue(runs[5], 'decode_tp', R)).toBe('16')
  })
  it('offers a field only when it has two real values', () => {
    const f = liveFields(runs, KEYS, R)
    expect(f).toEqual(['mode', 'num_instances', 'tp', 'dp', 'max_num_seqs', 'router_config'])
    // one P/D config: its pool fields cannot narrow anything
    expect(f).not.toContain('decode_tp')
  })
  it('reads every SLO unit the spec schema allows', () => {
    expect([sloMs('20ms'), sloMs('5s'), sloMs('250us'), sloMs('x')]).toEqual([20, 5000, 0.25, null])
  })
  it('sorts numbers numerically, defaults first, n/a last', () => {
    expect(sortValues(['16', 'n/a', '4', '8'])).toEqual(['4', '8', '16', 'n/a'])
    expect(sortValues(['fp8', 'default', 'auto'])).toEqual(['default', 'auto', 'fp8'])
    expect(sortValues(['default', '2048', '16384', '8192'])).toEqual(['default', '2048', '8192', '16384'])
  })
})

describe('crossfilter', () => {
  it('ANDs fields, ORs values, and can skip one field to count its own options', () => {
    let sel = toggle({}, 'tp', '4')
    sel = toggle(sel, 'tp', '8')
    expect(runs.filter((r) => matches(r, sel, R)).map((r) => r.id)).toEqual([0, 1, 2, 3])
    sel = toggle(sel, 'dp', '1')
    expect(runs.filter((r) => matches(r, sel, R)).map((r) => r.id)).toEqual([3])
    expect(runs.filter((r) => matches(r, sel, R, 'tp')).map((r) => r.id)).toEqual([3, 4])
    expect(toggle(toggle({}, 'tp', '4'), 'tp', '4')).toEqual({})
  })
  it('turns a config into filters for only the fields its mode has', () => {
    const f = liveFields(runs, KEYS, R)
    expect(configFilters(runs[0], f, R)).toEqual({ mode: ['aggregated'], num_instances: ['8'], tp: ['4'], dp: ['2'], max_num_seqs: ['64'], router_config: ['default'] })
  })
})

describe('ranking', () => {
  it('ranks each config at its best servable load, on the chosen metric', () => {
    expect(rankConfigs(runs, 'tpg').map((r) => r.id)).toEqual([1, 2, 3, 6])
  })
  it('ranks latency lowest first and ignores runs that miss an SLO', () => {
    const ids = rankConfigs(runs, 'ttft').map((r) => r.id)
    expect(ids[0]).toBe(3)
    expect(ids).not.toContain(4)
  })
  it('folds configs with identical results into one, so a no-op knob is not a runner-up', () => {
    const g = groupTies(rankConfigs(runs, 'tpg'))
    expect(g[0].lead.id).toBe(1)
    expect(g[0].ties.map((r) => r.id)).toEqual([2])
    expect(g).toHaveLength(3)
  })
})
