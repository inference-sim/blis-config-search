import { describe, expect, it } from 'vitest'
import type { ExploreData, Scalar } from './types'
import {
  configFilters, overrideParts, sloMs, fieldValue, groupTies, liveFields, matches, rankConfigs, resolvedConfig, routerLabel, sortValues, isNumeric, sweepLine, toggle, topoLabel, unpack, loadLabel,
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
    expect(topoLabel({ num_instances: 8, tp: 4, dp: 2, enable_expert_parallel: true })).toBe('8×TP4xDP2 + EP')
    expect(topoLabel({ prefill_instances: 2, prefill_tp: 8, decode_instances: 6, decode_tp: 8 })).toBe('2P/6D TP8')
  })
  it('captions a run with the variant the dashboard shows', () => {
    const v = unpack({ ...FIXTURE, variants: ['', 'max seqs 256'], variant: [0, 0, 1, 0, 0, 0, 0] })
    expect(v.map((r) => r.variant).slice(0, 3)).toEqual(['', '', 'max seqs 256'])
    expect(runs[0].variant).toBe('')
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
  it('names the value an unset knob ran at', () => {
    const d = unpack({ ...FIXTURE, defaults: { max_num_seqs: 256 } })
    const unset = { ...d[0], knobs: { ...d[0].knobs, max_num_seqs: null } }
    expect(fieldValue(unset, 'max_num_seqs', R)).toBe('256 (default)')
    expect(fieldValue(d[0], 'max_num_seqs', R)).toBe('64')
    expect(fieldValue({ ...d[0], knobs: { ...d[0].knobs, max_num_seqs: 256 } }, 'max_num_seqs', R)).toBe('256 (default)')
    expect(sortValues(['64', '256 (default)', '512'])).toEqual(['64', '256 (default)', '512'])
    expect(sortValues(['fp8', 'auto (default)'])).toEqual(['auto (default)', 'fp8'])
  })
  it('sorts numbers numerically, defaults first, n/a last', () => {
    expect(sortValues(['16', 'n/a', '4', '8'])).toEqual(['4', '8', '16', 'n/a'])
    expect(sortValues(['fp8', 'default', 'auto'])).toEqual(['default', 'auto', 'fp8'])
    expect(sortValues(['default', '2048', '16384', '8192'])).toEqual(['default', '2048', '8192', '16384'])
  })
  it('treats numbers as numeric despite defaults and n/a', () => {
    expect(isNumeric(['2', '3', '4', '5', '6', '7', '8'])).toBe(true)
    expect(isNumeric(['default', '256 (default)', '512', 'n/a'])).toBe(true)
    expect(isNumeric(['fp8', 'auto (default)'])).toBe(false)
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

describe('load axis', () => {
  it('writes a load as c<N> and never as a bare count', () => {
    expect(loadLabel(512)).toBe('c512')
    expect(loadLabel('64')).toBe('c64')
  })
})

describe('the arm behind a card', () => {
  const SPEC = [
    '# header', 'name: x', 'scenario:', '  model: m', 'sweeps:',
    '  - name: colo', '    narrative: first', '  - name: "pd"   # disaggregated', '    narrative: second', '',
  ].join('\n')
  it('finds the line of a sweep in the spec as written, plain or quoted, and only under sweeps', () => {
    expect(sweepLine(SPEC, 'colo')).toBe(6)
    expect(sweepLine(SPEC, 'pd')).toBe(8)
    expect(sweepLine(SPEC, 'missing')).toBeNull()
    expect(sweepLine('scenario:\n  - name: colo\nsweeps:\n', 'colo')).toBeNull()   // a name above `sweeps:` is not a sweep
    expect(sweepLine('', 'colo')).toBeNull()
  })
  it('does not read a regex character in a sweep name as a pattern', () => {
    expect(sweepLine('sweeps:\n  - name: a.b\n  - name: axb\n', 'axb')).toBe(3)
    expect(sweepLine('sweeps:\n  - name: axb\n', 'a.b')).toBeNull()
  })
  it('shows every setting the run carries, what it set, and the value an unset knob ran at', () => {
    const r = unpack({ ...FIXTURE, defaults: { max_num_seqs: 256 } })[0]
    const cfg = resolvedConfig(r, { max_model_len: 262144 }, FIXTURE.keys)
    const m = Object.fromEntries(cfg)
    expect(m.max_model_len).toBe('262144')            // shared by every run
    expect(m.tp).toBe('4')                            // set by this run
    expect(m.max_num_seqs).toBe('64')                 // set, differs from the default
    expect(cfg.map(([k]) => k)).not.toContain('prefill_tp')   // unset, no known default: not invented
  })
  it('names an unset knob with the default the report knows', () => {
    const r = unpack({ ...FIXTURE, defaults: { prefill_tp: 8 } })[0]
    expect(Object.fromEntries(resolvedConfig(r, {}, FIXTURE.keys)).prefill_tp).toBe('8 (default)')
  })
})

describe('lexicon wording', () => {
  it('has no sessions left for the load, in the shared lexicon', async () => {
    const lex = JSON.stringify((await import('../lexicon.json')).default)
    expect(lex).not.toMatch(/adding sessions|session count/)
  })
})
