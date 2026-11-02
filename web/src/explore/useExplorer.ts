// Explorer state: the crossfilter selection, the ranking target, what bars measure, and a picked config.
import { useMemo, useState } from 'react'
import type { ExploreData, MetricKey, Run } from './types'
import {
  configFilters, fieldValue, groupTies, liveFields, matches, mostInfluential, rankConfigs, toggle, unpack, usualValues, type Selection,
} from './model'

export type BarMeasure = 'best' | 'pass' | 'count'

export interface Explorer {
  data: ExploreData
  runs: Run[]
  fields: string[]
  usual: ReturnType<typeof usualValues>
  target: MetricKey
  setTarget: (m: MetricKey) => void
  bar: BarMeasure
  setBar: (b: BarMeasure) => void
  sel: Selection
  okOnly: boolean
  setOkOnly: (v: boolean) => void
  /** Runs in the current filter, optionally ignoring one field (for that field's own counts). */
  shown: (skip?: string) => Run[]
  toggleValue: (k: string, v: string) => void
  setField: (k: string, vals: string[]) => void
  clearField: (k: string) => void
  reset: () => void
  /** The picked config, by Run.cfg; picking it again restores the filters from before. */
  picked: string | null
  pick: (cfg: string) => void
  fv: (r: Run, k: string) => string
  ranked: ReturnType<typeof groupTies>
  /** The field the overview colors its points by. */
  colorBy: string | null
  setColorBy: (k: string) => void
  /** The config under the pointer on any chart, shared so every chart lights it up together. */
  hover: string | null
  setHover: (cfg: string | null) => void
  /** Undo what the charts set: the pick, the serving-mode filter and the SLO-only toggle. */
  resetCharts: () => void
}

export function useExplorer(data: ExploreData): Explorer {
  const runs = useMemo(() => unpack(data), [data])
  const fields = useMemo(() => liveFields(runs, data.keys, data.routers), [runs, data])
  const usual = useMemo(() => usualValues(runs, fields, data.routers), [runs, fields, data])
  const [target, setTarget] = useState<MetricKey>('tpg')
  const [bar, setBar] = useState<BarMeasure>('best')
  const [sel, setSel] = useState<Selection>({})
  const [okOnly, setOkOnly] = useState(false)
  const [picked, setPicked] = useState<string | null>(null)
  const [pickKeys, setPickKeys] = useState<string[]>([])
  // The selection as it was before the pick: what unpicking restores and what the cards rank on.
  const [beforePick, setBeforePick] = useState<Selection | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  const auto = useMemo(() => mostInfluential(runs, fields, target, data.routers), [runs, fields, target, data.routers])
  const [colorChoice, setColorBy] = useState<string | null>(null)
  const colorBy = colorChoice && fields.includes(colorChoice) ? colorChoice : auto

  const fv = (r: Run, k: string) => fieldValue(r, k, data.routers)
  const shown = (skip?: string) => runs.filter((r) => (!okOnly || r.ok) && matches(r, sel, data.routers, skip))

  const clearPick = () => { setPicked(null); setPickKeys([]); setBeforePick(null) }
  // Editing a field the pick set detaches the pick; the remaining filters stay.
  const detach = (k: string) => { if (picked && pickKeys.includes(k)) clearPick() }
  const toggleValue = (k: string, v: string) => { detach(k); setSel((s) => toggle(s, k, v)) }
  const setField = (k: string, vals: string[]) => {
    detach(k)
    setSel((s) => { const o = { ...s }; if (vals.length) o[k] = vals; else delete o[k]; return o })
  }
  const clearField = (k: string) => setField(k, [])
  const reset = () => { setSel({}); setOkOnly(false); clearPick() }
  const resetCharts = () => {
    const base = { ...(beforePick ?? sel) }
    if (colorBy) delete base[colorBy]
    setSel(base); setOkOnly(false); clearPick(); setHover(null)
  }

  const pick = (cfg: string) => {
    const base = beforePick ?? sel
    if (picked === cfg) { setSel(base); clearPick(); return }
    const r = runs.find((x) => x.cfg === cfg)
    if (!r) return
    const f = configFilters(r, fields, data.routers)
    setBeforePick(base)
    setSel({ ...base, ...f })
    setPicked(cfg)
    setPickKeys(Object.keys(f))
  }

  // The cards rank on the selection from before the pick, so they stay put while one is picked.
  const rankSel = beforePick ?? sel
  const ranked = useMemo(
    () => groupTies(rankConfigs(runs.filter((r) => (!okOnly || r.ok) && matches(r, rankSel, data.routers)), target)),
    [runs, rankSel, okOnly, target, data.routers],
  )

  return {
    data, runs, fields, usual, target, setTarget, bar, setBar, sel, okOnly, setOkOnly,
    shown, toggleValue, setField, clearField, reset, picked, pick, fv, ranked, hover, setHover, resetCharts, colorBy, setColorBy,
  }
}
