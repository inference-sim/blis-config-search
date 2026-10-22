import type { ReactNode } from 'react'
import { dash, fnum, sortRows, useSort } from '../lib'

export interface Col<T> {
  key: string
  label: string
  /** sortable/raw value */
  get: (r: T) => string | number | null | undefined
  render?: (r: T) => ReactNode
  num?: boolean
}

export function CatalogTable<T>({ rows, cols, rowKey, caption }: { rows: T[]; cols: Col<T>[]; rowKey: (r: T) => string; caption: string }) {
  const { sort, toggle } = useSort()
  const getter = (r: T, k: string) => cols.find((c) => c.key === k)?.get(r)
  const sorted = sortRows(rows, sort, getter)
  return (
    <div className="tscroll hscroll">
      <table className="readout">
        <caption className="sr">{caption}</caption>
        <thead>
          <tr className="col">
            {cols.map((c) => (
              <th key={c.key} aria-sort={sort?.key === c.key ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}>
                <button type="button" className="sortbtn" onClick={() => toggle(c.key)}>
                  {c.label}
                  <span className="arrow">{sort?.key === c.key ? (sort.dir === 1 ? '▲' : '▼') : ''}</span>
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={rowKey(r)}>
              {cols.map((c) => {
                const v = c.get(r)
                const content = c.render ? c.render(r) : typeof v === 'number' ? fnum(v, 3) : (v ?? dash)
                return <td key={c.key} className={c.num ? 'num' : undefined}>{c.num ? <span className="num">{content}</span> : content}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
