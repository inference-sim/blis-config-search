// Shared chart frame: one viewBox and margin for every SVG chart, plus the axis and controls they share.
import type { ReactNode } from 'react'
import { fmt, niceStep } from './model'

export const W = 520, H = 280, M = { l: 54, r: 12, t: 12, b: 34 }

export function YAxis({ ym, py, label }: { ym: number; py: (v: number) => number; label: string }) {
  const st = niceStep(ym)
  const ticks: number[] = []
  for (let v = 0; v <= ym + 1e-9; v += st) ticks.push(v)
  return (
    <g>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={M.l} x2={W - M.r} y1={py(v)} y2={py(v)} className="grid" />
          <text x={M.l - 5} y={py(v) + 3} textAnchor="end" className="tick">{v >= 10000 ? `${fmt(v / 1000, 0)}k` : fmt(v, v < 10 && st < 1 ? 1 : 0)}</text>
        </g>
      ))}
      <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} className="axis" />
      <text transform={`translate(11 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="axlabel">{label}</text>
    </g>
  )
}

/** A full-plot backdrop: clicking it (empty space) resets the charts. */
export function Backdrop({ onClick }: { onClick: () => void }) {
  return <rect className="xf-backdrop" x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} onClick={onClick} />
}

/** A legend entry that does something: filters, toggles, or hovers a config. */
export function LegendBtn({ on, off, onClick, onHover, children, label }: {
  on?: boolean; off?: boolean; onClick: () => void; onHover?: (h: boolean) => void; children: ReactNode; label: string
}) {
  return (
    <button type="button" className={`xf-lg${on ? ' on' : ''}${off ? ' off' : ''}`} aria-pressed={!!on} aria-label={label} onClick={onClick}
      onPointerEnter={() => onHover?.(true)} onPointerLeave={() => onHover?.(false)} onFocus={() => onHover?.(true)} onBlur={() => onHover?.(false)}>
      {children}
    </button>
  )
}
