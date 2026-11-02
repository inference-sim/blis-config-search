import { configLabel } from '../lib'

/** A config as the dashboard card shows it: the topology large, the knob variant under it. */
export function ConfigName({ label, variant, reserve }: { label: string; variant?: string; reserve?: boolean }) {
  return (
    <span className="cfgname">
      <span className="win">{configLabel(label)}</span>
      {variant ? <span className="winsub">{variant}</span> : reserve && <span className="winsub-blank" aria-hidden="true">&nbsp;</span>}
    </span>
  )
}
