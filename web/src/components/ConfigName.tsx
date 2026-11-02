import { configLabel } from '../lib'

/** A config as the dashboard card shows it: the topology large, the knob variant under it. */
export function ConfigName({ label, variant }: { label: string; variant?: string }) {
  return (
    <span className="cfgname">
      <span className="win">{configLabel(label)}</span>
      {variant && <span className="winsub">{variant}</span>}
    </span>
  )
}
