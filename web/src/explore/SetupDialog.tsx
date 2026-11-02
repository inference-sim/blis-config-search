// The one place a report says how to install the stack that priced it and reproduce its sweep.
import { createContext, useContext, useEffect, useRef, useState } from 'react'
import type { Setup } from './types'
import { CodeBlock } from './CopyButton'

const GH = 'https://github.com/inference-sim'
const SETUP_DOC = `${GH}/blis-config-search/blob/main/SETUP.md`

export type SetupTab = 'install' | 'sweep'
/** Opens the report's setup dialog, on a tab, from anywhere on the page. */
export const OpenSetup = createContext<((tab?: SetupTab) => void) | null>(null)

/** Solid for the page's main call to action, icon-only when it has no text; `quiet` for a secondary pointer to it. */
export function SetupButton({ children, tab = 'install', quiet = false }: { children?: React.ReactNode; tab?: SetupTab; quiet?: boolean }) {
  const open = useContext(OpenSetup)
  if (!open) return null
  const cls = ['xf-setupbtn', quiet && 'quiet', !children && 'icon'].filter(Boolean).join(' ')
  return (
    <button type="button" className={cls} onClick={() => open(tab)} aria-label={children ? undefined : 'Reproduce this report'} title={children ? undefined : 'Reproduce this report'}>
      {children}<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path d="M6 3.5l4.5 4.5L6 12.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </button>
  )
}

export function useSetupDialog(setup: Setup | null | undefined) {
  const [tab, setTab] = useState<SetupTab | null>(null)
  const dialog = setup ? <SetupDialog setup={setup} tab={tab} setTab={setTab} onClose={() => setTab(null)} /> : null
  return { open: setup ? (t: SetupTab = 'install') => setTab(t) : null, dialog }
}

function SetupDialog({ setup, tab, setTab, onClose }: { setup: Setup; tab: SetupTab | null; setTab: (t: SetupTab) => void; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  const open = tab !== null
  useEffect(() => {
    const d = ref.current
    if (!d) return
    // jsdom has no showModal; fall back to the open attribute
    if (open && !d.open) (d.showModal ? d.showModal() : d.setAttribute('open', ''))
    if (!open && d.open) (d.close ? d.close() : d.removeAttribute('open'))
  }, [open])
  return (
    <dialog ref={ref} className="xf-dialog" aria-labelledby="setup-h" onClose={onClose}
      onClick={(e) => { if (e.target === ref.current) onClose() }}>
      <div className="xf-dialog-h">
        <h3 id="setup-h">Reproduce this report on your machine</h3>
        <button type="button" className="xf-close" onClick={onClose} aria-label="close"><svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg></button>
      </div>
      <div className="xf-seg" role="tablist" aria-label="Step">
        <button type="button" role="tab" aria-selected={tab !== 'sweep'} aria-pressed={tab !== 'sweep'} onClick={() => setTab('install')}>1. Install the pinned stack</button>
        <button type="button" role="tab" aria-selected={tab === 'sweep'} aria-pressed={tab === 'sweep'} onClick={() => setTab('sweep')}>2. Reproduce the experiment sweep</button>
      </div>
      {tab === 'sweep' ? <Sweep setup={setup} /> : <Install setup={setup} />}
    </dialog>
  )
}

function Install({ setup }: { setup: Setup }) {
  return (
    <>
      <p className="xf-sub">A run's numbers reproduce only on these exact versions. Run this once; every command on the page assumes it.</p>
      <table className="xf-stack">
        <tbody>
          {setup.stack.map((d) => (
            <tr key={d.repo}>
              <th>{d.name}</th>
              <td className="mono"><a href={`${GH}/${d.repo}/tree/${d.commit ?? d.ref}`} target="_blank" rel="noreferrer">{d.repo} {d.ref}</a></td>
              <td className="mono xf-sub">{d.commit ? d.commit.slice(0, 12) : 'built into BLIS'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="xf-cmd-h"><b>Install script</b></div>
      <CodeBlock text={setup.script} />
      <p className="xf-sub">Each tag is checked against the commit it had at sweep time. See <a href={SETUP_DOC} target="_blank" rel="noreferrer">SETUP.md</a> for prerequisites and troubleshooting.</p>
    </>
  )
}

function Sweep({ setup }: { setup: Setup }) {
  return (
    <>
      <p className="xf-sub">Runs this report's experiment spec through the same sweeper on your machine, with the stack from step 1 plus Python 3 and PyYAML. The dry run prints the grid and run count without calling BLIS.</p>
      <p className="xf-note">This only reproduces the numbers locally, into <span className="mono">repro/</span>; the report on this site is unchanged.{setup.runs ? ` ${setup.runs.toLocaleString('en-US')} BLIS runs${setup.hours ? `, about ${setup.hours} h serial` : ''}.` : ''}</p>
      <div className="xf-cmd-h"><b>Sweep commands</b></div>
      <CodeBlock text={setup.sweep} />
      <p className="xf-sub">Same spec, seed and stack give the same numbers as the committed ledger. To reproduce one config, use its command in the Runs table.</p>
    </>
  )
}
