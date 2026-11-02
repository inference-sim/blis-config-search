// The one place a report says how to install the stack that priced it and rerun its sweep.
import { createContext, useContext, useEffect, useRef, useState } from 'react'
import type { Setup } from './types'
import { CopyButton } from './CopyButton'

const GH = 'https://github.com/inference-sim'
const SETUP_DOC = `${GH}/blis-config-search/blob/main/SETUP.md`

export type SetupTab = 'install' | 'sweep'
/** Opens the report's setup dialog, on a tab, from anywhere on the page. */
export const OpenSetup = createContext<((tab?: SetupTab) => void) | null>(null)

export function SetupButton({ tab = 'install', children }: { tab?: SetupTab; children: React.ReactNode }) {
  const open = useContext(OpenSetup)
  if (!open) return null
  return <button type="button" className="xf-setupbtn" onClick={() => open(tab)}>{children}</button>
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
        <h3 id="setup-h">Reproduce this report</h3>
        <button type="button" className="linkbtn" onClick={onClose} aria-label="close">close</button>
      </div>
      <div className="xf-seg" role="tablist" aria-label="Step">
        <button type="button" role="tab" aria-selected={tab !== 'sweep'} aria-pressed={tab !== 'sweep'} onClick={() => setTab('install')}>1. Install the pinned stack</button>
        <button type="button" role="tab" aria-selected={tab === 'sweep'} aria-pressed={tab === 'sweep'} onClick={() => setTab('sweep')}>2. Rerun the whole sweep</button>
      </div>
      {tab === 'sweep' ? <Sweep setup={setup} /> : <Install setup={setup} />}
    </dialog>
  )
}

function Sweep({ setup }: { setup: Setup }) {
  return (
    <>
      <p className="xf-sub">Runs this report's spec through the deterministic sweeper on the stack from step 1, with Python 3 and PyYAML. The dry run prints the grid and run count without calling BLIS.</p>
      {setup.runs ? <p className="xf-note">{setup.runs.toLocaleString('en-US')} BLIS runs{setup.hours ? `, about ${setup.hours} h serial when this report was swept` : ''}. Runs go one at a time; to rerun a single config, use its command in the Runs table.</p> : null}
      <div className="xf-cmd-h"><b>Sweep commands</b><CopyButton text={setup.sweep} /></div>
      <pre className="xf-cmd-line">{setup.sweep}</pre>
      <p className="xf-sub">Same seed and stack, same numbers: compare the rerun with the committed ledger. See <a href={SETUP_DOC} target="_blank" rel="noreferrer">SETUP.md</a> for a smoke test of one run.</p>
    </>
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
      <div className="xf-cmd-h"><b>Install script</b><CopyButton text={setup.script} /></div>
      <pre className="xf-cmd-line">{setup.script}</pre>
      <p className="xf-sub">Each tag is checked against the commit it had at sweep time. See <a href={SETUP_DOC} target="_blank" rel="noreferrer">SETUP.md</a> for prerequisites and troubleshooting.</p>
    </>
  )
}
