// The one place a report says how to install the exact stack that priced it.
import { createContext, useContext, useEffect, useRef, useState } from 'react'
import type { Setup } from './types'
import { CopyButton } from './CopyButton'

const GH = 'https://github.com/inference-sim'
const SETUP_DOC = `${GH}/blis-config-search/blob/main/SETUP.md`

/** Opens the report's setup dialog from anywhere on the page. */
export const OpenSetup = createContext<(() => void) | null>(null)

export function SetupLink({ children = 'set up the pinned stack' }: { children?: React.ReactNode }) {
  const open = useContext(OpenSetup)
  if (!open) return null
  return <button type="button" className="linkbtn" onClick={open}>{children}</button>
}

export function useSetupDialog(setup: Setup | null | undefined) {
  const [open, setOpen] = useState(false)
  const dialog = setup ? <SetupDialog setup={setup} open={open} onClose={() => setOpen(false)} /> : null
  return { open: setup ? () => setOpen(true) : null, dialog }
}

function SetupDialog({ setup, open, onClose }: { setup: Setup; open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
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
        <h3 id="setup-h">Set up the stack that priced this report</h3>
        <button type="button" className="linkbtn" onClick={onClose} aria-label="close">close</button>
      </div>
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
    </dialog>
  )
}
