import { useState } from 'react'

const Clip = () => <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><rect x="5" y="5" width="8.5" height="8.5" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M10.5 3.5v-.5A1.5 1.5 0 0 0 9 1.5H4A1.5 1.5 0 0 0 2.5 3v5A1.5 1.5 0 0 0 4 9.5h.5" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
const Check = () => <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  const copy = () => navigator.clipboard?.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1500) }).catch(() => {})
  return (
    <button type="button" className={`xf-copy${done ? ' done' : ''}`} onClick={copy} aria-label={done ? 'copied' : label.toLowerCase()}>
      {done ? <Check /> : <Clip />}<span>{done ? 'Copied' : label}</span>
    </button>
  )
}

/** A code block with its copy button pinned to the top-right corner. */
export function CodeBlock({ text, className = '' }: { text: string; className?: string }) {
  return (
    <div className="xf-code">
      <CopyButton text={text} />
      <pre className={`xf-cmd-line ${className}`.trim()}>{text}</pre>
    </div>
  )
}
