import { useState } from 'react'

export function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  const copy = () => navigator.clipboard?.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1500) }).catch(() => {})
  return <button type="button" className="linkbtn" onClick={copy}>{done ? 'copied' : 'copy'}</button>
}
