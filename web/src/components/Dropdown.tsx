/**
 * A small styled dropdown that matches the page design — the native <select> open list is painted
 * by the browser and can't be themed, so we render our own button + popup listbox (jgchn review).
 * Keyboard: ↑/↓ move, Enter/Space select, Esc closes; click-outside dismisses. ARIA listbox roles.
 */
import { useEffect, useId, useRef, useState } from 'react'

export interface DropdownOption {
  value: string
  label: string
}

interface Props {
  value: string
  options: DropdownOption[]
  onChange: (value: string) => void
  /** Extra class on the trigger button (e.g. 'vllmsel' | 'runsel') so existing sizing rules apply. */
  className?: string
  /** Disable interaction (e.g. a single-run picker); the current value still shows. */
  disabled?: boolean
  ariaLabel?: string
}

export function Dropdown({ value, options, onChange, className = '', disabled = false, ariaLabel }: Props) {
  const [open, setOpen] = useState(false)
  // The option the keyboard is highlighting while open (not yet committed).
  const [active, setActive] = useState(0)
  const rootRef = useRef<HTMLDivElement>(null)
  const listId = useId()

  const selectedIndex = Math.max(0, options.findIndex((o) => o.value === value))
  const current = options[selectedIndex] ?? options[0]

  // Close on any click outside the component.
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const openList = () => {
    if (disabled) return
    setActive(selectedIndex)
    setOpen(true)
  }
  const commit = (i: number) => {
    const opt = options[i]
    if (opt) onChange(opt.value)
    setOpen(false)
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return
    if (!open) {
      if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        openList()
      }
      return
    }
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setActive((a) => Math.min(options.length - 1, a + 1))
        break
      case 'ArrowUp':
        e.preventDefault()
        setActive((a) => Math.max(0, a - 1))
        break
      case 'Enter':
      case ' ':
        e.preventDefault()
        commit(active)
        break
      case 'Escape':
        e.preventDefault()
        setOpen(false)
        break
      case 'Tab':
        setOpen(false)
        break
    }
  }

  return (
    <div className={`dd${open ? ' open' : ''}${disabled ? ' disabled' : ''}`} ref={rootRef}>
      <button
        type="button"
        className={`dd-trigger box ${className}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        disabled={disabled}
        onClick={() => (open ? setOpen(false) : openList())}
        onKeyDown={onKeyDown}
      >
        <span className="dd-label">{current?.label ?? ''}</span>
        <span className="dd-car" aria-hidden="true">
          ▾
        </span>
      </button>
      {open && (
        <ul className="dd-list" role="listbox" id={listId} tabIndex={-1}>
          {options.map((o, i) => (
            <li
              key={o.value}
              role="option"
              aria-selected={o.value === value}
              className={`dd-opt${i === active ? ' active' : ''}${o.value === value ? ' sel' : ''}`}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                // mousedown (not click) so it fires before the outside-click handler closes us.
                e.preventDefault()
                commit(i)
              }}
            >
              {o.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
