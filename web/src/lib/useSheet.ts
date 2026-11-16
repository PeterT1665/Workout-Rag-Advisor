// For bottom sheets and dialogs: the page behind doesn't scroll (iOS drags it otherwise), and Escape closes.
import { useEffect, useRef } from 'react'

let locks = 0
export function useSheet(onClose?: () => void) {
  const close = useRef(onClose)
  useEffect(() => { close.current = onClose })
  useEffect(() => {
    const root = document.documentElement
    if (locks++ === 0) root.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close.current?.() }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      if (--locks === 0) root.style.overflow = ''
    }
  }, [])
}
