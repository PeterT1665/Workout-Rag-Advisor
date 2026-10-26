import { useSyncExternalStore } from 'react'

// True when the media query matches. Defaults to false (phone layout) where matchMedia is missing.
export function useMedia(query: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      if (typeof matchMedia === 'undefined') return () => {}
      const m = matchMedia(query)
      m.addEventListener('change', cb)
      return () => m.removeEventListener('change', cb)
    },
    () => (typeof matchMedia === 'undefined' ? false : matchMedia(query).matches),
    () => false,
  )
}

export const DESKTOP_QUERY = '(min-width: 1024px)'
