// Image slots: the user's own uploaded photo wins; otherwise the build's folder (licensed demo photos, or Peter's
// private /public/personal in local builds). Every use must work without either: the hook returns null then.
// A failed load is retried (bad signal at the field, app waking up) instead of hiding the photo for the session.
import { useEffect, useState } from 'react'
import { IMAGE_DIR } from './demo'
import { usePhotos, type Slot } from './photos'

const loaded = new Set<string>()
const inflight = new Map<string, Promise<boolean>>()
const RETRY_MS = [1500, 5000, 15000]

function probe(src: string): Promise<boolean> {
  if (loaded.has(src)) return Promise.resolve(true)
  let p = inflight.get(src)
  if (!p) {
    p = new Promise<boolean>((res) => {
      const img = new Image()
      img.onload = () => { loaded.add(src); res(true) }
      img.onerror = () => res(false)
      img.src = src
    }).finally(() => inflight.delete(src))
    inflight.set(src, p)
  }
  return p
}

export function usePersonalImage(name: string): string | null {
  const own = usePhotos()[name.replace(/\.jpg$/, '') as Slot]
  const src = `${import.meta.env.BASE_URL}${IMAGE_DIR}/${name}`
  const [ok, setOk] = useState(() => loaded.has(src))
  useEffect(() => {
    if (loaded.has(src)) { setOk(true); return }
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const attempt = (n: number) => {
      void probe(src).then((good) => {
        if (!alive) return
        if (good) setOk(true)
        else if (n < RETRY_MS.length) timer = setTimeout(() => attempt(n + 1), RETRY_MS[n])
      })
    }
    attempt(0)
    // coming back to the app or back online: try again straight away
    const again = () => { if (document.visibilityState === 'visible' && !loaded.has(src)) attempt(RETRY_MS.length) }
    document.addEventListener('visibilitychange', again)
    window.addEventListener('online', again)
    return () => { alive = false; clearTimeout(timer); document.removeEventListener('visibilitychange', again); window.removeEventListener('online', again) }
  }, [src])
  return own ?? (ok ? src : null)
}
