// Image slots: the user's own uploaded photo wins; otherwise the build's folder (licensed demo photos, or Peter's
// private /public/personal in local builds). Every use must work without either: the hook returns null then.
import { useEffect, useState } from 'react'
import { IMAGE_DIR } from './demo'
import { usePhotos, type Slot } from './photos'

const known = new Map<string, boolean>()

export function usePersonalImage(name: string): string | null {
  const own = usePhotos()[name.replace(/\.jpg$/, '') as Slot]
  const src = `${import.meta.env.BASE_URL}${IMAGE_DIR}/${name}`
  const [ok, setOk] = useState(known.get(src) ?? false)
  useEffect(() => {
    if (known.has(src)) { setOk(known.get(src)!); return }
    const img = new Image()
    img.onload = () => { known.set(src, true); setOk(true) }
    img.onerror = () => { known.set(src, false) }
    img.src = src
  }, [src])
  return own ?? (ok ? src : null)
}
