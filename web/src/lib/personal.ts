// Peter's private images live in /public/personal (git-ignored). The public build has none,
// so every use must work without them: this hook returns null when the file is missing.
import { useEffect, useState } from 'react'
import { IMAGE_DIR } from './demo'

const known = new Map<string, boolean>()

export function usePersonalImage(name: string): string | null {
  const src = `${import.meta.env.BASE_URL}${IMAGE_DIR}/${name}`
  const [ok, setOk] = useState(known.get(src) ?? false)
  useEffect(() => {
    if (known.has(src)) { setOk(known.get(src)!); return }
    const img = new Image()
    img.onload = () => { known.set(src, true); setOk(true) }
    img.onerror = () => { known.set(src, false) }
    img.src = src
  }, [src])
  return ok ? src : null
}
