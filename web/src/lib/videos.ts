// Video instructions per exercise. For now: links each player saves for themselves (TikTok, YouTube,
// Instagram), private to them and synced as 'videos'. Not shared or published: the creators own them.
// Later the shared exercise library carries licensed videos (docs/system-design.md).
import { useSyncExternalStore } from 'react'
import { load, save } from './storage'

type Videos = Record<string, string> // exercise name (lower case) -> link

const keyOf = (name: string) => name.replace(/\s*\([^)]*\)/g, '').trim().toLowerCase()
let cache: Videos = load<Videos>('videos', {})
const listeners = new Set<() => void>()
const emit = () => listeners.forEach((l) => l())
if (typeof window !== 'undefined') window.addEventListener('ta-sync', () => { cache = load<Videos>('videos', {}); emit() })

export const videoFor = (name: string): string | undefined => cache[keyOf(name)]
export function setVideo(name: string, url: string | null) {
  const next = { ...load<Videos>('videos', {}) }
  if (url) next[keyOf(name)] = url
  else delete next[keyOf(name)]
  save('videos', next)
  cache = next
  emit()
}
export function useVideo(name: string) {
  return useSyncExternalStore((fn) => { listeners.add(fn); return () => { listeners.delete(fn) } }, () => videoFor(name))
}

// A link people paste, cleaned up: must be http(s).
export function cleanLink(raw: string): string | null {
  const t = raw.trim().match(/https?:\/\/\S+/)?.[0]
  if (!t) return null
  try { return new URL(t).toString() } catch { return null }
}

// Embeddable player URL when the site allows it; otherwise null (open the link instead).
export function embedFor(url: string): { src: string; tall: boolean } | null {
  let u: URL
  try { u = new URL(url) } catch { return null }
  const host = u.hostname.replace(/^www\.|^m\./, '')
  if (host === 'tiktok.com') {
    const id = u.pathname.match(/\/video\/(\d+)/)?.[1]
    return id ? { src: `https://www.tiktok.com/embed/v2/${id}`, tall: true } : null
  }
  if (host === 'youtube.com' || host === 'youtu.be') {
    const id = host === 'youtu.be' ? u.pathname.slice(1) : u.pathname.startsWith('/shorts/') ? u.pathname.split('/')[2] : u.searchParams.get('v')
    return id ? { src: `https://www.youtube-nocookie.com/embed/${id}`, tall: u.pathname.startsWith('/shorts/') } : null
  }
  if (host === 'instagram.com') {
    const m = u.pathname.match(/\/(reel|p)\/([\w-]+)/)
    return m ? { src: `https://www.instagram.com/${m[1]}/${m[2]}/embed`, tall: true } : null
  }
  return null
}

export const searchLink = (name: string) => `https://www.tiktok.com/search?q=${encodeURIComponent(`${keyOf(name)} exercise`)}`
