// Speech, beeps and clip playback. iOS only allows audio after a tap, so unlockAudio()
// must be called from a click handler before anything plays.
import { pickNoRepeat } from './logic'
import { clipsOf } from './clips'
import type { Clip, ClipKind, Settings } from './types'

let ctx: AudioContext | null = null

export function unlockAudio() {
  try {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    ctx = ctx || new AC()
    if (ctx.state === 'suspended') void ctx.resume()
  } catch {
    ctx = null
  }
  try { speechSynthesis.cancel(); speechSynthesis.resume() } catch { /* ignore */ }
}

export function beep(s: Settings, freq: number, ms: number) {
  if (!ctx || !s.voice) return
  try {
    const o = ctx.createOscillator()
    const g = ctx.createGain()
    o.frequency.value = freq
    g.gain.value = 0.25 * s.volume
    o.connect(g)
    g.connect(ctx.destination)
    o.start()
    o.stop(ctx.currentTime + ms / 1000)
  } catch { /* ignore */ }
}

export function say(s: Settings, text: string, force = false) {
  if (typeof speechSynthesis === 'undefined' || (!s.voice && !force)) return
  try {
    speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.lang = 'en-GB'
    u.rate = 1.05
    u.volume = s.volume
    speechSynthesis.speak(u)
  } catch { /* ignore */ }
}

const clipAudio: HTMLAudioElement | null = typeof Audio !== 'undefined' ? new Audio() : null
let clipUrl: string | null = null
let lastClip: string | null = null

export function stopClip() {
  if (!clipAudio) return
  clipAudio.onended = null
  try { clipAudio.pause() } catch { /* ignore */ }
  if (clipUrl) { URL.revokeObjectURL(clipUrl); clipUrl = null }
}

export function playBlob(s: Settings, blob: Blob, onEnd?: () => void) {
  if (!clipAudio) { onEnd?.(); return }
  stopClip()
  clipUrl = URL.createObjectURL(blob)
  clipAudio.src = clipUrl
  clipAudio.volume = s.clipVolume
  clipAudio.onended = () => onEnd?.()
  const p = clipAudio.play()
  if (p && p.catch) p.catch(() => onEnd?.())
}

// Plays a random clip of that kind (no immediate repeats). Returns the clip, or null if none.
export function playClip(s: Settings, kind: ClipKind, onEnd?: () => void): Clip | null {
  const clip = pickNoRepeat(clipsOf(kind), lastClip, Math.random, (c) => c.id)
  if (!clip) { onEnd?.(); return null }
  lastClip = clip.id
  playBlob(s, clip.blob, onEnd)
  return clip
}

export type AudioDeps = {
  say: (text: string) => void
  beep: (freq: number, ms: number) => void
  playClip: (kind: ClipKind, onEnd?: () => void) => Clip | null
  stopClip: () => void
  hasClips: (kind: ClipKind) => boolean
}

export const realAudio = (getSettings: () => Settings): AudioDeps => ({
  say: (t) => say(getSettings(), t),
  beep: (f, ms) => beep(getSettings(), f, ms),
  playClip: (k, onEnd) => playClip(getSettings(), k, onEnd),
  stopClip,
  hasClips: (k) => clipsOf(k).length > 0,
})
