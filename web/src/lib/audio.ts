// Speech and beeps. iOS only allows audio after a tap, so unlockAudio()
// must be called from a click handler before anything plays.
import type { Settings } from './types'
import { voiceLang } from '../i18n'

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
    u.lang = voiceLang() // the app's language; iPhone picks its voice for it
    const v = pickVoice(u.lang)
    if (v) u.voice = v
    u.rate = 1.05
    u.volume = s.volume
    speechSynthesis.speak(u)
  } catch { /* ignore */ }
}

// Best installed voice for the language (an exact match like vi-VN first, then any voice of that language).
function pickVoice(lang: string): SpeechSynthesisVoice | undefined {
  try {
    const all = speechSynthesis.getVoices()
    return all.find((v) => v.lang.replace('_', '-') === lang) ?? all.find((v) => v.lang.slice(0, 2) === lang.slice(0, 2))
  } catch { return undefined }
}

export type AudioDeps = {
  say: (text: string) => void
  beep: (freq: number, ms: number) => void
}

export const realAudio = (getSettings: () => Settings): AudioDeps => ({
  say: (t) => say(getSettings(), t),
  beep: (f, ms) => beep(getSettings(), f, ms),
})
