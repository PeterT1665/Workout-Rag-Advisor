// AirPods / earbud and lock-screen controls during a session (Media Session API).
// iOS only sends earbud presses to a page that is playing audio, so a silent looping track plays while a
// session runs: one press is the session's main action (Set done in a gym set, otherwise pause/resume),
// a double press skips, a triple press goes back. The track keeps playing so every press arrives the same way.
// It must start inside a tap (Kick off), so begin() is called there and the player is attached later.

type Controls = { press: () => void; next: () => void; prev: () => void }

let el: HTMLAudioElement | null = null
let url: string | null = null
let controls: Controls | null = null

// 1 second of 8 kHz 8-bit mono silence as a WAV.
function silentWav(): string {
  const n = 8000
  const b = new Uint8Array(44 + n)
  const v = new DataView(b.buffer)
  const s = (o: number, t: string) => [...t].forEach((c, i) => (b[o + i] = c.charCodeAt(0)))
  s(0, 'RIFF'); v.setUint32(4, 36 + n, true); s(8, 'WAVE'); s(12, 'fmt ')
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true)
  v.setUint32(24, 8000, true); v.setUint32(28, 8000, true); v.setUint16(32, 1, true); v.setUint16(34, 8, true)
  s(36, 'data'); v.setUint32(40, n, true); b.fill(128, 44)
  return URL.createObjectURL(new Blob([b], { type: 'audio/wav' }))
}

const ms = () => (typeof navigator !== 'undefined' && 'mediaSession' in navigator ? navigator.mediaSession : null)

function setState(playing: boolean) {
  const m = ms()
  if (m) m.playbackState = playing ? 'playing' : 'paused'
  if (!el) return
  if (playing) void el.play().catch(() => {})
  else el.pause()
}

export function beginRemote(title: string) {
  const m = ms()
  if (!m || typeof Audio === 'undefined' || typeof URL.createObjectURL !== 'function') return
  url ??= silentWav()
  el ??= Object.assign(new Audio(url), { loop: true })
  void el.play().catch(() => {})
  try {
    m.metadata = new MediaMetadata({ title, artist: 'Football Personal Coach', artwork: [{ src: '/icon-512.png', sizes: '512x512', type: 'image/png' }] })
  } catch { /* MediaMetadata missing */ }
  // iOS sends play or pause depending on what it thinks is playing; both mean "the main action" here
  const press = () => { controls?.press(); setState(true) }
  const handlers: [MediaSessionAction, MediaSessionActionHandler][] = [
    ['play', press],
    ['pause', press],
    ['nexttrack', () => controls?.next()],
    ['previoustrack', () => controls?.prev()],
  ]
  for (const [a, h] of handlers) { try { m.setActionHandler(a, h) } catch { /* unsupported action */ } }
  m.playbackState = 'playing'
}

// Connect the running player.
export function attachRemote(c: Controls) { controls = c }

export function endRemote() {
  controls = null
  el?.pause()
  const m = ms()
  if (!m) return
  for (const a of ['play', 'pause', 'nexttrack', 'previoustrack'] as MediaSessionAction[]) { try { m.setActionHandler(a, null) } catch { /* ignore */ } }
  m.metadata = null
  m.playbackState = 'none'
}
