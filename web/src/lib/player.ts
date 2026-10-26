// Session player engine: timing, cues and clips, no React. Views subscribe to its state.
import { flattenSession, pickCue, planCues, restClipWindow, totalSeconds, type Rng } from './logic'
import type { AudioDeps } from './audio'
import type { Direction, Session, Settings, Step } from './types'

export type PlayerState = {
  steps: Step[]
  i: number
  running: boolean
  remainingMs: number
  elapsed: number // seconds of planned time already covered
  total: number
  cue: Direction | null
  cueKey: number
  finished: boolean
  siuu: boolean
  startedAt: number
  cuesCalled: number
  drillsDone: number
}

export type PlayerOptions = {
  audio: AudioDeps
  getSettings: () => Settings
  now?: () => number
  rng?: Rng
  onFinish?: (s: PlayerState) => void
}

export type Player = ReturnType<typeof createPlayer>

const TICK_MS = 150
const CUE_SHOW_MS = 1100

export function createPlayer(session: Session, opts: PlayerOptions) {
  const { audio, getSettings, onFinish } = opts
  const now = opts.now ?? (() => Date.now())
  const rng = opts.rng ?? Math.random
  const steps = flattenSession(session)

  let state: PlayerState = {
    steps, i: -1, running: false, remainingMs: 0, elapsed: 0, total: totalSeconds(steps),
    cue: null, cueKey: 0, finished: false, siuu: false, startedAt: now(), cuesCalled: 0, drillsDone: 0,
  }
  let endAt = 0
  let lastBeepSec: number | null = null
  let tick: ReturnType<typeof setInterval> | null = null
  let timers: ReturnType<typeof setTimeout>[] = []
  const cueHistory: Direction[] = []
  const drillsSeen = new Set<number>()
  const listeners = new Set<() => void>()

  const set = (patch: Partial<PlayerState>) => {
    state = { ...state, ...patch }
    listeners.forEach((l) => l())
  }
  const clearTimers = () => { timers.forEach(clearTimeout); timers = [] }
  const later = (ms: number, fn: () => void) => { timers.push(setTimeout(fn, ms)) }

  function elapsedAt(i: number, remainingMs: number) {
    let done = 0
    for (let k = 0; k < i; k++) done += steps[k].seconds
    return done + steps[i].seconds - Math.max(0, remainingMs / 1000)
  }

  function scheduleCues(seconds: number, firstDelay: number) {
    const s = getSettings()
    planCues(seconds, s.cueMin, s.cueMax, rng, firstDelay, 2).forEach((t) => later(t * 1000, fireCue))
  }

  function fireCue() {
    if (!state.running) return
    const dir = pickCue(cueHistory, rng, 3)
    cueHistory.push(dir)
    const key = state.cueKey + 1
    set({ cue: dir, cueKey: key, cuesCalled: state.cuesCalled + 1 })
    audio.say(dir)
    setTimeout(() => { if (state.cueKey === key) set({ cue: null }) }, CUE_SHOW_MS)
  }

  // "1v1 reaction (voice cues)" is read as "1v1 reaction"
  const spoken = (name: string) => name.replace(/\s*\([^)]*\)/g, '')

  function announce(i: number) {
    const s = steps[i]
    if (s.type === 'rest') {
      const nxt = steps[i + 1]
      audio.say('Rest.' + (nxt ? ' Next: ' + spoken(nxt.name) : ''))
    } else {
      audio.say(spoken(s.name) + (s.rounds > 1 ? `. Round ${s.round} of ${s.rounds}` : ''))
    }
  }

  function goto(i: number) {
    if (state.finished) return
    clearTimers()
    audio.stopClip()
    if (i >= steps.length) return finish()
    i = Math.max(0, i)
    const s = steps[i]
    if (s.type === 'work' && s.kind === 'drill') drillsSeen.add(s.blockIdx)
    const remainingMs = s.seconds * 1000
    endAt = now() + remainingMs
    lastBeepSec = null
    set({ i, running: true, remainingMs, elapsed: elapsedAt(i, remainingMs), cue: null })
    announce(i)
    if (s.type === 'work' && s.cue) scheduleCues(s.seconds, 3)
    const st = getSettings()
    if (s.type === 'rest' && st.clipRest && audio.hasClips('mindset')) {
      const w = restClipWindow(s.seconds, st.restClipChance, rng)
      if (w) {
        later(w.start * 1000, () => { if (state.running) audio.playClip('mindset') })
        later(w.stop * 1000, audio.stopClip)
      }
    }
  }

  function update() {
    if (state.finished || !state.running) return
    const s = steps[state.i]
    const rem = endAt - now()
    const sec = Math.ceil(rem / 1000)
    if (sec <= 3 && sec >= 1 && s.seconds > 3 && lastBeepSec !== sec) {
      lastBeepSec = sec
      audio.beep(880, 120)
    }
    if (rem <= 0) {
      audio.beep(1320, 350)
      goto(state.i + 1)
      return
    }
    set({ remainingMs: rem, elapsed: elapsedAt(state.i, rem) })
  }

  function togglePause() {
    if (state.finished) return
    if (state.running) {
      clearTimers()
      audio.stopClip()
      set({ running: false, remainingMs: endAt - now(), cue: null })
    } else {
      endAt = now() + state.remainingMs
      set({ running: true })
      const s = steps[state.i]
      if (s.type === 'work' && s.cue) scheduleCues(state.remainingMs / 1000, 2)
    }
  }

  function finish() {
    if (tick) clearInterval(tick)
    clearTimers()
    audio.stopClip()
    const st = getSettings()
    const siuu = st.siuuEnd && audio.hasClips('siuu')
    if (siuu) audio.playClip('siuu')
    else audio.say('Session complete. Nice work.')
    set({ finished: true, running: false, siuu, cue: null, elapsed: state.total, remainingMs: 0, drillsDone: drillsSeen.size })
    onFinish?.(state)
  }

  return {
    getState: () => state,
    subscribe(fn: () => void) {
      listeners.add(fn)
      return () => { listeners.delete(fn) }
    },
    start() {
      goto(0)
      tick = setInterval(update, TICK_MS)
    },
    togglePause,
    skip: () => goto(state.i + 1),
    back: () => goto(state.i - 1),
    stop() {
      if (tick) clearInterval(tick)
      clearTimers()
      audio.stopClip()
      try { speechSynthesis.cancel() } catch { /* ignore */ }
    },
  }
}
