// Session player engine: timing, cues and clips, no React. Views subscribe to its state.
import { isWordCue, pickWord } from './cues'
import { flattenSession, pickCue, planCues, selfPaced, totalSeconds, type Rng } from './logic'
import { t } from '../i18n'
import type { AudioDeps } from './audio'
import { editableFrom } from './logic'
import type { Block, Direction, Session, Settings, Step } from './types'

export type PlayerState = {
  steps: Step[]
  i: number
  running: boolean
  remainingMs: number
  elapsed: number // seconds of planned time already covered (skipped steps included)
  trained: number // seconds the session actually ran (not paused): what the log saves, so skipping through isn't training
  total: number
  cue: Direction | number | string | null // a number: the number drill (even: pass back, odd: turn); other strings: a word from lib/cues.ts (its id)
  cueKey: number
  finished: boolean
  startedAt: number
  cuesCalled: number
  drillsDone: number
  setMs: number // time in the current self-paced set (sets with reps)
}

export type PlayerOptions = {
  audio: AudioDeps
  getSettings: () => Settings
  now?: () => number
  rng?: Rng
  onFinish?: (s: PlayerState) => void
  resume?: Snapshot // continue a session saved before the app was closed or reloaded
  moves?: string[] // the player's signature skill moves, for drills where the app calls a move
}

// Enough to pick a session up again after a reload (see lib/activeSession.ts).
export type Snapshot = { i: number; remainingMs: number; startedAt: number; cuesCalled: number; drills: number[]; trainedMs?: number }

export type Player = ReturnType<typeof createPlayer>

// Minutes for the log and the full-time screen: time actually trained, never the time skipped past.
export const trainedMinutes = (st: PlayerState) => Math.max(1, Math.round(st.trained / 60))

const TICK_MS = 150
const CUE_SHOW_MS = 1100

export function createPlayer(session: Session, opts: PlayerOptions) {
  const { audio, getSettings, onFinish } = opts
  const now = opts.now ?? (() => Date.now())
  const rng = opts.rng ?? Math.random
  let steps = flattenSession(session)

  let state: PlayerState = {
    steps, i: -1, running: false, remainingMs: 0, elapsed: 0, trained: 0, total: totalSeconds(steps),
    cue: null, cueKey: 0, finished: false, startedAt: opts.resume?.startedAt ?? now(), cuesCalled: opts.resume?.cuesCalled ?? 0, drillsDone: 0, setMs: 0,
  }
  let endAt = 0
  let setStart = 0 // self-paced set: when it (re)started counting
  let setDone = 0 // self-paced set: time counted before a pause
  let lastBeepSec: number | null = null
  let tick: ReturnType<typeof setInterval> | null = null
  let timers: ReturnType<typeof setTimeout>[] = []
  const cueHistory: Direction[] = []
  let lastNumber = 0
  let lastWord: string | null = null
  const drillsSeen = new Set<number>(opts.resume?.drills ?? [])
  // real training time: counts while running, stops on pause
  let trainedMs = opts.resume?.trainedMs ?? 0
  let runFrom: number | null = null
  const clockOn = () => { if (runFrom === null) runFrom = now() }
  const clockOff = () => { if (runFrom !== null) { trainedMs += now() - runFrom; runFrom = null } }
  const trainedNow = () => trainedMs + (runFrom === null ? 0 : now() - runFrom)
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
    if (state.steps[state.i]?.cueKind === 'numbers') {
      let n = 1 + Math.floor(rng() * 9)
      if (n === lastNumber) n = (n % 9) + 1 // never the same number twice in a row
      lastNumber = n
      const key = state.cueKey + 1
      set({ cue: n, cueKey: key, cuesCalled: state.cuesCalled + 1 })
      audio.say(String(n))
      setTimeout(() => { if (state.cueKey === key) set({ cue: null }) }, CUE_SHOW_MS)
      return
    }
    const kind = state.steps[state.i]?.cueKind
    if (isWordCue(kind)) {
      const w = pickWord(kind, lastWord, rng, opts.moves)
      lastWord = w.id
      const key = state.cueKey + 1
      set({ cue: w.id, cueKey: key, cuesCalled: state.cuesCalled + 1 })
      audio.say(w.word)
      setTimeout(() => { if (state.cueKey === key) set({ cue: null }) }, CUE_SHOW_MS)
      return
    }
    const dir = pickCue(cueHistory, rng, 3)
    cueHistory.push(dir)
    const key = state.cueKey + 1
    set({ cue: dir, cueKey: key, cuesCalled: state.cuesCalled + 1 })
    audio.say(dir === 'left' ? t('left') : t('right'))
    setTimeout(() => { if (state.cueKey === key) set({ cue: null }) }, CUE_SHOW_MS)
  }

  // "1v1 reaction (voice cues)" is read as "1v1 reaction"
  const spoken = (name: string) => name.replace(/\s*\([^)]*\)/g, '')

  function announce(i: number) {
    const s = steps[i]
    if (s.type === 'rest') {
      const nxt = steps[i + 1]
      audio.say(nxt ? t('Rest. Next: {name}', { name: spoken(nxt.name) }) : t('Rest.'))
    } else {
      audio.say(spoken(s.name) + (s.rounds > 1 ? '. ' + t('Round {n} of {total}', { n: s.round, total: s.rounds }) : ''))
    }
  }

  function goto(i: number) {
    if (state.finished) return
    clearTimers()
    if (i >= steps.length) return finish()
    i = Math.max(0, i)
    const s = steps[i]
    if (s.type === 'work' && s.kind === 'drill') drillsSeen.add(s.blockIdx)
    const remainingMs = s.seconds * 1000
    endAt = now() + remainingMs
    setStart = now(); setDone = 0
    lastBeepSec = null
    clockOn()
    set({ i, running: true, remainingMs, elapsed: elapsedAt(i, remainingMs), cue: null, setMs: 0 })
    announce(i)
    if (s.type === 'work' && s.cue) scheduleCues(s.seconds, 3)
  }

  function update() {
    if (state.finished || !state.running) return
    const s = steps[state.i]
    if (selfPaced(s)) { set({ setMs: setDone + now() - setStart }); return } // ends on Set done, not on time
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
      const paced = selfPaced(steps[state.i])
      if (paced) setDone += now() - setStart
      clockOff()
      set({ running: false, remainingMs: paced ? state.remainingMs : endAt - now(), cue: null })
    } else {
      endAt = now() + state.remainingMs
      setStart = now()
      clockOn()
      set({ running: true })
      const s = steps[state.i]
      if (s.type === 'work' && s.cue) scheduleCues(state.remainingMs / 1000, 2)
    }
  }

  // early: ended by the player before the last drill; the time actually done counts
  function finish(early = false) {
    if (tick) clearInterval(tick)
    clearTimers()
    clockOff()
    audio.say(t('Session complete. Nice work.'))
    set({ finished: true, running: false, cue: null, elapsed: early ? state.elapsed : state.total, trained: Math.round(trainedMs / 1000), remainingMs: 0, drillsDone: drillsSeen.size })
    onFinish?.(state)
  }

  return {
    getState: () => state,
    subscribe(fn: () => void) {
      listeners.add(fn)
      return () => { listeners.delete(fn) }
    },
    start() {
      const r = opts.resume
      if (r && r.i >= 0 && r.i < steps.length) {
        // back where it was, paused: the player taps Resume when ready
        set({ i: r.i, running: false, remainingMs: r.remainingMs, elapsed: elapsedAt(r.i, r.remainingMs) })
      } else goto(0)
      tick = setInterval(update, TICK_MS)
    },
    snapshot: (): Snapshot => ({
      i: state.i, remainingMs: state.running ? Math.max(0, endAt - now()) : state.remainingMs,
      startedAt: state.startedAt, cuesCalled: state.cuesCalled, drills: [...drillsSeen], trainedMs: trainedNow(),
    }),
    endEarly: () => finish(true),
    getSession: () => session,
    // the exercises not started yet, as they stand now (reorder, swap or remove them mid-session)
    upcoming: () => session.blocks.slice(editableFrom(steps, state.i)),
    editUpcoming(blocks: Block[]) {
      if (state.finished) return
      const from = editableFrom(steps, state.i)
      session = { ...session, blocks: [...session.blocks.slice(0, from), ...blocks] }
      steps = flattenSession(session)
      if (state.i >= steps.length) { set({ steps }); return finish() }
      set({ steps, total: totalSeconds(steps), elapsed: elapsedAt(state.i, state.remainingMs) })
    },
    // change the number of sets of an exercise, the current one included (never below the set under way)
    setRounds(bi: number, n: number) {
      if (state.finished || !session.blocks[bi]) return
      const cur = steps[state.i]
      const doing = cur ? steps.slice(0, state.i + 1).filter((x) => x.type === 'work' && x.blockIdx === bi).length : 0
      const rounds = Math.max(1, Math.min(10, Math.round(n)))
      if (rounds < doing) return
      session = { ...session, blocks: session.blocks.map((b, k) => (k === bi ? { ...b, rounds } : b)) }
      const next = flattenSession(session)
      // the same step in the new list: same kind, exercise (or superset), round, and how many of its kind came before
      const same = (a: typeof cur, b: typeof cur) => a.type === b.type && a.round === b.round && (a.type === 'work' ? a.blockIdx === b.blockIdx : (a.pair ?? '') === (b.pair ?? ''))
      const nth = steps.slice(0, state.i).filter((x) => same(x, cur)).length
      let j = -1
      for (let k = 0, seen = 0; k < next.length; k++) if (same(next[k], cur)) { if (seen === nth) { j = k; break } seen++ }
      steps = next
      const i = j >= 0 ? j : Math.min(state.i, steps.length - 1)
      set({ steps, i, total: totalSeconds(steps), elapsed: elapsedAt(i, state.remainingMs) })
    },
    // self-paced set finished: on to the rest
    setDone: () => { if (selfPaced(steps[state.i])) { audio.beep(1320, 200); goto(state.i + 1) } },
    // one button / earbud press: ends a self-paced set, otherwise pause or resume
    press: () => (selfPaced(steps[state.i]) && state.running ? (audio.beep(1320, 200), goto(state.i + 1)) : togglePause()),
    togglePause,
    skip: () => goto(state.i + 1),
    back: () => goto(state.i - 1),
    stop() {
      if (tick) clearInterval(tick)
      clearTimers()
      try { speechSynthesis.cancel() } catch { /* ignore */ }
    },
  }
}
