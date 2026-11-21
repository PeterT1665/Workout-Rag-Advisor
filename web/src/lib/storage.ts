// localStorage helpers. Every access is guarded: private mode or blocked storage must not break the app.
import type { Settings } from './types'
import { mark } from '../i18n'
import { bestsFrom, cleanLifts, scoreLifts, sessionEffort, type LiftLog } from './lifts'

export function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v ? (JSON.parse(v) as T) : fallback
  } catch {
    return fallback
  }
}

let saveHook: ((key: string) => void) | null = null
export const setSaveHook = (fn: (key: string) => void) => { saveHook = fn }

export function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* ignore */
  }
  saveHook?.(key)
}

export const DEFAULT_SETTINGS: Settings = {
  voice: true, volume: 1, cueMin: 3, cueMax: 7,
  cards: true, earbudControls: false,
}

export const loadSettings = (): Settings => ({ ...DEFAULT_SETTINGS, ...load<Partial<Settings>>('settings', {}) })

// Commonly attributed to Ronaldo. Editable in Clips & quotes.
export const DEFAULT_QUOTES = [
  mark('Talent without working hard is nothing.'),
  mark('Your love makes me strong. Your hate makes me unstoppable.'),
  mark("Dreams are not what you see in your sleep. Dreams are the things that don't let you sleep."),
  mark("I don't have to show anything to anyone. There is nothing to prove."),
]
export const loadQuotes = () => load<string[]>('quotes', DEFAULT_QUOTES)

export type Legs = 'fresh' | 'ok' | 'heavy'
export type HistoryEntry = {
  id: string; title: string; date: string; minutes: number
  type?: string // session type when it was done (football, gym, home, recovery)
  // filled by the 20-second reflection after the session
  rpe?: number; legs?: Legs; sore?: string[]; load?: number
  lifts?: LiftLog[] // weight and reps per set (gym), logged after the session
  effort?: number // session effort: what you felt and what your sets say (lib/lifts.ts); the load uses it
  bw?: number // body weight that day, for bodyweight exercises
  drills?: number; cues?: number
  at?: number // last edited (ms): when two devices disagree, the later edit wins
  deleted?: true // removed by the player; kept as a marker so syncing doesn't bring it back
}
const allHistory = () => load<HistoryEntry[]>('history', [])
export const loadHistory = () => allHistory().filter((h) => !h.deleted)
export function addHistory(e: HistoryEntry) {
  save('history', [...allHistory(), e].slice(-200))
}
// Removes a logged session (e.g. one saved by mistake). The account's copy is merged, so it's marked, not dropped.
const same = (a: HistoryEntry, b: HistoryEntry) => !a.deleted && a.date === b.date && a.id === b.id && a.minutes === b.minutes && a.title === b.title
export function deleteHistory(e: HistoryEntry) {
  const all = allHistory()
  const at = all.findIndex((x) => same(x, e))
  if (at < 0) return
  save('history', all.map((x, i) => (i === at ? { ...x, deleted: true as const, at: Date.now() } : x)))
}

export type Reflection = { rpe: number; legs?: Legs; sore?: string[]; lifts?: LiftLog[]; bw?: number }
// The check-in, scored: your sets against your own bests from earlier sessions, effort, and load.
function scored(all: HistoryEntry[], at: number, r: Reflection): HistoryEntry {
  const e = { ...all[at], ...r }
  const lifts = r.lifts ? cleanLifts(r.lifts) : e.lifts
  const live = all.slice(0, at).filter((x) => !x.deleted)
  const { measured } = scoreLifts(lifts ?? [], r.rpe, bestsFrom(live, e.bw), e.bw)
  const effort = sessionEffort(r.rpe, measured)
  const out: HistoryEntry = { ...e, effort, load: sessionLoad(e.minutes, effort), at: Date.now() }
  if (lifts?.length) out.lifts = lifts; else delete out.lifts
  if (!out.bw) delete out.bw
  return out
}
// Adds the reflection to the most recent session.
export function reflectLast(r: Reflection) {
  const h = allHistory()
  const at = h.findLastIndex((x) => !x.deleted)
  if (at < 0) return null
  const updated = scored(h, at, r)
  save('history', h.map((x, i) => (i === at ? updated : x)))
  return updated
}
// Edit a logged session later (Record → tap it): sets, effort.
export function editHistory(e: HistoryEntry, r: Reflection) {
  const h = allHistory()
  const at = h.findIndex((x) => same(x, e))
  if (at < 0) return null
  const updated = scored(h, at, { legs: e.legs, sore: e.sore, ...r })
  save('history', h.map((x, i) => (i === at ? updated : x)))
  return updated
}

// Session-RPE load (Foster): minutes x effort 1-10, the standard way team sports track training load.
// The effort is what you felt, checked against what your logged sets say (lib/lifts.ts).
export const sessionLoad = (minutes: number, effort: number) => Math.round(minutes * effort)

// Load per day for the 7 days of the week containing `now` (Monday first).
export function weekLoads(history: HistoryEntry[], now = new Date()): number[] {
  const monday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - ((now.getDay() + 6) % 7))
  const out = [0, 0, 0, 0, 0, 0, 0]
  for (const e of history) {
    if (!e.load) continue
    const [y, m, d] = e.date.split('-').map(Number)
    const diff = Math.round((new Date(y, m - 1, d).getTime() - monday.getTime()) / 864e5)
    if (diff >= 0 && diff < 7) out[diff] += e.load
  }
  return out
}
