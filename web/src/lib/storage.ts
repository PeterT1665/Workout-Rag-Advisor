// localStorage helpers. Every access is guarded: private mode or blocked storage must not break the app.
import type { Settings } from './types'

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
  clipStart: true, clipRest: true, siuuEnd: true, cards: true, clipVolume: 1, restClipChance: 0.5,
}

export const loadSettings = (): Settings => ({ ...DEFAULT_SETTINGS, ...load<Partial<Settings>>('settings', {}) })

// Commonly attributed to Ronaldo. Editable in Clips & quotes.
export const DEFAULT_QUOTES = [
  'Talent without working hard is nothing.',
  'Your love makes me strong. Your hate makes me unstoppable.',
  "Dreams are not what you see in your sleep. Dreams are the things that don't let you sleep.",
  "I don't have to show anything to anyone. There is nothing to prove.",
]
export const loadQuotes = () => load<string[]>('quotes', DEFAULT_QUOTES)

export type Legs = 'fresh' | 'ok' | 'heavy'
export type HistoryEntry = {
  id: string; title: string; date: string; minutes: number
  // filled by the 20-second reflection after the session
  rpe?: number; legs?: Legs; sore?: string[]; load?: number
  drills?: number; cues?: number
}
export const loadHistory = () => load<HistoryEntry[]>('history', [])
export function addHistory(e: HistoryEntry) {
  save('history', [...loadHistory(), e].slice(-200))
}
// Adds the reflection to the most recent session.
export function reflectLast(r: { rpe: number; legs: Legs; sore: string[] }) {
  const h = loadHistory()
  const last = h.at(-1)
  if (!last) return null
  const updated = { ...last, ...r, load: sessionLoad(last.minutes, r.rpe) }
  save('history', [...h.slice(0, -1), updated])
  return updated
}

// Session-RPE load (Foster): minutes x effort 1-10. The standard way team sports track training load.
export const sessionLoad = (minutes: number, rpe: number) => Math.round(minutes * rpe)

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
