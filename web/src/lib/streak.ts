// Plan streak (like Duolingo's): days in a row you kept to the week plan.
// - A planned training day is kept by any session that day (no pitch? gym. No gym? home. Nothing proper? recovery).
// - Rest days, game days and days changed to rest keep it too.
// - A missed training day can be made up: an extra session (MAKEUP_MIN+ minutes) within the next MAKEUP_DAYS days,
//   on top of that day's own plan (or on a rest day). Each extra session makes up one missed day.
//   Until the deadline the streak isn't broken; Home asks for the make-up.
// - Today only adds once it's kept; an unfinished training day today doesn't break the streak.
import { buildWeek, todayIndex, type Game, type SlotKind } from './weekplan'
import type { HistoryEntry } from './storage'
import type { WeekChanges } from './weekChanges'

export const MAKEUP_MIN = 20
export const MAKEUP_DAYS = 2

const ymd = (d: Date) => d.toLocaleDateString('en-CA')
const addDays = (d: Date, n: number) => { const x = new Date(d); x.setDate(x.getDate() + n); return x }

export type MakeUp = { date: string; by: string; kind: SlotKind | string } // missed day, last day to make it up (YYYY-MM-DD)
export type Streak = { days: number; today: boolean; makeup?: MakeUp } // today: today already counts (the flame glows)

export function planStreak(history: HistoryEntry[], games: Game[], gym: number, changes: WeekChanges, now = new Date()): Streak {
  if (!history.length) return { days: 0, today: false }
  const first = history.reduce((a, h) => (h.date < a ? h.date : a), history[0].date)
  const slots = buildWeek(games, { gym }).slots
  const byDate = new Map<string, HistoryEntry[]>()
  for (const h of history) byDate.set(h.date, [...(byDate.get(h.date) ?? []), h])

  // what the plan wanted that day: null = no training (rest, game, cheer)
  const planned = (d: Date): string | null => {
    const change = changes[ymd(d)]
    if (change) return 'rest' in change ? null : 'session'
    const day = slots.filter((s) => s.day === todayIndex(d))
    if (day.some((s) => s.kind === 'game')) return null
    return day.find((s) => s.kind !== 'cheer')?.kind ?? null
  }

  // walk from the first logged day to today: which days are kept, and which missed days were made up
  const todayKey = ymd(now)
  const status = new Map<string, 'kept' | 'missed' | 'pending'>()
  const missed: string[] = [] // waiting for a make-up, oldest first
  const madeUp = new Set<string>()
  for (let d = new Date(`${first}T12:00:00`); ymd(d) <= todayKey; d = addDays(d, 1)) {
    const key = ymd(d)
    const done = byDate.get(key) ?? []
    const plan = planned(d)
    // a training day uses its longest session; anything else of MAKEUP_MIN+ minutes is extra
    const ordered = [...done].sort((a, b) => b.minutes - a.minutes)
    const extra = (plan ? ordered.slice(1) : ordered).filter((h) => h.minutes >= MAKEUP_MIN)
    extra.forEach(() => {
      // each extra session makes up the oldest missed day still inside its window
      const open = missed.find((m) => !madeUp.has(m) && key <= ymd(addDays(new Date(`${m}T12:00:00`), MAKEUP_DAYS)))
      if (open) madeUp.add(open)
    })
    if (!plan || done.length) status.set(key, 'kept')
    else if (key === todayKey) status.set(key, 'pending')
    else { status.set(key, 'missed'); missed.push(key) }
  }

  const keptOrOpen = (key: string) => {
    const s = status.get(key)
    if (s === 'kept' || madeUp.has(key)) return true
    // missed but the make-up window is still open
    return s === 'missed' && todayKey <= ymd(addDays(new Date(`${key}T12:00:00`), MAKEUP_DAYS))
  }

  const today = status.get(todayKey) === 'kept'
  let days = today ? 1 : 0
  let makeup: MakeUp | undefined
  for (let d = addDays(now, -1); ymd(d) >= first; d = addDays(d, -1)) {
    const key = ymd(d)
    if (!keptOrOpen(key)) break
    days++
    if (status.get(key) === 'missed' && !madeUp.has(key)) {
      makeup = { date: key, by: ymd(addDays(d, MAKEUP_DAYS)), kind: planned(d) ?? 'session' } // the oldest open one wins
    }
  }
  return { days, today, ...(makeup ? { makeup } : {}) }
}
