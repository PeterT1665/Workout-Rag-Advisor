// Plan streak (like Duolingo's): days in a row you kept to the week plan.
// - A planned training day is kept by STREAK_MIN+ minutes (RECOVERY_MIN+ for a recovery session) of any training that day (no pitch? gym. No gym? home.
//   Nothing proper? recovery). A shorter session never counts, not even two of them together.
// - Rest days, game days and days changed to rest keep it too.
// - A missed training day can be made up: an extra session (STREAK_MIN+ minutes) within the next MAKEUP_DAYS days,
//   on top of that day's own plan (or on a rest day). Each extra session makes up one missed day.
//   Until the deadline the streak isn't broken; Home asks for the make-up.
// - Today only adds once it's kept; an unfinished training day today doesn't break the streak.
import { buildWeek, todayIndex, type Game, type SlotKind } from './weekplan'
import type { HistoryEntry } from './storage'
import type { WeekChanges } from './weekChanges'

export const STREAK_MIN = 20
export const RECOVERY_MIN = 15 // a recovery session is shorter by design (and rests get skipped)
// how long a session must be to count toward the streak
export const minFor = (type?: string) => (type === 'recovery' ? RECOVERY_MIN : STREAK_MIN)
export const counts = (h: Pick<HistoryEntry, 'minutes' | 'type'>) => h.minutes >= minFor(h.type)
export const MAKEUP_DAYS = 2

const ymd = (d: Date) => d.toLocaleDateString('en-CA')
const addDays = (d: Date, n: number) => { const x = new Date(d); x.setDate(x.getDate() + n); return x }

export type MakeUp = { date: string; by: string; kind: SlotKind | string } // missed day, last day to make it up (YYYY-MM-DD)
export type Streak = { days: number; today: boolean; makeup?: MakeUp; status?: Record<string, DayStatus> } // today: today already counts (the flame glows)
// each day since the first log: kept (trained, or rested on a rest day), made up later, missed, or today still to do
export type DayStatus = 'kept' | 'made-up' | 'missed' | 'pending'

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
    // only sessions of STREAK_MIN+ minutes (RECOVERY_MIN+ for recovery) count (short ones never do, not even added together):
    // one keeps a training day, any other is extra
    const real = done.filter(counts)
    const used = plan && real.length ? 1 : 0
    const enough = !plan || used === 1
    const extra = real.slice(used)
    extra.forEach(() => {
      // each extra session makes up the oldest missed day still inside its window
      const open = missed.find((m) => !madeUp.has(m) && key <= ymd(addDays(new Date(`${m}T12:00:00`), MAKEUP_DAYS)))
      if (open) madeUp.add(open)
    })
    if (!plan || enough) status.set(key, 'kept')
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
  const days_ = Object.fromEntries([...status.entries()].map(([k, v]) => [k, v === 'missed' && madeUp.has(k) ? 'made-up' : v])) as Record<string, DayStatus>
  return { days, today, ...(makeup ? { makeup } : {}), status: days_ }
}
