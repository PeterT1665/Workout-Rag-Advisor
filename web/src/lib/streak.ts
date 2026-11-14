// Plan streak (like Duolingo's): days in a row you kept to the week plan.
// A day is kept if you trained, or if the plan said no training (rest day, game day, or a day changed to rest).
// Today only adds once it's kept; an unfinished training day today doesn't break the streak yet.
import { buildWeek, todayIndex, type Game } from './weekplan'
import type { HistoryEntry } from './storage'
import type { WeekChanges } from './weekChanges'

const ymd = (d: Date) => d.toLocaleDateString('en-CA')

export type Streak = { days: number; today: boolean } // today: today already counts (the flame is lit)

export function planStreak(history: HistoryEntry[], games: Game[], gym: number, changes: WeekChanges, now = new Date()): Streak {
  if (!history.length) return { days: 0, today: false }
  const trained = new Set(history.map((h) => h.date))
  const first = history.reduce((a, h) => (h.date < a ? h.date : a), history[0].date)
  const slots = buildWeek(games, { gym }).slots
  const training = (d: Date) => {
    const change = changes[ymd(d)]
    if (change) return !('rest' in change)
    const day = slots.filter((s) => s.day === todayIndex(d))
    return day.some((s) => s.kind !== 'game' && s.kind !== 'cheer') && !day.some((s) => s.kind === 'game')
  }
  const kept = (d: Date) => trained.has(ymd(d)) || !training(d)

  const d = new Date(now)
  const today = trained.has(ymd(d)) || (!training(d) && ymd(d) >= first)
  let days = today ? 1 : 0
  for (let k = 0; k < 400; k++) {
    d.setDate(d.getDate() - 1)
    if (ymd(d) < first || !kept(d)) break
    days++
  }
  return { days, today }
}
