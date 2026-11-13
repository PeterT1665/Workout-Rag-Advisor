// Manual changes to the week: a day can get any session (recommended or own) or rest, instead of what the
// planner picked. Stored by calendar date so a change only applies to that week. Synced as 'weekPlan'.
import { load, save } from './storage'
import { todayIndex } from './weekplan'
import type { Session } from './types'

export type DayChange = { sessionId: string } | { rest: true }
export type WeekChanges = Record<string, DayChange> // key: YYYY-MM-DD

const ymd = (d: Date) => d.toLocaleDateString('en-CA')

// Calendar date of a weekday (0 = Monday) in the current week.
export function dateOfDay(day: number, now = new Date()) {
  const d = new Date(now)
  d.setDate(d.getDate() - todayIndex(now) + day)
  return ymd(d)
}

export const loadWeekChanges = () => load<WeekChanges>('weekPlan', {})

export function saveWeekChange(day: number, change: DayChange | null, now = new Date()) {
  const all = { ...loadWeekChanges() }
  const key = dateOfDay(day, now)
  if (change) all[key] = change
  else delete all[key]
  // keep roughly the last 8 weeks
  const cutoff = ymd(new Date(now.getTime() - 56 * 864e5))
  for (const k of Object.keys(all)) if (k < cutoff) delete all[k]
  save('weekPlan', all)
  return all
}

export function changeFor(day: number, changes: WeekChanges, now = new Date()): DayChange | null {
  return changes[dateOfDay(day, now)] ?? null
}

// The session a change points to (null for rest, or if that session was deleted).
export function changedSession(change: DayChange | null, all: Session[]): Session | null {
  if (!change || 'rest' in change) return null
  return all.find((s) => s.id === change.sessionId) ?? null
}
