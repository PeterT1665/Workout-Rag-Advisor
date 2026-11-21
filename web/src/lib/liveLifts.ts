// Sets logged during the session (in the rest after an exercise's last set), kept until the check-in saves them.
// Device-only and tied to the session's start time, so a reload mid-session keeps them and an old session's don't leak.
import type { LiftLog } from './lifts'

const KEY = 'liveLifts'
type Live = { startedAt: number; lifts: LiftLog[] }

export function liveLifts(startedAt: number): LiftLog[] {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Live | null
    return v && v.startedAt === startedAt ? v.lifts : []
  } catch { return [] }
}
// Replaces these exercises' sets, keeps the others.
export function saveLiveLifts(startedAt: number, lifts: LiftLog[]) {
  const keep = liveLifts(startedAt).filter((l) => !lifts.some((x) => x.name === l.name))
  try { localStorage.setItem(KEY, JSON.stringify({ startedAt, lifts: [...keep, ...lifts] })) } catch { /* ignore */ }
}
export function clearLiveLifts() {
  try { localStorage.removeItem(KEY) } catch { /* ignore */ }
}
// The check-in's form with what was already logged during the session.
export const withLive = (planned: LiftLog[], live: LiftLog[]) => planned.map((l) => live.find((x) => x.name === l.name) ?? l)
