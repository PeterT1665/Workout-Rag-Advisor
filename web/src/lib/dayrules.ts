// Rules for what suits which day, from the drills research (brief 28/11, section 3): the day before a game is short and sharp,
// a recovery day is touch work only, headers are capped, high-volume shooting is limited, and every week has one true
// max-speed run. Pure functions on sessions; lib/today.ts uses them to pick the week's football sessions.
import type { Session } from './types'
import { exerciseByName } from './library'

export const MAX_HEADERS_PER_SESSION = 10 // English FA adult amateur guidance, counted across all cards in a session
export const MAX_HEADING_SESSIONS_PER_WEEK = 1
export const MAX_HIGH_VOLUME_SHOOTING_DAYS = 3 // 2 to 3 days a week
export const HIGH_VOLUME_SHOTS = 30
export const MD1_MAX_MINUTES = 45
export const MD1_MAX_SHOTS = 8

export const headersIn = (s: Session) => s.blocks.reduce((a, b) => a + (b.headers ?? 0), 0)
export const hasHeading = (s: Session) => headersIn(s) > 0

const firstNumber = (r?: string) => { const m = r?.match(/\d+/); return m ? Number(m[0]) : 0 }
// shots in a session: each finishing block's rounds x the reps ("4 shots", "6 each foot" counts both feet)
export function shotsIn(s: Session) {
  return s.blocks.reduce((a, b) => (b.category === 'finishing' ? a + (b.rounds ?? 1) * (firstNumber(b.reps) || 6) * (/each/.test(b.reps ?? '') ? 2 : 1) : a), 0)
}
export const isHighVolumeShooting = (s: Session) => shotsIn(s) >= HIGH_VOLUME_SHOTS

// a true max-speed exposure: a curved sprint or a flying 20
export const hasMaxSpeed = (s: Session) => s.blocks.some((b) => /curved sprint|flying 20/i.test(b.name))

const WARM = new Set(['warmup', 'cooldown', 'mobility'])
const NOT_MD1 = new Set(['crossing', 'conditioning', 'gym-conditioning'])
const minutesOf = (s: Session) => Math.round(s.blocks.reduce((a, b) => a + (b.rounds ?? 1) * (b.work + (b.rest ?? 0)), 0) / 60)

// The day before a game: at most 45 minutes, 6 to 8 shots at most, no heading, crossing blocks or conditioning, and every drill tagged for it.
export function md1Safe(s: Session): boolean {
  if (minutesOf(s) > MD1_MAX_MINUTES || shotsIn(s) > MD1_MAX_SHOTS || hasHeading(s)) return false
  return s.blocks.every((b) => b.kind !== 'drill' || WARM.has(b.category ?? '') || (!NOT_MD1.has(b.category ?? '') && !!exerciseByName(b.name)?.gameDay?.includes('md-1')))
}

// A recovery day: ball mastery, easy wall work and juggling only, every drill tagged for recovery. No sprints.
export function recoverySafe(s: Session): boolean {
  return s.blocks.every((b) => b.kind !== 'drill' || WARM.has(b.category ?? '') || (!!exerciseByName(b.name)?.gameDay?.includes('recovery') && !['speed', 'sprints'].includes(b.category ?? '')))
}
