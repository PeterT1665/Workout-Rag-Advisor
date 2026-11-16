// What today looks like: the week plan's slot for today, its session, and the matchday label.
import { DEFAULT_GYM, buildWeek, matchdayLabel, todayIndex, type Game, type GymFocus, type Slot, type WeekPlan } from './weekplan'
import { load, save } from './storage'
import type { Session } from './types'
import { changedSession, type DayChange } from './weekChanges'
import { dayKindOf } from './library'

// Recommended gym sessions for each job (MD-4+ legs, MD-3 core, MD-2 upper power, MD-1 activation).
export const GYM_FOR: Record<GymFocus, string[]> = {
  strength: ['gym-contrast', 'gym-lower-power', 'gym-explosive', 'gym-single-leg'],
  'upper-power': ['gym-upper-power', 'gym-upper-core'],
  core: ['gym-core-prevention'],
  activation: ['gym-activation'],
}

export const DEFAULT_GAMES: Game[] = [{ day: 2, kind: 'competitive' }] // futsal, Wednesday night
export const loadGames = () => load<Game[]>('games', DEFAULT_GAMES)
export const saveGames = (g: Game[]) => save('games', g)

export type Today = {
  day: number
  plan: WeekPlan
  slot: Slot | null
  md: string | null
  session: Session // what Kick off starts
  scheduled: boolean // true if the session is today's planned one
  nextDay: number | null
  changed?: 'session' | 'rest' // the player changed today by hand (Week screen or a session's Do it today)
}

const soloFootball = (all: Session[]) => all.filter((s) => s.type === 'football' && !s.id.includes('partner'))

export function sessionForSlot(slot: Slot, plan: WeekPlan, all: Session[]): Session | null {
  if (slot.kind === 'recovery') return all.find((s) => s.type === 'recovery') ?? null
  // rotate so the week's sessions differ (gym: among days with the same job)
  const nth = plan.slots.filter((s) => s.kind === slot.kind && s.day < slot.day && s.focus === slot.focus).length
  if (slot.kind === 'football') {
    const pool = soloFootball(all)
    return pool[nth % pool.length] ?? null
  }
  if (slot.kind === 'gym') {
    const gym = all.filter((s) => s.type === 'gym')
    // in the listed order: the first is the main session for that job
    const pick = (ids: string[]) => ids.map((id) => gym.find((s) => s.id === id)).filter((s): s is Session => !!s)
    // each gym job has its sessions (lib/weekplan.ts gymFor); built-ins first, then rotation
    const pool = pick(GYM_FOR[slot.focus ?? 'strength'])
    return (pool.length ? pool : gym)[nth % (pool.length || gym.length)] ?? null
  }
  return null
}

export function planToday(all: Session[], games: Game[], now = new Date(), gym = DEFAULT_GYM): Today {
  const day = todayIndex(now)
  const plan = buildWeek(games, { gym })
  const slots = plan.slots.filter((s) => s.day === day)
  // a game day is a game day, even when a short recovery follows the game
  const slot = slots.find((s) => s.kind === 'game') ?? slots.find((s) => s.kind !== 'game') ?? null
  const todays = slot ? sessionForSlot(slot, plan, all) : null
  const next = plan.slots.find((s) => s.day > day && sessionForSlot(s, plan, all))
  const fallback = (next && sessionForSlot(next, plan, all)) ?? soloFootball(all)[0] ?? all[0]
  return {
    day, plan, slot, md: matchdayLabel(day, games),
    session: todays ?? fallback, scheduled: !!todays, nextDay: next?.day ?? null,
  }
}

// Apply a manual change for today on top of the planner's choice.
export function withChange(t: Today, change: DayChange | null, library: Session[]): Today {
  if (!change) return t
  if ('rest' in change) return { ...t, slot: t.slot?.kind === 'game' ? t.slot : null, scheduled: false, changed: 'rest' }
  const s = changedSession(change, library)
  if (!s) return t
  const kind = dayKindOf(s)
  return { ...t, session: s, scheduled: true, changed: 'session', slot: t.slot?.kind === 'game' ? t.slot : { day: t.day, kind } }
}
