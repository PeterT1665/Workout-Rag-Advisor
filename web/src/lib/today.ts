// What today looks like: the week plan's slot for today, its session, and the matchday label.
import { DEFAULT_GYM, buildWeek, matchdayLabel, todayIndex, type Game, type Slot, type WeekPlan } from './weekplan'
import { load, save } from './storage'
import type { Session } from './types'
import { changedSession, type DayChange } from './weekChanges'
import { dayKindOf } from './library'

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
  // rotate so the week's sessions differ (gym: within the same kind of day, light or full)
  const nth = plan.slots.filter((s) => s.kind === slot.kind && s.day < slot.day && !!s.light === !!slot.light).length
  if (slot.kind === 'football') {
    const pool = soloFootball(all)
    return pool[nth % pool.length] ?? null
  }
  if (slot.kind === 'gym') {
    const gym = all.filter((s) => s.type === 'gym')
    const pick = (ids: string[]) => gym.filter((s) => ids.includes(s.id))
    // the day before a game: upper body only; other light days: the lighter full-body session
    const pool = slot.light ? pick(slot.note?.includes('no heavy legs') ? ['gym-upper-core'] : ['gym-full-light', 'gym-upper-core'])
      : gym.filter((s) => s.id !== 'gym-full-light')
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
