// What today looks like: the week plan's slot for today, its session, and the matchday label.
import { buildWeek, matchdayLabel, todayIndex, type Game, type Slot, type WeekPlan } from './weekplan'
import { load, save } from './storage'
import type { Session } from './types'

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
}

const soloFootball = (all: Session[]) => all.filter((s) => s.type === 'football' && !s.id.includes('partner'))

export function sessionForSlot(slot: Slot, plan: WeekPlan, all: Session[]): Session | null {
  if (slot.kind === 'recovery') return all.find((s) => s.type === 'recovery') ?? null
  if (slot.kind === 'football') {
    const pool = soloFootball(all)
    const nth = plan.slots.filter((s) => s.kind === 'football' && s.day < slot.day).length
    return pool[nth % pool.length] ?? null
  }
  return null
}

export function planToday(all: Session[], games: Game[], now = new Date()): Today {
  const day = todayIndex(now)
  const plan = buildWeek(games)
  const slots = plan.slots.filter((s) => s.day === day)
  const slot = slots.find((s) => s.kind !== 'game') ?? slots[0] ?? null
  const todays = slot ? sessionForSlot(slot, plan, all) : null
  const next = plan.slots.find((s) => s.day > day && sessionForSlot(s, plan, all))
  const fallback = (next && sessionForSlot(next, plan, all)) ?? soloFootball(all)[0] ?? all[0]
  return {
    day, plan, slot, md: matchdayLabel(day, games),
    session: todays ?? fallback, scheduled: !!todays, nextDay: next?.day ?? null,
  }
}
