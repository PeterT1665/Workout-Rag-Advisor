// Calories, estimated (research 24/11, calorie-research.md sections 4 and 7). Two ways:
//  1. Per exercise, summed: a max sprint, a max jump or a max isometric hold costs a fixed amount however fresh you felt after;
//     rests count at a low MET; strength and football blocks use a MET set by the session effort. Scaled to the minutes done.
//  2. Whole session, when the session isn't known: MET by type and effort.
// Always shown as a range: heart-rate watches read lifting and power sessions higher, and the honest error is about ±20%.
import type { Block, Session } from './types'

export type Kcal = { low: number; mid: number; high: number }
const tens = (x: number) => Math.round(x / 10) * 10
const range = (base: number, epoc: number): Kcal => ({ low: tens(base * 0.9), mid: tens(base * (1 + epoc)), high: tens(base * (1 + epoc) * 1.15) })

// Whole-session MET by type and effort (1-10); a missing effort counts as 6.
export function sessionMet(type: string | undefined, effort: number | undefined, power = false) {
  const e = effort ?? 6
  if (type === 'recovery') return 2.3
  if (type === 'football') return 4 + 0.5 * e
  if (type === 'home') return 2.5 + 0.4 * e
  return power ? 3 + 0.45 * e : 3 + 0.3 * e
}

const num = (s: string | undefined) => { const m = s?.match(/\d+(?:\.\d+)?/g); return m ? m.reduce((a, b) => a + Number(b), 0) / m.length : undefined }
const BALL = new Set(['ball-mastery', 'first-touch', 'passing', 'finishing', '1v1', 'crossing', 'defending'])
const STRENGTH = new Set(['lower-strength', 'upper-strength', 'single-limb', 'core', 'gym-conditioning'])
const SPRINT = new Set(['sprints', 'speed', 'conditioning', 'off-ball'])
const CONTINUOUS_JUMPS = /pogo|bound|hurdle|skater|hops|squat jump|split jump|burpee/i

// kcal for one block: its work, its rests, at the intensity of what it is.
export function blockKcal(b: Block, kg: number, effort: number): number {
  const rounds = b.rounds ?? 1, work = b.work * rounds, rest = (b.rest ?? 0) * rounds
  const h = (s: number) => s / 3600
  const mets = (met: number, s: number) => met * kg * h(s)
  const c = b.category ?? ''
  if (b.kind === 'warmup' || c === 'warmup') return mets(5, work + rest)
  if (b.kind === 'cooldown' || c === 'cooldown' || c === 'mobility') return mets(2.3, work + rest)
  if (c === 'isometrics') return mets(b.work >= 40 ? 5.5 : 7, work) + mets(1.8, rest) // a max hold costs the same however you felt after
  if (c === 'plyometrics') {
    const jumps = num(b.reps)
    if (!CONTINUOUS_JUMPS.test(b.name) && jumps && !/s\b/.test(b.reps ?? '')) return rounds * jumps * 0.009 * kg + mets(2, rest + work)
    return mets(9, work) + mets(2, rest)
  }
  if (c === 'explosive' || c === 'olympic') return mets(c === 'olympic' ? 6 : 8, work) + mets(2, rest)
  if (SPRINT.has(c)) {
    if (b.work <= 15) { const dist = num(b.reps?.includes('m') ? b.reps : undefined) ?? 20; return rounds * 0.037 * kg * (dist / 20) * (/sled/i.test(b.name) ? 1.5 : 1) + mets(2.4, rest + work) }
    return mets(8, work) + mets(2.4, rest)
  }
  if (BALL.has(c)) return mets(sessionMet('football', effort), work + rest)
  if (STRENGTH.has(c)) return mets(sessionMet('gym', effort), work + rest)
  return mets(sessionMet(b.kind === 'drill' ? 'gym' : 'recovery', effort), work + rest)
}

// The estimate for a session someone did. `session`: what was planned (when known): the exercises are summed, then scaled to the
// minutes actually trained (0.5 to 1.5 of the plan). Without it: whole-session MET.
export function estimateKcal(o: { session?: Session; type?: string; minutes: number; effort?: number; kg: number }): Kcal {
  const { session, type, minutes, kg } = o
  const effort = o.effort ?? 6
  if (session && session.blocks.length) {
    const planned = session.blocks.reduce((a, b) => a + ((b.rounds ?? 1) * (b.work + (b.rest ?? 0))), 0)
    if (planned > 0) {
      const scale = Math.min(1.5, Math.max(0.5, (minutes * 60) / planned))
      const sum = session.blocks.reduce((a, b) => a + blockKcal(b, kg, effort), 0)
      return range(sum * scale, 0.05)
    }
  }
  return range(sessionMet(type, effort) * kg * (minutes / 60), type === 'recovery' ? 0 : type === 'football' ? 0.05 : 0.08)
}

export const kcalText = (k: Kcal) => `~${k.low}–${k.high}`
