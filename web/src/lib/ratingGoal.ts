// What it takes to reach a goal overall rating: which stats to raise, and the result to hit in each test.
import { CARD_STATS, OVERALL_WEIGHTS, overall, positionGroup, type CardStats } from '../components/PlayerCard'
import { TESTS, resultForRating, resultText, type Stat } from './testday'

export const GOAL_DEFAULT = 75 // gold: everyone works toward a gold card unless they choose otherwise
export const GOAL_TIERS = [65, 75, 85] as const // silver, gold, top
export const goalOf = (p: { goal?: number }) => p.goal ?? GOAL_DEFAULT
export const tierOf = (ovr: number) => (ovr >= 75 ? 'gold' : ovr >= 65 ? 'silver' : 'bronze') as 'gold' | 'silver' | 'bronze'

export type GoalStep = { stat: Stat; from: number; to: number; target: string }
export type GoalPlan = { now: number; goal: number; reached: boolean; next: number | null; steps: GoalStep[]; far: boolean; after: number } // far: the steps alone don't reach the goal; after: the overall they do reach

const MAX_STEP = 15 // rating points to ask of one stat in one go: a long way, but a realistic one
const TOP = 95 // a stat at or above this has no more to give
const FEWEST = [10, 12, 15] // use as few stats as possible: one if it needs at most +10, two if +12 each, then three

// Raise the stats that count most for the position (the cheapest points), the same number of rating points on each, until the
// overall reaches the goal. `skip`: stats to leave out (shooting from a session scored on the old full-goal test).
export function planToGoal(stats: CardStats, positions: readonly string[] | undefined, goal: number, skip: Stat[] = []): GoalPlan {
  const now = overall(stats, positions)
  if (now >= goal) return { now, goal, reached: true, next: ([85, 95] as number[]).find((g) => g > goal) ?? null, steps: [], far: false, after: now }
  const w = OVERALL_WEIGHTS[positionGroup(positions) ?? '']
  const present = CARD_STATS.filter((k) => typeof stats[k] === 'number')
  const total = present.reduce((a, k) => a + (w ? w[k] : 1), 0)
  const weight = (k: Stat) => (w ? w[k] : 1) / total
  // the overall shows rounded, so 74.5 already reads 75
  const need = goal - 0.5 - present.reduce((a, k) => a + stats[k]! * weight(k), 0)
  const cap = (k: Stat) => Math.min(stats[k]! + MAX_STEP, k === 'PAC' || k === 'DRI' || k === 'DEF' ? 99 : TESTS.find((x) => x.stat === k)!.anchors.reduce((m, [, r]) => Math.max(m, r), 0))
  const order = present.filter((k) => !skip.includes(k) && stats[k]! < TOP && cap(k) > stats[k]!).sort((a, b) => weight(b) - weight(a) || stats[a]! - stats[b]!)

  let best: { to: Partial<Record<Stat, number>>; remaining: number; biggest: number } | null = null
  for (let n = 1; n <= Math.min(3, order.length); n++) {
    let active = order.slice(0, n), remaining = need
    const to: Partial<Record<Stat, number>> = {}
    while (active.length && remaining > 1e-9) {
      const sw = active.reduce((a, k) => a + weight(k), 0), d = remaining / sw
      const capped = active.filter((k) => (to[k] ?? stats[k]!) + d > cap(k))
      if (!capped.length) { for (const k of active) { to[k] = Math.min(cap(k), stats[k]! + Math.ceil(d)); remaining -= weight(k) * (to[k]! - stats[k]!) } break }
      for (const k of capped) { remaining -= weight(k) * (cap(k) - stats[k]!); to[k] = cap(k) }
      active = active.filter((k) => !capped.includes(k))
    }
    const biggest = Math.max(0, ...Object.entries(to).map(([k, v]) => v! - stats[k as Stat]!))
    best = { to, remaining, biggest }
    if (remaining <= 1e-9 && biggest <= FEWEST[n - 1]) break
  }
  const steps: GoalStep[] = []
  for (const k of order) {
    const to = best?.to[k]
    if (to === undefined || to <= stats[k]!) continue
    const r = resultForRating(k, to)
    if (r !== null) steps.push({ stat: k, from: stats[k]!, to, target: resultText(k, r) })
  }
  const raised = { ...stats }; for (const st of steps) raised[st.stat] = st.to
  return { now, goal, reached: false, next: null, steps, far: !best || best.remaining > 1e-9, after: overall(raised, positions) }
}
