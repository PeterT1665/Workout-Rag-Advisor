import { describe, expect, test } from 'vitest'
import { planToGoal, goalOf, tierOf } from './ratingGoal'
import { overall } from '../components/PlayerCard'
import { resultForRating, score, TESTS } from './testday'

const stats = { PAC: 47, DRI: 71, SHO: 63, PAS: 55, DEF: 49, PHY: 78 }

describe('goal plan', () => {
  test('the goal defaults to 75 (gold) and tiers follow FIFA', () => {
    expect(goalOf({})).toBe(75)
    expect(goalOf({ goal: 85 })).toBe(85)
    expect(tierOf(75)).toBe('gold'); expect(tierOf(74)).toBe('silver'); expect(tierOf(65)).toBe('silver'); expect(tierOf(64)).toBe('bronze')
  })
  test('a small gap: as few stats as possible, pace first for a winger, and the steps reach the goal', () => {
    const plan = planToGoal(stats, ['winger'], 65)
    expect(plan.reached).toBe(false); expect(plan.far).toBe(false)
    expect(plan.steps.length).toBeLessThanOrEqual(2)
    expect(plan.steps[0].stat).toBe('PAC') // heaviest weight with room (ties: lowest first)
    const after = { ...stats }; for (const s of plan.steps) after[s.stat] = s.to
    expect(overall(after, ['winger'])).toBeGreaterThanOrEqual(65)
    expect(plan.after).toBeGreaterThanOrEqual(65)
  })
  test('a big gap: three stats at most, +15 each at most, and it says how far the steps get you', () => {
    const plan = planToGoal(stats, ['winger'], 75)
    expect(plan.far).toBe(true)
    expect(plan.steps.length).toBeLessThanOrEqual(3)
    expect(plan.steps.every((s) => s.to > s.from && s.to - s.from <= 15)).toBe(true)
    expect(plan.steps.some((s) => s.stat === 'PHY')).toBe(false) // the jump is not what moves a winger's card
    expect(plan.after).toBeGreaterThan(plan.now); expect(plan.after).toBeLessThan(75)
  })
  test('already at the goal: reached, and the next tier is offered', () => {
    const plan = planToGoal({ PAC: 90, DRI: 90, SHO: 90, PAS: 90, DEF: 90, PHY: 90 }, ['winger'], 75)
    expect(plan.reached).toBe(true); expect(plan.next).toBe(85); expect(plan.steps).toEqual([])
  })
  test('a stat to skip is left out, and a skipped test is never advised', () => {
    const plan = planToGoal({ PAC: 47, DRI: 71, SHO: 63, PAS: 55 }, ['striker'], 75, ['SHO'])
    expect(plan.steps.some((s) => s.stat === 'SHO')).toBe(false)
    expect(plan.steps.some((s) => s.stat === 'DEF' || s.stat === 'PHY')).toBe(false) // no result for them: nothing to raise
  })
  test('a target result earns the rating it was worked out for', () => {
    for (const t of TESTS) for (const r of [50, 60, 70, 80]) {
      const v = resultForRating(t.stat, r)!
      expect(Math.abs(score(t, t.stat === 'SHO' || t.stat === 'PAS' ? Math.round(v) : v) - r)).toBeLessThanOrEqual(t.stat === 'SHO' ? 4 : 3)
    }
    expect(resultForRating('PHY', 99)).toBeNull() // counted tests stop at their top anchor
    expect(resultForRating('PAC', 60)).toBeCloseTo(3.34, 1)
  })
})
