import { describe, expect, test } from 'vitest'
import { buildWeek, LOAD_TABLE, matchdayLabel, type Game } from './weekplan'

const g = (...days: number[]): Game[] => days.map((day) => ({ day, kind: 'casual' }))
const daysOf = (games: Game[], kind: string) => buildWeek(games).slots.filter((s) => s.kind === kind).map((s) => s.day)

describe('week plan (same cases as weekplan.py)', () => {
  test('football sessions follow the game count: 1 -> 2, 2 -> 1, 3 -> 0', () => {
    expect(daysOf(g(2), 'football')).toHaveLength(2)
    expect(daysOf(g(2, 0), 'football')).toHaveLength(1)
    expect(daysOf(g(0, 2, 4), 'football')).toHaveLength(0)
  })
  test('three games get recovery instead', () => {
    expect(daysOf(g(0, 2, 4), 'recovery')).toHaveLength(LOAD_TABLE[3].recovery)
  })
  test('no hard session the day before a game when avoidable', () => {
    const hard = [...daysOf(g(2), 'football'), ...daysOf(g(2), 'gym')]
    expect(hard).not.toContain(1)
  })
  test('one main session per day besides games', () => {
    const plan = buildWeek([{ day: 2, kind: 'competitive' }, { day: 2, kind: 'casual' }])
    for (let d = 0; d < 7; d++) expect(plan.slots.filter((s) => s.day === d && s.kind !== 'game').length).toBeLessThanOrEqual(1)
  })
  test('Saturday is cheerleading, recovery follows a game', () => {
    expect(daysOf(g(2), 'cheer')).toEqual([5])
    expect(daysOf(g(2), 'recovery')).toContain(3)
  })
  test('the first reason explains the week', () => {
    expect(buildWeek(g(2)).reasons[0]).toMatch(/1 game/)
  })
  test('matchday labels', () => {
    expect(matchdayLabel(0, g(2))).toBe('MD-2')
    expect(matchdayLabel(2, g(2))).toBe('MD')
    expect(matchdayLabel(3, g(2))).toBe('MD+1')
    expect(matchdayLabel(3, [])).toBeNull()
  })
})
