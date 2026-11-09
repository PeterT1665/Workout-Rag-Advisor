import { describe, expect, test } from 'vitest'
import { FOOTBALL_BY_GAMES, RECOVERY_BY_GAMES, buildWeek, matchdayLabel, type Game } from './weekplan'

const g = (...days: number[]): Game[] => days.map((day) => ({ day, kind: 'casual' }))
const of = (games: Game[], kind: string, gym?: number) => buildWeek(games, { gym }).slots.filter((s) => s.kind === kind)
const daysOf = (games: Game[], kind: string, gym?: number) => of(games, kind, gym).map((s) => s.day)

describe('week plan v3', () => {
  test('football sessions follow the game count: 1 -> 2, 2 -> 1, 3 -> 0', () => {
    expect(daysOf(g(2), 'football')).toHaveLength(2)
    expect(daysOf(g(2, 5), 'football')).toHaveLength(1)
    expect(daysOf(g(0, 2, 4), 'football')).toHaveLength(0)
    expect(FOOTBALL_BY_GAMES[0]).toBe(3)
  })
  test('gym count comes from the player (default 3), never on a game day', () => {
    for (const gym of [0, 3, 4, 5]) {
      const days = daysOf(g(1, 4), 'gym', gym)
      expect(days).toHaveLength(gym)
      expect(days).not.toContain(1)
      expect(days).not.toContain(4)
    }
    expect(daysOf(g(2), 'gym')).toHaveLength(3)
  })
  test('one or two games with 3-4 gym sessions fit the week', () => {
    for (const games of [g(2), g(1, 5), g(2, 6)]) {
      for (const gym of [3, 4]) {
        const plan = buildWeek(games, { gym })
        expect(plan.slots.filter((s) => s.kind === 'gym')).toHaveLength(gym)
        expect(plan.slots.filter((s) => s.kind === 'football')).toHaveLength(FOOTBALL_BY_GAMES[games.length])
      }
    }
  })
  test('heavy weeks (2+ games) make every gym session light', () => {
    expect(of(g(1, 5), 'gym', 4).every((s) => s.light)).toBe(true)
    expect(of(g(2), 'gym', 3).some((s) => !s.light)).toBe(true)
  })
  test('gym the day before a game is upper body only', () => {
    for (const games of [g(2), g(1, 5), g(0, 2, 4)]) {
      for (const s of of(games, 'gym', 5)) {
        if (games.some((x) => x.day === s.day + 1)) expect(s.note).toMatch(/no heavy legs/)
      }
    }
  })
  test('gym the day after a game (only when the week is full) is light too', () => {
    const thu = of(g(2), 'gym', 4).find((s) => s.day === 3)
    if (thu) expect(thu.note).toMatch(/day after a game/)
  })
  test('no football the day before a game when avoidable', () => {
    expect(daysOf(g(2), 'football')).not.toContain(1)
  })
  test('recovery follows a game; with a full week it moves onto the game day', () => {
    expect(daysOf(g(2), 'recovery')).toContain(3)
    const full = buildWeek(g(0, 2, 4), { gym: 4 })
    const rec = full.slots.filter((s) => s.kind === 'recovery')
    expect(rec).toHaveLength(RECOVERY_BY_GAMES[3])
  })
  test('a doubled day is explained', () => {
    const plan = buildWeek(g(0, 2, 4), { gym: 5 })
    expect(plan.slots.filter((s) => s.kind === 'gym')).toHaveLength(4) // only 4 non-game days exist
    expect(plan.reasons.join(' ')).toMatch(/No day left for another gym session/)
  })
  test('the first reason explains the week', () => {
    expect(buildWeek(g(2)).reasons[0]).toMatch(/1 game this week: 2 football, 3 gym/)
    expect(buildWeek(g(1, 5)).reasons[0]).toMatch(/lighter/)
  })
  test('matchday labels', () => {
    expect(matchdayLabel(0, g(2))).toBe('MD-2')
    expect(matchdayLabel(2, g(2))).toBe('MD')
    expect(matchdayLabel(3, g(2))).toBe('MD+1')
    expect(matchdayLabel(3, [])).toBeNull()
  })
})
