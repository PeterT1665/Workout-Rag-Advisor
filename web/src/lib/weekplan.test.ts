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
  test('Saturday game, like the pro week: legs MD-4, core MD-3, upper power MD-2, activation MD-1', () => {
    const focus = (gym: number) => of(g(5), 'gym', gym).map((s) => `${s.day}:${s.focus}`)
    expect(focus(3)).toEqual(['1:strength', '2:core', '3:upper-power'])
    expect(focus(4)).toEqual(['1:strength', '2:core', '3:upper-power', '4:activation'])
    expect(daysOf(g(5), 'recovery')).toEqual([6]) // Sunday, the day after
    expect(daysOf(g(5), 'football')).not.toContain(4) // never the day before
  })
  test('the day before a game is only ever activation; the day after a game gets no gym unless the week is full', () => {
    for (const games of [g(2), g(5), g(1, 5), g(2, 6)]) {
      for (const s of of(games, 'gym', 4)) {
        if (games.some((x) => x.day === (s.day + 1) % 7)) expect(s.focus).toBe('activation')
      }
      expect(of(games, 'gym', 3).some((s) => games.some((x) => x.day === (s.day + 6) % 7))).toBe(false)
    }
  })
  test('congested weeks (Wednesday + Saturday) drop heavy legs', () => {
    expect(of(g(2, 5), 'gym', 3).some((s) => s.focus === 'strength')).toBe(false)
  })
  test('no games: the jobs rotate and gym days are spread out', () => {
    const gym = of([], 'gym', 3)
    expect(gym.map((s) => s.focus)).toEqual(['strength', 'upper-power', 'core'])
    const days = gym.map((s) => s.day)
    for (let k = 1; k < days.length; k++) expect(days[k] - days[k - 1]).toBeGreaterThan(1)
  })
  test('matchday distance wraps around the week', () => {
    expect(matchdayLabel(4, g(2))).toBe('MD+2') // Friday after a Wednesday game
    expect(matchdayLabel(5, g(2))).toBe('MD-4') // Saturday: counting down to next Wednesday
    expect(matchdayLabel(6, g(2))).toBe('MD-3') // Sunday: three days before next Wednesday
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
    expect(buildWeek(g(1, 5)).reasons[0]).toMatch(/job set by how far it is from the game/)
  })
  test('matchday labels', () => {
    expect(matchdayLabel(0, g(2))).toBe('MD-2')
    expect(matchdayLabel(2, g(2))).toBe('MD')
    expect(matchdayLabel(3, g(2))).toBe('MD+1')
    expect(matchdayLabel(3, [])).toBeNull()
  })
})
