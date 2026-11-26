import { describe, expect, test } from 'vitest'
import { FOOTBALL_IN_SEASON, FOOTBALL_OFF_SEASON, RECOVERY_BY_GAMES, buildWeek, defaultFootball, defaultGymFor, matchdayLabel, planOptsOf, type Game, type PlanOpts } from './weekplan'

const g = (...days: number[]): Game[] => days.map((day) => ({ day, kind: 'casual' }))
const of = (games: Game[], kind: string, gym?: number, o: PlanOpts = {}) => buildWeek(games, { gym, ...o }).slots.filter((s) => s.kind === kind)
const daysOf = (games: Game[], kind: string, gym?: number, o: PlanOpts = {}) => of(games, kind, gym, o).map((s) => s.day)

describe('week plan v3', () => {
  test('football is a real part of every week: in season most days (1 game: 4, 2: 3, 3: 2, none: 5), off season 3 to 4', () => {
    expect(daysOf(g(2), 'football')).toHaveLength(4)
    expect(daysOf(g(2, 5), 'football')).toHaveLength(3)
    expect(daysOf(g(0, 2, 4), 'football')).toHaveLength(2)
    expect(daysOf([], 'football')).toHaveLength(5)
    expect(FOOTBALL_IN_SEASON[1]).toBe(4); expect(FOOTBALL_OFF_SEASON[0]).toBe(4)
    expect(daysOf(g(2), 'football', undefined, { season: 'off' })).toHaveLength(3)
    expect(defaultFootball(1, 'off')).toBe(3)
  })
  test('football sessions a week is the player\'s to set, and never lands on a game day', () => {
    for (const football of [0, 2, 5]) {
      const days = daysOf(g(2), 'football', undefined, { football })
      expect(days.length).toBeLessThanOrEqual(football); expect(days).not.toContain(2)
    }
    expect(daysOf(g(2), 'football', undefined, { football: 2 })).toHaveLength(2)
  })
  test('in season gym is 1 to 2 (default 2), off season 3 to 4; the player\'s own number wins', () => {
    expect(defaultGymFor('in')).toBe(2); expect(defaultGymFor('off', 1)).toBe(3); expect(defaultGymFor('off', 0)).toBe(4)
    expect(daysOf(g(2), 'gym')).toHaveLength(2)
    expect(daysOf(g(2), 'gym', undefined, { season: 'off' })).toHaveLength(3)
    expect(daysOf(g(2), 'gym', 4, { season: 'in' })).toHaveLength(4)
  })
  test('double days: football then gym on the same day; with doubles off no day has both', () => {
    const plan = (o: PlanOpts) => buildWeek(g(2), o).slots
    const both = (slots: ReturnType<typeof plan>) => [...new Set(slots.map((s) => s.day))].filter((d) => slots.some((s) => s.day === d && s.kind === 'football') && slots.some((s) => s.day === d && s.kind === 'gym'))
    expect(both(plan({ gym: 3 })).length).toBeGreaterThan(0)
    expect(plan({ gym: 3 }).filter((s) => s.day === both(plan({ gym: 3 }))[0]).map((s) => s.kind)).toEqual(['football', 'gym']) // football first
    expect(both(plan({ gym: 3, doubles: false }))).toEqual([])
    expect(both(plan({ gym: 3, doubles: false, season: 'off' }))).toEqual([])
    expect(planOptsOf({ gym: 3, footballPerWeek: 2, season: 'off', doubleDays: false })).toEqual({ gym: 3, football: 2, season: 'off', doubles: false })
  })
  test('the day before and the day after a game keep their jobs: no football there when another day is free', () => {
    for (const games of [g(2), g(5)]) {
      for (const day of daysOf(games, 'football')) {
        expect(games.some((x) => (x.day + 6) % 7 === day)).toBe(false) // MD-1
        expect(games.some((x) => (x.day + 1) % 7 === day)).toBe(false) // MD+1
      }
    }
  })
  test('gym count comes from the player (default 3), never on a game day', () => {
    for (const gym of [0, 3, 4, 5]) {
      const days = daysOf(g(1, 4), 'gym', gym)
      expect(days).toHaveLength(gym)
      expect(days).not.toContain(1)
      expect(days).not.toContain(4)
    }
    expect(daysOf(g(2), 'gym', 3)).toHaveLength(3)
  })
  test('one or two games with 3-4 gym sessions fit the week', () => {
    for (const games of [g(2), g(1, 5), g(2, 6)]) {
      for (const gym of [3, 4]) {
        const plan = buildWeek(games, { gym })
        expect(plan.slots.filter((s) => s.kind === 'gym')).toHaveLength(gym)
        expect(plan.slots.filter((s) => s.kind === 'football').length).toBeGreaterThanOrEqual(FOOTBALL_IN_SEASON[games.length] - 1)
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
  test('hard day hard: MD-4 has the legs session and the high-intensity football together; never football the day before', () => {
    for (const games of [g(5), g(2)]) {
      const plan = buildWeek(games, { gym: 3 }).slots
      const md4 = (games[0].day + 3) % 7
      expect(plan.filter((s) => s.day === md4).map((s) => s.kind).sort()).toEqual(['football', 'gym'])
      expect(plan.some((s) => s.day === (games[0].day + 6) % 7 && s.kind === 'football')).toBe(false) // never the day before
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
    expect(buildWeek(g(2)).reasons[0]).toMatch(/1 game this week: 4 football, 2 gym/)
    expect(buildWeek(g(1, 5)).reasons[0]).toMatch(/job set by how far it is from the game/)
  })
  test('matchday labels', () => {
    expect(matchdayLabel(0, g(2))).toBe('MD-2')
    expect(matchdayLabel(2, g(2))).toBe('MD')
    expect(matchdayLabel(3, g(2))).toBe('MD+1')
    expect(matchdayLabel(3, [])).toBeNull()
  })
})
