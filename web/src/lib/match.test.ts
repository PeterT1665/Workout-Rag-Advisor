import { describe, expect, test } from 'vitest'
import { matchDay, matchForm, matchRating, resultStreaks, withForm, type MatchReview } from './match'

const base = { us: 1, them: 1, minutes: 90, position: 'winger' as const, goals: 0, assists: 0, dribbles: 0, keyPasses: 0, duels: 0, firstTouch: 6 }
const review = (date: string, p: Partial<MatchReview> = {}): MatchReview =>
  ({ ...base, date, tiring: 5, self: 6, moments: '', rating: 6, savedAt: 0, ...p })

describe('match rating', () => {
  test('a quiet draw is a 6.0; goals, assists and a win lift it', () => {
    expect(matchRating(base)).toBe(6)
    expect(matchRating({ ...base, us: 2, them: 1, goals: 1, assists: 1, keyPasses: 2 })).toBeGreaterThan(7.5)
    expect(matchRating({ ...base, us: 0, them: 3 })).toBeLessThan(6)
  })
  test('a defender gets credit for a clean sheet and is marked down for goals conceded', () => {
    const def = { ...base, position: 'defender' as const }
    expect(matchRating({ ...def, us: 1, them: 0 })).toBeGreaterThan(matchRating({ ...base, us: 1, them: 0 }))
    expect(matchRating({ ...def, us: 0, them: 4 })).toBeLessThan(matchRating({ ...base, us: 0, them: 4 }))
  })
  test('a goal in 20 minutes counts for more than a goal in 90, and the rating stays between 3 and 10', () => {
    expect(matchRating({ ...base, goals: 1, minutes: 20 })).toBeGreaterThan(matchRating({ ...base, goals: 1, minutes: 90 }))
    expect(matchRating({ ...base, goals: 9, assists: 9, us: 18, them: 0 })).toBe(10)
    expect(matchRating({ ...base, us: 0, them: 9, firstTouch: 1, position: 'defender' })).toBeGreaterThanOrEqual(3)
  })
  test('a full futsal game is a full game: 40 of 40 rates like 90 of 90, 40 of 90 is boosted', () => {
    const g = { ...base, us: 6, them: 4, goals: 3, keyPasses: 5, dribbles: 3, duels: 3, firstTouch: 8 }
    expect(matchRating({ ...g, minutes: 40, length: 40 })).toBe(matchRating({ ...g, minutes: 90, length: 90 }))
    expect(matchRating({ ...g, minutes: 40 })).toBeGreaterThan(matchRating({ ...g, minutes: 40, length: 40 }))
    const r = matchRating({ ...g, minutes: 40, length: 40 })
    expect(r).toBeGreaterThan(9)
    expect(r).toBeLessThan(10)
  })
})

describe('results streaks', () => {
  test('wins in a row, and unbeaten (draws keep it going)', () => {
    const m = [review('2026-11-01', { us: 0, them: 2 }), review('2026-11-08', { us: 1, them: 1 }), review('2026-11-15', { us: 2, them: 0 }), review('2026-11-22', { us: 3, them: 1 })]
    expect(resultStreaks(m)).toEqual({ wins: 2, unbeaten: 3 })
    expect(resultStreaks([])).toEqual({ wins: 0, unbeaten: 0 })
  })
})

describe('match form on the card', () => {
  test('scoring lifts SHO, by 3 at most; old games are ignored', () => {
    const now = new Date('2026-11-20T12:00:00')
    const form = matchForm([review('2026-11-15', { goals: 4 }), review('2026-11-08', { goals: 3 })], now)
    expect(form.SHO).toBe(3)
    expect(matchForm([review('2026-08-01', { goals: 4 })], now)).toEqual({})
    const card = { PAC: 70, SHO: 98, PAS: 60, DRI: 60, DEF: 40, PHY: 60 }
    expect(withForm(card, form).SHO).toBe(99)
  })
})

describe('when the review is open', () => {
  const games = [{ day: 2, kind: 'competitive' as const }] // Wednesday
  const wed = (hhmm: string) => new Date(`2026-11-18T${hhmm}:00`)
  test('game day morning: set your times; then it opens 15 min after full time', () => {
    expect(matchDay(wed('09:00'), games, {}, [])).toEqual({ state: 'times', date: '2026-11-18' })
    const times = { '2026-11-18': { start: '15:00', end: '16:30' } }
    expect(matchDay(wed('16:40'), games, times, [])?.state).toBe('set')
    expect(matchDay(wed('16:45'), games, times, [])).toEqual({ state: 'review', date: '2026-11-18' })
  })
  test('no times: 9 pm. Still open the next day, gone once saved or after that', () => {
    expect(matchDay(wed('20:59'), games, {}, [])?.state).toBe('times')
    expect(matchDay(wed('21:00'), games, {}, [])?.state).toBe('review')
    expect(matchDay(new Date('2026-11-19T22:00:00'), games, {}, [])).toEqual({ state: 'review', date: '2026-11-18' })
    expect(matchDay(new Date('2026-11-19T22:00:00'), games, {}, [review('2026-11-18')])).toBeNull()
    expect(matchDay(new Date('2026-11-20T10:00:00'), games, {}, [])).toBeNull()
    expect(matchDay(wed('21:30'), games, {}, [review('2026-11-18')])).toBeNull()
  })
})
