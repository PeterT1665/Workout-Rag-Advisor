import { describe, expect, test } from 'vitest'
import { TESTS, looksOff, ratingsFrom, score, startingPoint, toCardStats } from './testday'
import { overall } from '../components/PlayerCard'

const t = (stat: string) => TESTS.find((x) => x.stat === stat)!

describe('rating session scoring', () => {
  test('faster times score higher for timed tests, more reps/cm score higher for counted ones', () => {
    expect(score(t('PAC'), 3.0)).toBeGreaterThan(score(t('PAC'), 3.6))
    expect(score(t('DRI'), 10)).toBeGreaterThan(score(t('DRI'), 14))
    expect(score(t('PAS'), 25)).toBeGreaterThan(score(t('PAS'), 12))
    expect(score(t('PHY'), 260)).toBeGreaterThan(score(t('PHY'), 200))
  })
  test('anchors are hit exactly and in-between results interpolate', () => {
    expect(score(t('PAC'), 3.5)).toBe(74)
    expect(score(t('SHO'), 6)).toBe(70)
  })
  test('ratings stay between 30 and 99', () => {
    expect(score(t('PAC'), 2.0)).toBe(99)
    expect(score(t('PAC'), 9)).toBe(30)
    expect(score(t('SHO'), 10)).toBeLessThanOrEqual(99)
  })
  test('a card needs 3 tests; a skipped test stays blank (never filled from an older session)', () => {
    expect(toCardStats(ratingsFrom({ PAC: 3.3, SHO: 7 }))).toBeNull()
    const card = toCardStats(ratingsFrom({ PAC: 3.3, SHO: 7, PAS: 20 }))!
    expect(card.PAC).toBe(82)
    expect(card.DEF).toBeUndefined()
    expect(overall(card)).toBe(Math.round((82 + 76 + 77) / 3)) // the overall averages the tests done
  })
  test('the first session is the starting point', () => {
    const first = ratingsFrom({ PAC: 3.3 }, '2026-11-14'), second = ratingsFrom({ PAC: 3.5 }, '2026-11-22')
    expect(startingPoint({ ratings: second, baseRatings: first })).toBe(first)
    expect(startingPoint({ ratings: second, prevRatings: first })).toBe(first) // older profiles
    expect(startingPoint({ ratings: first })).toBeUndefined()
  })
  test('results that look off are flagged: impossible ones, and ones far from the player\'s others', () => {
    // Peter's first try-out: a 35 cm broad jump, a 3.26 s sprint next to a 97 pass rating
    const tryout = ratingsFrom({ PAC: 3.26, DRI: 4.58, SHO: 10, PAS: 34, DEF: 5.09, PHY: 35 })
    expect(looksOff(tryout)).toContainEqual({ stat: 'PHY', why: 'impossible' })
    expect(looksOff(tryout).some((o) => o.stat === 'DRI')).toBe(true)
    // a believable session flags nothing
    expect(looksOff(ratingsFrom({ PAC: 3.5, DRI: 12, SHO: 6, PAS: 20, DEF: 5.2, PHY: 235 }))).toEqual([])
  })
  test('empty or zero results are ignored', () => {
    expect(ratingsFrom({ PAC: 0, SHO: Number.NaN }).stats).toEqual({})
  })
})
