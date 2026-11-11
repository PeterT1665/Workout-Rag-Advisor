import { describe, expect, test } from 'vitest'
import { TESTS, ratingsFrom, score, toCardStats } from './testday'

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
  test('a card needs 3 tests; skipped ones are filled from the last test day, then the average', () => {
    const first = ratingsFrom({ PAC: 3.3, SHO: 7 })
    expect(toCardStats(first)).toBeNull()
    const full = ratingsFrom({ PAC: 3.3, SHO: 7, PAS: 20 })
    const card = toCardStats(full)!
    expect(card.PAC).toBe(82)
    expect(card.DEF).toBe(Math.round((82 + 76 + 77) / 3))
    const next = ratingsFrom({ DRI: 12, DEF: 4.9, PHY: 235 })
    expect(toCardStats(next, full)).toMatchObject({ PAC: 82, DRI: 80, DEF: 80, PHY: 76 })
  })
  test('empty or zero results are ignored', () => {
    expect(ratingsFrom({ PAC: 0, SHO: Number.NaN }).stats).toEqual({})
  })
})
