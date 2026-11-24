import { describe, expect, test } from 'vitest'
import { TESTS, checkResult, looksOff, ratingsFrom, rescored, score, startingPoint, toCardStats } from './testday'
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
    expect(score(t('PAC'), 3.5)).toBe(52)
    expect(score(t('PAC'), 3.05)).toBe(80)
    expect(score(t('SHO'), 6)).toBe(80)
    expect(score(t('PHY'), 245)).toBe(74)
    expect(score(t('PAC'), 3.4)).toBe(57) // halfway between 62 and 52
  })
  test('ratings stay between 30 and 99', () => {
    expect(score(t('PAC'), 2.0)).toBe(99)
    expect(score(t('PAC'), 9)).toBe(30)
    expect(score(t('SHO'), 10)).toBeLessThanOrEqual(99)
  })
  test('a card needs 3 tests; a skipped test stays blank (never filled from an older session)', () => {
    expect(toCardStats(ratingsFrom({ PAC: 3.3, SHO: 7 }))).toBeNull()
    const card = toCardStats(ratingsFrom({ PAC: 3.3, SHO: 7, PAS: 20 }))!
    expect(card.PAC).toBe(62)
    expect(card.DEF).toBeUndefined()
    expect(overall(card)).toBe(Math.round((62 + 86 + 77) / 3)) // no position: a plain average of the tests done
  })
  test('the first session is the starting point', () => {
    const first = ratingsFrom({ PAC: 3.3 }, '2026-11-14'), second = ratingsFrom({ PAC: 3.5 }, '2026-11-22')
    expect(startingPoint({ ratings: second, baseRatings: first })).toBe(first)
    expect(startingPoint({ ratings: second, prevRatings: first })).toBe(first) // older profiles
    expect(startingPoint({ ratings: first })).toBeUndefined()
  })
  test('limits: impossible results are refused, world-class ones need proof', () => {
    expect(checkResult('PAC', 2.4)).toBe('impossible')
    expect(checkResult('PAC', 2.8)).toBe('proof')
    expect(checkResult('PAC', 3.3)).toBeNull()
    expect(checkResult('PHY', 35)).toBe('impossible')
    expect(checkResult('PHY', 310)).toBe('proof')
    expect(checkResult('SHO', 10)).toBe('proof')
    expect(checkResult('SHO', 8)).toBeNull()
    expect(checkResult('PAS', 34)).toBe('proof')
  })
  test('results that look off are flagged: impossible, world-class, or far from the player\'s others', () => {
    // Peter's first try-out: a 35 cm broad jump, a 4.58 s weave, 34 passes, 10/10 shots
    const odd = looksOff(ratingsFrom({ PAC: 3.26, DRI: 4.58, SHO: 10, PAS: 34, DEF: 5.09, PHY: 35 }))
    expect(odd).toContainEqual({ stat: 'PHY', why: 'impossible' })
    expect(odd).toContainEqual({ stat: 'DRI', why: 'impossible' })
    expect(odd).toContainEqual({ stat: 'PAS', why: 'needs proof' })
    expect(odd.some((o) => o.stat === 'PAC')).toBe(false)
    // a believable session flags nothing
    expect(looksOff(ratingsFrom({ PAC: 3.5, DRI: 12, SHO: 6, PAS: 20, DEF: 5.2, PHY: 235 }))).toEqual([])
    // one result far from the rest
    expect(looksOff(ratingsFrom({ PAC: 3.05, DRI: 9, SHO: 8, PAS: 23, DEF: 4.75, PHY: 160 })).map((o) => o.stat)).toEqual(['PHY'])
  })
  test('sessions are scored again from the raw results with the current anchors', () => {
    const old = { date: '2026-11-22', raw: { PAC: 3.66, DRI: 6.19, SHO: 8, PAS: 13, DEF: 5.66, PHY: 252 }, stats: { PAC: 69, DRI: 99, SHO: 83, PAS: 61, DEF: 62, PHY: 83 }, proof: true }
    const r = rescored(old)!
    expect(r.stats).toEqual({ PAC: 47, DRI: 99, SHO: 63, PAS: 55, DEF: 49, PHY: 78 }) // an old session: shots in a full goal
    expect(r.proof).toBe(true)
    expect(rescored({ ...old, v: 2 })!.stats.SHO).toBe(91) // the same 8/10 with the corner-zones test
    expect(toCardStats(old)).toEqual(r.stats)
  })
  test('the overall is weighted for the position, a plain average without one', () => {
    const s = { PAC: 90, SHO: 50, PAS: 50, DRI: 90, DEF: 44, PHY: 60 }
    expect(overall(s)).toBe(64)
    expect(overall(s, ['winger'])).toBe(Math.round(90 * .25 + 50 * .15 + 50 * .2 + 90 * .25 + 44 * .05 + 60 * .1))
    expect(overall(s, ['cb'])).toBe(Math.round(90 * .2 + 50 * .05 + 50 * .15 + 90 * .05 + 44 * .3 + 60 * .25))
    expect(overall(s, ['striker'])).toBeLessThan(overall(s, ['winger']))
    expect(overall({ PAC: 80, SHO: 60, PAS: 70 }, ['striker'])).toBe(Math.round((80 * .2 + 60 * .3 + 70 * .1) / .6)) // a skipped test is left out
    expect(overall(s, ['gk'])).toBe(64)
  })
  test('empty or zero results are ignored', () => {
    expect(ratingsFrom({ PAC: 0, SHO: Number.NaN }).stats).toEqual({})
  })
})
