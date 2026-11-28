import { beforeEach, describe, expect, test } from 'vitest'
import { SCENARIOS, answerDaily, answerFor, dailyStreak, dayNumber, optionOrder, scenarioFor, shareText } from './daily'

beforeEach(() => localStorage.clear())
const day = (n: number) => new Date(2026, 10, n, 9)

describe('daily play', () => {
  test('the pool is hand-written and checked: every scenario has 3 or 4 options, exactly one best, and sits on the pitch', () => {
    expect(SCENARIOS.length).toBeGreaterThanOrEqual(20)
    expect(new Set(SCENARIOS.map((s) => s.id)).size).toBe(SCENARIOS.length)
    for (const s of SCENARIOS) {
      expect(s.options.length, s.id).toBeGreaterThanOrEqual(3); expect(s.options.length).toBeLessThanOrEqual(4)
      expect(s.best, s.id).toBeGreaterThanOrEqual(0); expect(s.best).toBeLessThan(s.options.length)
      for (const [x, y] of [s.you, ...s.mates, ...s.opps, ...s.options.map((o) => o.to)]) { expect(x).toBeGreaterThanOrEqual(0); expect(x).toBeLessThanOrEqual(100); expect(y).toBeGreaterThanOrEqual(0); expect(y).toBeLessThanOrEqual(64) }
      for (const o of s.options) { expect(o.label.length).toBeGreaterThan(8); expect(o.why.length, `${s.id} ${o.label}`).toBeGreaterThan(20) }
      expect(s.for.length).toBeGreaterThan(0)
    }
  })
  test('the same scenario for everyone on a day, a new one the next day, and it cycles through the pool', () => {
    expect(scenarioFor(day(20)).id).toBe(scenarioFor(day(20)).id)
    expect(scenarioFor(day(20)).id).not.toBe(scenarioFor(day(21)).id)
    expect(new Set(Array.from({ length: SCENARIOS.length }, (_, i) => scenarioFor(day(1 + i)).id)).size).toBe(SCENARIOS.length)
    expect(dayNumber(new Date(2026, 0, 1))).toBe(1)
  })
  test('the options are shuffled the same way for everyone, and all of them stay', () => {
    const a = optionOrder(300, 4), b = optionOrder(300, 4)
    expect(a).toEqual(b); expect([...a].sort()).toEqual([0, 1, 2, 3])
    const firsts = new Set(Array.from({ length: 40 }, (_, d) => optionOrder(d, 4)[0])); expect(firsts.size).toBeGreaterThan(1)
  })
  test('an answer is saved with whether it was right; the streak counts days played', () => {
    const s = scenarioFor(day(20))
    expect(answerFor(day(20))).toBeNull()
    expect(answerDaily(s.best, day(20)).correct).toBe(true)
    expect(answerFor(day(20))).toMatchObject({ id: s.id, picked: s.best, correct: true })
    answerDaily((scenarioFor(day(21)).best + 1) % 3, day(21))
    expect(dailyStreak(day(21))).toBe(2) // right or wrong, both days were played
    expect(dailyStreak(day(22))).toBe(2) // today still to play: yesterday keeps it
    expect(dailyStreak(day(24))).toBe(0)
  })
  test('the share text is short and has no answer in it', () => {
    expect(shareText(day(20), { id: 'x', picked: 0, correct: true }, 3)).toBe(`Daily play #${dayNumber(day(20))} ✅ 🔥3`)
    expect(shareText(day(20), { id: 'x', picked: 0, correct: false }, 1)).toMatch(/❌$/)
  })
})
