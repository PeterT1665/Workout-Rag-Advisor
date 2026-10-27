import { beforeEach, describe, expect, test } from 'vitest'
import { buildDemoHistory, seedDemo } from './demo'
import { loadHistory, weekLoads } from './storage'

beforeEach(() => localStorage.clear())

describe('demo sample data', () => {
  test('history is in the past, consistent, and uses real sessions', () => {
    const now = new Date(2026, 9, 28, 12) // Wednesday
    const h = buildDemoHistory(now)
    expect(h.length).toBeGreaterThanOrEqual(10)
    for (const e of h) {
      expect(e.date <= '2026-10-28').toBe(true)
      expect(e.load).toBe(Math.round(e.minutes * (e.rpe ?? 0)))
      expect(e.title).not.toBe(e.id) // every id maps to a real session title
    }
  })
  test('the weekly chart always has data, including on a Monday', () => {
    for (const day of [26, 27, 28, 29, 30, 31, 25]) {
      const now = new Date(2026, 9, day, 12)
      localStorage.clear()
      seedDemo(now)
      expect(weekLoads(loadHistory(), now).some((v) => v > 0)).toBe(true)
    }
  })
  test('seeds once per day and can be forced', () => {
    const now = new Date(2026, 9, 28, 12)
    seedDemo(now)
    localStorage.setItem('history', '[]')
    seedDemo(now) // same day: keeps the visitor's state
    expect(loadHistory()).toEqual([])
    seedDemo(now, true)
    expect(loadHistory().length).toBeGreaterThan(5)
  })
})
