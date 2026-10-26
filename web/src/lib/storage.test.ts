import { beforeEach, describe, expect, test } from 'vitest'
import { addHistory, loadHistory, reflectLast, sessionLoad, weekLoads } from './storage'

beforeEach(() => localStorage.clear())

describe('training load', () => {
  test('session load is minutes x effort', () => {
    expect(sessionLoad(64, 7)).toBe(448)
  })
  test('reflection updates the latest session only', () => {
    addHistory({ id: 'a', title: 'A', date: '2026-10-26', minutes: 30 })
    addHistory({ id: 'b', title: 'B', date: '2026-10-27', minutes: 60 })
    const e = reflectLast({ rpe: 6, legs: 'heavy', sore: ['Knee'] })
    expect(e?.load).toBe(360)
    const h = loadHistory()
    expect(h[0].load).toBeUndefined()
    expect(h[1]).toMatchObject({ rpe: 6, legs: 'heavy', sore: ['Knee'], load: 360 })
  })
  test('week loads are bucketed Monday to Sunday', () => {
    const wed = new Date(2026, 9, 28, 12) // Wednesday 28 Oct 2026
    const loads = weekLoads([
      { id: 'x', title: 'X', date: '2026-10-26', minutes: 60, load: 300 }, // Mon
      { id: 'y', title: 'Y', date: '2026-10-28', minutes: 40, load: 200 }, // Wed
      { id: 'z', title: 'Z', date: '2026-10-25', minutes: 40, load: 999 }, // previous Sunday: excluded
      { id: 'w', title: 'W', date: '2026-10-28', minutes: 40 },            // no check-in: ignored
    ], wed)
    expect(loads).toEqual([300, 0, 200, 0, 0, 0, 0])
  })
})
