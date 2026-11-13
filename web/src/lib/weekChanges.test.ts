import { beforeEach, expect, test } from 'vitest'
import { changeFor, changedSession, dateOfDay, loadWeekChanges, saveWeekChange } from './weekChanges'
import { BUILTIN_SESSIONS } from './library'

beforeEach(() => localStorage.clear())
const thu = new Date(2026, 10, 12, 9) // Thursday 12 Nov 2026

test('days map to this week\'s dates, Monday first', () => {
  expect(dateOfDay(0, thu)).toBe('2026-11-09')
  expect(dateOfDay(6, thu)).toBe('2026-11-15')
})

test('a change applies to that date only, and can be reset', () => {
  saveWeekChange(3, { sessionId: 'gym-lower-power' }, thu)
  const all = loadWeekChanges()
  expect(changedSession(changeFor(3, all, thu), BUILTIN_SESSIONS)?.id).toBe('gym-lower-power')
  expect(changeFor(3, all, new Date(2026, 10, 19, 9))).toBeNull() // next week's Thursday is untouched
  saveWeekChange(3, null, thu)
  expect(changeFor(3, loadWeekChanges(), thu)).toBeNull()
})

test('old changes are dropped after about 8 weeks', () => {
  saveWeekChange(0, { rest: true }, new Date(2026, 6, 1))
  saveWeekChange(0, { rest: true }, thu)
  expect(Object.keys(loadWeekChanges())).toEqual(['2026-11-09'])
})
