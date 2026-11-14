import { describe, expect, test } from 'vitest'
import { planStreak } from './streak'
import { buildWeek } from './weekplan'
import type { HistoryEntry } from './storage'

const games = [{ day: 2, kind: 'competitive' as const }] // Wednesday game
const h = (date: string): HistoryEntry => ({ id: 's', title: 'S', date, minutes: 30 })
const at = (date: string) => new Date(`${date}T12:00:00`)
// 2026-11-16 is a Monday
const kinds = buildWeek(games, { gym: 3 }).slots

describe('plan streak', () => {
  test('nothing logged: no streak', () => {
    expect(planStreak([], games, 3, {}, at('2026-11-18'))).toEqual({ days: 0, today: false })
  })

  test('training days need a session; rest and game days keep the streak', () => {
    const week = ['2026-11-16', '2026-11-17', '2026-11-18', '2026-11-19', '2026-11-20', '2026-11-21', '2026-11-22']
    const planned = (i: number) => kinds.some((s) => s.day === i && s.kind !== 'game') && !kinds.some((s) => s.day === i && s.kind === 'game')
    const log = week.filter((_, i) => planned(i)).map(h)
    expect(planStreak(log, games, 3, {}, at('2026-11-22')).days).toBe(7 - week.findIndex((_, i) => planned(i)))
  })

  test('today not done yet does not break the streak', () => {
    const t = planStreak([h('2026-11-15')], games, 3, {}, at('2026-11-16'))
    expect(t.today).toBe(false)
    expect(t.days).toBe(1)
  })

  test('a day changed to rest counts as kept', () => {
    const s = planStreak([h('2026-11-16')], games, 3, { '2026-11-17': { rest: true } }, at('2026-11-17'))
    expect(s).toEqual({ days: 2, today: true })
  })

  // week of 2026-11-16: Mon football, Tue gym, Wed game, Thu recovery, Fri football, Sat gym, Sun gym
  const h2 = (date: string, minutes: number): HistoryEntry => ({ id: 's', title: 'S', date, minutes })

  test('any session keeps a training day (no pitch, gym; no gym, home)', () => {
    expect(planStreak([h2('2026-11-16', 25), h2('2026-11-17', 15)], games, 3, {}, at('2026-11-17'))).toMatchObject({ days: 2, today: true })
  })

  test('a missed day is open for a make-up for 2 days, then breaks the streak', () => {
    const log = [h2('2026-11-16', 40)] // Tuesday missed
    const wed = planStreak(log, games, 3, {}, at('2026-11-18'))
    expect(wed).toMatchObject({ days: 3, makeup: { date: '2026-11-17', by: '2026-11-19', kind: 'gym' } })
    // Thursday's own session isn't extra: Tuesday's window closes, the streak restarts after it (Wed, Thu)
    expect(planStreak([...log, h2('2026-11-19', 20)], games, 3, {}, at('2026-11-20'))).toEqual({ days: 2, today: false })
  })

  test('an extra session in time makes the missed day up', () => {
    const log = [h2('2026-11-16', 40), h2('2026-11-18', 30)] // Tue missed, extra home workout on game day Wed
    const s = planStreak(log, games, 3, {}, at('2026-11-20'))
    expect(s).toMatchObject({ days: 4, makeup: { date: '2026-11-19' } }) // Mon, Tue made up, Wed, Thu (missed, still open)
    const two = [h2('2026-11-16', 40), h2('2026-11-19', 20), h2('2026-11-19', 30)] // Thu: recovery + an extra 30 min
    expect(planStreak(two, games, 3, {}, at('2026-11-19'))).toMatchObject({ days: 4, today: true })
    expect(planStreak(two, games, 3, {}, at('2026-11-19')).makeup).toBeUndefined()
  })

  test('a short extra session does not count as a make-up', () => {
    const log = [h2('2026-11-16', 40), h2('2026-11-18', 10)]
    expect(planStreak(log, games, 3, {}, at('2026-11-20')).makeup?.date).toBe('2026-11-19') // Tue gone (window closed), Thu open
  })
})
