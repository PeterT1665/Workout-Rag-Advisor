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

  test('a missed training day breaks it; today not done yet does not', () => {
    const missed = kinds.find((s) => s.kind !== 'game' && s.day > 0 && s.day < 6)!.day // a planned training day
    const dates = ['2026-11-16', '2026-11-17', '2026-11-18', '2026-11-19', '2026-11-20', '2026-11-21']
    const log = dates.filter((_, i) => i !== missed).map(h)
    const s = planStreak(log, games, 3, {}, at('2026-11-21'))
    expect(s.days).toBeLessThanOrEqual(6 - missed)
    // today (a training day) not done: yesterday's streak still stands, flame not lit
    const t = planStreak([h('2026-11-15')], games, 3, {}, at('2026-11-16'))
    expect(t.today).toBe(false)
    expect(t.days).toBe(1)
  })

  test('a day changed to rest counts as kept', () => {
    const s = planStreak([h('2026-11-16')], games, 3, { '2026-11-17': { rest: true } }, at('2026-11-17'))
    expect(s).toEqual({ days: 2, today: true })
  })
})
