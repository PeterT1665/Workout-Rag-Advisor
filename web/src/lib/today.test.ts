import { describe, expect, test } from 'vitest'
import { BUILTIN_SESSIONS as sessions } from './library'
import { planToday, sessionForSlot } from './today'
import { kitFor } from './kit'
import { buildWeek } from './weekplan'
import type { Session } from './types'

// 2026-10-26 is a Monday
const monday = new Date(2026, 9, 26, 9)
const wednesday = new Date(2026, 9, 28, 9)
const thursday = new Date(2026, 9, 29, 9)

describe('today', () => {
  test('a football day gets a solo football session and a matchday label', () => {
    const games = [{ day: 2, kind: 'competitive' as const }]
    const fbDay = buildWeek(games).slots.find((x) => x.kind === 'football' && !buildWeek(games).slots.some((y) => y.day === x.day && y.kind === 'gym'))!.day
    const t = planToday(sessions, games, new Date(2026, 9, 26 + fbDay, 9))
    expect(t.slot?.kind).toBe('football')
    expect(t.scheduled).toBe(true)
    expect(t.session.type).toBe('football')
    expect(t.session.id).not.toContain('partner')
    expect(t.md).toMatch(/^MD[+-]\d$/)
  })
  test('a football + gym day moves on to the gym once the football is done', () => {
    const games = [{ day: 2, kind: 'competitive' as const }]
    const slots = buildWeek(games).slots
    const both = slots.find((x) => x.kind === 'football' && slots.some((y) => y.day === x.day && y.kind === 'gym'))!.day
    const date = new Date(2026, 9, 26 + both, 9)
    expect(planToday(sessions, games, date).session.type).toBe('football')
    const after = planToday(sessions, games, date, undefined, ['football'])
    expect(after.slot?.kind).toBe('gym')
    expect(after.session.type).toBe('gym')
    expect(after.scheduled).toBe(true)
    expect(after.done).toBeUndefined()
    expect(planToday(sessions, games, date, undefined, ['football', 'gym']).done).toBe(true)
  })
  test('game day is not a training day; Kick off offers the next session', () => {
    const t = planToday(sessions, [{ day: 2, kind: 'competitive' }], wednesday)
    expect(t.slot?.kind).toBe('game')
    expect(t.scheduled).toBe(false)
    expect(t.md).toBe('MD')
  })
  test('the day after a game is recovery', () => {
    const t = planToday(sessions, [{ day: 2, kind: 'competitive' }], thursday)
    expect(t.slot?.kind).toBe('recovery')
    expect(t.session.type).toBe('recovery')
  })
  test('kit follows the session needs', () => {
    const ids = (s: Session) => kitFor(s).map((k) => k.id)
    expect(ids(sessions.find((s) => s.id === 'fb-close-control-1v1')!)).toEqual(['boots', 'ball', 'discs', 'water'])
    expect(ids(sessions.find((s) => s.type === 'recovery')!)).toContain('mat')
  })
  test('gym days get a recommended gym session', () => {
    const t = planToday(sessions, [{ day: 2, kind: 'competitive' }], monday)
    const gym = t.plan.slots.find((s) => s.kind === 'gym')!
    expect(sessionForSlot(gym, t.plan, sessions)?.type).toBe('gym')
  })
})
