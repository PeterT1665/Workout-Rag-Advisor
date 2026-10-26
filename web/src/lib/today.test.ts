import { describe, expect, test } from 'vitest'
import data from '../data/sessions.json'
import { planToday } from './today'
import { kitFor } from './kit'
import type { Session } from './types'

const sessions = data.sessions as Session[]
// 2026-10-26 is a Monday
const monday = new Date(2026, 9, 26, 9)
const wednesday = new Date(2026, 9, 28, 9)
const thursday = new Date(2026, 9, 29, 9)

describe('today', () => {
  test('a football day gets a solo football session and a matchday label', () => {
    const t = planToday(sessions, [{ day: 2, kind: 'competitive' }], monday)
    expect(t.slot?.kind).toBe('football')
    expect(t.scheduled).toBe(true)
    expect(t.session.type).toBe('football')
    expect(t.session.id).not.toContain('partner')
    expect(t.md).toBe('MD-2')
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
})
