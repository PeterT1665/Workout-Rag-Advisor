import { describe, expect, test } from 'vitest'
import { BUILTIN_SESSIONS } from './library'
import { MAX_HEADERS_PER_SESSION, hasHeading, hasMaxSpeed, headersIn, isHighVolumeShooting, md1Safe, recoverySafe, shotsIn } from './dayrules'
import { buildWeek } from './weekplan'
import { headingAllowed, planToday, sessionForSlot, setPlanHeading, setPlanPositions } from './today'
import { injurySafe } from './injuries'
import { sessionMinutes } from './logic'
import type { Injury } from './profile'

const by = (id: string) => BUILTIN_SESSIONS.find((s) => s.id === id)!
const game = [{ day: 2, kind: 'competitive' as const }]

describe('day rules from the drills research', () => {
  test('the sharp session for the day before a game: at most 45 min, 8 shots, no heading, every drill tagged for it', () => {
    const s = by('fb-sharp-session')
    expect(md1Safe(s)).toBe(true)
    expect(sessionMinutes(s)).toBeLessThanOrEqual(45); expect(shotsIn(s)).toBeLessThanOrEqual(8)
    for (const id of ['fb-heading-session', 'fb-winger-session', 'fb-wall-weakfoot-shooting']) expect(md1Safe(by(id)), id).toBe(false)
  })
  test('a recovery day: the touch session has ball mastery, wall work and juggling only', () => {
    expect(recoverySafe(by('recovery-touch'))).toBe(true)
    expect(recoverySafe(by('fb-speed-acceleration'))).toBe(false)
  })
  test('heading: sessions keep to 10 headers, the heading session counts 9, and nothing else has headers', () => {
    expect(MAX_HEADERS_PER_SESSION).toBe(10)
    for (const s of BUILTIN_SESSIONS) expect(headersIn(s), s.id).toBeLessThanOrEqual(MAX_HEADERS_PER_SESSION)
    expect(BUILTIN_SESSIONS.filter(hasHeading).map((s) => s.id)).toEqual(['fb-heading-session'])
    expect(headersIn(by('fb-heading-session'))).toBe(9)
  })
  test('the week: a heading session at most once, only when allowed; the day before a game is the sharp session', () => {
    const plan = buildWeek(game, { football: 5, gym: 0 })
    setPlanPositions(['cb']); setPlanHeading(true)
    const picks = plan.slots.filter((s) => s.kind === 'football').map((s) => sessionForSlot(s, plan, BUILTIN_SESSIONS)!)
    expect(picks.filter(hasHeading).length).toBeLessThanOrEqual(1)
    expect(picks.filter(isHighVolumeShooting).length).toBeLessThanOrEqual(3)
    setPlanHeading(false)
    expect(plan.slots.filter((s) => s.kind === 'football').map((s) => sessionForSlot(s, plan, BUILTIN_SESSIONS)!).some(hasHeading)).toBe(false)
    // football forced onto the day before the game: the sharp session
    const eve = { day: 1, kind: 'football' as const }
    const crowded = { slots: [{ day: 2, kind: 'game' as const }, eve], reasons: [] }
    expect(sessionForSlot(eve, crowded, BUILTIN_SESSIONS)?.id).toBe('fb-sharp-session')
    setPlanHeading(true); setPlanPositions([])
  })
  test('every in-season week has a true max-speed run', () => {
    for (const pos of [[], ['winger'], ['striker'], ['cb'], ['cm'], ['fb']]) {
      setPlanPositions(pos); setPlanHeading(true)
      for (const games of [game, [{ day: 5, kind: 'competitive' as const }], [{ day: 2, kind: 'casual' as const }, { day: 5, kind: 'competitive' as const }]]) {
        const plan = buildWeek(games, {})
        const picks = plan.slots.filter((s) => s.kind === 'football').map((s) => sessionForSlot(s, plan, BUILTIN_SESSIONS)!)
        expect(picks.some(hasMaxSpeed), `${pos} ${games.length}`).toBe(true)
      }
    }
    setPlanPositions([])
  })
  test('heading is allowed only when cleared and none in the last 7 days', () => {
    const now = new Date(2026, 10, 25)
    const log = (date: string) => [{ id: 'fb-heading-session', title: 'H', date, minutes: 40 }]
    expect(headingAllowed(true, BUILTIN_SESSIONS, [], now)).toBe(false)
    expect(headingAllowed(false, BUILTIN_SESSIONS, [], now)).toBe(true)
    expect(headingAllowed(false, BUILTIN_SESSIONS, log('2026-11-22'), now)).toBe(false)
    expect(headingAllowed(false, BUILTIN_SESSIONS, log('2026-11-10'), now)).toBe(true)
  })
  test('football sessions run warm-up, then sprints and footwork, then technical work, then conditioning', () => {
    const TECH = new Set(['ball-mastery', 'first-touch', 'passing', 'finishing', '1v1', 'crossing', 'defending', 'off-ball'])
    for (const s of BUILTIN_SESSIONS.filter((x) => x.type === 'football' && x.id.startsWith('fb-') && x.positions)) {
      const cats = s.blocks.map((b) => b.category ?? '')
      const lastCond = cats.lastIndexOf('conditioning')
      if (lastCond >= 0) expect(cats.slice(lastCond).some((c) => TECH.has(c)), s.id).toBe(false)
      expect(s.blocks[0].kind === 'warmup' || s.blocks[0].category === 'warmup', s.id).toBe(true)
    }
  })
  test('an ankle or knee injury on file: the full 11+ style warm-up replaces the usual one', () => {
    const inj: Injury = { id: 'a', part: 'ankles', tissue: 'joint', kind: 'pain', severity: 'light', since: '2026-11-26' }
    const s = injurySafe(by('fb-wall-session'), [inj])
    expect(s.session.blocks[0].name).toBe('Warm-up 11+ style: Level 1')
    expect(s.session.blocks.filter((b) => b.kind === 'warmup' || b.category === 'warmup').length).toBe(1)
    expect(s.swaps.some((x) => x.to === 'Warm-up 11+ style: Level 1')).toBe(true)
    const none = injurySafe(by('fb-wall-session'), [{ ...inj, part: 'hamstrings' }])
    expect(none.session.blocks[0].name).not.toMatch(/11\+/)
  })
  test('today still plans a session on a football day (smoke)', () => {
    expect(planToday(BUILTIN_SESSIONS, game, new Date(2026, 10, 23, 9)).session.type).toBe('football')
  })
})
