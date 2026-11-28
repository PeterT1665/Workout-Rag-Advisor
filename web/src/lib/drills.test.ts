import { describe, expect, test } from 'vitest'
import { FAMILIES, drillExercises, drillId } from './drills'
import { BUILTIN_SESSIONS, CATEGORY_LABELS, EXERCISES } from './library'
import { MOVES } from './demo3d/moves'
import { DEMOS } from './demo3d/catalog'
import { planToday, setPlanPositions } from './today'

describe('drill families', () => {
  test('every family and level is complete, with a known category, demo move and equipment', () => {
    expect(FAMILIES.length).toBeGreaterThanOrEqual(25)
    expect(new Set(FAMILIES.map((f) => f.id)).size).toBe(FAMILIES.length)
    for (const f of FAMILIES) {
      expect(CATEGORY_LABELS[f.category], f.id).toBeTruthy()
      expect(MOVES[f.demo], `${f.id} demo`).toBeTruthy()
      expect(f.levels.length, f.id).toBeGreaterThan(0)
      for (const l of f.levels) for (const k of ['name', 'how', 'mistake', 'fix'] as const) expect(l[k].length, `${f.id} ${l.name} ${k}`).toBeGreaterThan(k === 'name' ? 1 : 4)
      for (const l of f.levels) { expect(l.work).toBeGreaterThan(0); expect(l.rounds).toBeGreaterThan(0) }
    }
  })
  test('each level is a library exercise with a demo and a mistake and fix; no heading drills yet (the heading cap is not built)', () => {
    const ex = drillExercises()
    expect(ex.length).toBe(FAMILIES.reduce((a, f) => a + f.levels.length, 0))
    for (const e of ex) { expect(EXERCISES.some((x) => x.id === e.id), e.id).toBe(true); expect(DEMOS[e.id], e.id).toBeTruthy(); expect(e.mistake).toBeTruthy() }
    expect(ex.filter((e) => /\bhead(er|ing)s?\b/i.test(e.how)).map((e) => e.id)).toEqual([])
    expect(drillId(FAMILIES[0], 0)).toBe('wall-ladder-1')
  })
  test('position sessions use the families and the football picker prefers the player\'s position', () => {
    const fb = BUILTIN_SESSIONS.filter((s) => s.positions)
    expect(fb.map((s) => s.id).sort()).toEqual(['fb-defender-session', 'fb-fullback-session', 'fb-heading-session', 'fb-midfielder-session', 'fb-striker-session', 'fb-winger-session'])
    const wednesdayGame = [{ day: 2, kind: 'competitive' as const }]
    const monday = new Date(2026, 10, 23, 9) // a football day in season
    setPlanPositions(['striker'])
    const sessions = [planToday(BUILTIN_SESSIONS, wednesdayGame, monday)]
    expect(sessions[0].session.id).toBe('fb-striker-session')
    setPlanPositions(['cb'])
    expect(planToday(BUILTIN_SESSIONS, wednesdayGame, monday).session.id).toBe('fb-defender-session')
    setPlanPositions([])
    expect(planToday(BUILTIN_SESSIONS, wednesdayGame, monday).session.positions).toBeUndefined() // no position: only the general sessions
  })
})
