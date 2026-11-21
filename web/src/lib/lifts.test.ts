import { describe, expect, test } from 'vitest'
import { bestsFrom, cleanLifts, e1rm, loggable, moved, planLifts, scoreLifts, sessionEffort, type LiftLog } from './lifts'
import type { HistoryEntry } from './storage'
import type { Session } from './types'

const bench = (sets: [number | null, number | null][]): LiftLog => ({ ex: 'bench-press', name: 'Bench press', sets: sets.map(([kg, reps]) => ({ kg, reps })) })
const entry = (lifts: LiftLog[], rpe: number, bw?: number): HistoryEntry => ({ id: 'gym', title: 'Gym', date: '2026-11-01', minutes: 50, rpe, lifts, bw })

describe('what gets logged', () => {
  test('strength and loaded power, not holds, sprints or med-ball throws', () => {
    expect(loggable({ name: 'Bench press', reps: '6' })).toBe(true)
    expect(loggable({ name: 'Bulgarian split squat', reps: '8 each leg' })).toBe(true)
    expect(loggable({ name: 'Plank', reps: '45 s' })).toBe(false)
    expect(loggable({ name: 'Box jump', reps: '3' })).toBe(false)
    expect(loggable({ name: 'Med ball slam', reps: '8' })).toBe(false)
    expect(loggable({ name: 'Juggling' })).toBe(false)
  })
  test('bodyweight exercises count your body; barbell lifts need a weight', () => {
    expect(moved({ ex: 'push-up', name: 'Push-ups', sets: [] }, { kg: null, reps: 15 }, 80)).toBeCloseTo(52)
    expect(moved({ ex: 'pull-up', name: 'Pull-ups', sets: [] }, { kg: 10, reps: 6 }, 80)).toBeCloseTo(86)
    expect(moved(bench([]), { kg: null, reps: 6 })).toBeNull()
    expect(moved(bench([]), { kg: 60, reps: 0 })).toBeNull()
  })
})

describe('effort is measured against your own max', () => {
  test('first time: the effort you report calibrates your max', () => {
    const { lifts, measured } = scoreLifts([bench([[60, 6], [60, 6], [60, 6]])], 8, new Map())
    expect(lifts[0].max).toBeCloseTo(e1rm(60, 8)) // effort 8 ≈ 2 reps left
    expect(lifts[0].effort).toBe(8)
    expect(measured).toBe(8)
    expect(lifts[0].best).toBe(false) // nothing to beat yet
  })
  test('the same 60 kg is harder for a weaker player', () => {
    const strong = bestsFrom([entry([bench([[100, 5]])], 9)])
    const weak = bestsFrom([entry([bench([[65, 5]])], 9)])
    const s = scoreLifts([bench([[60, 6]])], 7, strong).lifts[0]
    const w = scoreLifts([bench([[60, 6]])], 7, weak).lifts[0]
    expect(w.intensity).toBeGreaterThan(s.intensity)
    expect(w.effort).toBeGreaterThan(s.effort)
    expect(s.effort).toBeLessThan(5)
    expect(w.effort).toBe(7) // 60 of a 78 kg max: about 3 reps left
  })
  test('lifting more than before is a new best', () => {
    const prior = bestsFrom([entry([bench([[60, 6]])], 8)])
    const r = scoreLifts([bench([[65, 6]])], 8, prior).lifts[0]
    expect(r.best).toBe(true)
    expect(r.intensity).toBeCloseTo(65 / r.max)
    expect(r.volume).toBe(390)
  })
  test('session effort: half what you felt, half what your sets say', () => {
    expect(sessionEffort(6, 8)).toBe(7)
    expect(sessionEffort(6, null)).toBe(6)
  })
  test('sets without anything to measure are ignored', () => {
    expect(scoreLifts([bench([[null, 6]])], 7, new Map())).toEqual({ lifts: [], measured: null })
  })
})

describe('the check-in form', () => {
  const session: Session = {
    id: 'g', title: 'Gym', type: 'gym', summary: '', needs: [],
    blocks: [
      { name: 'Jog + dynamic leg swings', kind: 'warmup', work: 180 },
      { name: 'Bench press', kind: 'drill', work: 40, reps: '6', rounds: 4 },
      { name: 'Plank', kind: 'drill', work: 45, reps: '45 s', rounds: 3 },
    ],
  }
  test("plan reps, or last time's numbers", () => {
    expect(planLifts(session, [])).toEqual([{ ex: 'bench-press', name: 'Bench press', sets: Array(4).fill({ kg: null, reps: 6 }) }])
    const last = planLifts(session, [entry([bench([[60, 6], [62.5, 5]])], 8)])
    expect(last[0].sets).toEqual([{ kg: 60, reps: 6 }, { kg: 62.5, reps: 5 }, { kg: 62.5, reps: 5 }, { kg: 62.5, reps: 5 }])
  })
  test('emptied rows are sets not done', () => {
    expect(cleanLifts([bench([[60, 6], [60, null]]), bench([[null, null]])])).toEqual([bench([[60, 6]])])
  })
})

test('adding your body weight later is not a new best on the same pull-ups', () => {
  const pull = (reps: number[]): LiftLog => ({ ex: 'pull-up', name: 'Pull-ups', sets: reps.map((r) => ({ kg: null, reps: r })) })
  const prior = bestsFrom([entry([pull([8, 7])], 8)], 72.5) // saved before any weigh-in
  expect(scoreLifts([pull([8, 7, 7])], 8, prior, 72.5).lifts[0].best).toBe(false)
})
