import { describe, expect, test } from 'vitest'
import { BUILTIN_SESSIONS } from './library'
import { activeInjuries, injurySafe, loadsPart } from './injuries'
import type { Injury } from './profile'

const day = new Date().toLocaleDateString('en-CA')
const knee: Injury = { id: 'k', part: 'knees', tissue: 'joint', kind: 'pain', since: day }
const ham: Injury = { id: 'h', part: 'hamstrings', tissue: 'muscle', kind: 'strain', since: day }

describe('kick off with an injury', () => {
  test('exercises that load the injured part are swapped for safe ones of the same kind', () => {
    const s = BUILTIN_SESSIONS.find((x) => x.id === 'gym-lower-power')!
    const { session, swaps } = injurySafe(s, [knee])
    expect(swaps.length).toBeGreaterThan(0)
    for (const b of session.blocks.filter((x) => x.kind === 'drill')) expect(loadsPart(b, 'knees'), b.name).toBe(false)
    expect(session.blocks.length).toBe(s.blocks.length - swaps.filter((w) => !w.to).length)
    expect(session.blocks[0].kind).toBe('warmup')
  })
  test('a moderate leg injury also takes out jumps, sprints and Olympic lifts', () => {
    const s = BUILTIN_SESSIONS.find((x) => x.id === 'gym-explosive')!
    const { session } = injurySafe(s, [{ ...ham, severity: 'moderate' }])
    expect(session.blocks.filter((b) => ['plyometrics', 'sprints', 'olympic', 'explosive'].includes(b.category ?? ''))).toEqual([])
  })
  test('light: only exercises that mainly work the part go; serious: no session', () => {
    const s = BUILTIN_SESSIONS.find((x) => x.id === 'gym-lower-power')!
    const light = injurySafe(s, [{ ...knee, severity: 'light' }])
    const moderate = injurySafe(s, [{ ...knee, severity: 'moderate' }])
    expect(light.swaps.length).toBeLessThan(moderate.swaps.length)
    expect(injurySafe(s, [{ ...knee, severity: 'serious' }]).blocked?.id).toBe('k')
  })
  test('light injuries clear after 2 days', () => {
    const old = { ...knee, severity: 'light' as const, since: '2026-01-01' }
    expect(activeInjuries({ injuries: [old] } as never, new Date(2026, 0, 2))).toHaveLength(1)
    expect(activeInjuries({ injuries: [old] } as never, new Date(2026, 0, 5))).toHaveLength(0)
  })
  test('no injury, no change', () => {
    const s = BUILTIN_SESSIONS[0]
    expect(injurySafe(s, []).session).toBe(s)
  })
  test('supersets stay valid when an exercise is swapped or left out', () => {
    const s = BUILTIN_SESSIONS.find((x) => x.id === 'gym-contrast')!
    const { session } = injurySafe(s, [ham, knee])
    session.blocks.forEach((b, k) => { if (b.superset) expect(session.blocks[k - 1]?.kind).toBe('drill') })
  })
})
