import { describe, expect, test } from 'vitest'
import { BUILTIN_SESSIONS, EXERCISES, TEMPLATES, categories, inferType, toSession } from './library'

describe('library', () => {
  test('every template only uses exercises that exist', () => {
    const ids = new Set(EXERCISES.map((e) => e.id))
    for (const t of TEMPLATES) for (const it of t.items) expect(ids.has(it.ex), `${t.id} -> ${it.ex}`).toBe(true)
  })
  test('exercise ids are unique', () => {
    expect(new Set(EXERCISES.map((e) => e.id)).size).toBe(EXERCISES.length)
  })
  test('item overrides win over exercise defaults; needs are derived when not given', () => {
    const s = toSession({ id: 'my-x', title: 'X', type: 'football', summary: '', needs: [], items: [{ ex: 'cone-weave', rounds: 2 }, { ex: 'nope' }] })
    expect(s.blocks).toHaveLength(1) // unknown exercise skipped
    expect(s.blocks[0]).toMatchObject({ name: 'Cone weave dribble', rounds: 2, work: 45 })
    expect(s.needs).toContain('ball')
  })
  test('session type follows content', () => {
    expect(inferType([{ ex: 'back-squat' }, { ex: 'cone-weave' }])).toBe('football')
    expect(inferType([{ ex: 'back-squat' }])).toBe('gym')
    expect(inferType([{ ex: 'glute-bridge' }])).toBe('recovery')
  })
  test('each sport has categories and the gym has recommended sessions', () => {
    expect(categories('football')).toContain('ball-mastery')
    const gym = categories('gym')
    expect(gym[0]).toBe('olympic')
    expect(gym.slice(-2)).toEqual(['warmup', 'cooldown']) // warm-up and cool-down sit at the end of the picker
    for (const c of ['plyometrics', 'sprints', 'isometrics', 'single-limb']) expect(gym).toContain(c)
    expect(BUILTIN_SESSIONS.filter((s) => s.type === 'gym').length).toBeGreaterThanOrEqual(3)
  })
})
