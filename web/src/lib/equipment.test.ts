import { describe, expect, test } from 'vitest'
import { BUILTIN_SESSIONS } from './library'
import { DB_SWAP, hasBarbellWork, withDumbbells } from './equipment'
import { EXERCISES } from './library'

describe('dumbbells instead of a barbell', () => {
  test('every swap points at real exercises', () => {
    const ids = new Set(EXERCISES.map((e) => e.id))
    for (const [from, to] of Object.entries(DB_SWAP)) { expect(ids.has(from), from).toBe(true); expect(ids.has(to), to).toBe(true) }
  })
  test('a barbell session swaps its lifts and keeps sets, rest and supersets', () => {
    const s = BUILTIN_SESSIONS.find((x) => x.id === 'gym-contrast')!
    expect(hasBarbellWork(s)).toBe(true)
    const db = withDumbbells(s)
    const i = s.blocks.findIndex((b) => b.name === 'Back squat')
    expect(db.blocks[i].name).toBe('Goblet squat')
    expect(db.blocks[i].rounds).toBe(s.blocks[i].rounds)
    expect(db.blocks.map((b) => !!b.superset)).toEqual(s.blocks.map((b) => !!b.superset))
    expect(db.blocks.length).toBe(s.blocks.length)
  })
  test('sessions without barbell work are left alone', () => {
    const s = BUILTIN_SESSIONS.find((x) => x.type === 'football')!
    expect(hasBarbellWork(s)).toBe(false)
    expect(withDumbbells(s)).toBe(s)
  })
})
