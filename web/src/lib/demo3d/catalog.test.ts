import { describe, expect, test } from 'vitest'
import data from '../../data/library.json'
import { DEMOS, demoFor } from './catalog'
import { MOVES } from './moves'
import { poseAt } from './pose'

describe('3D demos', () => {
  test('every exercise in the library has a demo, and every demo a move', () => {
    const ids = (data.exercises as { id: string }[]).map((e) => e.id)
    expect(ids.filter((id) => !DEMOS[id])).toEqual([])
    expect(Object.values(DEMOS).filter((d) => !MOVES[d.move]).map((d) => d.move)).toEqual([])
  })
  test('found by the names sessions use', () => {
    expect(demoFor('Bench press')?.move).toBe('bench-press')
    expect(demoFor('1v1 reaction (voice cues)')?.move).toBe('1v1')
    expect(demoFor('Something I made up')).toBeNull()
  })
  test('keyframes are in order and inside the loop', () => {
    for (const [id, m] of Object.entries(MOVES)) {
      const ts = m.keys.map((k) => k[0])
      expect(ts.every((t, i) => t >= 0 && t < 1 && (i === 0 || t > ts[i - 1])), id).toBe(true)
      expect(poseAt(m.keys, 0.37).pose).toBeTruthy()
    }
  })
})
