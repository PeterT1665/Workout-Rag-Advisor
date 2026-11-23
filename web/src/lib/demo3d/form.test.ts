import { describe, expect, test } from 'vitest'
import { FORM_MOVES, formFor, withMistake } from './form'
import { MOVES } from './moves'

describe('form check', () => {
  test('every form entry is a real move, with a cue', () => {
    for (const id of FORM_MOVES()) {
      expect(MOVES[id], id).toBeTruthy()
      expect(formFor(id)!.cue.length).toBeGreaterThan(10)
    }
  })
  test('a mistake grows with the rep: none at the lockout, full at the bottom', () => {
    const m = formFor('rdl')!.mistakes[0]
    const bad = withMistake(MOVES.rdl, m)
    const spine = (i: number) => (bad.keys[i][1].spine ?? 0) - (MOVES.rdl.keys[i][1].spine ?? 0)
    const added = bad.keys.map((_, i) => spine(i))
    expect(Math.min(...added)).toBe(0) // standing tall: no rounding
    expect(Math.max(...added)).toBeCloseTo(40) // deepest point: the full mistake
  })
  test('a start mistake only changes the first key (pull-up half reps)', () => {
    const bad = withMistake(MOVES['pull-up'], formFor('pull-up')!.mistakes[0])
    expect(bad.keys[0][1]).not.toEqual(MOVES['pull-up'].keys[0][1])
    bad.keys.slice(1).forEach(([, p], i) => expect(p).toEqual(MOVES['pull-up'].keys[i + 1][1]))
  })
  test('the move itself is left alone', () => {
    const before = JSON.stringify(MOVES['squat-back'])
    withMistake(MOVES['squat-back'], formFor('squat-back')!.mistakes[0])
    expect(JSON.stringify(MOVES['squat-back'])).toBe(before)
  })
})
