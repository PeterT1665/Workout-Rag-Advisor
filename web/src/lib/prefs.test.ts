import { beforeEach, describe, expect, test } from 'vitest'
import { BUILTIN_SESSIONS, exerciseBlock } from './library'
import { diff, fitsDay, learn, loadPrefs, withPrefs } from './prefs'
import type { Session } from './types'

const base = () => BUILTIN_SESSIONS.find((s) => s.id === 'gym-upper-core')!
const names = (s: Session) => s.blocks.map((b) => b.name)
const swapIn = (s: Session, from: string, to: string): Session => ({ ...s, blocks: s.blocks.map((b) => (b.name === from ? { ...exerciseBlock(to)!, rounds: b.rounds } : b)) })

describe('your way of doing a session', () => {
  beforeEach(() => localStorage.clear())
  test('edits are found: swap, drop, add, sets', () => {
    const s = base()
    let done = swapIn(s, 'Pull-ups', 'seated-cable-row')
    done = { ...done, blocks: done.blocks.filter((b) => b.name !== 'Face pull') }
    done = { ...done, blocks: done.blocks.map((b) => (b.name === 'Bench press' ? { ...b, rounds: 5 } : b)) }
    const i = done.blocks.findIndex((b) => b.kind === 'cooldown')
    done.blocks.splice(i, 0, exerciseBlock('copenhagen-plank')!)
    const d = diff(s, done)
    expect(d.swap).toEqual({ 'pull-up': 'seated-cable-row' })
    expect(d.drop).toEqual(['face-pull'])
    expect(d.add?.map((a) => a.ex)).toEqual(['copenhagen-plank'])
    expect(d.rounds).toEqual({ 'bench-press': 5 })
  })
  test('learned once, it comes back next time', () => {
    const s = base()
    learn(s, swapIn(s, 'Pull-ups', 'seated-cable-row'))
    const { session, applied } = withPrefs(s)
    expect(names(session)).toContain('Seated cable row')
    expect(names(session)).not.toContain('Pull-ups')
    expect(applied.length).toBe(1)
    // swapping it back undoes it
    learn(session, swapIn(session, 'Seated cable row', 'pull-up'))
    expect(loadPrefs()['gym-upper-core'].swap).toEqual({})
  })
  test('the match-day rules win: no heavy legs added the day before a game', () => {
    const s = BUILTIN_SESSIONS.find((x) => x.id === 'gym-activation')!
    const i = s.blocks.findIndex((b) => b.kind === 'cooldown')
    const done = { ...s, blocks: [...s.blocks.slice(0, i), exerciseBlock('back-squat')!, ...s.blocks.slice(i)] }
    learn(s, done)
    expect(names(withPrefs(s, 'MD-4').session)).toContain('Back squat')
    expect(names(withPrefs(s, 'MD-1').session)).not.toContain('Back squat')
    expect(fitsDay('olympic', 'MD-2')).toBe(false)
    expect(fitsDay('upper-strength', 'MD-2')).toBe(true)
  })
  test('applying it twice changes nothing more', () => {
    const s = base()
    learn(s, swapIn(s, 'Pull-ups', 'seated-cable-row'))
    const once = withPrefs(s).session
    expect(withPrefs(once).session.blocks).toEqual(once.blocks)
  })
})
