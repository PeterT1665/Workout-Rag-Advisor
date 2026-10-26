import { describe, expect, test } from 'vitest'
import * as L from './logic'
import type { Direction, Session } from './types'

const session: Session = {
  id: 's', title: 'S', type: 'football', summary: '', needs: [],
  blocks: [
    { name: 'Warm', kind: 'warmup', work: 60 },
    { name: 'Juggle', kind: 'drill', work: 30, rest: 10, rounds: 3 },
    { name: '1v1', kind: 'drill', work: 20, rest: 20, rounds: 2, cue: true },
  ],
}

describe('session logic', () => {
  test('flatten: rounds and rests, no trailing rest', () => {
    const steps = L.flattenSession(session)
    expect(steps.filter((s) => s.type === 'work')).toHaveLength(6)
    expect(steps.at(-1)!.type).toBe('work')
    expect(L.totalSeconds(steps)).toBe(60 + 90 + 30 + 40 + 20)
  })

  test('only cue blocks carry cues', () => {
    const names = new Set(L.flattenSession(session).filter((s) => s.cue).map((s) => s.name))
    expect([...names]).toEqual(['1v1'])
  })

  test('cue direction never repeats more than 3 times in a row', () => {
    const hist: Direction[] = []
    for (let i = 0; i < 2000; i++) hist.push(L.pickCue(hist))
    for (let i = 3; i < hist.length; i++) {
      expect(hist[i] === hist[i - 1] && hist[i] === hist[i - 2] && hist[i] === hist[i - 3]).toBe(false)
    }
    expect(hist).toContain('left')
    expect(hist).toContain('right')
  })

  test('cue gaps stay in the window and end before the step does', () => {
    const times = L.planCues(30, 3, 7)
    expect(times.length).toBeGreaterThanOrEqual(3)
    times.forEach((t) => expect(t).toBeLessThan(28))
    for (let i = 1; i < times.length; i++) {
      const gap = times[i] - times[i - 1]
      expect(gap).toBeGreaterThanOrEqual(3 - 1e-9)
      expect(gap).toBeLessThanOrEqual(7 + 1e-9)
    }
  })

  test('ics has required fields and alarm', () => {
    const ics = L.makeICS([{ title: 'Football, skills', start: new Date(Date.UTC(2026, 9, 26, 17, 0)), minutes: 65 }])
    expect(ics).toMatch(/DTSTART:20261026T170000Z/)
    expect(ics).toMatch(/DTEND:20261026T180500Z/)
    expect(ics).toMatch(/SUMMARY:Football\\, skills/)
    expect(ics).toMatch(/TRIGGER:-PT30M/)
  })

  test('fmtTime', () => {
    expect(L.fmtTime(65)).toBe('01:05')
    expect(L.fmtTime(0.2)).toBe('00:01')
  })

  test('pickNoRepeat never repeats when there is a choice', () => {
    const items = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
    let last: string | null = null
    for (let i = 0; i < 500; i++) {
      const x: { id: string } = L.pickNoRepeat(items, last, Math.random, (c) => c.id)!
      expect(x.id).not.toBe(last)
      last = x.id
    }
    expect(L.pickNoRepeat([], null)).toBeNull()
  })

  test('rest clips only on long rests, stopping before the beeps', () => {
    expect(L.restClipWindow(15, 1, () => 0)).toBeNull()
    expect(L.restClipWindow(45, 0, () => 0.5)).toBeNull()
    expect(L.restClipWindow(45, 1, () => 0.5)).toEqual({ start: 2.5, stop: 41 })
  })
})
