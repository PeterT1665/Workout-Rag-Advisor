import { describe, expect, test } from 'vitest'
import { DEFAULT_MOVES, cueWords, pickWord, wordFor, isWordCue } from './cues'
import { flattenSession } from './logic'
import type { Session } from './types'

describe('voice cue words', () => {
  test('every word set has words, ids are unique, and a word is never called twice in a row', () => {
    for (const kind of ['touches', 'scan', 'target', 'colours', 'corner', 'run', 'turn', 'go', 'move'] as const) {
      const ws = cueWords(kind)
      expect(ws.length, kind).toBeGreaterThan(0)
      expect(new Set(ws.map((w) => w.id)).size).toBe(ws.length)
      if (ws.length > 1) { let last: string | null = null; for (let i = 0; i < 60; i++) { const w = pickWord(kind, last); expect(w.id).not.toBe(last); last = w.id } }
    }
    expect(cueWords('target').map((w) => w.word)).toEqual(['Left', 'Middle', 'Right'])
  })
  test('called moves are the player\'s signature moves, or three well-known ones', () => {
    expect(cueWords('move').map((w) => w.word)).toEqual(DEFAULT_MOVES)
    expect(cueWords('move', ['Puskas', 'Lift']).map((w) => w.word)).toEqual(['Puskas', 'Lift'])
    expect(wordFor('m-Lift', 'move', ['Puskas', 'Lift'])?.word).toBe('Lift')
  })
  test('a block\'s cue reaches its steps; true and numbers stay as they were', () => {
    const s: Session = { id: 'x', title: 'X', type: 'football', summary: '', needs: [], blocks: [
      { name: 'A', kind: 'drill', work: 30, cue: 'touches' }, { name: 'B', kind: 'drill', work: 30, cue: true }, { name: 'C', kind: 'drill', work: 30, cue: 'numbers' }] }
    const work = flattenSession(s).filter((x) => x.type === 'work')
    expect(work.map((x) => x.cueKind)).toEqual(['touches', undefined, 'numbers'])
    expect(work.map((x) => x.cue)).toEqual([true, true, true])
    expect(isWordCue('touches')).toBe(true); expect(isWordCue('numbers')).toBe(false)
  })
})
