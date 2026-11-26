import { expect, test } from 'vitest'
import { sessionShare } from './shareStats'
import type { HistoryEntry } from './storage'

const set = (kg: number, reps: number) => ({ kg, reps })
const prev: HistoryEntry = { id: 'g', title: 'Gym', type: 'gym', date: '2026-11-10', minutes: 50, rpe: 7, lifts: [{ ex: 'bench-press', name: 'Bench press', sets: [set(70, 5)] }] }
const gym: HistoryEntry = { id: 'g', title: 'Gym', type: 'gym', date: '2026-11-20', minutes: 52, rpe: 8, effort: 8, lifts: [{ ex: 'bench-press', name: 'Bench press', sets: [set(60, 8), set(80, 5)] }] }

test('gym card: time and calories only, the streak flame is the picture (no best set, no set bars)', () => {
  const s = sessionShare(gym, [prev, gym], 72)
  expect(s.stats.map((x) => x.value)).toEqual(['52 min', expect.stringMatching(/^~\d+–\d+$/)]) // calories: a range
  expect(s.graphic).toEqual({ type: 'flame' })
})
test('football card: shots on target and goals from the Full time tally, with the shot map', () => {
  const fb: HistoryEntry = { id: 'f', title: 'Finishing', type: 'football', date: '2026-11-21', minutes: 45, drills: 6, tally: { shots: 24, onTarget: 17, goals: 9 } }
  const s = sessionShare(fb, [fb], 72)
  expect(s.stats.map((x) => x.value)).toEqual(['45 min', '17/24', '9'])
  expect(s.graphic).toEqual({ type: 'shots', shots: 24, onTarget: 17, goals: 9 })
})
