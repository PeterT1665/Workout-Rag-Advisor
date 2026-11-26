import { describe, expect, test } from 'vitest'
import { resolveShowcase, showcaseOptions } from './showcase'
import type { HistoryEntry } from './storage'
import type { MatchReview } from './match'
import { ratingsFrom } from './testday'

const lift = (ex: string, name: string, sets: { kg: number | null; reps: number }[]) => ({ ex, name, sets })
const h = (date: string, lifts: HistoryEntry['lifts']): HistoryEntry => ({ id: 'g', title: 'Gym', type: 'gym', date, minutes: 50, lifts })
const m = (goals: number, assists: number) => ({ goals, assists }) as MatchReview

describe('showcase stats', () => {
  test('only stats the app has logged can be picked', () => {
    expect(showcaseOptions([], [], {})).toEqual([])
    const o = showcaseOptions([h('2026-11-10', [lift('bench-press', 'Bench press', [{ kg: 60, reps: 8 }, { kg: 80, reps: 3 }])])], [m(2, 1), m(1, 0)], { ratings: ratingsFrom({ PAC: 3.45 }) })
    expect(o.map((x) => x.k)).toEqual(['goals', 'assists', 'sprint', 'pr:bench-press'])
    expect(o.find((x) => x.k === 'goals')!.value).toBe('3'); expect(o.find((x) => x.k === 'assists')!.value).toBe('1')
    expect(o.find((x) => x.k === 'sprint')!.value).toBe('3.45 s')
    expect(o.find((x) => x.k === 'pr:bench-press')!.value).toBe('80 kg × 3') // the set with the highest estimated max
  })
  test('a picked key with no data any more is dropped, order kept, six at most', () => {
    const items = resolveShowcase(['assists', 'sprint', 'goals'], [], [m(2, 1)], {})
    expect(items).toEqual([{ k: 'assists', v: '1' }, { k: 'goals', v: '2' }])
    expect(resolveShowcase(undefined, [], [], {})).toEqual([])
  })
})
