// Rating session ("test day"): six solo tests, each turned into a 1-99 card stat.
// Scores interpolate between anchor points (result -> rating), clamped to 30-99. Anchors are rough amateur
// benchmarks so a fit club player lands around silver/gold; they can be tuned without touching the UI.
import { CARD_STATS, type CardStats } from '../components/PlayerCard'

export type Stat = (typeof CARD_STATS)[number]
export type Timer = 'stopwatch' | 'countdown' | 'none'

export type Test = {
  stat: Stat
  name: string
  setup: string
  how: string
  unit: string // shown next to the input
  timer: Timer
  seconds?: number // countdown length
  step: number // input step
  anchors: [number, number][] // [result, rating], any order
}

export const TESTS: Test[] = [
  {
    stat: 'PAC', name: '20 m sprint', setup: 'Two markers 20 m apart (about 26 big steps).',
    how: 'Start still, one foot forward. Sprint flat out past the second marker. A friend times it, or tap Start as you go and Stop as you pass. Take the best of two.',
    unit: 's', timer: 'stopwatch', step: 0.01,
    anchors: [[2.9, 97], [3.1, 90], [3.3, 82], [3.5, 74], [3.8, 64], [4.2, 52], [4.8, 40]],
  },
  {
    stat: 'DRI', name: 'Cone weave', setup: '6 markers in a line, 1 m apart.',
    how: 'Dribble through all of them and back, touching the ball between every marker. Time from first touch to crossing the start line. Best of two.',
    unit: 's', timer: 'stopwatch', step: 0.1,
    anchors: [[9, 96], [10.5, 88], [12, 80], [13.5, 72], [15, 64], [18, 50], [22, 38]],
  },
  {
    stat: 'SHO', name: 'Shots on target', setup: 'A goal or a target on a wall, about 16 m away.',
    how: 'Ten shots with your stronger foot, ball stopped. Count how many hit the target.',
    unit: '/ 10', timer: 'none', step: 1,
    anchors: [[0, 35], [3, 52], [5, 64], [7, 76], [8, 83], [9, 90], [10, 96]],
  },
  {
    stat: 'PAS', name: 'Wall passes', setup: 'A wall, stand 3 m back.',
    how: 'Pass against the wall for 30 seconds, two touches, alternating feet. Count clean passes back to the wall.',
    unit: 'passes', timer: 'countdown', seconds: 30, step: 1,
    anchors: [[8, 45], [12, 58], [16, 68], [20, 77], [24, 85], [28, 92], [32, 97]],
  },
  {
    stat: 'DEF', name: '5-10-5 shuffle', setup: 'Three markers in a line, 5 m apart.',
    how: 'Start at the middle marker. Sprint 5 m to one side, touch the line, 10 m to the other side, touch, then 5 m back through the middle. Defending footwork and turns. Best of two.',
    unit: 's', timer: 'stopwatch', step: 0.01,
    anchors: [[4.3, 96], [4.6, 88], [4.9, 80], [5.2, 72], [5.6, 63], [6.2, 50], [7, 38]],
  },
  {
    stat: 'PHY', name: 'Broad jump', setup: 'A flat line to jump from and something to measure with.',
    how: 'Both feet behind the line, swing the arms, jump as far as you can and stick the landing. Measure to the back of the heel. Best of three.',
    unit: 'cm', timer: 'none', step: 1,
    anchors: [[160, 45], [190, 58], [215, 68], [235, 76], [255, 84], [275, 91], [295, 97]],
  },
]

export function score(test: Test, result: number): number {
  const pts = [...test.anchors].sort((a, b) => a[0] - b[0])
  const lowerIsBetter = pts[0][1] > pts[pts.length - 1][1]
  let r: number
  if (result <= pts[0][0]) r = pts[0][1] + (lowerIsBetter ? (pts[0][0] - result) * 10 : 0)
  else if (result >= pts[pts.length - 1][0]) r = pts[pts.length - 1][1] - (lowerIsBetter ? (result - pts[pts.length - 1][0]) * 5 : 0)
  else {
    const i = pts.findIndex(([x]) => x >= result)
    const [x0, y0] = pts[i - 1]
    const [x1, y1] = pts[i]
    r = y0 + ((result - x0) / (x1 - x0)) * (y1 - y0)
  }
  return Math.round(Math.max(30, Math.min(99, r)))
}

export type Ratings = { date: string; raw: Partial<Record<Stat, number>>; stats: Partial<CardStats> }

// A card needs at least 3 tests; missing stats are taken from the previous test day if there is one.
export function toCardStats(r: Ratings | undefined, prev?: Ratings): CardStats | null {
  if (!r) return null
  const merged = { ...(prev?.stats ?? {}), ...r.stats }
  if (Object.keys(merged).length < 3) return null
  const known = Object.values(merged) as number[]
  const fill = Math.round(known.reduce((a, b) => a + b, 0) / known.length)
  return Object.fromEntries(CARD_STATS.map((k) => [k, merged[k] ?? fill])) as CardStats
}

export function ratingsFrom(raw: Partial<Record<Stat, number>>, date = new Date().toLocaleDateString('en-CA')): Ratings {
  const stats: Partial<CardStats> = {}
  for (const t of TESTS) {
    const v = raw[t.stat]
    if (v !== undefined && Number.isFinite(v) && v > 0) stats[t.stat] = score(t, v)
  }
  return { date, raw, stats }
}
