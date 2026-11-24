// Rating session ("test day"): six solo tests, each turned into a 1-99 card stat.
// Scores interpolate between anchor points (result -> rating), clamped to 30-99. Anchors follow FIFA's scale (research 24/11,
// rating-research.md): 75 = an average professional in that quality, 85 = a top-flight standout, 95 = among the best in the
// world. Sprint, jump and shuttle anchors come from published pro data (gate-timed); dribbling, shooting and passing have no
// norms for these exact tests, so those are judgment, to be re-fitted to real users' results. Tune here without touching the UI.
import { CARD_STATS, type CardStats } from '../components/PlayerCard'
import { t } from '../i18n'

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
    stat: 'PAC', name: t('20 m sprint'), setup: t('Two markers 20 m apart (about 26 big steps).'),
    how: t('Start still, one foot forward. Sprint flat out past the second marker. A friend times it, or tap Start as you go and Stop as you pass. Take the best of two.'),
    unit: 's', timer: 'stopwatch', step: 0.01,
    anchors: [[2.85, 97], [2.95, 90], [3.05, 80], [3.15, 72], [3.3, 62], [3.5, 52], [3.9, 40], [4.5, 30]],
  },
  {
    stat: 'DRI', name: t('Cone weave'), setup: t('6 markers in a line, 1 m apart.'),
    how: t('Dribble through all of them and back, touching the ball between every marker. Time from first touch to crossing the start line. Best of two.'),
    unit: 's', timer: 'stopwatch', step: 0.1,
    anchors: [[8, 96], [9, 88], [10.5, 78], [12, 68], [13.5, 58], [16, 46], [20, 34]],
  },
  {
    stat: 'SHO', name: t('Corner shots'), setup: t('A goal with a 1 m wide zone marked inside each post (tape, cones or bags), shooting from about 16 m.'),
    how: t('Ten shots with your stronger foot, ball stopped. Count how many land in one of the two corner zones.'),
    unit: '/ 10', timer: 'none', step: 1,
    anchors: [[0, 30], [1, 40], [2, 50], [3, 58], [4, 66], [5, 73], [6, 80], [7, 86], [8, 91], [9, 95], [10, 98]],
  },
  {
    stat: 'PAS', name: t('Wall passes'), setup: t('A wall, stand 3 m back.'),
    how: t('Pass against the wall for 30 seconds, two touches, alternating feet. Count clean passes back to the wall.'),
    unit: 'passes', timer: 'countdown', seconds: 30, step: 1,
    anchors: [[6, 34], [10, 46], [14, 58], [17, 68], [20, 77], [23, 86], [26, 93], [28, 97]],
  },
  {
    stat: 'DEF', name: t('5-10-5 agility shuttle'), setup: t('Three markers in a line, 5 m apart.'),
    how: t('Start at the middle marker. Sprint 5 m to one side, touch the line, 10 m to the other side, touch, then 5 m back through the middle. It measures how fast you change direction, which is what defending needs. Best of two.'),
    unit: 's', timer: 'stopwatch', step: 0.01,
    anchors: [[4.25, 97], [4.45, 89], [4.75, 77], [5.05, 66], [5.4, 55], [5.9, 44], [6.6, 34]],
  },
  {
    stat: 'PHY', name: t('Broad jump'), setup: t('A flat line to jump from and something to measure with.'),
    how: t('Both feet behind the line, swing the arms, jump as far as you can and stick the landing. Measure to the back of the heel. Best of three.'),
    unit: 'cm', timer: 'none', step: 1,
    anchors: [[160, 34], [190, 45], [215, 56], [230, 65], [245, 74], [260, 82], [280, 90], [300, 97]],
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

export type Ratings = { date: string; raw: Partial<Record<Stat, number>>; stats: Partial<CardStats>; proof?: boolean; v?: 2 } // proof: a photo was added; v 2: shot with the corner-zones test (older sessions used the full goal)

// Sessions before 24/11 counted shots on target in a full goal: 10/10 doesn't separate levels, so it tops out at 80.
const FULL_GOAL_ANCHORS: [number, number][] = [[0, 30], [3, 40], [5, 40], [7, 55], [8, 63], [9, 72], [10, 80]]

// Scores are always worked out again from the raw results, so every saved session follows the current anchors.
export function rescored(r: Ratings | undefined): Ratings | undefined {
  return r ? { ...ratingsFrom(r.raw, r.date, r.v !== 2), ...(r.proof ? { proof: true } : {}) } : undefined
}

// A card needs at least 3 tests. A test you skipped stays blank (it is never filled in from an older session).
export function toCardStats(saved: Ratings | undefined): CardStats | null {
  const r = rescored(saved)
  if (!r) return null
  const stats: CardStats = {}
  for (const k of CARD_STATS) if (r.stats[k] !== undefined) stats[k] = r.stats[k]
  return Object.keys(stats).length >= 3 ? stats : null
}

// The first rating session is the starting point: progress is measured from it. (Older profiles kept `prevRatings`, the
// session before the last, which is the same thing until the third session.)
export const startingPoint = (p: { ratings?: Ratings; baseRatings?: Ratings; prevRatings?: Ratings }) => p.baseRatings ?? p.prevRatings

// The result that earns a rating (the other way round from score): null when that rating can't be reached on the test.
// Timed tests go on past their best anchor (10 rating points a second), counted ones stop at their top anchor.
export function resultForRating(stat: Stat, rating: number): number | null {
  const test = TESTS.find((x) => x.stat === stat)!
  const pts = [...test.anchors].sort((a, b) => a[0] - b[0])
  const lowerIsBetter = pts[0][1] > pts[pts.length - 1][1]
  const best = lowerIsBetter ? pts[0][1] : pts[pts.length - 1][1], worst = lowerIsBetter ? pts[pts.length - 1][1] : pts[0][1]
  if (rating > best) return lowerIsBetter ? pts[0][0] - (rating - pts[0][1]) / 10 : null
  if (rating <= worst) return lowerIsBetter ? pts[pts.length - 1][0] : pts[0][0]
  for (let i = 1; i < pts.length; i++) {
    const [xa, ya] = pts[i - 1], [xb, yb] = pts[i]
    if (Math.min(ya, yb) <= rating && rating <= Math.max(ya, yb)) return xa + ((rating - ya) / (yb - ya)) * (xb - xa)
  }
  return null
}
// A target as the player reads it: "3.23 s", "6 of 10", "19 clean", "247 cm".
export function resultText(stat: Stat, value: number): string {
  if (stat === 'PAC' || stat === 'DEF') return `${value.toFixed(2)} s`
  if (stat === 'DRI') return `${value.toFixed(1)} s`
  if (stat === 'SHO') return t('{n} of 10', { n: Math.round(value) })
  if (stat === 'PAS') return t('{n} clean', { n: Math.round(value) })
  return `${Math.round(value)} cm`
}
export const STAT_NAMES = (): Record<Stat, string> => ({ PAC: t('Pace'), SHO: t('Shooting'), PAS: t('Passing'), DRI: t('Dribbling'), DEF: t('Defending'), PHY: t('Physical') })

// What a person can physically do in each test (research 24/11, rating-research.md section 3). Beyond lo/hi: impossible, the
// result is refused. Past proofBelow / proofAbove: world-class, so the player is asked for a photo and a friend to check it.
export type Limits = { lo: number; hi: number; proofBelow?: number; proofAbove?: number }
export const PLAUSIBLE: Record<Stat, Limits> = {
  PAC: { lo: 2.6, hi: 9, proofBelow: 2.85 }, DRI: { lo: 5.5, hi: 60, proofBelow: 8 }, SHO: { lo: 0, hi: 10, proofAbove: 9.5 },
  PAS: { lo: 0, hi: 40, proofAbove: 28 }, DEF: { lo: 3.9, hi: 14, proofBelow: 4.25 }, PHY: { lo: 80, hi: 340, proofAbove: 300 },
}
export function checkResult(stat: Stat, v: number): 'impossible' | 'proof' | null {
  const l = PLAUSIBLE[stat]
  if (v < l.lo || v > l.hi) return 'impossible'
  if ((l.proofBelow !== undefined && v < l.proofBelow) || (l.proofAbove !== undefined && v > l.proofAbove)) return 'proof'
  return null
}
export type Odd = { stat: Stat; why: 'impossible' | 'needs proof' | 'out of line' }
export function looksOff(saved: Ratings): Odd[] {
  const r = rescored(saved)!
  const out: Odd[] = []
  const done = TESTS.filter((x) => r.stats[x.stat] !== undefined).length
  for (const x of TESTS) {
    const v = r.raw[x.stat], sc = r.stats[x.stat]
    if (v === undefined || sc === undefined) continue
    const c = checkResult(x.stat, v)
    if (c) { out.push({ stat: x.stat, why: c === 'impossible' ? 'impossible' : 'needs proof' }); continue }
    // far from the player's other results (needs four or more tests to compare)
    const others = TESTS.filter((y) => y.stat !== x.stat).map((y) => r.stats[y.stat]).filter((n): n is number => n !== undefined).sort((a, b) => a - b)
    if (done >= 4 && others.length >= 3) { const med = others[Math.floor(others.length / 2)]; if (Math.abs(sc - med) > 35) out.push({ stat: x.stat, why: 'out of line' }) }
  }
  return out
}

export function ratingsFrom(raw: Partial<Record<Stat, number>>, date = new Date().toLocaleDateString('en-CA'), legacy = false): Ratings {
  const stats: Partial<CardStats> = {}
  for (const t of TESTS) {
    const v = raw[t.stat]
    if (v !== undefined && Number.isFinite(v) && v > 0) stats[t.stat] = score(legacy && t.stat === 'SHO' ? { ...t, anchors: FULL_GOAL_ANCHORS } : t, v)
  }
  return { date, raw, stats, ...(legacy ? {} : { v: 2 as const }) }
}

// Rate yourself weekly (same day, same warm-up) so the card follows real progress.
export const RATE_EVERY_DAYS = 7
export function nextRatingDue(r: Ratings | undefined): string | null {
  if (!r) return null
  const [y, m, d] = r.date.split('-').map(Number)
  return new Date(y, m - 1, d + RATE_EVERY_DAYS).toLocaleDateString('en-CA')
}

// A proper warm-up before the tests, so the results are real and comparable week to week.
// Gentle to sharp: jog, build-up runs, the ball, jumps, then change of direction. `demo` = the library exercise shown in 3D.
export type WarmStep = { name: string; how: string; minutes: number; demo: string }
export const WARMUP: WarmStep[] = [
  { name: t('Easy jog'), how: t('Jog around the area, then leg swings front and side. Get warm, not tired.'), minutes: 3, demo: 'Jog + dynamic leg swings' },
  { name: t('Build-up runs'), how: t('Three 20 m runs at about 50%, 70% and 85% speed. Walk back between them.'), minutes: 2, demo: 'Build-up sprints (no ball)' },
  { name: t('Wall passes'), how: t('Easy two-touch passes against the wall with both feet. Find your touch.'), minutes: 2, demo: 'Wall passes: ground, both feet' },
  { name: t('Dribbling'), how: t('Dribble through the markers at half pace, then a little quicker. Close touches.'), minutes: 2, demo: 'Cone weave dribble' },
  { name: t('Jumps'), how: t('10 pogo hops, then 3 easy broad jumps at about 70%.'), minutes: 1, demo: 'Pogo hops' },
  { name: t('Change of direction'), how: t('Shuffle and cut between the markers at 70%, then two sharp ones. Now you are ready.'), minutes: 2, demo: 'Sprint + change of direction (no ball)' },
]
