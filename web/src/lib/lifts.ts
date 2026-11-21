// Sets logged after a gym session (weight and reps), and how hard they were for *you*.
// No DOM, unit-tested (lifts.test.ts).
//
// The idea: the same 60 kg bench is easy for one player and a max for another, so every set is measured
// against your own estimated max for that exercise (Epley: weight × (1 + reps/30)), learned from what you log.
//   intensity  = set weight ÷ your estimated max               (0.8 = 80% of your max)
//   reps left  = reps you could have done at that weight − reps done
//   set effort = 10 − reps left                                (the RIR-based RPE scale lifters use)
// Your first time on an exercise there is no max yet, so the effort you report calibrates it
// (effort 8 ≈ 2 reps left in the tank). After that, your sets speak for themselves.
// Session effort = half what you felt, half what your sets say; session load = minutes × session effort.
import type { Block, Session } from './types'
import type { HistoryEntry } from './storage'
import data from '../data/library.json'

export type SetLog = { kg: number | null; reps: number | null }
export type LiftLog = { ex?: string; name: string; sets: SetLog[] }

type RawExercise = { id: string; name: string; category: string; reps?: string; rounds?: number }
const RAW = data.exercises as RawExercise[]
const byName = new Map(RAW.map((e) => [e.name, e]))

// Exercises worth a weight and reps: strength and loaded power work. Not sprints, jumps, holds or med-ball throws.
const LOGGED = new Set(['lower-strength', 'upper-strength', 'single-limb', 'olympic', 'explosive'])
const timeOrDistance = (reps: string) => /\d\s*(s|m)\b|taps/.test(reps)
export function loggable(b: Pick<Block, 'name' | 'reps' | 'category'>): boolean {
  const raw = byName.get(b.name)
  const cat = b.category ?? raw?.category
  const reps = raw?.reps ?? b.reps
  return !!reps && !!cat && LOGGED.has(cat) && !timeOrDistance(reps) && !/med-ball/.test(raw?.id ?? '')
}

// The share of your body weight you move on bodyweight exercises (rough, from the biomechanics literature),
// so 15 push-ups count as real work, and a belt with 10 kg on pull-ups adds to your body, not to zero.
const BODY_SHARE: Record<string, number> = {
  'push-up': 0.65, 'home-push-up': 0.65, 'home-pike-push-up': 0.7, 'pull-up': 0.95, 'home-table-row': 0.6, 'home-chair-dip': 0.8,
  'nordic-curl': 0.6, 'home-hamstring-slide': 0.4, 'home-glute-bridge-sl': 0.6,
  'home-tempo-squat': 0.85, 'jump-squat': 0.85, 'skater-squat': 0.9, 'step-up': 0.9, 'bulgarian-split-squat': 0.9,
  'home-reverse-lunge': 0.9, 'home-sofa-split-squat': 0.9, 'home-cossack': 0.85, 'calf-raise': 1, 'single-leg-rdl': 0.5,
}
export const bodyShare = (l: Pick<LiftLog, 'ex'>) => (l.ex ? BODY_SHARE[l.ex] ?? 0 : 0)
export const isBodyweight = (l: Pick<LiftLog, 'ex'>) => bodyShare(l) > 0

const DEFAULT_BW = 70 // when no body weight is logged; only bodyweight exercises use it, and ratios barely move
const key = (l: LiftLog) => l.ex ?? l.name
const firstInt = (s?: string) => { const m = s?.match(/\d+/); return m ? Number(m[0]) : null }

// The weight actually moved in a set; null when there's nothing to measure (a barbell lift without a weight).
export function moved(l: LiftLog, s: SetLog, bw = DEFAULT_BW): number | null {
  const share = bodyShare(l)
  if (!s.reps || s.reps <= 0) return null
  if (!share) return s.kg && s.kg > 0 ? s.kg : null
  return (s.kg ?? 0) + share * bw
}
export const e1rm = (load: number, reps: number) => load * (1 + Math.min(reps, 15) / 30)
const repsLeftGuess = (rpe: number) => Math.min(4, Math.max(0, 10 - rpe)) // effort 8 ≈ 2 reps in the tank

// Your estimated max per exercise from one session's sets.
function sessionBests(lifts: LiftLog[], rpe: number, bw: number): Map<string, number> {
  const out = new Map<string, number>()
  const rir = repsLeftGuess(rpe)
  for (const l of lifts) for (const s of l.sets) {
    const w = moved(l, s, bw)
    if (w === null) continue
    const m = e1rm(w, s.reps! + rir)
    if (m > (out.get(key(l)) ?? 0)) out.set(key(l), m)
  }
  return out
}
// Your estimated max per exercise from every logged session. Sessions saved before you logged your body weight
// use `bw` (today's), so adding your weight doesn't turn the same pull-ups into a new best.
export function bestsFrom(history: HistoryEntry[], bw?: number): Map<string, number> {
  const out = new Map<string, number>()
  for (const h of history) {
    if (!h.lifts?.length || !h.rpe) continue
    for (const [k, v] of sessionBests(h.lifts, h.rpe, h.bw ?? bw ?? DEFAULT_BW)) if (v > (out.get(k) ?? 0)) out.set(k, v)
  }
  return out
}

export type LiftScore = {
  name: string; ex?: string
  sets: number // sets with something to measure
  top: SetLog // heaviest set
  intensity: number // heaviest set ÷ your estimated max, 0..1
  effort: number // 1..10, how hard the exercise was for you
  max: number // your estimated max after this session (kg moved, body share included)
  best: boolean // a new estimated max
  volume: number // kg × reps, what you put on the bar
}
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x))
const round1 = (x: number) => Math.round(x * 10) / 10

export function scoreLifts(lifts: LiftLog[], rpe: number, prior: Map<string, number>, bw = DEFAULT_BW): { lifts: LiftScore[]; measured: number | null } {
  const now = sessionBests(lifts, rpe, bw)
  const out: LiftScore[] = []
  for (const l of lifts) {
    const before = prior.get(key(l)) ?? 0
    const max = Math.max(before, now.get(key(l)) ?? 0)
    const sets = l.sets.map((s) => ({ s, w: moved(l, s, bw) })).filter((x): x is { s: SetLog; w: number } => x.w !== null)
    if (!sets.length || !max) continue
    const efforts = sets.map(({ s, w }) => {
      const canDo = 30 * (max / w - 1) // reps possible at this weight
      return clamp(10 - Math.max(0, canDo - s.reps!), 1, 10)
    })
    const hardest = Math.max(...efforts)
    const mean = efforts.reduce((a, b) => a + b, 0) / efforts.length
    const top = sets.reduce((a, b) => (b.w > a.w || (b.w === a.w && b.s.reps! > a.s.reps!) ? b : a))
    out.push({
      name: l.name, ex: l.ex, sets: sets.length, top: top.s,
      intensity: top.w / max,
      effort: round1(0.6 * hardest + 0.4 * mean), // the hardest set is what you remember
      max, best: before > 0 && max > before * 1.005,
      volume: Math.round(sets.reduce((a, { s }) => a + (s.kg ?? 0) * s.reps!, 0)),
    })
  }
  const n = out.reduce((a, x) => a + x.sets, 0)
  return { lifts: out, measured: n ? round1(out.reduce((a, x) => a + x.effort * x.sets, 0) / n) : null }
}

// What you felt, checked against what your sets say.
export const sessionEffort = (rpe: number, measured: number | null) => (measured === null ? rpe : round1((rpe + measured) / 2))

// The check-in form: every loggable exercise of the session, filled in with last time's sets
// (or the plan's reps), so most days it's a quick look and a tap.
export function planLifts(session: Session, history: HistoryEntry[]): LiftLog[] {
  const last = new Map<string, SetLog[]>()
  for (const h of history) for (const l of h.lifts ?? []) if (l.sets.some((s) => s.reps)) last.set(key(l), l.sets)
  const out: LiftLog[] = []
  for (const b of session.blocks) {
    if (!loggable(b) || out.some((x) => x.name === b.name)) continue
    const raw = byName.get(b.name)
    const l: LiftLog = { ex: raw?.id, name: b.name, sets: [] }
    const rounds = b.rounds || raw?.rounds || 1
    const prev = last.get(key(l))
    const reps = firstInt(b.reps) ?? firstInt(raw?.reps)
    l.sets = Array.from({ length: rounds }, (_, i) => {
      const p = prev?.[Math.min(i, prev.length - 1)]
      return { kg: p?.kg ?? null, reps: p?.reps ?? reps }
    })
    out.push(l)
  }
  return out
}

// Only sets with reps are kept (an emptied row is a set you didn't do).
export const cleanLifts = (lifts: LiftLog[]) =>
  lifts.map((l) => ({ ...l, sets: l.sets.filter((s) => s.reps && s.reps > 0).map((s) => ({ kg: s.kg && s.kg > 0 ? s.kg : null, reps: s.reps })) }))
    .filter((l) => l.sets.length)
