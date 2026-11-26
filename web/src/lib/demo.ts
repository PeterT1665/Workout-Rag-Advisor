// Public demo mode (build flag VITE_DEMO=1): no login, no sync, sample data, only licensed photos.
// The private build never seeds anything. Dates are relative to "today" so the demo always looks current.
import { DEFAULT_SETTINGS, save, type HistoryEntry } from './storage'
import { buildWeek, type Game } from './weekplan'
import { BUILTIN_SESSIONS as sessions } from './library'
import { mark } from '../i18n'
import type { LiftLog } from './lifts'
import { getProfile, setProfile } from './profile'

export const DEMO = import.meta.env.VITE_DEMO === '1'
export const IMAGE_DIR = DEMO ? 'demo-img' : 'personal'

export const DEMO_QUOTES = [
  mark('Show up. Do the reps. Repeat.'),
  mark('Fast feet, calm head.'),
  mark('The ball only goes where you tell it. Tell it with both feet.'),
  mark('Rest is part of the plan, not a break from it.'),
]

const title = (id: string) => sessions.find((s) => s.id === id)?.title ?? id
const ymd = (d: Date) => d.toLocaleDateString('en-CA')

type Row = [offset: number, id: string, minutes: number, rpe: number, legs: 'fresh' | 'ok' | 'heavy', sore: string[], drills: number, cues: number]
const ROWS: Row[] = [
  [-21, 'fb-close-control-1v1', 63, 7, 'ok', [], 8, 44],
  [-19, 'recovery-hips', 22, 3, 'fresh', ['Hips'], 5, 0],
  [-17, 'fb-wall-weakfoot-shooting', 58, 6, 'ok', [], 9, 0],
  [-14, 'fb-partner-passing', 46, 6, 'fresh', [], 6, 0],
  [-12, 'fb-close-control-1v1', 64, 8, 'heavy', ['Hamstring'], 8, 52],
  [-11, 'recovery-hips', 24, 3, 'fresh', [], 5, 0],
  [-9, 'fb-wall-weakfoot-shooting', 57, 7, 'ok', [], 9, 0],
  [-7, 'fb-close-control-1v1', 62, 7, 'ok', [], 8, 46],
  [-5, 'recovery-hips', 21, 2, 'fresh', ['Hips'], 5, 0],
  [-4, 'fb-wall-weakfoot-shooting', 59, 6, 'ok', [], 9, 0],
  [-2, 'fb-close-control-1v1', 64, 7, 'ok', [], 8, 48],
]

// Gym days with logged sets, getting a little stronger each time (so Record, the coach and New best have something to show).
const sets = (kg: number | null, reps: number[]) => reps.map((r) => ({ kg, reps: r }))
const GYM: [offset: number, id: string, minutes: number, rpe: number, lifts: LiftLog[]][] = [
  [-20, 'gym-lower-power', 55, 8, [{ ex: 'back-squat', name: 'Back squat', sets: sets(70, [5, 5, 5, 4]) }, { ex: 'romanian-deadlift', name: 'Romanian deadlift', sets: sets(60, [8, 8, 7]) }]],
  [-16, 'gym-upper-core', 50, 7, [{ ex: 'bench-press', name: 'Bench press', sets: sets(55, [6, 6, 6, 5]) }, { ex: 'pull-up', name: 'Pull-ups', sets: sets(null, [7, 6, 6]) }]],
  [-13, 'gym-lower-power', 56, 8, [{ ex: 'back-squat', name: 'Back squat', sets: sets(72.5, [5, 5, 5, 5]) }, { ex: 'romanian-deadlift', name: 'Romanian deadlift', sets: sets(62.5, [8, 8, 8]) }]],
  [-6, 'gym-upper-core', 52, 8, [{ ex: 'bench-press', name: 'Bench press', sets: sets(57.5, [6, 6, 6, 6]) }, { ex: 'pull-up', name: 'Pull-ups', sets: sets(null, [8, 7, 7]) }]],
  [-3, 'gym-lower-power', 54, 8, [{ ex: 'back-squat', name: 'Back squat', sets: sets(75, [5, 5, 5, 4]) }, { ex: 'romanian-deadlift', name: 'Romanian deadlift', sets: sets(65, [8, 8, 7]) }]],
]
// The sample player's weigh-ins (a slow lean gain).
export function demoBody(now = new Date()) {
  return [[-27, 71.2], [-20, 71.6], [-13, 71.9], [-6, 72.3], [-1, 72.4]].map(([off, kg]) => ({ date: ymd(new Date(now.getFullYear(), now.getMonth(), now.getDate() + off)), kg }))
}

export function buildDemoHistory(now = new Date()): HistoryEntry[] {
  const rows = [...ROWS]
  // Monday and Tuesday have no earlier day of this week in the sample above, so add one: keeps the weekly chart non-empty.
  const weekday = (now.getDay() + 6) % 7
  if (weekday <= 1) rows.push([weekday === 0 ? 0 : -1, 'fb-wall-weakfoot-shooting', 58, 6, 'ok', [], 9, 0])
  // the sample player sticks to the plan (so the streak flame has a number): fill planned training days of the last 9
  const plan = buildWeek([{ day: 2, kind: 'competitive' }], { gym: 3 }).slots
  const FILL: Record<string, [string, number, number]> = { football: ['fb-close-control-1v1', 40, 5], gym: ['gym-full-light', 45, 6], recovery: ['recovery-hips', 20, 2] }
  for (let off = -1; off >= -9; off--) {
    if (rows.some((r) => r[0] === off) || GYM.some((g) => g[0] === off)) continue
    const day = (weekday + off + 70) % 7
    const slot = plan.find((x) => x.day === day && x.kind !== 'game')
    const fill = slot && !plan.some((x) => x.day === day && x.kind === 'game') ? FILL[slot.kind] : undefined
    if (fill) rows.push([off, fill[0], fill[1], fill[2], 'ok', [], 5, 0])
  }
  const day = (off: number) => ymd(new Date(now.getFullYear(), now.getMonth(), now.getDate() + off))
  const type = (id: string) => sessions.find((s) => s.id === id)?.type
  const out: (HistoryEntry & { off: number })[] = rows.map(([off, id, minutes, rpe, legs, sore, drills, cues]) =>
    ({ off, id, title: title(id), type: type(id), date: day(off), minutes, rpe, legs, sore, load: Math.round(minutes * rpe), drills, cues }))
  for (const [off, id, minutes, rpe, lifts] of GYM) {
    if (rows.some((r) => r[0] === off)) continue
    out.push({ off, id, title: title(id), type: type(id), date: day(off), minutes, rpe, effort: rpe, legs: 'ok', sore: [], load: Math.round(minutes * rpe), drills: 6, cues: 0, lifts, bw: 72 })
  }
  return out.sort((x, y) => x.off - y.off).map(({ off: _off, ...h }) => h)
}

// Rewrites the sample data once per day (so dates stay recent). Visitors' own changes during a day are kept.
export function seedDemo(now = new Date(), force = false) {
  const today = ymd(now)
  try { if (!force && localStorage.getItem('demo:seedDay') === today) return } catch { /* continue */ }
  save('history', buildDemoHistory(now))
  save('games', [{ day: 2, kind: 'competitive' }])
  save('quotes', DEMO_QUOTES)
  save('settings', { ...DEFAULT_SETTINGS })
  save(`kit:${today}`, ['water', 'ball'])
  const p = getProfile()
  if (p.sample) setProfile({ ...p, heightCm: 176, body: demoBody(now) })
  try { localStorage.setItem('demo:seedDay', today) } catch { /* ignore */ }
}

// Demo visitor made their own player: start from an empty log with their game days (no daily reseeding).
export function startOwnDemo(games: Game[]) {
  save('history', [])
  save('games', games)
  save('quotes', DEMO_QUOTES)
  save('settings', { ...DEFAULT_SETTINGS })
  try { localStorage.removeItem('demo:seedDay') } catch { /* ignore */ }
}
// Back to the welcome screen: forget the player and their demo data (built sessions stay).
export function startOverDemo() {
  for (const k of ['profile', 'history', 'games', 'quotes', 'demo:seedDay']) {
    try { localStorage.removeItem(k) } catch { /* ignore */ }
  }
}
