// Public demo mode (build flag VITE_DEMO=1): no login, no sync, sample data, only licensed photos.
// The private build never seeds anything. Dates are relative to "today" so the demo always looks current.
import { DEFAULT_SETTINGS, save, type HistoryEntry } from './storage'
import data from '../data/sessions.json'
import type { Session } from './types'

export const DEMO = import.meta.env.VITE_DEMO === '1'
export const IMAGE_DIR = DEMO ? 'demo-img' : 'personal'
// Quotes in the private build are attributed to the role model; the public demo uses neutral lines with no attribution.
export const QUOTE_BY = DEMO ? '' : 'Cristiano Ronaldo'

export const DEMO_QUOTES = [
  'Show up. Do the reps. Repeat.',
  'Fast feet, calm head.',
  'The ball only goes where you tell it. Tell it with both feet.',
  'Rest is part of the plan, not a break from it.',
]

const sessions = data.sessions as Session[]
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

export function buildDemoHistory(now = new Date()): HistoryEntry[] {
  const rows = [...ROWS]
  // Monday and Tuesday have no earlier day of this week in the sample above, so add one: keeps the weekly chart non-empty.
  const weekday = (now.getDay() + 6) % 7
  if (weekday <= 1) rows.push([weekday === 0 ? 0 : -1, 'fb-wall-weakfoot-shooting', 58, 6, 'ok', [], 9, 0])
  return rows.map(([off, id, minutes, rpe, legs, sore, drills, cues]) => {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() + off)
    return { id, title: title(id), date: ymd(d), minutes, rpe, legs, sore, load: Math.round(minutes * rpe), drills, cues }
  })
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
  try { localStorage.setItem('demo:seedDay', today) } catch { /* ignore */ }
}
