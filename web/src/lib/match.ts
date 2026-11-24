// Match day: a short review after each game (a journal), the app's own match rating, results streaks,
// and the times that decide when the review opens (supabase/migrations/0009_match_day.sql sends the reminders).
import { load, save } from './storage'
import { todayIndex, type Game } from './weekplan'
import { familyOf, type Position } from './profile'
import type { CardStats } from '../components/PlayerCard'

export type MatchReview = {
  date: string // the game day, YYYY-MM-DD
  us: number
  them: number
  minutes: number // minutes you played
  length?: number // how long the game was (90 eleven-a-side, 40 futsal...); older reviews without it were 90
  position: Position
  goals: number
  assists: number
  dribbles: number // completed
  keyPasses: number
  duels: number // 1v1s won
  firstTouch: number // 1-10
  tiring: number // 1-10
  self: number // your own rating, 1-10
  moments: string
  rating: number // the app's, 3.0-10.0
  savedAt: number
}
export type MatchTimes = Record<string, { start: string; end: string }> // date -> 'HH:MM'
export type Result = 'W' | 'D' | 'L'

const ymd = (d: Date) => d.toLocaleDateString('en-CA')
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

export const loadMatches = (): MatchReview[] => load<MatchReview[]>('matches', []).slice().sort((a, b) => a.date.localeCompare(b.date))
export function saveMatch(m: MatchReview) {
  save('matches', [...loadMatches().filter((x) => x.date !== m.date), m])
}
export const deleteMatch = (date: string) => save('matches', loadMatches().filter((x) => x.date !== date))

export const loadTimes = (): MatchTimes => load<MatchTimes>('matchTimes', {})
export function saveTimes(date: string, start: string, end: string) {
  // keep the last few weeks only; the newest one prefills next time
  const kept = Object.entries(loadTimes()).sort(([a], [b]) => b.localeCompare(a)).slice(0, 8)
  save('matchTimes', { ...Object.fromEntries(kept), [date]: { start, end } })
}
export const gameLength = (m: Pick<MatchReview, 'length'>) => m.length || 90
// Prefill: the length of your last game (you usually play the same format), else 90.
export const lastLength = () => { const all = loadMatches(); return all.length ? gameLength(all[all.length - 1]) : 90 }
// Prefill: your last game's times, else a typical afternoon game.
export function lastTimes(): { start: string; end: string } {
  const all = Object.entries(loadTimes()).sort(([a], [b]) => b.localeCompare(a))
  return all[0]?.[1] ?? { start: '15:00', end: '16:30' }
}

export const resultOf = (m: Pick<MatchReview, 'us' | 'them'>): Result => (m.us > m.them ? 'W' : m.us < m.them ? 'L' : 'D')

// ---- the app's match rating (like the ratings sports apps give players after a game) ----
// Starts at 6.0. What you did adds to it, weighted by where you played (a defender's goal is rarer than a striker's,
// and a clean sheet matters more to a defender). Playing part of the game counts more per minute, up to 1.5x
// (a full futsal game is a full game, not 40 of 90). Each extra contribution adds a little less, so a 10 is rare.
// The result and your first touch move it a little. Calibrated to sports apps: a goal and an assist in a win is about 7.6,
// a hat-trick with chances made around 9.5. Your own rating and how tiring it was don't change it.
const WEIGHTS: Record<'winger' | 'striker' | 'midfielder' | 'defender', { goal: number; assist: number; key: number; dribble: number; duel: number; cleanSheet: number }> = {
  striker: { goal: 0.8, assist: 0.55, key: 0.15, dribble: 0.08, duel: 0.08, cleanSheet: 0 },
  winger: { goal: 0.8, assist: 0.6, key: 0.16, dribble: 0.1, duel: 0.08, cleanSheet: 0 },
  midfielder: { goal: 0.9, assist: 0.6, key: 0.18, dribble: 0.09, duel: 0.1, cleanSheet: 0.25 },
  defender: { goal: 1.0, assist: 0.65, key: 0.2, dribble: 0.08, duel: 0.15, cleanSheet: 0.6 },
}
const SOFT = 8 // diminishing returns: raw 1.4 -> 1.3, raw 3.7 -> 3.0, never more than 8
export function matchRating(m: Omit<MatchReview, 'rating' | 'savedAt' | 'date' | 'moments' | 'self' | 'tiring'>): number {
  const fam = familyOf(m.position)
  const w = WEIGHTS[fam === 'goalkeeper' ? 'defender' : fam] ?? WEIGHTS.midfielder
  const length = clamp(gameLength(m), 10, 120)
  const minutes = clamp(m.minutes || 0, 1, length)
  let raw = m.goals * w.goal + m.assists * w.assist + m.keyPasses * w.key + m.dribbles * w.dribble + m.duels * w.duel
  raw *= Math.min(1.5, Math.sqrt(length / Math.max(minutes, length * 0.22)))
  const plus = SOFT * (1 - Math.exp(-raw / SOFT))
  let r = 6 + plus + (m.firstTouch - 6) * 0.1
  const res = resultOf(m)
  r += res === 'W' ? 0.3 : res === 'L' ? -0.3 : 0
  if (m.them === 0) r += w.cleanSheet
  else if (fam === 'defender' || fam === 'goalkeeper') r -= 0.12 * Math.max(0, m.them - 1)
  if (minutes < length * 0.22 && plus === 0) r = Math.min(r, 6.3) // a quiet cameo stays around 6
  return Math.round(clamp(r, 3, 10) * 10) / 10
}

// Colour bands like the ratings on match apps: red below 6, orange 6+, yellow 7+, green 8+, deep green 9+.
export const ratingBand = (r: number) => (r >= 9 ? 'top' : r >= 8 ? 'great' : r >= 7 ? 'good' : r >= 6 ? 'ok' : 'poor')
export const BAND_COLOR: Record<ReturnType<typeof ratingBand>, { bg: string; fg: string }> = {
  top: { bg: '#0f8a43', fg: '#fff' }, great: { bg: '#2fb350', fg: '#fff' }, good: { bg: '#f2c230', fg: '#111' },
  ok: { bg: '#f08a1c', fg: '#111' }, poor: { bg: '#e2452b', fg: '#fff' },
}

// ---- results streaks: wins in a row, and games without losing (a draw keeps that one going) ----
export function resultStreaks(matches: MatchReview[]): { wins: number; unbeaten: number } {
  const newestFirst = matches.slice().sort((a, b) => b.date.localeCompare(a.date)).map(resultOf)
  const run = (ok: (r: Result) => boolean) => { let n = 0; for (const r of newestFirst) { if (!ok(r)) break; n++ } return n }
  return { wins: run((r) => r === 'W'), unbeaten: run((r) => r !== 'L') }
}

// ---- match form on the player card ----
// The last 5 games (within 60 days) move SHO, PAS, DRI, DEF and PHY by up to 3 points each, up or down,
// against what a steady game looks like. Pace only comes from the rating session.
export function matchForm(matches: MatchReview[], now = new Date()): Partial<CardStats> {
  const since = ymd(new Date(now.getTime() - 60 * 864e5))
  const recent = matches.filter((m) => m.date >= since).sort((a, b) => b.date.localeCompare(a.date)).slice(0, 5)
  if (!recent.length) return {}
  const avg = (f: (m: MatchReview) => number) => recent.reduce((a, m) => a + f(m), 0) / recent.length
  const shift = (score: number, par: number) => clamp(Math.round((score - par) * 2), -3, 3)
  return {
    SHO: shift(avg((m) => m.goals * 1.5), 0.5),
    PAS: shift(avg((m) => m.assists + m.keyPasses * 0.35), 0.8),
    DRI: shift(avg((m) => m.dribbles * 0.3 + (m.firstTouch - 6) * 0.3), 0.8),
    DEF: shift(avg((m) => m.duels * 0.35 + (m.them === 0 ? 0.8 : 0)), 0.9),
    PHY: shift(avg((m) => clamp(m.minutes / gameLength(m), 0, 1) * 1.2 - (m.tiring - 5) * 0.05), 1.0),
  }
}
export function withForm(stats: CardStats, form: Partial<CardStats>): CardStats {
  const out = { ...stats }
  for (const [k, d] of Object.entries(form) as [keyof CardStats, number][]) if (out[k] !== undefined) out[k] = clamp(out[k]! + d, 1, 99) // a blank stat stays blank
  return out
}

// ---- when the review is open ----
// Game day: before the review opens you set your times ('times', or 'set' once entered). It opens 15 minutes after
// full time (9 pm without times) and stays open until the end of the next day, or until it's saved.
export type MatchDay =
  | { state: 'times'; date: string }
  | { state: 'set'; date: string; start: string; end: string; opensAt: number }
  | { state: 'review'; date: string }

const at = (date: string, hhmm: string) => new Date(`${date}T${hhmm}:00`).getTime()
export const reviewOpensAt = (date: string, times?: { start: string; end: string }) =>
  times ? at(date, times.end) + 15 * 60000 : at(date, '21:00')

export function matchDay(now: Date, games: Game[], times: MatchTimes, matches: MatchReview[]): MatchDay | null {
  const isGame = (d: Date) => games.some((g) => g.day === todayIndex(d))
  const reviewed = (date: string) => matches.some((m) => m.date === date)
  const today = ymd(now)
  if (isGame(now) && !reviewed(today)) {
    const tm = times[today]
    const opens = reviewOpensAt(today, tm)
    if (now.getTime() >= opens) return { state: 'review', date: today }
    return tm ? { state: 'set', date: today, start: tm.start, end: tm.end, opensAt: opens } : { state: 'times', date: today }
  }
  const y = new Date(now.getTime() - 864e5)
  if (isGame(y) && !reviewed(ymd(y))) return { state: 'review', date: ymd(y) }
  return null
}
