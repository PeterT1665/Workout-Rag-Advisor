// Week skeleton: which kind of session goes on which day, built around the week's games.
// Started as a port of ../../../weekplan.py; v3 adds the gym rules below (the Python file keeps v2).

import { locale, t } from '../i18n'

// Day names in the app's language (1 Jan 2024 was a Monday).
const dayNames = (weekday: 'long' | 'short' | 'narrow') =>
  Array.from({ length: 7 }, (_, i) => new Date(2024, 0, 1 + i).toLocaleDateString(locale(), { weekday }))
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
export const DAYS = dayNames('long').map(cap)
export const DAY_SHORT = dayNames('narrow').map(cap)
export const DAY_ABBR = dayNames('short').map((d) => cap(d.replace(/\.$/, '')))

// Football rule (Peter, 2026-10-23): 1 game -> 2 football sessions, 2 games -> 1, 3 games -> 0.
// v3 (2026-11-09, D33): footballers also lift 3-4 times a week. v4 (2026-11-16): gym days get a job by matchday
// distance (see gymFor), modelled on a pro footballer's week; recovery can share a game day.
// The Python weekplan.py keeps the original v2 rules for the CLI; this file is the app's source of truth.
export const FOOTBALL_BY_GAMES: Record<number, number> = { 0: 3, 1: 2, 2: 1, 3: 0 }
export const RECOVERY_BY_GAMES: Record<number, number> = { 0: 1, 1: 1, 2: 1, 3: 2 }
export const DEFAULT_GYM = 3

export type GameKind = 'casual' | 'competitive'
export type Game = { day: number; kind: GameKind } // day: 0 = Monday
export type SlotKind = 'game' | 'football' | 'gym' | 'recovery' | 'cheer'
// v4 (2026-11-16, modelled on a pro's week): each gym day has a job set by its distance to the game.
export type GymFocus = 'strength' | 'upper-power' | 'core' | 'activation'
export type Slot = { day: number; kind: SlotKind; note?: string; focus?: GymFocus }
export type WeekPlan = { slots: Slot[]; reasons: string[] }
export const kindLabel = (k: SlotKind | string) =>
  ({ game: t('Game'), football: t('Football'), gym: t('Gym'), recovery: t('Recovery'), home: t('Home') } as Record<string, string>)[k] ?? k

export const FOCUS_LABEL: Record<GymFocus, string> = {
  strength: t('Strength & contrast: legs'), 'upper-power': t('Upper body power'), core: t('Core & injury prevention'), activation: t('Match-eve activation'),
}
export const FOCUS_SHORT: Record<GymFocus, string> = { strength: t('Legs'), 'upper-power': t('Upper'), core: t('Core'), activation: t('Activation') }
const FOCUS_NOTE: Record<GymFocus, string> = {
  strength: t('heavy legs and contrast lifts, far enough from the game'),
  'upper-power': t('submax upper body and power, legs stay fresh'),
  core: t('core and injury prevention (hamstrings, groin, trunk)'),
  activation: t('short isometrics and a few fast reps: wakes the legs up without tiring them'),
}

// Days to the next game and since the last one. Games repeat every week, so it wraps (Wednesday's game makes
// next Saturday MD-4). 0 = game day; Infinity when there are no games.
export function matchDistance(day: number, games: Game[]) {
  const days = [...new Set(games.map((g) => g.day))]
  if (!days.length) return { ahead: Infinity, behind: Infinity }
  return { ahead: Math.min(...days.map((g) => (g - day + 7) % 7)), behind: Math.min(...days.map((g) => (day - g + 7) % 7)) }
}

// The gym job for a day, and how good a gym day it is (lower is better).
function gymFor(day: number, games: Game[]): { focus: GymFocus; score: number } {
  const { ahead, behind } = matchDistance(day, games)
  if (behind === 1) return { focus: 'core', score: 9 } // MD+1 is for recovery: gym only if nothing else is left
  if (ahead === 1) return { focus: 'activation', score: 4 }
  if (ahead === 2) return { focus: 'upper-power', score: 2 }
  if (ahead === 3) return { focus: 'core', score: 3 }
  if (behind === 2) return { focus: 'core', score: 6 } // MD+2: active recovery, so only light gym
  return { focus: 'strength', score: 1 }
}

export function buildWeek(games: Game[], opts: { gym?: number } = {}): WeekPlan {
  const n = Math.min(games.length, 3)
  const footballCount = FOOTBALL_BY_GAMES[n]
  const gymCount = Math.max(0, Math.min(5, opts.gym ?? DEFAULT_GYM))
  const slots: Slot[] = []
  const reasons: string[] = []
  const gameDays = new Set(games.map((g) => g.day))
  const on = (d: number, kind: SlotKind) => slots.some((s) => s.day === d && s.kind === kind)
  const free = (d: number) => !slots.some((s) => s.day === d)
  const week = DAYS.map((_, d) => d)

  for (const g of games) slots.push({ day: g.day, kind: 'game', note: g.kind === 'casual' ? t('casual game') : t('competitive game') })

  // Recovery first: the day after a game (MD+1) is kept for it; back-to-back games put it on the game day.
  for (let i = 0; i < RECOVERY_BY_GAMES[n]; i++) {
    const after = [...gameDays].sort((a, b) => a - b).map((g) => (g + 1) % 7).find((d) => free(d))
    if (after !== undefined) { slots.push({ day: after, kind: 'recovery', note: t('hips, hamstrings, quads, light mobility') }); continue }
    const gameDay = [...gameDays].sort((a, b) => a - b).find((d) => !on(d, 'recovery'))
    const d = gameDay ?? week.find(free)
    if (d === undefined) continue
    slots.push({ day: d, kind: 'recovery', note: gameDays.has(d) ? t('after the game: 15 min mobility and stretching') : t('light mobility') })
  }

  // Gym: the most useful jobs first (legs far from the game, upper power at MD-2, core at MD-3, activation at MD-1),
  // never on a game day, away from recovery days and from each other when there's a choice.
  const gymDays: number[] = []
  for (let i = 0; i < gymCount; i++) {
    const cands = week.filter((d) => !gameDays.has(d) && !gymDays.includes(d))
    if (!cands.length) { reasons.push(t('No day left for another gym session this week.')); continue }
    const cost = (d: number) => (n ? gymFor(d, games).score : 0) + (on(d, 'recovery') ? 5 : 0)
      + (gymDays.includes(d - 1) || gymDays.includes(d + 1) ? (n ? 0.5 : 2) : 0) + d * 0.01
    gymDays.push(cands.reduce((a, b) => (cost(b) < cost(a) ? b : a)))
  }
  // No games: rotate the jobs through the week. With games: the job follows the matchday.
  const rotation: GymFocus[] = ['strength', 'upper-power', 'core', 'strength', 'upper-power']
  gymDays.sort((a, b) => a - b).forEach((d, i) => {
    const focus = n ? gymFor(d, games).focus : rotation[i]
    slots.push({ day: d, kind: 'gym', focus, note: FOCUS_NOTE[focus] })
  })

  // Football: hard days hard, easy days easy (both pro references). The high-intensity session goes with the MD-4
  // legs day (football + legs + speed), sharp work at MD-2, technique further out; MD-3 stays light, never MD-1.
  const footballCost = (d: number) => {
    const { ahead, behind } = matchDistance(d, games)
    const hardDay = n > 0 && ahead === 4
    return (ahead === 1 ? 10 : 0) + (behind === 1 ? 6 : 0) + (ahead === 2 ? 1 : 0) + (ahead === 3 ? 2 : 0) + (behind === 2 ? 1 : 0)
      + (on(d, 'gym') && !hardDay ? 3 : 0) + (on(d, 'recovery') ? 4 : 0) + (on(d - 1, 'football') || on(d + 1, 'football') ? 1.5 : 0) + d * 0.01
  }
  let doubled = 0
  for (let i = 0; i < footballCount; i++) {
    const cands = week.filter((d) => !gameDays.has(d) && !on(d, 'football'))
    if (!cands.length) { reasons.push(t('No free day for a football session; skipped to protect recovery.')); continue }
    const best = cands.reduce((a, b) => (footballCost(b) < footballCost(a) ? b : a))
    if (footballCost(best) >= 6) reasons.push(t('Football on {day} is close to a game: no cleaner day was free.', { day: DAYS[best] }))
    if (on(best, 'gym')) doubled++
    const { ahead } = matchDistance(best, games)
    const note = ahead === 4 ? t('high intensity: 1v1, finishing, sprints (then legs in the gym)') : ahead === 3 ? t('high intensity: 1v1, finishing') : ahead === 2 ? t('sharp: passing and first touch') : t('technique and touches')
    slots.push({ day: best, kind: 'football', note: n ? note : undefined })
  }

  slots.sort((a, b) => a.day - b.day || order(a.kind) - order(b.kind))
  const label = n === 1 ? t('1 game') : n < 3 ? t('{n} games', { n }) : t('3+ games')
  let why = t('{games} this week: {football} football, {gym} gym, {recovery} recovery.', { games: label, football: footballCount, gym: gymCount, recovery: RECOVERY_BY_GAMES[n] })
  if (n) why += ' ' + t('Each gym day has a job set by how far it is from the game: legs early in the week, then core, upper body, and a short activation the day before. Hard days are hard (football, legs and speed together at MD-4), easy days are easy.')
  if (n >= 3) why += ' ' + t('Games carry the football load, so the focus is gym and recovery.')
  if (doubled) why += ' ' + (doubled > 1 ? t('{n} football sessions share a day with the gym: ball work first, gym after.', { n: doubled }) : t('1 football session shares a day with the gym: ball work first, gym after.'))

  reasons.unshift(why)
  return { slots, reasons }
}
const order = (k: SlotKind) => ['game', 'football', 'gym', 'recovery', 'cheer'].indexOf(k)

// Monday = 0
export const todayIndex = (d = new Date()) => (d.getDay() + 6) % 7

// Matchday label: MD-2 = two days before the next game, MD+1 = day after the last one (wraps around the week).
export function matchdayLabel(day: number, games: Game[]): string | null {
  const { ahead, behind } = matchDistance(day, games)
  if (ahead === Infinity) return null
  if (ahead === 0) return 'MD'
  // football convention: MD+1 and MD+2 after a game, then count down to the next one
  return behind <= 2 && behind < ahead ? `MD+${behind}` : `MD-${ahead}`
}
