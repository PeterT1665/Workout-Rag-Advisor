// Week skeleton: which kind of session goes on which day, built around the week's games.
// Started as a port of ../../../weekplan.py; v3 adds the gym rules below (the Python file keeps v2).

export const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
export const DAY_SHORT = ['M', 'T', 'W', 'T', 'F', 'S', 'S']

// Football rule (Peter, 2026-10-23): 1 game -> 2 football sessions, 2 games -> 1, 3 games -> 0.
// v3 (2026-11-09, D33): footballers also lift 3-4 times a week; heavy game weeks lighten the gym instead of
// cutting it, no heavy legs the day before a game, and recovery can share a game day.
// The Python weekplan.py keeps the original v2 rules for the CLI; this file is the app's source of truth.
export const FOOTBALL_BY_GAMES: Record<number, number> = { 0: 3, 1: 2, 2: 1, 3: 0 }
export const RECOVERY_BY_GAMES: Record<number, number> = { 0: 1, 1: 1, 2: 1, 3: 2 }
export const DEFAULT_GYM = 3

export type GameKind = 'casual' | 'competitive'
export type Game = { day: number; kind: GameKind } // day: 0 = Monday
export type SlotKind = 'game' | 'football' | 'gym' | 'recovery' | 'cheer'
export type Slot = { day: number; kind: SlotKind; note?: string; light?: boolean }
export type WeekPlan = { slots: Slot[]; reasons: string[] }

export function buildWeek(games: Game[], opts: { gym?: number } = {}): WeekPlan {
  const n = Math.min(games.length, 3)
  const footballCount = FOOTBALL_BY_GAMES[n]
  const gymCount = Math.max(0, Math.min(5, opts.gym ?? DEFAULT_GYM))
  const slots: Slot[] = []
  const reasons: string[] = []
  const gameDays = new Set(games.map((g) => g.day))
  const on = (d: number, kind: SlotKind) => slots.some((s) => s.day === d && s.kind === kind)
  const free = (d: number) => !slots.some((s) => s.day === d)

  for (const g of games) slots.push({ day: g.day, kind: 'game', note: g.kind })

  // Football: free days, away from the day before a game, then the day after, then other football days.
  const footballPenalty = (d: number) =>
    (gameDays.has(d + 1) ? 10 : 0) + (gameDays.has(d - 1) ? 4 : 0) + (on(d - 1, 'football') || on(d + 1, 'football') ? 1 : 0)
  for (let i = 0; i < footballCount; i++) {
    const cands = DAYS.map((_, d) => d).filter(free)
    if (!cands.length) { reasons.push('No free day for a football session; skipped to protect recovery.'); continue }
    const best = cands.reduce((a, b) => (footballPenalty(b) < footballPenalty(a) ? b : a))
    if (footballPenalty(best) >= 4) reasons.push(`Football on ${DAYS[best]} is close to a game: no cleaner day was free.`)
    slots.push({ day: best, kind: 'football' })
  }

  // Gym: free days first, keeping the day after a game for recovery; a full week doubles up with a football day
  // (gym after the ball work). Never on a game day.
  const gymPenalty = (d: number) =>
    (gameDays.has(d + 1) ? 6 : 0) + (gameDays.has(d - 1) ? 7 : 0) + (on(d - 1, 'gym') || on(d + 1, 'gym') ? 2 : 0) + (on(d, 'football') ? 9 : 0)
  let doubled = 0
  for (let i = 0; i < gymCount; i++) {
    const cands = DAYS.map((_, d) => d).filter((d) => !gameDays.has(d) && !on(d, 'gym'))
    if (!cands.length) { reasons.push('No day left for another gym session this week.'); continue }
    const best = cands.reduce((a, b) => (gymPenalty(b) < gymPenalty(a) ? b : a))
    if (on(best, 'football')) doubled++
    slots.push({ day: best, kind: 'gym' })
  }

  // Heavy weeks (2+ games) lighten every gym session; MD-1, MD+1 and MD-2 never get heavy legs.
  for (const s of slots.filter((x) => x.kind === 'gym')) {
    if (gameDays.has(s.day + 1)) { s.light = true; s.note = 'light: upper body and core, no heavy legs' }
    else if (gameDays.has(s.day - 1)) { s.light = true; s.note = 'light: day after a game, upper body and core, no heavy legs' }
    else if (n >= 2 || gameDays.has(s.day + 2)) { s.light = true; s.note = 'light: fewer sets, stop well short of failure' }
  }

  // Recovery: the day after a game if it's free, otherwise straight after a game on game day.
  for (let i = 0; i < RECOVERY_BY_GAMES[n]; i++) {
    const after = [...gameDays].sort((a, b) => a - b).map((g) => g + 1).find((d) => d < 7 && free(d))
    if (after !== undefined) { slots.push({ day: after, kind: 'recovery', note: 'hips, hamstrings, quads, light mobility' }); continue }
    const gameDay = [...gameDays].sort((a, b) => a - b).find((d) => !on(d, 'recovery'))
    const anyFree = DAYS.map((_, d) => d).find(free)
    const d = gameDay ?? anyFree
    if (d === undefined) continue
    slots.push({ day: d, kind: 'recovery', note: gameDays.has(d) ? 'after the game: 15 min mobility and stretching' : 'light mobility' })
  }

  slots.sort((a, b) => a.day - b.day || order(a.kind) - order(b.kind))
  const label = n < 3 ? `${n} game${n === 1 ? '' : 's'}` : '3+ games'
  const light = slots.filter((s) => s.light).length
  let why = `${label} this week: ${footballCount} football, ${gymCount} gym${light ? ` (${light} light)` : ''}, ${RECOVERY_BY_GAMES[n]} recovery.`
  if (n >= 3) why += ' Games carry the football load, so the focus is gym and recovery.'
  else if (n === 2) why += ' Two games: gym stays, but lighter.'
  if (doubled) why += ` ${doubled} gym session${doubled > 1 ? 's share' : ' shares'} a day with football: ball work first, gym after.`
  reasons.unshift(why)
  return { slots, reasons }
}
const order = (k: SlotKind) => ['game', 'football', 'gym', 'recovery', 'cheer'].indexOf(k)

// Monday = 0
export const todayIndex = (d = new Date()) => (d.getDay() + 6) % 7

// Matchday label: MD-2 = two days before the next game, MD+1 = day after the last one.
export function matchdayLabel(day: number, games: Game[]): string | null {
  const days = [...new Set(games.map((g) => g.day))]
  if (days.includes(day)) return 'MD'
  const ahead = days.filter((g) => g > day).map((g) => g - day)
  if (ahead.length) return `MD-${Math.min(...ahead)}`
  const behind = days.filter((g) => g < day).map((g) => day - g)
  if (behind.length) return `MD+${Math.min(...behind)}`
  return null
}
