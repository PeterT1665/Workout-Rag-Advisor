// Week skeleton: which kind of session goes on which day, built around the week's games.
// TypeScript port of ../../../weekplan.py (same rules, same tests).

export const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
export const DAY_SHORT = ['M', 'T', 'W', 'T', 'F', 'S', 'S']

// Peter's rule (2026-10-23): 1 game -> 2 football, 2 games -> 1, 3 games -> 0.
export const LOAD_TABLE: Record<number, { football: number; gym: number; recovery: number }> = {
  0: { football: 3, gym: 1, recovery: 1 },
  1: { football: 2, gym: 1, recovery: 1 },
  2: { football: 1, gym: 1, recovery: 1 },
  3: { football: 0, gym: 1, recovery: 2 },
}
export const CHEER_DAY = 5 // Saturday morning

export type GameKind = 'casual' | 'competitive'
export type Game = { day: number; kind: GameKind } // day: 0 = Monday
export type SlotKind = 'game' | 'football' | 'gym' | 'recovery' | 'cheer'
export type Slot = { day: number; kind: SlotKind; note?: string }
export type WeekPlan = { slots: Slot[]; reasons: string[] }

export function buildWeek(games: Game[], cheer = true): WeekPlan {
  const n = Math.min(games.length, 3)
  const counts = LOAD_TABLE[n]
  const slots: Slot[] = []
  const reasons: string[] = []
  const gameDays = new Set(games.map((g) => g.day))
  const busy = new Set<number>()
  const busyHard = new Set<number>()

  for (const g of games) { slots.push({ day: g.day, kind: 'game', note: g.kind }); busy.add(g.day) }
  const cheerDay = cheer && !gameDays.has(CHEER_DAY) ? CHEER_DAY : null
  if (cheerDay !== null) { slots.push({ day: cheerDay, kind: 'cheer', note: 'Saturday morning' }); busy.add(cheerDay) }

  const penalty = (d: number) => {
    let p = 0
    if (gameDays.has(d + 1)) p += 10 // day before a game
    if (gameDays.has(d - 1)) p += 4 // day after a game
    if (cheerDay !== null) {
      if (d + 1 === cheerDay) p += 3
      if (d - 1 === cheerDay) p += 2
    }
    if (busyHard.has(d + 1) || busyHard.has(d - 1)) p += 1
    return p
  }
  const pick = (kind: string) => {
    const cands = [0, 1, 2, 3, 4, 5, 6].filter((d) => !busy.has(d))
    if (!cands.length) return null
    const best = cands.reduce((a, b) => (penalty(b) < penalty(a) ? b : a))
    if (penalty(best) >= 4) reasons.push(`${kind} placed on ${DAYS[best]}, close to a game/cheer day: no cleaner day was free.`)
    return best
  }

  for (const kind of ['football', 'gym'] as const) {
    for (let i = 0; i < counts[kind]; i++) {
      const d = pick(kind)
      if (d === null) { reasons.push(`No free day for a ${kind} session; skipped to protect recovery.`); continue }
      slots.push({ day: d, kind }); busy.add(d); busyHard.add(d)
    }
  }
  for (let i = 0; i < counts.recovery; i++) {
    const afterGame = [...gameDays].sort((a, b) => a - b).map((g) => g + 1).filter((d) => d < 7 && !busy.has(d))
    const d = afterGame.length ? afterGame[0] : pick('recovery')
    if (d === null) continue
    slots.push({ day: d, kind: 'recovery', note: 'hips, hamstrings, quads, light mobility' }); busy.add(d)
  }

  slots.sort((a, b) => a.day - b.day)
  const label = n < 3 ? `${n} game${n === 1 ? '' : 's'}` : '3+ games'
  let why = `${label} this week: ${counts.football} football, ${counts.gym} gym, ${counts.recovery} recovery.`
  if (n >= 3) why += ' Games already carry the football load, so the focus is gym and recovery.'
  reasons.unshift(why)
  return { slots, reasons }
}

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
