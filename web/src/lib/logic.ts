// Pure session logic: no DOM, unit-tested.
import type { Direction, Session, Step } from './types'

export type Rng = () => number

export function flattenSession(session: Session): Step[] {
  const steps: Step[] = []
  const blocks = session.blocks
  let letter = 0
  for (let bi = 0; bi < blocks.length;) {
    // a superset is this block plus the following ones marked superset: their rounds alternate, A1 A2 rest A1 A2...
    const group = [bi]
    while (bi + group.length < blocks.length && blocks[bi + group.length].superset) group.push(bi + group.length)
    const isSet = group.length > 1
    const tag = isSet ? String.fromCharCode(65 + letter++) : ''
    const maxRounds = Math.max(...group.map((g) => blocks[g].rounds || 1))
    const lastBlock = bi + group.length >= blocks.length
    for (let r = 1; r <= maxRounds; r++) {
      const active = group.filter((g) => r <= (blocks[g].rounds || 1))
      active.forEach((g, k) => {
        const b = blocks[g]
        steps.push({
          type: 'work', blockIdx: g, name: b.name, kind: b.kind, round: r, rounds: b.rounds || 1,
          seconds: b.work, cue: !!b.cue, how: b.how || '', needs: b.needs || [], reps: b.reps, category: b.category,
          ...(isSet ? { pair: `${tag}${group.indexOf(g) + 1}` } : {}),
        })
        const endOfRound = k === active.length - 1
        const rest = isSet ? blocks[group[group.length - 1]].rest : b.rest
        if (endOfRound && rest && !(lastBlock && r === maxRounds)) {
          steps.push({
            type: 'rest', blockIdx: g, name: 'Rest', kind: b.kind, round: r, rounds: maxRounds,
            seconds: rest, cue: false, how: '', needs: [],
          })
        }
      })
    }
    bi += group.length
  }
  return steps
}

// First block that can still be changed mid-session: everything done or under way (the whole current superset) is locked.
export function editableFrom(steps: Step[], i: number): number {
  if (i < 0) return 0
  let last = -1
  const letter = steps[i]?.pair?.[0]
  for (let k = 0; k <= i && k < steps.length; k++) last = Math.max(last, steps[k].blockIdx)
  if (letter) for (const s of steps) if (s.pair?.[0] === letter) last = Math.max(last, s.blockIdx)
  // a superset's rest belongs to the pair: rounds still to come keep it locked
  const cur = steps[i]
  if (cur?.type === 'rest') {
    const prevPair = steps[i - 1]?.pair?.[0]
    if (prevPair) for (const s of steps) if (s.pair?.[0] === prevPair) last = Math.max(last, s.blockIdx)
  }
  return last + 1
}

export const totalSeconds = (steps: Step[]) => steps.reduce((a, s) => a + s.seconds, 0)
export const sessionMinutes = (s: Session) => Math.round(totalSeconds(flattenSession(s)) / 60)

// Random delay (seconds) until the next cue, uniform in [min, max].
export function nextCueDelay(min: number, max: number, rng: Rng = Math.random) {
  if (max < min) max = min
  return min + rng() * (max - min)
}

// Random left/right, never the same direction more than maxRun times in a row.
export function pickCue(history: Direction[], rng: Rng = Math.random, maxRun = 3): Direction {
  const last = history[history.length - 1]
  let run = 0
  for (let i = history.length - 1; i >= 0 && history[i] === last; i--) run++
  if (last && run >= maxRun) return last === 'left' ? 'right' : 'left'
  return rng() < 0.5 ? 'left' : 'right'
}

// Cue times (seconds from step start) for one work step. Stops `tail` seconds before the end.
export function planCues(stepSeconds: number, min: number, max: number, rng: Rng = Math.random, firstDelay = 3, tail = 2) {
  const times: number[] = []
  let t = firstDelay
  while (t < stepSeconds - tail) {
    times.push(t)
    t += nextCueDelay(min, max, rng)
  }
  return times
}

// Random item, never the same one twice in a row (when there is a choice).
export function pickNoRepeat<T>(items: T[], lastKey: unknown, rng: Rng = Math.random, keyOf: (x: T) => unknown = (x) => x): T | null {
  if (!items.length) return null
  const pool = items.length > 1 ? items.filter((x) => keyOf(x) !== lastKey) : items
  return pool[Math.floor(rng() * pool.length)]
}

// Rest clips only on rests long enough to hear something and still get the "next" call.
export const REST_CLIP = { start: 2.5, stopBeforeEnd: 4, minRest: 20 }
export function restClipWindow(restSeconds: number, chance: number, rng: Rng = Math.random) {
  if (restSeconds < REST_CLIP.minRest || rng() >= chance) return null
  return { start: REST_CLIP.start, stop: restSeconds - REST_CLIP.stopBeforeEnd }
}

// ---- calendar export (.ics) ----
const pad = (n: number) => String(n).padStart(2, '0')
const icsDate = (d: Date) =>
  `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}T${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}Z`
const icsEscape = (s: string) =>
  String(s).replace(/\\/g, '\\\\').replace(/;/g, '\;').replace(/,/g, '\\,').replace(/\r?\n/g, '\\n')

export type IcsEvent = { title: string; start: Date; minutes: number; description?: string; alertMinutes?: number; uid?: string }
export function makeICS(events: IcsEvent[]) {
  const lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//Football Personal Coach//EN', 'CALSCALE:GREGORIAN']
  events.forEach((e, i) => {
    const end = new Date(e.start.getTime() + e.minutes * 60000)
    lines.push(
      'BEGIN:VEVENT',
      'UID:' + (e.uid || `${icsDate(e.start)}-${i}@training-agent`),
      'DTSTAMP:' + icsDate(new Date()),
      'DTSTART:' + icsDate(e.start),
      'DTEND:' + icsDate(end),
      'SUMMARY:' + icsEscape(e.title),
      'DESCRIPTION:' + icsEscape(e.description || ''),
      'BEGIN:VALARM', 'ACTION:DISPLAY', 'DESCRIPTION:' + icsEscape(e.title),
      `TRIGGER:-PT${e.alertMinutes ?? 30}M`, 'END:VALARM',
      'END:VEVENT')
  })
  lines.push('END:VCALENDAR')
  return lines.join('\r\n') + '\r\n'
}

export function fmtTime(sec: number) {
  sec = Math.max(0, Math.ceil(sec))
  return `${pad(Math.floor(sec / 60))}:${pad(sec % 60)}`
}

// "8" -> "8 reps"; descriptive values like "8 each leg" or "45 s" stay as written.
export const fmtReps = (r: string) => (/^\d+(-\d+)?$/.test(r) ? `${r} reps` : r)

// Sets with reps run at the player's own pace: no countdown, the set ends when they tap Set done.
export const selfPaced = (s: Step | undefined) => !!s && s.type === 'work' && !!s.reps

// One-line tempo cue for a self-paced set, by exercise category.
export function tempoFor(category?: string): string {
  switch (category) {
    case 'olympic': case 'explosive': case 'plyometrics': case 'sprints': case 'speed':
      return 'Max intent: explode up, reset between reps'
    case 'isometrics': return 'Hold it steady and breathe'
    case 'core': return 'Slow and controlled'
    default: return '3 seconds down, drive up fast'
  }
}
