// Pure session logic: no DOM, unit-tested.
import type { Direction, Session, Step } from './types'

export type Rng = () => number

export function flattenSession(session: Session): Step[] {
  const steps: Step[] = []
  session.blocks.forEach((b, bi) => {
    const rounds = b.rounds || 1
    for (let r = 1; r <= rounds; r++) {
      steps.push({
        type: 'work', blockIdx: bi, name: b.name, kind: b.kind, round: r, rounds,
        seconds: b.work, cue: !!b.cue, how: b.how || '', needs: b.needs || [],
      })
      const isLast = bi === session.blocks.length - 1 && r === rounds
      if (b.rest && !isLast) {
        steps.push({
          type: 'rest', blockIdx: bi, name: 'Rest', kind: b.kind, round: r, rounds,
          seconds: b.rest, cue: false, how: '', needs: [],
        })
      }
    }
  })
  return steps
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
  const lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//Sport Personal Coach//EN', 'CALSCALE:GREGORIAN']
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
