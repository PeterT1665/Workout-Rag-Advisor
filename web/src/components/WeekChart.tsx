import { useState } from 'react'
import { DAYS, DAY_SHORT, todayIndex } from '../lib/weekplan'
import { allSessions } from '../lib/library'
import type { HistoryEntry } from '../lib/storage'

// Minutes trained per day this week, stacked by kind of session. Colours: validated categorical palette
// (dataviz default, dark steps, checked against the app's surface #161618). Fixed order, never cycled.
export const KINDS = [
  { id: 'football', label: 'Football', color: '#3987e5' },
  { id: 'gym', label: 'Gym', color: '#d95926' },
  { id: 'recovery', label: 'Recovery & mobility', color: '#199e70' },
  { id: 'home', label: 'Home', color: '#c98500' },
] as const
type Kind = (typeof KINDS)[number]['id']

// What kind a logged session was: saved with it since v3, otherwise looked up from the session.
export function kindOf(h: HistoryEntry): Kind {
  const t = h.type ?? allSessions().find((s) => s.id === h.id)?.type ?? h.id.split('-')[0]
  return t === 'gym' || t === 'home' || t === 'recovery' ? t : 'football'
}

const ymd = (d: Date) => d.toLocaleDateString('en-CA')

export function weekByKind(history: HistoryEntry[], now = new Date()): Record<Kind, number>[] {
  const monday = new Date(now)
  monday.setDate(now.getDate() - todayIndex(now))
  return DAYS.map((_, i) => {
    const d = new Date(monday)
    d.setDate(monday.getDate() + i)
    const day = { football: 0, gym: 0, recovery: 0, home: 0 }
    for (const h of history) if (h.date === ymd(d)) day[kindOf(h)] += h.minutes
    return day
  })
}

export function WeekChart({ history }: { history: HistoryEntry[] }) {
  const days = weekByKind(history)
  const [open, setOpen] = useState<number | null>(null)
  const totals = days.map((d) => KINDS.reduce((a, k) => a + d[k.id], 0))
  const max = Math.max(...totals, 30)
  const week = totals.reduce((a, b) => a + b, 0)
  const byKind = KINDS.map((k) => days.reduce((a, d) => a + d[k.id], 0))
  const today = todayIndex()
  const describe = (i: number) => `${DAYS[i]}: ${totals[i]} min${totals[i] ? ` (${KINDS.filter((k) => days[i][k.id]).map((k) => `${k.label} ${days[i][k.id]}`).join(', ')})` : ''}`

  return (
    <figure className="m-0 rounded-[20px] bg-surface p-4">
      <figcaption className="flex items-baseline justify-between">
        <span className="font-semibold">Minutes trained this week</span>
        <span className="font-display text-xl tabular-nums">{week}</span>
      </figcaption>

      <ul className="m-0 mt-3 flex list-none flex-wrap gap-x-4 gap-y-1.5 p-0 text-xs text-muted">
        {KINDS.map((k, i) => (
          <li key={k.id} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: k.color }} aria-hidden="true" />
            {k.label} <span className="tabular-nums text-ink">{byKind[i]}</span>
          </li>
        ))}
      </ul>

      <div className="relative mt-6 grid h-36 grid-cols-7 items-end gap-2" role="img" aria-label={`Minutes by day: ${days.map((_, i) => describe(i)).join('; ')}`}>
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-[color-mix(in_oklab,var(--text)_14%,transparent)]" />
        {days.map((d, i) => (
          <button key={i} aria-label={describe(i)} onClick={() => setOpen(open === i ? null : i)}
            onMouseEnter={() => setOpen(i)} onMouseLeave={() => setOpen(null)}
            className="relative flex h-full flex-col items-center justify-end">
            {totals[i] > 0 && (open === null || open === i) && (
              <span className="mb-1 text-[11px] font-semibold tabular-nums text-muted">{totals[i]}</span>
            )}
            {totals[i] === 0
              ? <span className="h-1 w-full max-w-7 rounded-t-[4px] bg-surface-2" />
              : <span className="flex w-full max-w-7 flex-col-reverse gap-[2px] overflow-hidden rounded-t-[4px]" style={{ height: `${Math.max(8, (totals[i] / max) * 100)}%` }}>
                  {KINDS.filter((k) => d[k.id]).map((k) => (
                    <span key={k.id} style={{ flex: d[k.id], background: k.color }} />
                  ))}
                </span>}
            {open === i && totals[i] > 0 && (
              <span className="absolute bottom-full left-1/2 z-10 mb-6 w-max -translate-x-1/2 rounded-xl bg-surface-2 px-3 py-2 text-left text-xs shadow-lg">
                <span className="mb-1 block font-semibold">{DAYS[i]}</span>
                {KINDS.filter((k) => d[k.id]).map((k) => (
                  <span key={k.id} className="flex items-center gap-1.5 text-muted">
                    <span className="h-2 w-2 rounded-[2px]" style={{ background: k.color }} />{k.label} <span className="tabular-nums text-ink">{d[k.id]} min</span>
                  </span>
                ))}
              </span>
            )}
          </button>
        ))}
      </div>
      <div className="mt-2 grid grid-cols-7 gap-2 text-center font-display text-[11px] text-muted">
        {DAY_SHORT.map((s, i) => <span key={i} className={i === today ? 'text-ink' : ''}>{s}</span>)}
      </div>
      {week === 0 && <p className="m-0 mt-3 text-sm text-muted">Finish a session and it shows up here, coloured by kind.</p>}
    </figure>
  )
}
