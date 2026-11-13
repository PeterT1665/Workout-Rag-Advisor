import { Button, Pill } from './ui'
import { isMine } from '../lib/library'
import { DAYS } from '../lib/weekplan'
import type { DayChange } from '../lib/weekChanges'
import type { Session } from '../lib/types'

// Pick what a day holds: any of your sessions or a recommended one, rest, or back to the plan.
export function DayPicker({ day, planned, current, library, md, onPick, onClose }: {
  day: number
  planned: string // what the planner put on this day
  current: DayChange | null
  library: Session[]
  md: string | null
  onPick: (change: DayChange | null) => void
  onClose: () => void
}) {
  const mine = library.filter((s) => isMine(s.id))
  const rest = library.filter((s) => !isMine(s.id))
  const chosen = current && 'sessionId' in current ? current.sessionId : null
  const row = (s: Session) => (
    <li key={s.id}>
      <button onClick={() => onPick({ sessionId: s.id })} aria-pressed={chosen === s.id}
        className={`flex min-h-12 w-full items-center justify-between gap-3 rounded-2xl px-4 py-2.5 text-left ${chosen === s.id ? 'bg-red text-on-red' : 'bg-surface-2'}`}>
        <span className="font-semibold">{s.title}</span>
        <span className={`shrink-0 text-xs uppercase ${chosen === s.id ? '' : 'text-muted'}`}>{s.type}</span>
      </button>
    </li>
  )
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label={`Change ${DAYS[day]}`}>
      <button aria-label="Close" className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="rise relative max-h-[86dvh] w-full max-w-xl overflow-y-auto rounded-t-[28px] bg-surface px-5 pt-3 pb-[max(20px,env(safe-area-inset-bottom))]">
        <div className="mx-auto mb-4 h-1.5 w-10 rounded-full bg-surface-2" aria-hidden="true" />
        <h2 className="m-0 font-display text-2xl uppercase">{DAYS[day]}</h2>
        <p className="m-0 mt-1 text-sm text-muted">Planned: {planned}{md ? ` · ${md}` : ''}</p>
        {md === 'MD-1' && <p className="m-0 mt-3"><Pill tone="red">Day before a game: keep it light</Pill></p>}
        <div className="mt-4 flex gap-2">
          {current && <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => onPick(null)}>Back to plan</Button>}
          <Button variant="ghost" size="sm" className={current && 'rest' in current ? 'bg-red text-on-red' : 'bg-surface-2'} onClick={() => onPick({ rest: true })}>Rest</Button>
        </div>
        {mine.length > 0 && <>
          <h3 className="mt-5 mb-2 font-display text-sm uppercase text-muted">My sessions</h3>
          <ul className="m-0 flex list-none flex-col gap-1.5 p-0">{mine.map(row)}</ul>
        </>}
        <h3 className="mt-5 mb-2 font-display text-sm uppercase text-muted">Recommended</h3>
        <ul className="m-0 flex list-none flex-col gap-1.5 p-0">{rest.map(row)}</ul>
      </div>
    </div>
  )
}
