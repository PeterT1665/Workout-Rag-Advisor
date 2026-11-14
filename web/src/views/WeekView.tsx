import { Panel, Pill, SectionHead, Title } from '../components/ui'
import { DAYS, DAY_SHORT, DEFAULT_GYM, matchdayLabel, type Game } from '../lib/weekplan'
import { changeFor, changedSession, type DayChange, type WeekChanges } from '../lib/weekChanges'
import { DayPicker } from '../components/DayPicker'
import { useState } from 'react'
import { setProfile, useProfile } from '../lib/profile'
import { sessionForSlot, type Today } from '../lib/today'
import type { Session } from '../lib/types'

const PencilIcon = () => (
  <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0 text-red-ink" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M4 20h4L19 9l-4-4L4 16v4Z" /><path d="m13.5 6.5 4 4" />
  </svg>
)

const NEXT: Record<string, Game['kind'] | null> = { none: 'casual', casual: 'competitive', competitive: null }

export function WeekView({ today, games, sessions, library, changes, onChange, onGames, onOpen, wide = false }: {
  today: Today; games: Game[]; sessions: Session[]; library: Session[]; changes: WeekChanges
  onChange: (day: number, change: DayChange | null) => void; onGames: (g: Game[]) => void; onOpen: (s: Session) => void; wide?: boolean
}) {
  const [picking, setPicking] = useState<number | null>(null)
  const [editing, setEditing] = useState(false) // Edit: tap any day to change it
  const me = useProfile()
  const gym = me.gym ?? DEFAULT_GYM
  const setGym = (n: number) => setProfile({ ...me, gym: Math.max(0, Math.min(5, n)) })
  const kindOn = (day: number) => games.find((g) => g.day === day)?.kind ?? 'none'
  const cycle = (day: number) => {
    const next = NEXT[kindOn(day)]
    const rest = games.filter((g) => g.day !== day)
    onGames(next ? [...rest, { day, kind: next }].sort((a, b) => a.day - b.day) : rest)
  }
  return (
    <main className={wide ? 'px-10 pt-10 pb-12' : 'mx-auto max-w-xl px-4 pt-[max(20px,var(--sat))] pb-32'}>
      <Title>This week</Title>
      <div className={wide ? 'mt-2 grid grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-10' : ''}>
      <div>
      <p className="mt-2 text-muted">Tap the days you play.</p>
      <div className="mt-4 grid grid-cols-7 gap-1.5">
        {DAY_SHORT.map((d, day) => {
          const k = kindOn(day)
          return (
            <button key={day} onClick={() => cycle(day)} aria-label={`${DAYS[day]}: ${k === 'none' ? 'no game' : k + ' game'}`}
              className={`grid h-16 place-items-center rounded-[14px] font-display text-[9px] uppercase transition-transform duration-150 ease-snap active:scale-95 ${k === 'none' ? 'bg-surface text-muted' : 'bg-red text-white'}`}>
              <span className="text-sm">{d}</span>
              <span>{k === 'none' ? '—' : k === 'casual' ? 'Casual' : 'Comp'}</span>
            </button>
          )
        })}
      </div>

      <div className="mt-4 flex items-center justify-between rounded-[16px] bg-surface px-4 py-2">
        <span className="font-semibold">Gym sessions a week</span>
        <span className="flex items-center gap-1">
          <button aria-label="Fewer gym sessions" onClick={() => setGym(gym - 1)} disabled={gym <= 0} className="h-10 w-10 rounded-full bg-surface-2 text-lg disabled:opacity-30">−</button>
          <span className="w-8 text-center font-display text-lg tabular-nums" aria-live="polite">{gym}</span>
          <button aria-label="More gym sessions" onClick={() => setGym(gym + 1)} disabled={gym >= 5} className="h-10 w-10 rounded-full bg-surface-2 text-lg disabled:opacity-30">+</button>
        </span>
      </div>

      <Panel className="mt-4">
        {today.plan.reasons.map((r, i) => <p key={i} className={`m-0 ${i ? 'mt-2 text-sm text-muted' : ''}`}>{r}</p>)}
      </Panel>

      </div>
      <div>
      <SectionHead title="The plan" right={
        <button onClick={() => { setEditing(!editing); setPicking(null) }} aria-pressed={editing}
          className={`min-h-9 rounded-full px-4 text-sm font-semibold ${editing ? 'bg-red text-on-red' : 'bg-surface-2 text-ink'}`}>{editing ? 'Done' : 'Edit'}</button>} />
      {editing && <p className="m-0 mb-3 text-sm text-muted">Tap a day to change it.</p>}
      <ol className="m-0 flex list-none flex-col gap-2 p-0">
        {DAYS.map((name, day) => {
          const slots = today.plan.slots.filter((s) => s.day === day)
          const change = changeFor(day, changes)
          const changed = changedSession(change, library)
          const gameSlots = slots.filter((s) => s.kind === 'game')
          const planned = slots.map((s) => (s.kind === 'game' ? `${s.note} game` : sessionForSlot(s, today.plan, sessions)?.title ?? s.kind)).join(' + ') || 'Rest'
          // links: session titles open the session (outside Edit, where the whole row changes the day)
          const title = (session: Session, links: boolean) => links
            ? <button onClick={() => onOpen(session)} className="text-left font-semibold underline decoration-red decoration-2 underline-offset-4">{session.title}</button>
            : <span className="font-semibold">{session.title}</span>
          const row = (links: boolean) => change
            ? <>
                {gameSlots.map((g, i) => <Pill key={i} tone="red">{`${g.note} game`}</Pill>)}
                {changed ? title(changed, links) : <span className="text-muted">Rest</span>}
                <Pill>Changed</Pill>
              </>
            : <>
                {slots.length === 0 && <span className="text-muted">Rest</span>}
                {slots.map((s, i) => {
                  const session = sessionForSlot(s, today.plan, sessions)
                  return session
                    ? <span key={i} className="flex items-center gap-2">
                        {title(session, links)}
                        {s.light && <span title={s.note}><Pill>Light</Pill></span>}
                      </span>
                    : <Pill key={i} tone={s.kind === 'game' ? 'red' : 'muted'}>{s.kind === 'game' ? `${s.note} game` : s.kind}</Pill>
                })}
              </>
          return (
            <li key={day} className={`relative rounded-[16px] ${day === today.day ? 'bg-surface-2' : 'bg-surface'} ${editing ? 'ring-1 ring-red/50' : ''}`}>
              {editing
                ? <button onClick={() => setPicking(day)} aria-label={`Change ${name}`}
                    className="flex min-h-14 w-full items-center gap-3 rounded-[16px] py-2 pr-3 pl-4 text-left transition-transform duration-150 ease-snap active:scale-[0.98]">
                    <span className="w-10 font-display text-xs uppercase text-muted">{name.slice(0, 3)}</span>
                    <span className="flex flex-1 flex-wrap items-center gap-2">{row(false)}</span>
                    <PencilIcon />
                  </button>
                : <div className="flex min-h-14 items-center gap-3 py-2 pr-2 pl-4">
                    <span className="w-10 font-display text-xs uppercase text-muted">{name.slice(0, 3)}</span>
                    <span className="flex flex-1 flex-wrap items-center gap-2">{row(true)}</span>
                  </div>}
              {picking === day && (
                <DayPicker day={day} planned={planned} current={change} library={library} md={matchdayLabel(day, games)}
                  onPick={(c) => { onChange(day, c); setPicking(null) }} onClose={() => setPicking(null)} />
              )}
            </li>
          )
        })}
      </ol>
      </div>
      </div>
    </main>
  )
}
