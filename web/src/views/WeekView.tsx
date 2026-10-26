import { Panel, Pill, SectionHead, Title } from '../components/ui'
import { DAYS, DAY_SHORT, type Game } from '../lib/weekplan'
import { sessionForSlot, type Today } from '../lib/today'
import type { Session } from '../lib/types'

const NEXT: Record<string, Game['kind'] | null> = { none: 'casual', casual: 'competitive', competitive: null }

export function WeekView({ today, games, sessions, onGames, onOpen, wide = false }: {
  today: Today; games: Game[]; sessions: Session[]; onGames: (g: Game[]) => void; onOpen: (s: Session) => void; wide?: boolean
}) {
  const kindOn = (day: number) => games.find((g) => g.day === day)?.kind ?? 'none'
  const cycle = (day: number) => {
    const next = NEXT[kindOn(day)]
    const rest = games.filter((g) => g.day !== day)
    onGames(next ? [...rest, { day, kind: next }].sort((a, b) => a.day - b.day) : rest)
  }
  return (
    <main className={wide ? 'px-10 pt-10 pb-12' : 'mx-auto max-w-xl px-4 pt-[max(20px,env(safe-area-inset-top))] pb-32'}>
      <Title>This week</Title>
      <div className={wide ? 'mt-2 grid grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-10' : ''}>
      <div>
      <p className="mt-2 text-muted">Tap the days you play. The week rebuilds around your games.</p>
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

      <Panel className="mt-4">
        {today.plan.reasons.map((r, i) => <p key={i} className={`m-0 ${i ? 'mt-2 text-sm text-muted' : ''}`}>{r}</p>)}
      </Panel>

      </div>
      <div>
      <SectionHead title="The plan" />
      <ol className="m-0 flex list-none flex-col gap-2 p-0">
        {DAYS.map((name, day) => {
          const slots = today.plan.slots.filter((s) => s.day === day)
          return (
            <li key={day} className={`flex items-center gap-3 rounded-[16px] px-4 py-3 ${day === today.day ? 'bg-surface-2' : 'bg-surface'}`}>
              <span className="w-10 font-display text-xs uppercase text-muted">{name.slice(0, 3)}</span>
              <span className="flex flex-1 flex-wrap gap-2">
                {slots.length === 0 && <span className="text-muted">Rest</span>}
                {slots.map((s, i) => {
                  const session = sessionForSlot(s, today.plan, sessions)
                  return session
                    ? <button key={i} onClick={() => onOpen(session)} className="text-left font-semibold underline decoration-red decoration-2 underline-offset-4">{session.title}</button>
                    : <Pill key={i} tone={s.kind === 'game' ? 'red' : 'muted'}>{s.kind === 'game' ? `${s.note} game` : s.kind}</Pill>
                })}
              </span>
            </li>
          )
        })}
      </ol>
      </div>
      </div>
    </main>
  )
}
