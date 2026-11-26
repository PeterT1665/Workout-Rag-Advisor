import { Pill, SectionHead, Title } from '../components/ui'
import { FitLabel } from '../components/TabBar'
import { DAY_ABBR, DAYS, DAY_SHORT, FOCUS_SHORT, defaultFootball, defaultGymFor, kindLabel, matchdayLabel, todayIndex, type Game } from '../lib/weekplan'
import { locale, t } from '../i18n'
import { changeFor, changedSession, type DayChange, type WeekChanges } from '../lib/weekChanges'
import { DayPicker } from '../components/DayPicker'
import { WeekGuide } from '../components/WeekGuide'
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

export function WeekView({ today, games, sessions, library, changes, onChange, onGames, onOpen, onCoach, wide = false }: {
  today: Today; games: Game[]; sessions: Session[]; library: Session[]; changes: WeekChanges
  onChange: (day: number, change: DayChange | null, week?: 0 | 1) => void; onGames: (g: Game[]) => void; onOpen: (s: Session) => void; onCoach?: () => void; wide?: boolean
}) {
  const [picking, setPicking] = useState<string | null>(null) // `${week}-${day}`
  // Sunday: next week's plan shows under this week's, ready to plan
  const showNext = todayIndex() === 6
  const nextMonday = new Date(Date.now() + 864e5).toLocaleDateString(locale(), { weekday: 'short', day: 'numeric', month: 'short' })
  const [editing, setEditing] = useState(false) // Edit: tap any day to change it
  const me = useProfile()
  const season = me.season ?? 'in'
  const gym = me.gym ?? defaultGymFor(season, games.length)
  const football = me.footballPerWeek ?? defaultFootball(games.length, season)
  const setGym = (n: number) => setProfile({ ...me, gym: Math.max(0, Math.min(5, n)) })
  const setFootball = (n: number) => setProfile({ ...me, footballPerWeek: Math.max(0, Math.min(6, n)) })
  // a new season starts from that season's numbers: the player's own counts are cleared
  const setSeason = (x: 'in' | 'off') => setProfile({ ...me, season: x, gym: undefined, footballPerWeek: undefined })
  const kindOn = (day: number) => games.find((g) => g.day === day)?.kind ?? 'none'
  const cycle = (day: number) => {
    const next = NEXT[kindOn(day)]
    const rest = games.filter((g) => g.day !== day)
    onGames(next ? [...rest, { day, kind: next }].sort((a, b) => a.day - b.day) : rest)
  }
  const md = (day: number) => matchdayLabel(day, games)
  // one week's plan, this week (0) or next (1): same planner, its own changes by date
  const planList = (week: 0 | 1) => {
    const when = week ? new Date(Date.now() + 7 * 864e5) : new Date()
    return (
        <ol className="m-0 flex list-none flex-col gap-2 p-0">
          {DAYS.map((name, day) => {
            const slots = today.plan.slots.filter((s) => s.day === day)
            const change = changeFor(day, changes, when)
            const changed = changedSession(change, library)
            const gameSlots = slots.filter((s) => s.kind === 'game')
            const planned = slots.map((s) => (s.kind === 'game' ? s.note : sessionForSlot(s, today.plan, sessions)?.title ?? kindLabel(s.kind))).join(' + ') || t('Rest')
            // links: session titles open the session (outside Edit, where the whole row changes the day)
            const title = (session: Session, links: boolean) => links
              ? <button onClick={() => onOpen(session)} className="text-left font-semibold underline decoration-red decoration-2 underline-offset-4">{session.title}</button>
              : <span className="font-semibold">{session.title}</span>
            const row = (links: boolean) => change
              ? <>
                  {gameSlots.map((g, i) => <Pill key={i} tone="red">{g.note}</Pill>)}
                  {changed ? title(changed, links) : <span className="text-muted">{t('Rest')}</span>}
                  <Pill>{t('Changed')}</Pill>
                </>
              : <>
                  {slots.length === 0 && <span className="text-muted">{t('Rest')}</span>}
                  {slots.map((s, i) => {
                    const session = sessionForSlot(s, today.plan, sessions)
                    return session
                      ? <span key={i} className="flex items-center gap-2">
                          {title(session, links)}
                          {s.focus && <span title={s.note}><Pill>{FOCUS_SHORT[s.focus]}</Pill></span>}
                        </span>
                      : <Pill key={i} tone={s.kind === 'game' ? 'red' : 'muted'}>{s.kind === 'game' ? s.note : kindLabel(s.kind)}</Pill>
                  })}
                </>
            return (
              <li key={day} className={`relative rounded-[16px] ${week === 0 && day === today.day ? 'bg-surface-2' : 'bg-surface'} ${editing ? 'ring-1 ring-red/50' : ''}`}>
                {editing
                  ? <button onClick={() => setPicking(`${week}-${day}`)} aria-label={week ? t('Change next {day}', { day: name }) : t('Change {day}', { day: name })}
                      className="flex min-h-14 w-full items-center gap-3 rounded-[16px] py-2 pr-3 pl-4 text-left transition-transform duration-150 ease-snap active:scale-[0.98]">
                      <span className="w-12 shrink-0 font-display text-xs uppercase text-muted">{DAY_ABBR[day]}{md(day) && <span className={`block text-[10px] ${md(day) === 'MD' ? 'text-red-ink' : ''}`}>{md(day)}</span>}</span>
                      <span className="flex flex-1 flex-wrap items-center gap-2">{row(false)}</span>
                      <PencilIcon />
                    </button>
                  : <div className="flex min-h-14 items-center gap-3 py-2 pr-2 pl-4">
                      <span className="w-12 shrink-0 font-display text-xs uppercase text-muted">{DAY_ABBR[day]}{md(day) && <span className={`block text-[10px] ${md(day) === 'MD' ? 'text-red-ink' : ''}`}>{md(day)}</span>}</span>
                      <span className="flex flex-1 flex-wrap items-center gap-2">{row(true)}</span>
                    </div>}
                {picking === `${week}-${day}` && (
                  <DayPicker day={day} planned={planned} current={change} library={library} md={matchdayLabel(day, games)}
                    onPick={(c) => { onChange(day, c, week); setPicking(null) }} onClose={() => setPicking(null)} />
                )}
              </li>
            )
          })}
        </ol>
    )
  }
  return (
    <main className={wide ? 'px-10 pt-10 pb-12' : 'mx-auto max-w-xl px-4 pt-[max(20px,var(--sat))] pb-32'}>
      <div className="flex items-center justify-between gap-3">
        <Title>{t('This week')}</Title>
        {onCoach && <button onClick={onCoach} className="min-h-10 shrink-0 rounded-full bg-surface-2 px-4 font-display text-xs uppercase text-ink active:scale-95">{t('AI coach')}</button>}
      </div>
      <div className={wide ? 'mt-2 grid grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-10' : ''}>
      <div>
      <div className="mt-4 grid grid-cols-[repeat(7,minmax(0,1fr))] gap-1.5">
        {DAY_SHORT.map((d, day) => {
          const k = kindOn(day)
          return (
            <button key={day} onClick={() => cycle(day)} aria-label={`${DAYS[day]}: ${k === 'none' ? t('no game') : k === 'casual' ? t('casual game') : t('competitive game')}`}
              className={`grid h-16 place-items-center rounded-[14px] font-display text-[9px] uppercase transition-transform duration-150 ease-snap active:scale-95 ${k === 'none' ? 'bg-surface text-muted' : 'bg-red text-white'}`}>
              <span className="text-sm">{d}</span>
              <span className="w-full overflow-hidden text-center"><FitLabel base={9} className="font-medium [font-stretch:100%]" text={k === 'none' ? '—' : k === 'casual' ? t('Casual|strip') : t('Comp')} /></span>
            </button>
          )
        })}
      </div>

      <div role="radiogroup" aria-label={t('Season')} className="mt-4 grid grid-cols-2 gap-1 rounded-full bg-surface p-1">
        {([['in', t('In season')], ['off', t('Off season')]] as const).map(([id, label]) => (
          <button key={id} role="radio" aria-checked={season === id} onClick={() => setSeason(id)}
            className={`min-h-10 rounded-full font-display text-xs uppercase ${season === id ? 'bg-red text-on-red' : 'text-muted'}`}>{label}</button>
        ))}
      </div>
      <Counter label={t('Football a week')} value={football} min={0} max={6} onChange={setFootball} less={t('Fewer football sessions')} more={t('More football sessions')} />
      <Counter label={t('Gym a week')} value={gym} min={0} max={5} onChange={setGym} less={t('Fewer gym sessions')} more={t('More gym sessions')} />
      <label className="mt-2 flex items-center justify-between gap-3 rounded-[16px] bg-surface px-4 py-3">
        <span className="font-semibold">{t('Football and gym on the same day')}</span>
        <input type="checkbox" checked={me.doubleDays !== false} onChange={(e) => setProfile({ ...me, doubleDays: e.target.checked })} className="h-6 w-6 accent-[var(--red)]" />
      </label>

      <WeekGuide plan={today.plan} games={games} />

      </div>
      <div>
      <SectionHead title={t('The plan')} right={
        <button onClick={() => { setEditing(!editing); setPicking(null) }} aria-pressed={editing}
          className={`min-h-9 rounded-full px-4 text-sm font-semibold ${editing ? 'bg-red text-on-red' : 'bg-surface-2 text-ink'}`}>{editing ? t('Done') : t('Edit')}</button>} />
      {planList(0)}
      {showNext && (
        <section aria-label={t('Next week')}>
          <SectionHead title={t('Next week')} right={<span className="text-sm text-muted">{t('from {date}', { date: nextMonday })}</span>} />
          {planList(1)}
        </section>
      )}
      </div>
      </div>
    </main>
  )
}

function Counter({ label, value, min, max, onChange, less, more }: { label: string; value: number; min: number; max: number; onChange: (n: number) => void; less: string; more: string }) {
  return (
    <div className="mt-2 flex items-center justify-between rounded-[16px] bg-surface px-4 py-2">
      <span className="font-semibold">{label}</span>
      <span className="flex items-center gap-1">
        <button aria-label={less} onClick={() => onChange(value - 1)} disabled={value <= min} className="h-10 w-10 rounded-full bg-surface-2 text-lg disabled:opacity-30">−</button>
        <span className="w-8 text-center font-display text-lg tabular-nums" aria-live="polite">{value}</span>
        <button aria-label={more} onClick={() => onChange(value + 1)} disabled={value >= max} className="h-10 w-10 rounded-full bg-surface-2 text-lg disabled:opacity-30">+</button>
      </span>
    </div>
  )
}
