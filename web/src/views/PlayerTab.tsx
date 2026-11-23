import { Button, Panel, SectionHead, Title } from '../components/ui'
import { FlipCard } from '../components/FlipCard'
import { nextRatingDue } from '../lib/testday'
import { cardStatsFor } from '../lib/cardStats'
import { usePersonalImage } from '../lib/personal'
import { setPhoto } from '../lib/photos'
import { CropSheet } from '../components/CropSheet'
import { useProfile } from '../lib/profile'
import { useState } from 'react'
import { deleteHistory, loadHistory, type HistoryEntry } from '../lib/storage'
import { counts, minFor } from '../lib/streak'
import { KINDS, WeekChart, kindOf } from '../components/WeekChart'
import { locale, t } from '../i18n'
import { load, save } from '../lib/storage'
import { loadMatches, resultOf, resultStreaks, type MatchReview } from '../lib/match'
import { RatingBadge } from './MatchReview'
import { FlameIcon } from '../components/icons'
import { BodyPanel } from '../components/BodyPanel'
import { SessionSheet } from './SessionSheet'

const fmtDate = (iso: string) => {
  const [y, m, d] = iso.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString(locale(), { weekday: 'short', day: 'numeric', month: 'short' })
}

type Side = 'player' | 'matches'

// Record: your player (card, week, training log) and your matches (reviews, ratings, results streaks).
export function PlayerTab({ wide = false, onTestDay, onMatch }: { wide?: boolean; onTestDay?: () => void; onMatch?: (date: string) => void }) {
  const [side, setSideState] = useState<Side>(() => load<Side>('performance:side', 'player'))
  const setSide = (v: Side) => { setSideState(v); save('performance:side', v) }
  return (
    <main className={wide ? 'px-10 pt-10 pb-12' : 'mx-auto max-w-xl px-4 pt-[max(20px,var(--sat))] pb-32'}>
      <Title>{t('Record')}</Title>
      <div role="tablist" aria-label={t('Record')} className="mt-3 grid grid-cols-2 rounded-full bg-surface p-1">
        {([['player', t('Player')], ['matches', t('Matches')]] as const).map(([id, label]) => (
          <button key={id} role="tab" aria-selected={side === id} onClick={() => setSide(id)}
            className={`min-h-10 rounded-full font-display text-sm uppercase transition-colors duration-150 ${side === id ? 'bg-red text-on-red' : 'text-muted'}`}>{label}</button>
        ))}
      </div>
      {side === 'player' ? <PlayerSide wide={wide} onTestDay={onTestDay} /> : <MatchesSide onMatch={onMatch} />}
    </main>
  )
}

function MatchesSide({ onMatch }: { onMatch?: (date: string) => void }) {
  const matches = loadMatches().slice().reverse()
  const streaks = resultStreaks(matches)
  const avg = matches.length ? matches.reduce((a, m) => a + m.rating, 0) / matches.length : 0
  const tally = (r: string) => matches.filter((m) => resultOf(m) === r).length
  if (!matches.length) {
    return <Panel className="mt-6"><p className="m-0 font-semibold">{t('No matches yet.')}</p><p className="m-0 mt-1 text-sm text-muted">{t('On a game day, Home asks when you play. After full time you review the match here, and the app rates it.')}</p></Panel>
  }
  return (
    <>
      <div className="mt-5 grid grid-cols-3 gap-2">
        <div className="rounded-[16px] bg-surface p-3">
          <p className="m-0 flex items-center gap-1 font-display text-2xl tabular-nums"><FlameIcon lit={streaks.wins > 0} className="h-5 w-5" />{streaks.wins}</p>
          <p className="m-0 text-xs text-muted">{t('Winning streak')}</p>
        </div>
        <div className="rounded-[16px] bg-surface p-3">
          <p className="m-0 font-display text-2xl tabular-nums">{streaks.unbeaten}</p>
          <p className="m-0 text-xs text-muted">{t('Unbeaten')}</p>
        </div>
        <div className="rounded-[16px] bg-surface p-3">
          <p className="m-0 font-display text-2xl tabular-nums">{avg.toFixed(1)}</p>
          <p className="m-0 text-xs text-muted">{t('Average rating')}</p>
        </div>
      </div>
      <p className="mt-2 text-sm text-muted">{matches.length === 1 ? t('1 game · {w} W · {d} D · {l} L', { w: tally('W'), d: tally('D'), l: tally('L') }) : t('{n} games · {w} W · {d} D · {l} L', { n: matches.length, w: tally('W'), d: tally('D'), l: tally('L') })}</p>
      <SectionHead title={t('Matches')} />
      <ul className="m-0 flex list-none flex-col gap-2 p-0">
        {matches.map((m: MatchReview) => {
          const r = resultOf(m)
          return (
            <li key={m.date}>
              <button onClick={() => onMatch?.(m.date)} className="flex w-full items-center gap-3 rounded-[16px] bg-surface py-3 pr-3 pl-4 text-left active:scale-[0.99]">
                <span className={`grid h-8 w-8 shrink-0 place-items-center rounded-full font-display text-sm ${r === 'W' ? 'bg-[#12a150] text-white' : r === 'L' ? 'bg-[#e2452b] text-white' : 'bg-surface-2'}`}>
                  {r === 'W' ? t('W|win') : r === 'L' ? t('L|loss') : t('D|draw')}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block font-semibold tabular-nums">{m.us}–{m.them}</span>
                  <span className="text-sm text-muted">{fmtDate(m.date)}{m.goals || m.assists ? ` · ${t('{g} G, {a} A', { g: m.goals, a: m.assists })}` : ''}</span>
                </span>
                <RatingBadge value={m.rating} size="sm" />
              </button>
            </li>
          )
        })}
      </ul>
    </>
  )
}

function PlayerSide({ wide, onTestDay }: { wide: boolean; onTestDay?: () => void }) {
  const bg = usePersonalImage('card.jpg')
  const me = useProfile()
  const stats = cardStatsFor(me)
  const due = nextRatingDue(me.ratings)
  const todayIso = new Date().toLocaleDateString('en-CA')
  const [, setVersion] = useState(0)
  const [cropping, setCropping] = useState<Blob | null>(null)
  const [openEntry, setOpenEntry] = useState<HistoryEntry | null>(null)
  const all = loadHistory()
  const history = all.slice().reverse()
  const remove = (h: HistoryEntry) => {
    if (!confirm(t('Delete "{title}" ({date}, {n} min) from your log?', { title: t(h.title), date: fmtDate(h.date), n: h.minutes }))) return
    deleteHistory(h)
    setVersion((v) => v + 1)
  }
  return (
    <>
      <div className={wide ? 'grid grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] items-start gap-10' : ''}>
      <div>
      <div className="mx-auto mt-4 w-full max-w-[300px]">
        <FlipCard p={me} stats={stats} photo={bg} />
      </div>
      <p className="mt-4 text-center text-sm text-muted">
        {me.sample ? t('Example ratings.') : me.ratings ? t('Rated {date}. Tap the card for the results.', { date: new Date(me.ratings.date + 'T12:00').toLocaleDateString(locale(), { day: 'numeric', month: 'short' }) }) : t('Not rated yet.')}
        {' · '}
        <label className="cursor-pointer font-semibold text-ink underline decoration-red decoration-2 underline-offset-4">
          {t('Change photo')}
          <input type="file" accept="image/*" className="sr-only" aria-label={t('Change your card photo')}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) setCropping(f); e.target.value = '' }} />
        </label>
      </p>
      {cropping && <CropSheet file={cropping} title={t('Your card photo')} onCancel={() => setCropping(null)} onDone={(b) => { setCropping(null); void setPhoto('card', b) }} />}
      {onTestDay && <Button block className="mt-4" onClick={onTestDay}>{!me.ratings ? t('Start rating session') : due && due <= todayIso ? t('Rating session due: start') : t('New rating session')}</Button>}
      <SectionHead title={t('Body')} /><BodyPanel />

      </div>
      <div>
      <SectionHead title={t('This week')} />
      <WeekChart history={all} />
      <SectionHead title={t('Training log')} />
      {history.length === 0
        ? <Panel><p className="m-0 text-muted">{t('No sessions yet.')}</p></Panel>
        : <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {history.slice(0, 20).map((h, i) => (
              <li key={`${h.date}-${h.id}-${i}`} className={`flex items-center justify-between gap-1 rounded-[16px] bg-surface pr-2 ${!counts(h) ? 'opacity-60' : ''}`}>
                <button onClick={() => setOpenEntry(h)} className="flex min-w-0 flex-1 items-center gap-3 py-3 pl-4 text-left active:opacity-70">
                <span className="h-8 w-1 shrink-0 rounded-full" style={{ background: KINDS.find((k) => k.id === kindOf(h))!.color }} aria-hidden="true" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-semibold">{t(h.title)}</span>
                  <span className="text-sm text-muted">{fmtDate(h.date)} · {t('{n} min', { n: h.minutes })}{h.rpe ? ` · ${t('effort {n}', { n: h.effort ?? h.rpe })}` : ''}{h.lifts?.length ? ` · ${h.lifts.length === 1 ? t('1 lift') : t('{n} lifts', { n: h.lifts.length })}` : ''}{h.sore?.length ? ` · ${t('sore: {list}', { list: h.sore.map((x) => t(x)).join(', ').toLowerCase() })}` : ''}</span>
                  {!counts(h) && <span className="mt-1 inline-block rounded-full bg-surface-2 px-2 py-0.5 text-xs font-semibold text-muted">{t("Under {n} min: doesn't count", { n: minFor(h.type) })}</span>}
                </span>
                {h.load ? <span className="shrink-0 text-right"><span className="block font-display text-lg leading-none">{h.load}</span><span className="text-xs text-muted">{t('load')}</span></span> : null}
                </button>
                <button onClick={() => remove(h)} aria-label={t('Delete {title} on {date}', { title: t(h.title), date: fmtDate(h.date) })}
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-sm text-muted active:bg-surface-2">✕</button>
              </li>
            ))}
          </ul>}
      {openEntry && <SessionSheet entry={openEntry} date={fmtDate(openEntry.date)} onClose={() => setOpenEntry(null)} onSaved={() => { setOpenEntry(null); setVersion((v) => v + 1) }} />}
      </div>
      </div>
    </>
  )
}
