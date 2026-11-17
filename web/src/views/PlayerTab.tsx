import { Button, Panel, SectionHead, Title } from '../components/ui'
import { PlayerCard } from '../components/PlayerCard'
import { cardStatsFor } from '../lib/cardStats'
import { usePersonalImage } from '../lib/personal'
import { setPhoto } from '../lib/photos'
import { CropSheet } from '../components/CropSheet'
import { useProfile } from '../lib/profile'
import { useState } from 'react'
import { deleteHistory, loadHistory, type HistoryEntry } from '../lib/storage'
import { KINDS, WeekChart, kindOf } from '../components/WeekChart'
import { locale, t } from '../i18n'

const fmtDate = (iso: string) => {
  const [y, m, d] = iso.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString(locale(), { weekday: 'short', day: 'numeric', month: 'short' })
}

export function PlayerTab({ wide = false, onTestDay }: { wide?: boolean; onTestDay?: () => void }) {
  const bg = usePersonalImage('card.jpg')
  const me = useProfile()
  const stats = cardStatsFor(me)
  const [, setVersion] = useState(0)
  const [cropping, setCropping] = useState<Blob | null>(null)
  const all = loadHistory()
  const history = all.slice().reverse()
  const remove = (h: HistoryEntry) => {
    if (!confirm(t('Delete "{title}" ({date}, {n} min) from your log?', { title: t(h.title), date: fmtDate(h.date), n: h.minutes }))) return
    deleteHistory(h)
    setVersion((v) => v + 1)
  }
  return (
    <main className={wide ? 'px-10 pt-10 pb-12' : 'mx-auto max-w-xl px-4 pt-[max(20px,var(--sat))] pb-32'}>
      <Title>{t('Player')}</Title>
      <div className={wide ? 'grid grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] items-start gap-10' : ''}>
      <div>
      <div className="mx-auto mt-4 w-full max-w-[300px]">
        <PlayerCard p={me} stats={stats} photo={bg} />
      </div>
      <p className="mt-4 text-center text-sm text-muted">
        {me.sample ? t('Example ratings.') : me.ratings ? `Rated ${me.ratings.date}` : t('Not rated yet.')}
        {' · '}
        <label className="cursor-pointer font-semibold text-ink underline decoration-red decoration-2 underline-offset-4">
          {t('Change photo')}
          <input type="file" accept="image/*" className="sr-only" aria-label={t('Change your card photo')}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) setCropping(f); e.target.value = '' }} />
        </label>
      </p>
      {cropping && <CropSheet file={cropping} title={t('Your card photo')} onCancel={() => setCropping(null)} onDone={(b) => { setCropping(null); void setPhoto('card', b) }} />}
      {onTestDay && <Button block className="mt-4" onClick={onTestDay}>{me.ratings ? t('New rating session') : t('Start rating session')}</Button>}

      </div>
      <div>
      <SectionHead title={t('This week')} />
      <WeekChart history={all} />
      <SectionHead title={t('Training log')} />
      {history.length === 0
        ? <Panel><p className="m-0 text-muted">{t('No sessions yet.')}</p></Panel>
        : <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {history.slice(0, 20).map((h, i) => (
              <li key={`${h.date}-${h.id}-${i}`} className="flex items-center justify-between gap-3 rounded-[16px] bg-surface py-3 pr-2 pl-4">
                <span className="h-8 w-1 shrink-0 rounded-full" style={{ background: KINDS.find((k) => k.id === kindOf(h))!.color }} aria-hidden="true" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-semibold">{t(h.title)}</span>
                  <span className="text-sm text-muted">{fmtDate(h.date)} · {t('{n} min', { n: h.minutes })}{h.rpe ? ` · ${t('effort {n}', { n: h.rpe })}` : ''}{h.sore?.length ? ` · ${t('sore: {list}', { list: h.sore.map((x) => t(x)).join(', ').toLowerCase() })}` : ''}</span>
                </span>
                {h.load ? <span className="shrink-0 text-right"><span className="block font-display text-lg leading-none">{h.load}</span><span className="text-xs text-muted">{t('load')}</span></span> : null}
                <button onClick={() => remove(h)} aria-label={t('Delete {title} on {date}', { title: t(h.title), date: fmtDate(h.date) })}
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-sm text-muted active:bg-surface-2">✕</button>
              </li>
            ))}
          </ul>}
      </div>
      </div>
    </main>
  )
}
