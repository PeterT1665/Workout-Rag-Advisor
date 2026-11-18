import { useState } from 'react'
import { Button } from './ui'
import { lastTimes, reviewOpensAt, saveTimes, type MatchDay } from '../lib/match'
import { scheduleMatch } from '../lib/push'
import { locale, t } from '../i18n'

const clock = (ms: number) => new Date(ms).toLocaleTimeString(locale(), { hour: 'numeric', minute: '2-digit' })

// Home on a game day: in the morning, when do you play (the review reminder comes 15 min after full time);
// after the game, the review. Like the streak make-up box.
export function MatchDayBox({ day, onReview, onChanged }: { day: MatchDay; onReview: () => void; onChanged: () => void }) {
  const [editing, setEditing] = useState(false)
  const [times, setTimes] = useState(() => (day.state === 'set' ? { start: day.start, end: day.end } : lastTimes()))

  if (day.state === 'review') {
    return (
      <section aria-label={t('Match review')} className="rise mt-6 rounded-[22px] border border-red/50 bg-surface p-4">
        <p className="m-0 font-display text-xs uppercase text-red-ink">{t('Match review')}</p>
        <p className="m-0 mt-1.5 font-semibold">{t('How did the game go? Two minutes while it\'s fresh, and the app rates your match.')}</p>
        <Button block className="mt-3" onClick={onReview}>{t('Review my match')}</Button>
      </section>
    )
  }

  if (day.state === 'set' && !editing) {
    return (
      <section aria-label={t('Game day')} className="rise mt-6 flex items-center gap-3 rounded-[22px] bg-surface p-4">
        <div className="min-w-0 flex-1">
          <p className="m-0 font-display text-xs uppercase text-red-ink">{t('Game day')}</p>
          <p className="m-0 mt-1 font-semibold">{t('{start} to {end}', { start: day.start, end: day.end })}</p>
          <p className="m-0 text-sm text-muted">{t('Your match review opens at {time}.', { time: clock(day.opensAt) })}</p>
        </div>
        <Button variant="ghost" size="sm" className="shrink-0 bg-surface-2" onClick={() => setEditing(true)}>{t('Change')}</Button>
      </section>
    )
  }

  const save = () => {
    if (!times.start || !times.end) return
    saveTimes(day.date, times.start, times.end)
    void scheduleMatch(day.date, reviewOpensAt(day.date, times))
    setEditing(false)
    onChanged()
  }
  return (
    <section aria-label={t('Game day')} className="rise mt-6 rounded-[22px] border border-red/50 bg-surface p-4">
      <p className="m-0 font-display text-xs uppercase text-red-ink">{t('Game day')}</p>
      <p className="m-0 mt-1.5 font-semibold">{t('When do you play? We\'ll ask for your match review just after full time.')}</p>
      <div className="mt-3 grid grid-cols-2 gap-2">
        {([['start', t('Kick-off')], ['end', t('Full time')]] as const).map(([k, label]) => (
          <label key={k} className="flex flex-col gap-1 rounded-2xl bg-surface-2 px-3 py-2">
            <span className="text-xs text-muted">{label}</span>
            <input type="time" value={times[k]} onChange={(e) => setTimes({ ...times, [k]: e.target.value })}
              className="min-h-8 bg-transparent font-display text-xl tabular-nums text-ink" />
          </label>
        ))}
      </div>
      <Button block className="mt-3" onClick={save}>{t('Save times')}</Button>
    </section>
  )
}
