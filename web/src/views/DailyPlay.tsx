import { useMemo, useState } from 'react'
import { Button, Screen, Title, TopBar } from '../components/ui'
import { Pitch, type Mark } from '../components/Pitch'
import { POSITIONS } from '../lib/profile'
import { answerDaily, answerFor, dailyStreak, dayNumber, optionOrder, scenarioFor, shareText, type Answer } from '../lib/daily'
import { t } from '../i18n'

// One scenario a day. Pick an action; then the best one is shown with why. Same scenario for everyone: share your result.
export function DailyPlay({ onBack }: { onBack: () => void }) {
  const now = useMemo(() => new Date(), [])
  const s = scenarioFor(now)
  const order = useMemo(() => optionOrder(dayNumber(now), s.options.length), [now, s.options.length])
  const [done, setDone] = useState<Answer | null>(() => answerFor(now))
  const [note, setNote] = useState('')
  const pick = (i: number) => { if (!done) setDone(answerDaily(i, now)) }
  const marks: Mark[] = done ? [
    { to: s.options[s.best].to, kind: s.options[s.best].kind, tone: 'best' },
    ...(done.picked !== s.best ? [{ to: s.options[done.picked].to, kind: s.options[done.picked].kind, tone: 'wrong' as const }] : []),
  ] : []
  const streak = dailyStreak(now)
  const roles = s.for.map((x) => POSITIONS.find((p) => p.id === x)?.short ?? x).join(' · ')
  const share = async () => {
    const text = shareText(now, done!, streak)
    try { if (navigator.share) await navigator.share({ text }); else { await navigator.clipboard.writeText(text); setNote(t('Copied. Send it to a friend.')) } }
    catch { /* cancelled */ }
  }
  return (
    <Screen>
      <TopBar left={<Title>{t('Daily play')}</Title>} right={<Button variant="ghost" size="sm" onClick={onBack}>{t('Back')}</Button>} />
      <p className="m-0 flex items-center gap-2 text-sm text-muted"><span className="font-display tabular-nums">#{dayNumber(now)}</span>{roles && <span>· {roles}</span>}{streak > 0 && <span>· 🔥 {streak}</span>}</p>
      <h2 className="m-0 mt-2 font-display text-2xl uppercase leading-tight">{t(s.title)}</h2>
      <Pitch s={s} marks={marks} className="mt-3 w-full" />
      <p className="mt-3 text-sm">{t(s.setup)}</p>
      <p className="m-0 mt-2 font-semibold">{t('What do you do?')}</p>
      <ol className="m-0 mt-2 flex list-none flex-col gap-2 p-0">
        {order.map((oi) => {
          const o = s.options[oi]
          const chosen = done?.picked === oi, best = done && oi === s.best
          return (
            <li key={oi}>
              <button disabled={!!done} onClick={() => pick(oi)} aria-pressed={chosen}
                className={`flex min-h-14 w-full flex-col justify-center rounded-2xl px-4 py-2.5 text-left transition-transform duration-150 active:scale-[0.98] ${best ? 'bg-emerald-500/20 ring-2 ring-emerald-400' : chosen ? 'bg-red/20 ring-2 ring-red' : 'bg-surface'} ${done && !best && !chosen ? 'opacity-60' : ''}`}>
                <span className="font-semibold">{best && '✓ '}{t(o.label)}</span>
                {done && (best || chosen) && <span className="mt-1 text-sm text-muted">{t(o.why)}</span>}
              </button>
            </li>
          )
        })}
      </ol>
      {done && (
        <section className="rise mt-4 rounded-[22px] bg-surface p-4">
          <p className="m-0 font-display text-lg uppercase">{done.correct ? t('Spot on') : t('Not this time')}</p>
          <p className="m-0 mt-1 text-sm text-muted">{done.correct ? t('Same scenario for everyone today. Share yours.') : t('The green arrow is the best option. Same scenario for everyone today: share yours.')}</p>
          <Button block className="mt-3" onClick={() => void share()}>{t('Share my result')}</Button>
          {note && <p role="status" className="m-0 mt-2 text-center text-sm text-muted">{note}</p>}
          <p className="m-0 mt-3 text-center text-xs text-muted">{t('A new play every day.')}</p>
        </section>
      )}
    </Screen>
  )
}
