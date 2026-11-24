import { useState } from 'react'
import { Sheet } from './Sheet'
import { Button } from './ui'
import { DAY_ABBR, DAYS, FOCUS_LABEL, matchdayLabel, type Game, type WeekPlan } from '../lib/weekplan'
import { t } from '../i18n'

// The rules the planner follows (where each came from: docs/training-references.md).
const RULES = () => [
  { title: t('Every gym day has a job'), body: t('Set by how far it is from the game: legs at MD-4 or earlier, core & injury prevention at MD-3, upper body power at MD-2, a short activation at MD-1.') },
  { title: t('Hard days hard, easy days easy'), body: t('The hardest day (MD-4) puts high-intensity football, legs and speed together. The day after stays light.') },
  { title: t('Fresh legs for the game'), body: t('No heavy legs in the last three days. The day before is a 20-minute activation (isometric holds, a few fast jumps) or rest, never a hard session.') },
  { title: t('Recover first'), body: t('The day after a game is for recovery (mobility, hips, hamstrings). Training starts again at MD+2.') },
  { title: t('Games carry the football load'), body: t('1 game: 2 football sessions. 2 games: 1. 3 games: none, just gym and recovery. Busy weeks drop heavy legs.') },
]
const KIND_LABEL = (): Record<string, string> => ({ game: t('Game'), football: t('Football'), recovery: t('Recovery'), cheer: 'Cheer' }) // i18n-ignore: old cheer mode

export function WeekGuide({ plan, games }: { plan: WeekPlan; games: Game[] }) {
  const [open, setOpen] = useState(false)
  const count = (k: string) => plan.slots.filter((s) => s.kind === k).length
  const summary = [
    count('game') === 1 ? t('1 game') : t('{n} games', { n: count('game') }), t('{n} football', { n: count('football') }),
    t('{n} gym', { n: count('gym') }), t('{n} recovery', { n: count('recovery') }),
  ].join(' · ')
  const warnings = plan.reasons.slice(1)

  return (
    <>
      <button onClick={() => setOpen(true)} aria-haspopup="dialog"
        className="mt-4 flex w-full items-center gap-3 rounded-[16px] bg-surface px-4 py-3 text-left transition-transform duration-150 ease-snap active:scale-[0.98]">
        <span className="min-w-0 flex-1">
          <span className="block font-semibold">{t('How your week works')}</span>
          <span className="block truncate text-sm text-muted">{summary}</span>
        </span>
        {warnings.length > 0 && <span className="h-2 w-2 shrink-0 rounded-full bg-red" aria-label={t('Has notes')} />}
        <span className="shrink-0 text-red-ink" aria-hidden="true">→</span>
      </button>

      {open && (
        <Sheet label={t('How your week works')} onClose={() => setOpen(false)}>
            <h2 className="m-0 font-display text-xl uppercase">{t('How your week works')}</h2>
            <p className="m-0 mt-1 text-sm text-muted">{summary}. {count('game') === 1 ? t('Built around your game, the way pros plan theirs.') : t('Built around your games, the way pros plan theirs.')}</p>

            <h3 className="m-0 mt-5 mb-2 font-display text-sm uppercase text-muted">{t('Your week')}</h3>
            <ol className="m-0 flex list-none flex-col gap-1.5 p-0">
              {DAYS.map((_, day) => {
                const slots = plan.slots.filter((s) => s.day === day)
                const md = matchdayLabel(day, games)
                return (
                  <li key={day} className="flex gap-3 rounded-2xl bg-surface-2 px-3 py-2.5">
                    <span className="w-12 shrink-0 font-display text-xs uppercase text-muted">{DAY_ABBR[day]}{md && <span className={`block text-[10px] ${md === 'MD' ? 'text-red-ink' : ''}`}>{md}</span>}</span>
                    <span className="min-w-0 flex-1 text-sm">
                      {slots.length === 0 && <span className="text-muted">{t('Rest')}</span>}
                      {slots.map((s, i) => (
                        <span key={i} className="block">
                          <strong>{s.kind === 'gym' && s.focus ? t('Gym: {focus}', { focus: FOCUS_LABEL[s.focus] }) : KIND_LABEL()[s.kind] ?? s.kind}</strong>
                          {s.note && <span className="text-muted">: {s.note}</span>}
                        </span>
                      ))}
                    </span>
                  </li>
                )
              })}
            </ol>

            {warnings.length > 0 && (
              <div className="mt-4 rounded-2xl border border-red/50 p-3 text-sm">
                {warnings.map((w, i) => <p key={i} className="m-0">{w}</p>)}
              </div>
            )}

            <h3 className="m-0 mt-6 mb-2 font-display text-sm uppercase text-muted">{t('The rules')}</h3>
            <ul className="m-0 flex list-none flex-col gap-3 p-0">
              {RULES().map((r) => (
                <li key={r.title}>
                  <p className="m-0 font-semibold">{r.title}</p>
                  <p className="m-0 mt-0.5 text-sm text-muted">{r.body}</p>
                </li>
              ))}
            </ul>

            <Button block className="mt-5" onClick={() => setOpen(false)}>{t('Got it')}</Button>
        </Sheet>
      )}
    </>
  )
}
