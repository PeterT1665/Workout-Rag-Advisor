import { useState } from 'react'
import { Sheet } from './Sheet'
import { Button } from './ui'
import { MiniCard } from './MiniCard'
import { CARD_STATS, OVERALL_WEIGHTS, overall, positionGroup } from './PlayerCard'
import { setProfile, useProfile } from '../lib/profile'
import { cardStatsFor } from '../lib/cardStats'
import { GOAL_TIERS, goalOf, planToGoal, tierOf } from '../lib/ratingGoal'
import { STAT_NAMES, TESTS, resultForRating, resultText, type Stat } from '../lib/testday'
import { useSheet } from '../lib/useSheet'
import { t } from '../i18n'

// "How rating works": short words, the six stats, the gold / silver / bronze scale; a red button for what it takes to reach your
// goal, a small one for the full table (the same anchors the scoring uses, so this can't drift from the app).
export function RatingGuide({ onClose }: { onClose: () => void }) {
  const me = useProfile()
  const [view, setView] = useState<'guide' | 'table'>('guide')
  const [meaning, setMeaning] = useState(false)
  const stats = cardStatsFor(me)
  const ovr = stats ? overall(stats, me.positions) : null
  const names = STAT_NAMES()
  const group = positionGroup(me.positions)
  const counts = group === 'striker' ? t('shooting counts most') : group === 'winger' ? t('pace and dribbling count most') : group === 'midfielder' ? t('passing counts most') : group === 'defender' ? t('defending and physical count most') : null

  if (view === 'table') return <RatingTable onBack={() => setView('guide')} onClose={onClose} />
  return (
    <>
      <Sheet label={t('How your rating works')} onClose={onClose}>
        <h2 className="m-0 font-display text-xl uppercase">{t('How your rating works')}</h2>
        <p className="m-0 mt-1 text-sm text-muted">{t('Six tests, six stats.')}</p>
        <div className="mt-4 grid grid-cols-2 gap-2">
          {TESTS.map((x) => (
            <div key={x.stat} className="rounded-2xl bg-surface-2 px-3 py-3 text-center">
              <span className="block font-display text-sm uppercase">{names[x.stat]}</span>
              <span className="block text-[11px] text-muted">{x.name}</span>
            </div>
          ))}
        </div>
        <p className="m-0 mt-5 text-sm">{t("Same scale as FIFA, set against real pro results. For dribbling, shooting and passing it's our best estimate, and we'll tune it as more players rate.")}</p>

        <div className="mt-5 flex items-center gap-4">
          <MiniCard tier="gold" value={75} label={t('Gold')} className="w-[84px] shrink-0" />
          <div>
            <p className="m-0 font-semibold">{t('75 is an average pro.')}</p>
            <p className="m-0 mt-1 flex gap-2 text-sm text-muted">
              <span className="rounded-full bg-surface-2 px-3 py-1" style={{ color: '#d9dde0' }}>{t('Silver 65–74')}</span>
              <span className="rounded-full bg-surface-2 px-3 py-1" style={{ color: '#d39a63' }}>{t('Bronze below 65')}</span>
            </p>
          </div>
        </div>

        <p className="m-0 mt-5 text-sm">{t('Your overall is your six stats averaged, weighted for your position.')}{counts ? ` ${t('For you, {what}.', { what: counts })}` : ''}</p>
        {ovr !== null && <p className="m-0 mt-2 font-display text-lg uppercase">{t('Your overall: {n}', { n: ovr })}</p>}

        <Button block className="mt-5" onClick={() => setMeaning(true)}>{t('What does my rating mean?')}</Button>
        <p className="mt-3 mb-0 text-center text-sm">
          <button onClick={() => setView('table')} className="font-semibold text-ink underline decoration-red decoration-2 underline-offset-4">{t('Full table')}</button>
        </p>
      </Sheet>
      {meaning && <Meaning onClose={() => setMeaning(false)} />}
    </>
  )
}

// The small window: your card next to your goal card, and the quickest ways to get there.
function Meaning({ onClose }: { onClose: () => void }) {
  useSheet(onClose)
  const me = useProfile()
  const stats = cardStatsFor(me)
  const goal = goalOf(me)
  const now = stats ? overall(stats, me.positions) : null
  const names = STAT_NAMES()
  // a session scored on the old full-goal shooting test can't say what the corner test needs
  const plan = stats ? planToGoal(stats, me.positions, goal, me.ratings && me.ratings.v !== 2 ? ['SHO'] : []) : null
  const setGoal = (g: number) => setProfile({ ...me, goal: Math.max(50, Math.min(95, Math.round(g) || 75)) })
  const shown = plan?.reached ? (plan.next ?? goal) : goal
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-label={t('What does my rating mean?')}>
      <button aria-label={t('Close')} className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="rise relative max-h-[90dvh] w-full max-w-sm overflow-y-auto overscroll-contain rounded-[28px] bg-surface p-5">
        <div className="flex items-center justify-center gap-3">
          {now !== null ? <MiniCard tier={tierOf(now)} value={now} label={t('You')} className="w-[96px] scale-90 opacity-60 blur-[1.5px]" /> : <MiniCard tier="bronze" value="--" label={t('You')} className="w-[96px] scale-90 opacity-60 blur-[1.5px]" />}
          <span aria-hidden="true" className="text-2xl text-muted">→</span>
          <MiniCard tier={tierOf(shown)} value={shown} label={t('Goal')} className="w-[112px]" />
        </div>

        <div className="mt-4 flex items-center justify-center gap-2">
          <span className="text-sm text-muted">{t('Goal')}</span>
          <button aria-label={t('Lower the goal')} onClick={() => setGoal(goal - 1)} className="grid h-9 w-9 place-items-center rounded-full bg-surface-2 text-lg active:scale-95">−</button>
          <input inputMode="numeric" aria-label={t('Goal rating')} value={goal} onChange={(e) => { const v = Number(e.target.value.replace(/\D/g, '')); if (v) setProfile({ ...me, goal: Math.min(99, v) }) }}
            onBlur={() => setGoal(goal)} className="min-h-10 w-16 rounded-2xl bg-surface-2 text-center font-display text-xl text-ink" />
          <button aria-label={t('Raise the goal')} onClick={() => setGoal(goal + 1)} className="grid h-9 w-9 place-items-center rounded-full bg-surface-2 text-lg active:scale-95">+</button>
        </div>
        <div className="mt-2 flex justify-center gap-1.5">
          {GOAL_TIERS.map((g) => <button key={g} onClick={() => setGoal(g)} aria-pressed={goal === g} className={`min-h-9 rounded-full px-3.5 text-xs font-semibold active:scale-95 ${goal === g ? 'bg-ink text-bg' : 'bg-surface-2 text-muted'}`}>{g === 65 ? t('Silver {n}', { n: g }) : g === 75 ? t('Gold {n}', { n: g }) : t('Top {n}', { n: g })}</button>)}
        </div>

        <div className="mt-4 text-sm">
          {!plan || now === null ? <p className="m-0 text-muted">{t('Do a rating session first, then this shows what it takes to reach your goal.')}</p>
            : plan.reached ? (
              <>
                <p className="m-0 font-semibold">{t("You're already there. Aim higher?")}</p>
                {plan.next && <Button block size="sm" className="mt-3" onClick={() => setGoal(plan.next!)}>{t('Aim for {n}', { n: plan.next })}</Button>}
              </>
            ) : (
              <>
                <p className="m-0 font-semibold">{t('You need +{n}. The quickest ways:', { n: plan.goal - plan.now })}</p>
                <ul className="m-0 mt-2 flex list-none flex-col gap-1.5 p-0">
                  {plan.steps.map((s) => (
                    <li key={s.stat} className="rounded-2xl bg-surface-2 px-3 py-2">
                      <strong>{t('{stat} {from} → {to}', { stat: names[s.stat], from: s.from, to: s.to })}</strong>
                      <span className="block text-muted">{TESTS.find((x) => x.stat === s.stat)!.name}: {s.target}</span>
                    </li>
                  ))}
                </ul>
                {plan.far && <p className="m-0 mt-2 text-muted">{t("That's a long way. These take you to about {n}. Redo your rating session as you improve and this updates.", { n: plan.after })}</p>}
                {me.ratings && me.ratings.v !== 2 && <p className="m-0 mt-2 text-muted">{t('Shooting: redo it with the corner test to see what it needs.')}</p>}
              </>
            )}
        </div>
        <Button variant="ghost" block className="mt-4 bg-surface-2" onClick={onClose}>{t('Close')}</Button>
      </div>
    </div>
  )
}

// The full table: the result each rating needs in every test, and how the overall is weighted.
function RatingTable({ onBack, onClose }: { onBack: () => void; onClose: () => void }) {
  const names = STAT_NAMES()
  const rows = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]
  const cell = (stat: Stat, r: number) => { const v = resultForRating(stat, r); return v === null ? '–' : resultText(stat, v).replace(/ clean$| of 10$/, (m) => (m === ' of 10' ? '/10' : '')) }
  const weights: [string, string][] = [['striker', t('Striker')], ['winger', t('Winger')], ['midfielder', t('Midfielder')], ['defender', t('Defender')]]
  return (
    <Sheet label={t('Full table')} onClose={onClose}>
      <h2 className="m-0 font-display text-xl uppercase">{t('Full table')}</h2>
      <p className="m-0 mt-1 text-sm text-muted">{t('The result you need in each test for each rating.')}</p>
      <div className="mt-3">
        <table className="w-full border-collapse text-center text-[11px] tabular-nums">
          <thead>
            <tr className="text-muted"><th className="py-1.5 text-left font-semibold">{t('Rating')}</th>{TESTS.map((x) => <th key={x.stat} className="py-1.5 text-[10px] font-semibold">{names[x.stat]}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r} className={r === 75 ? 'bg-surface-2 font-semibold' : ''}>
                <td className="py-1.5 text-left font-display text-sm">{r}</td>
                {TESTS.map((x) => <td key={x.stat} className="py-1.5">{cell(x.stat, r)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="m-0 mt-2 text-xs text-muted">{t('Seconds for the sprint, weave and shuttle (lower is better). Corner shots out of 10, wall passes in 30 seconds, jump in cm.')}</p>

      <h3 className="m-0 mt-5 mb-2 font-display text-sm uppercase text-muted">{t('How the overall is weighted')}</h3>
      <table className="w-full border-collapse text-center text-xs tabular-nums">
        <thead><tr className="text-muted"><th className="py-1.5 text-left font-semibold" />{CARD_STATS.map((k) => <th key={k} className="py-1.5 font-semibold">{k}</th>)}</tr></thead>
        <tbody>
          {weights.map(([id, label]) => (
            <tr key={id}><td className="py-1.5 text-left font-semibold">{label}</td>{CARD_STATS.map((k) => <td key={k} className="py-1.5">{Math.round(OVERALL_WEIGHTS[id][k] * 100)}%</td>)}</tr>
          ))}
        </tbody>
      </table>
      <Button block className="mt-5" onClick={onBack}>{t('Back')}</Button>
    </Sheet>
  )
}
