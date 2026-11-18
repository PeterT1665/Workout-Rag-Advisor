import { useState, type ReactNode } from 'react'
import { Button, Panel, Screen, Title, TopBar } from '../components/ui'
import { FlameIcon } from '../components/icons'
import { POSITIONS, useProfile, type Position } from '../lib/profile'
import { BAND_COLOR, loadMatches, matchRating, ratingBand, resultOf, resultStreaks, saveMatch, type MatchReview as Review } from '../lib/match'
import { locale, t } from '../i18n'

const fmtDay = (iso: string) => {
  const [y, m, d] = iso.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString(locale(), { weekday: 'long', day: 'numeric', month: 'short' })
}

// − 2 + : big tap targets, no typing
function Counter({ label, value, onChange, step = 1, max = 99 }: { label: string; value: number; onChange: (v: number) => void; step?: number; max?: number }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1.5">
      <span className="min-w-0 font-semibold">{label}</span>
      <span className="flex shrink-0 items-center gap-1">
        <button type="button" aria-label={t('Less {what}', { what: label })} onClick={() => onChange(Math.max(0, value - step))}
          className="grid h-11 w-11 place-items-center rounded-full bg-surface-2 text-xl active:scale-95">−</button>
        <span className="w-10 text-center font-display text-xl tabular-nums" aria-live="polite">{value}</span>
        <button type="button" aria-label={t('More {what}', { what: label })} onClick={() => onChange(Math.min(max, value + step))}
          className="grid h-11 w-11 place-items-center rounded-full bg-surface-2 text-xl active:scale-95">+</button>
      </span>
    </div>
  )
}

function Scale({ id, label, low, high, value, onChange }: { id: string; label: string; low: string; high: string; value: number; onChange: (v: number) => void }) {
  return (
    <div>
      <label htmlFor={id} className="flex justify-between font-semibold"><span>{label}</span><span className="font-display text-xl tabular-nums">{value}</span></label>
      <input id={id} type="range" min={1} max={10} step={1} value={value} onChange={(e) => onChange(Number(e.target.value))} className="mt-1 w-full accent-[var(--red)]" />
      <div className="flex justify-between text-xs text-muted"><span>{low}</span><span>{high}</span></div>
    </div>
  )
}

const Section = ({ title, children }: { title: string; children: ReactNode }) => (
  <>
    <h2 className="mt-7 mb-2 font-display text-base uppercase">{title}</h2>
    <Panel className="flex flex-col gap-2">{children}</Panel>
  </>
)

// After a game: a quick journal. Saving shows the app's match rating.
export function MatchReview({ date, onDone }: { date: string; onDone: () => void }) {
  const me = useProfile()
  const old = loadMatches().find((m) => m.date === date)
  const [m, setM] = useState<Omit<Review, 'rating' | 'savedAt'>>(() => old ?? {
    date, us: 0, them: 0, minutes: 90, position: me.positions[0] ?? 'midfielder',
    goals: 0, assists: 0, dribbles: 0, keyPasses: 0, duels: 0, firstTouch: 6, tiring: 6, self: 6, moments: '',
  })
  const [saved, setSaved] = useState<Review | null>(null)
  const set = (p: Partial<Review>) => setM((x) => ({ ...x, ...p }))

  if (saved) return <MatchReport m={saved} onDone={onDone} fresh />

  const save = () => {
    const r: Review = { ...m, rating: matchRating(m), savedAt: Date.now() }
    saveMatch(r)
    setSaved(r)
  }
  const res = resultOf(m)
  return (
    <Screen className="pb-32">
      <TopBar left={<Title>{t('Match review')}</Title>} right={<Button variant="ghost" size="sm" onClick={onDone}>{t('Later')}</Button>} />
      <p className="m-0 text-sm text-muted">{fmtDay(date)}</p>

      <Section title={t('Result')}>
        <Counter label={t('Us|score')} value={m.us} onChange={(v) => set({ us: v })} />
        <Counter label={t('Them|score')} value={m.them} onChange={(v) => set({ them: v })} />
        <p className="m-0 text-sm text-muted">{res === 'W' ? t('A win.') : res === 'L' ? t('A loss.') : t('A draw.')}</p>
      </Section>

      <Section title={t('Your game')}>
        <Counter label={t('Minutes played')} value={m.minutes} step={5} max={120} onChange={(v) => set({ minutes: v })} />
        <div className="flex flex-wrap gap-2 py-1.5" role="radiogroup" aria-label={t('Position played')}>
          {POSITIONS.map((p) => (
            <button key={p.id} type="button" role="radio" aria-checked={m.position === p.id} onClick={() => set({ position: p.id as Position })}
              className={`min-h-10 rounded-full px-4 text-sm font-semibold ${m.position === p.id ? 'bg-red text-on-red' : 'bg-surface-2'}`}>{p.label}</button>
          ))}
        </div>
      </Section>

      <Section title={t('Your numbers')}>
        <Counter label={t('Goals')} value={m.goals} onChange={(v) => set({ goals: v })} />
        <Counter label={t('Assists')} value={m.assists} onChange={(v) => set({ assists: v })} />
        <Counter label={t('Dribbles completed')} value={m.dribbles} onChange={(v) => set({ dribbles: v })} />
        <Counter label={t('Key passes')} value={m.keyPasses} onChange={(v) => set({ keyPasses: v })} />
        <Counter label={t('1v1s won')} value={m.duels} onChange={(v) => set({ duels: v })} />
      </Section>

      <Section title={t('How it felt')}>
        <Scale id="m-touch" label={t('First touch')} low={t('Heavy')} high={t('Perfect')} value={m.firstTouch} onChange={(v) => set({ firstTouch: v })} />
        <Scale id="m-tiring" label={t('How tiring')} low={t('Easy')} high={t('Empty')} value={m.tiring} onChange={(v) => set({ tiring: v })} />
        <Scale id="m-self" label={t('Your rating')} low={t('Off day')} high={t('Best game')} value={m.self} onChange={(v) => set({ self: v })} />
      </Section>

      <h2 className="mt-7 mb-2 font-display text-base uppercase">{t('Key moments')}</h2>
      <textarea value={m.moments} onChange={(e) => set({ moments: e.target.value.slice(0, 1000) })} rows={4}
        placeholder={t('The goal, the chance you missed, what to work on…')} aria-label={t('Key moments')}
        className="w-full rounded-[20px] bg-surface p-4 text-base text-ink placeholder:text-muted" />

      <div className="fixed inset-x-0 bottom-0 z-30 bg-gradient-to-t from-bg via-bg to-transparent px-4 pt-6 pb-[max(16px,env(safe-area-inset-bottom))]">
        <Button block className="mx-auto max-w-xl" onClick={save}>{old ? t('Save changes') : t('Save and rate my match')}</Button>
      </div>
    </Screen>
  )
}

export function RatingBadge({ value, size = 'lg' }: { value: number; size?: 'lg' | 'sm' }) {
  return (
    <span className={`inline-grid place-items-center rounded-[14px] font-display tabular-nums text-white ${size === 'lg' ? 'h-24 w-28 text-5xl' : 'h-9 w-12 text-lg rounded-[10px]'}`}
      style={{ background: BAND_COLOR[ratingBand(value)] }}>{value.toFixed(1)}</span>
  )
}

// The match report: the app's rating, the result, your numbers and notes. Shown after saving and from Performance.
export function MatchReport({ m, onDone, onEdit, fresh = false }: { m: Review; onDone: () => void; onEdit?: () => void; fresh?: boolean }) {
  const res = resultOf(m)
  const streaks = resultStreaks(loadMatches())
  const stats: [string, number][] = [[t('Goals'), m.goals], [t('Assists'), m.assists], [t('Dribbles'), m.dribbles], [t('Key passes'), m.keyPasses], [t('1v1s won'), m.duels], [t('Minutes'), m.minutes]]
  return (
    <Screen className="pb-10">
      <TopBar left={<Title>{t('Match report')}</Title>} right={<Button variant="ghost" size="sm" onClick={onDone}>{t('Done')}</Button>} />
      <p className="m-0 text-sm text-muted">{fmtDay(m.date)} · {POSITIONS.find((p) => p.id === m.position)?.label}</p>
      <section className={`mt-5 flex items-center gap-5 rounded-[22px] bg-surface p-5 ${fresh ? 'rise' : ''}`}>
        <RatingBadge value={m.rating} />
        <div className="min-w-0">
          <p className="m-0 font-display text-sm uppercase text-muted">{t('Match rating')}</p>
          <p className="m-0 mt-1 font-display text-3xl uppercase tabular-nums">
            {res === 'W' ? t('Win') : res === 'L' ? t('Loss') : t('Draw')} {m.us}–{m.them}
          </p>
          <p className="m-0 mt-1 text-sm text-muted">{t('Your rating: {n}/10', { n: m.self })}</p>
        </div>
      </section>
      {(streaks.wins > 1 || streaks.unbeaten > 1) && (
        <div className="mt-3 flex flex-wrap gap-2">
          {streaks.wins > 1 && <span className="flex items-center gap-1.5 rounded-full bg-surface px-3 py-1.5 text-sm font-semibold"><FlameIcon className="h-4 w-4" />{t('{n} wins in a row', { n: streaks.wins })}</span>}
          {streaks.unbeaten > streaks.wins && <span className="rounded-full bg-surface px-3 py-1.5 text-sm font-semibold">{t('Unbeaten in {n}', { n: streaks.unbeaten })}</span>}
        </div>
      )}
      <dl className="m-0 mt-4 grid grid-cols-3 gap-2">
        {stats.map(([k, v]) => (
          <div key={k} className="rounded-[16px] bg-surface p-3">
            <dd className="m-0 font-display text-2xl tabular-nums">{v}</dd>
            <dt className="text-xs text-muted">{k}</dt>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-sm text-muted">{t('First touch {a}/10 · How tiring {b}/10', { a: m.firstTouch, b: m.tiring })}</p>
      {m.moments && <Panel className="mt-3"><p className="m-0 whitespace-pre-wrap">{m.moments}</p></Panel>}
      <p className="mt-4 text-xs text-muted">{t('The rating starts at 6.0 and goes up with what you did for your position. Short games count more per minute. Your last 5 games also nudge your card.')}</p>
      {onEdit && <Button variant="ghost" block className="mt-4 bg-surface-2" onClick={onEdit}>{t('Edit review')}</Button>}
    </Screen>
  )
}
