import { useEffect, useState } from 'react'
import { keepLiftsOnLast, loadHistory, reflectLast, weekLoads, type Legs, type Tally } from '../lib/storage'
import { setDayPhoto, useDayPhotos } from '../lib/dayPhotos'
import { sessionShare, type ShareItem } from '../lib/shareStats'
import { ShareSheet } from './CalendarView'
import { planStreak } from '../lib/streak'
import { loadGames } from '../lib/today'
import { loadWeekChanges } from '../lib/weekChanges'
import { DEFAULT_GYM } from '../lib/weekplan'
import { ACCENTS, getProfile } from '../lib/profile'
import { latestBody } from '../lib/body'
import { clearLiveLifts, liveLifts, withLive } from '../lib/liveLifts'
import { bestsFrom, planLifts, scoreLifts, type LiftLog, type LiftScore } from '../lib/lifts'
import { bodyWeight } from '../lib/body'
import { LiftsEditor } from '../components/LiftsEditor'
import { EFFORT_WORDS, LiftSummary } from '../components/LiftSummary'
import { trainedMinutes, type PlayerState } from '../lib/player'
import type { Session } from '../lib/types'
import { AnimateDigits } from '../components/AnimateDigits'
import { minFor } from '../lib/streak'
import { mark, t } from '../i18n'
import { tr } from '../i18n/rich'

const SORE = [mark('Hips'), mark('Knee'), mark('Hamstring'), mark('Quad'), mark('Ankle'), mark('Other')]

// Count up once, on the rare full-time moment (delight tier): a few scoreboard steps the digits roll through.
// Lands on the real number immediately when motion is reduced.
function useCountUp(target: number, run: boolean) {
  const [v, setV] = useState(0)
  useEffect(() => {
    if (!run) return
    if (matchMedia?.('(prefers-reduced-motion: reduce)').matches) { setV(target); return }
    const steps = [0.35, 0.6, 0.8, 0.92, 1]
    const timers = steps.map((f, i) => setTimeout(() => setV(Math.round(target * f)), 120 * (i + 1)))
    return () => timers.forEach(clearTimeout)
  }, [target, run])
  return v
}

// Shots, on target, goals: three − / + counters, kept in order (goals ≤ on target ≤ shots).
function TallyBox({ tally, onChange }: { tally: Required<Tally>; onChange: (t: Required<Tally>) => void }) {
  const set = (k: keyof Tally, d: number) => {
    const n = { ...tally, [k]: Math.max(0, Math.min(200, tally[k] + d)) }
    if (k === 'shots') { n.onTarget = Math.min(n.onTarget, n.shots); n.goals = Math.min(n.goals, n.onTarget) }
    if (k === 'onTarget') { n.shots = Math.max(n.shots, n.onTarget); n.goals = Math.min(n.goals, n.onTarget) }
    if (k === 'goals') { n.onTarget = Math.max(n.onTarget, n.goals); n.shots = Math.max(n.shots, n.onTarget) }
    onChange(n)
  }
  const rows: [keyof Tally, string][] = [['shots', t('Shots')], ['onTarget', t('On target')], ['goals', t('Goals')]]
  return (
    <div className="grid grid-cols-3 gap-2">
      {rows.map(([k, label]) => (
        <div key={k} className="rounded-[18px] bg-white/[.06] p-2.5 text-center">
          <p className="m-0 text-xs text-white/60">{label}</p>
          <p className="m-0 font-display text-3xl tabular-nums" aria-live="polite">{tally[k]}</p>
          <div className="mt-1 grid grid-cols-2 gap-1">
            <button onClick={() => set(k, -1)} aria-label={t('{name}: one less', { name: label })} className="h-10 rounded-xl bg-white/10 text-lg active:scale-95">−</button>
            <button onClick={() => set(k, 1)} aria-label={t('{name}: one more', { name: label })} className="h-10 rounded-xl bg-white/10 text-lg active:scale-95">+</button>
          </div>
        </div>
      ))}
    </div>
  )
}

function Choice({ on, onClick, children, className = '' }: { on: boolean; onClick: () => void; children: React.ReactNode; className?: string }) {
  return (
    <button onClick={onClick} aria-pressed={on}
      className={`min-h-11 rounded-full font-semibold transition-[transform,background-color,color] duration-150 ease-snap active:scale-[0.96] ${on ? 'bg-white text-black' : 'bg-white/10 text-white'} ${className}`}>
      {children}
    </button>
  )
}

export function FullTime({ session, st, onExit }: { session: Session; st: PlayerState; onExit: () => void }) {
  const minutes = trainedMinutes(st) // as the log saves it: pauses and skipped drills don't count
  const [rpe, setRpe] = useState<number | null>(null)
  const [legs, setLegs] = useState<Legs | null>(null)
  const [sore, setSore] = useState<string[]>([])
  // gym sets: last time's weights and reps (or the plan's reps) filled in, the player fixes what changed
  const [lifts, setLifts] = useState<LiftLog[]>(() => withLive(planLifts(session, loadHistory()), liveLifts(st.startedAt)))
  const loggedLive = liveLifts(st.startedAt).length > 0
  const [saved, setSaved] = useState<{ load: number; week: number; effort: number; measured: number | null; lifts: LiftScore[] } | null>(null)
  const shown = useCountUp(saved?.load ?? 0, !!saved)
  // finishing in the session: count shots, on target and goals (they go on the share card)
  const shooting = session.blocks.some((b) => b.category === 'finishing' || /shoot|finish|volley/i.test(b.name))
  const [tally, setTally] = useState<Required<Tally>>({ shots: 0, onTarget: 0, goals: 0 })
  const day = new Date().toLocaleDateString('en-CA')
  const photo = useDayPhotos()[day]
  const [sharing, setSharing] = useState<ShareItem | null>(null)
  const openShare = () => {
    const all = loadHistory(), last = all[all.length - 1]
    if (last) setSharing(sessionShare(last, all, latestBody(getProfile())?.kg ?? 70))
  }

  const toggleSore = (s: string) => setSore((x) => (x.includes(s) ? x.filter((y) => y !== s) : [...x, s]))
  const save = () => {
    if (!rpe || !legs) return
    const e = reflectLast({ rpe, legs, sore, lifts, bw: bodyWeight(), ...(tally.shots ? { tally } : {}) })
    clearLiveLifts()
    const all = loadHistory()
    const week = weekLoads(all).reduce((a, b) => a + b, 0)
    const scored = scoreLifts(e?.lifts ?? [], rpe, bestsFrom(all.slice(0, -1), e?.bw), e?.bw)
    setSaved({ load: e?.load ?? minutes * rpe, week, effort: e?.effort ?? rpe, ...scored })
  }

  return (
    <div className="fixed inset-0 overflow-y-auto bg-[#0a0c0b] text-white">
      <div className="mx-auto flex min-h-full max-w-xl flex-col px-5 pt-[max(28px,var(--sat))] pb-[max(28px,env(safe-area-inset-bottom))]">
        <h1 className="slam m-0 font-display text-[clamp(3.5rem,19vw,6rem)] uppercase leading-[.85]">
          {tr('Full <r>time</r>', { r: (x) => <><br /><span className="text-[var(--red-ink)]">{x}</span></> })}
        </h1>
        <p className="m-0 mt-3 text-white/70">{session.title}</p>

        <dl className="m-0 mt-6 grid grid-cols-3 divide-x divide-white/10 rounded-[22px] bg-white/[.06] py-4">
          {[[t('Minutes'), minutes], [t('Drills'), st.drillsDone], [t('Cues called'), st.cuesCalled]].map(([k, v]) => (
            <div key={k as string} className="px-4">
              <dd className="m-0 font-display text-3xl">{v}</dd>
              <dt className="mt-1 text-sm text-white/60">{k}</dt>
            </div>
          ))}
        </dl>
        {minutes < minFor(session.type) && <p className="m-0 mt-3 text-sm text-white/60">{t("Under {n} min: it's in your log, but it doesn't count toward your streak.", { n: minFor(session.type) })}</p>}

        {!saved ? (
          <section className="mt-8 flex flex-1 flex-col">
            <h2 className="m-0 font-display text-xl uppercase">{lifts.length ? t('Check-in') : t('20-second check-in')}</h2>
            <p className="m-0 mt-1 text-sm text-white/60">{lifts.length ? (loggedLive ? t('Your sets from the session are in: check them, fill in any you missed.') : t("Your sets first: last time's numbers are filled in, change what's different.")) : t("Tap, don't type.")}</p>

            {lifts.length > 0 && <>
              <h3 className="m-0 mt-6 mb-2 font-semibold">{t('Your sets')}</h3>
              <LiftsEditor lifts={lifts} onChange={setLifts} />
            </>}

            {shooting && <>
              <h3 className="m-0 mt-6 mb-2 font-semibold">{t('Your shooting')}</h3>
              <TallyBox tally={tally} onChange={setTally} />
            </>}

            <h3 className="m-0 mt-6 mb-2 font-semibold">{lifts.length ? t('How hard was the whole session?') : t('How hard was it?')}{rpe ? <span className="ml-2 font-normal text-white/60">{EFFORT_WORDS()[rpe]}</span> : null}</h3>
            <div className="grid grid-cols-5 gap-2">
              {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
                <button key={n} onClick={() => setRpe(n)} aria-pressed={rpe === n} aria-label={t('Effort {n}', { n })}
                  className={`h-12 rounded-2xl font-display text-lg tabular-nums transition-[transform,background-color] duration-150 ease-snap active:scale-[0.94] ${rpe === n ? 'bg-[var(--red)] text-white' : 'bg-white/10'}`}>
                  {n}
                </button>
              ))}
            </div>

            <h3 className="m-0 mt-6 mb-2 font-semibold">{t('Legs?')}</h3>
            <div className="grid grid-cols-3 gap-2">
              {(['fresh', 'ok', 'heavy'] as Legs[]).map((l) => (
                <Choice key={l} on={legs === l} onClick={() => setLegs(l)} className="capitalize">{l === 'ok' ? t('OK') : l === 'fresh' ? t('fresh') : t('heavy')}</Choice>
              ))}
            </div>

            <h3 className="m-0 mt-6 mb-2 font-semibold">{t('Anything sore?')}</h3>
            <div className="flex flex-wrap gap-2">
              {SORE.map((s) => <Choice key={s} on={sore.includes(s)} onClick={() => toggleSore(s)} className="px-5">{t(s)}</Choice>)}
            </div>

            <h3 className="m-0 mt-6 mb-2 font-semibold">{t('Photo of today')} <span className="font-normal text-white/50">{t('(optional)')}</span></h3>
            <label className="flex cursor-pointer items-center gap-3 rounded-[18px] bg-white/[.06] p-2.5 active:scale-[0.99]">
              <span className="grid h-16 w-16 shrink-0 place-items-center overflow-hidden rounded-[14px] bg-white/10">
                {photo ? <img src={photo} alt="" className="h-full w-full object-cover" /> : <svg viewBox="0 0 24 24" className="h-6 w-6 text-white/70" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M4 8h3l1.5-2h7L17 8h3v11H4V8Z" /><circle cx="12" cy="13" r="3.5" /></svg>}
              </span>
              <span className="text-sm">
                <span className="block font-semibold">{photo ? t('Change photo') : t('Add a photo of today')}</span>
                <span className="block text-white/55">{t('It goes on your streak calendar and your share picture. It stays on this phone.')}</span>
              </span>
              <input type="file" accept="image/*" className="sr-only" onChange={(e) => { const f = e.target.files?.[0]; if (f) void setDayPhoto(day, f) }} />
            </label>

            <div className="mt-auto flex flex-col gap-2 pt-8">
              <button onClick={save} disabled={!rpe || !legs}
                className="min-h-14 rounded-full bg-[var(--red)] font-display text-sm uppercase tracking-wide text-white transition-[transform,opacity] duration-150 ease-snap active:scale-[0.97] disabled:opacity-40">
                {t('Save check-in')}
              </button>
              <button onClick={() => { if (loggedLive) keepLiftsOnLast(liveLifts(st.startedAt), bodyWeight()); clearLiveLifts(); onExit() }} className="min-h-11 text-sm font-semibold text-white/60">{t('Skip for now')}</button>
            </div>
          </section>
        ) : (
          <section className="rise mt-10 flex flex-1 flex-col">
            <p className="m-0 font-semibold text-white/70">{t('Session load')}</p>
            <p className="m-0 font-display text-[clamp(4rem,24vw,7rem)] leading-none"><AnimateDigits value={String(shown)} /></p>
            <p className="m-0 mt-2 text-white/70">{tr('{minutes} min × effort {rpe}. This week so far: <b>{week}</b>.', {}, { minutes, rpe: saved.effort, week: saved.week })}</p>
            {saved.measured !== null && (
              <>
                <p className="m-0 mt-1 text-sm text-white/60">{t('Effort {e}: you said {rpe}, your sets say {m}.', { e: saved.effort, rpe: rpe ?? 0, m: saved.measured })}</p>
                <h2 className="m-0 mt-6 mb-2 font-display text-lg uppercase">{t('Your sets')}</h2>
                <LiftSummary dark lifts={saved.lifts} />
              </>
            )}
            {sore.length > 0 && (
              <p className="m-0 mt-4 rounded-[18px] bg-white/[.06] p-4 text-sm">
                {t('Noted: {list}. Your plan keeps an eye on it.', { list: sore.map((x) => t(x)).join(', ').toLowerCase() })}
              </p>
            )}
            <div className="mt-auto flex flex-col gap-2 pt-8">
              <button onClick={openShare}
                className="min-h-14 w-full rounded-full bg-white font-display text-sm uppercase tracking-wide text-black transition-transform duration-150 ease-snap active:scale-[0.97]">
                {t('Share')}
              </button>
              <button onClick={onExit}
                className="min-h-14 w-full rounded-full bg-[var(--red)] font-display text-sm uppercase tracking-wide text-white transition-transform duration-150 ease-snap active:scale-[0.97]">
                {t('Back home')}
              </button>
            </div>
          </section>
        )}
      </div>
      {sharing && <ShareSheet item={sharing} photo={photo} accent={ACCENTS.find((a) => a.id === getProfile().accent)?.hex ?? '#e01b35'}
        streak={planStreak(loadHistory(), loadGames(), getProfile().gym ?? DEFAULT_GYM, loadWeekChanges()).days} onClose={() => setSharing(null)} />}
    </div>
  )
}
