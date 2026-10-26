import { useEffect, useState } from 'react'
import { loadHistory, reflectLast, weekLoads, type Legs } from '../lib/storage'
import type { PlayerState } from '../lib/player'
import type { Session } from '../lib/types'
import { AnimateDigits } from '../components/AnimateDigits'

const SORE = ['Hips', 'Knee', 'Hamstring', 'Quad', 'Ankle', 'Other']
const EFFORT_WORDS = ['', 'Very easy', 'Easy', 'Easy', 'Moderate', 'Moderate', 'Hard', 'Hard', 'Very hard', 'Very hard', 'Max']

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

function Choice({ on, onClick, children, className = '' }: { on: boolean; onClick: () => void; children: React.ReactNode; className?: string }) {
  return (
    <button onClick={onClick} aria-pressed={on}
      className={`min-h-11 rounded-full font-semibold transition-[transform,background-color,color] duration-150 ease-snap active:scale-[0.96] ${on ? 'bg-white text-black' : 'bg-white/10 text-white'} ${className}`}>
      {children}
    </button>
  )
}

export function FullTime({ session, st, onExit }: { session: Session; st: PlayerState; onExit: () => void }) {
  const minutes = Math.max(1, Math.round((Date.now() - st.startedAt) / 60000))
  const [rpe, setRpe] = useState<number | null>(null)
  const [legs, setLegs] = useState<Legs | null>(null)
  const [sore, setSore] = useState<string[]>([])
  const [saved, setSaved] = useState<{ load: number; week: number } | null>(null)
  const shown = useCountUp(saved?.load ?? 0, !!saved)

  const toggleSore = (s: string) => setSore((x) => (x.includes(s) ? x.filter((y) => y !== s) : [...x, s]))
  const save = () => {
    if (!rpe || !legs) return
    const e = reflectLast({ rpe, legs, sore })
    const week = weekLoads(loadHistory()).reduce((a, b) => a + b, 0)
    setSaved({ load: e?.load ?? minutes * rpe, week })
  }

  return (
    <div className="fixed inset-0 overflow-y-auto bg-[#0a0c0b] text-white">
      <div className="mx-auto flex min-h-full max-w-xl flex-col px-5 pt-[max(28px,env(safe-area-inset-top))] pb-[max(28px,env(safe-area-inset-bottom))]">
        <h1 className="slam m-0 font-display text-[clamp(3.5rem,19vw,6rem)] uppercase leading-[.85]">
          {st.siuu ? <>Siuuu<span className="text-[var(--red-ink)]">!</span></> : <>Full<br /><span className="text-[var(--red-ink)]">time</span></>}
        </h1>
        <p className="m-0 mt-3 text-white/70">{session.title}</p>

        <dl className="m-0 mt-6 grid grid-cols-3 divide-x divide-white/10 rounded-[22px] bg-white/[.06] py-4">
          {[['Minutes', minutes], ['Drills', st.drillsDone], ['Cues called', st.cuesCalled]].map(([k, v]) => (
            <div key={k as string} className="px-4">
              <dd className="m-0 font-display text-3xl">{v}</dd>
              <dt className="mt-1 text-sm text-white/60">{k}</dt>
            </div>
          ))}
        </dl>

        {!saved ? (
          <section className="mt-8 flex flex-1 flex-col">
            <h2 className="m-0 font-display text-xl uppercase">20-second check-in</h2>
            <p className="m-0 mt-1 text-sm text-white/60">Tap, don't type. It shapes next week's plan.</p>

            <h3 className="m-0 mt-6 mb-2 font-semibold">How hard was it?{rpe ? <span className="ml-2 font-normal text-white/60">{EFFORT_WORDS[rpe]}</span> : null}</h3>
            <div className="grid grid-cols-5 gap-2">
              {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
                <button key={n} onClick={() => setRpe(n)} aria-pressed={rpe === n} aria-label={`Effort ${n}`}
                  className={`h-12 rounded-2xl font-display text-lg tabular-nums transition-[transform,background-color] duration-150 ease-snap active:scale-[0.94] ${rpe === n ? 'bg-[var(--red)] text-white' : 'bg-white/10'}`}>
                  {n}
                </button>
              ))}
            </div>

            <h3 className="m-0 mt-6 mb-2 font-semibold">Legs?</h3>
            <div className="grid grid-cols-3 gap-2">
              {(['fresh', 'ok', 'heavy'] as Legs[]).map((l) => (
                <Choice key={l} on={legs === l} onClick={() => setLegs(l)} className="capitalize">{l === 'ok' ? 'OK' : l}</Choice>
              ))}
            </div>

            <h3 className="m-0 mt-6 mb-2 font-semibold">Anything sore?</h3>
            <div className="flex flex-wrap gap-2">
              {SORE.map((s) => <Choice key={s} on={sore.includes(s)} onClick={() => toggleSore(s)} className="px-5">{s}</Choice>)}
            </div>

            <div className="mt-auto flex flex-col gap-2 pt-8">
              <button onClick={save} disabled={!rpe || !legs}
                className="min-h-14 rounded-full bg-[var(--red)] font-display text-sm uppercase tracking-wide text-white transition-[transform,opacity] duration-150 ease-snap active:scale-[0.97] disabled:opacity-40">
                Save check-in
              </button>
              <button onClick={onExit} className="min-h-11 text-sm font-semibold text-white/60">Skip for now</button>
            </div>
          </section>
        ) : (
          <section className="rise mt-10 flex flex-1 flex-col">
            <p className="m-0 font-semibold text-white/70">Session load</p>
            <p className="m-0 font-display text-[clamp(4rem,24vw,7rem)] leading-none"><AnimateDigits value={String(shown)} /></p>
            <p className="m-0 mt-2 text-white/70">{minutes} min × effort {rpe}. This week so far: <b className="text-white">{saved.week}</b>.</p>
            {sore.length > 0 && (
              <p className="m-0 mt-4 rounded-[18px] bg-white/[.06] p-4 text-sm">
                Noted: {sore.join(', ').toLowerCase()}. Your plan keeps an eye on it{sore.includes('Hips') || sore.includes('Knee') ? ', and and recovery days stay in.' : '.'}
              </p>
            )}
            <button onClick={onExit}
              className="mt-auto min-h-14 rounded-full bg-[var(--red)] font-display text-sm uppercase tracking-wide text-white transition-transform duration-150 ease-snap active:scale-[0.97]">
              Back home
            </button>
          </section>
        )}
      </div>
    </div>
  )
}
