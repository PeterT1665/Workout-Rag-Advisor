import { useState, type ReactNode } from 'react'
import { Button, Screen } from '../components/ui'
import { ACCENTS, POSITIONS, applyAccent, positionTag, type Foot, type Profile } from '../lib/profile'
import { DAYS, type Game } from '../lib/weekplan'
import { usePersonalImage } from '../lib/personal'
import { InstallButton } from '../components/InstallButton'

// Make your player: four quick screens with a live player card on top. Used by guests ("Try it"),
// by new accounts, and by Edit player. `welcome` adds the first screen with the sample-player shortcut.
const STEPS = ['player', 'play', 'week', 'style'] as const

export function Onboarding({ initial, initialGames = [], welcome = false, onDone, onSample, onSignIn, onSignUp, onCancel }: {
  initial?: Profile
  initialGames?: Game[]
  welcome?: boolean
  onDone: (p: Profile, games: Game[]) => void
  onSample?: () => void
  onSignIn?: () => void
  onSignUp?: () => void
  onCancel?: () => void
}) {
  const [p, setP] = useState<Profile>(initial ?? { name: '', number: 10, positions: [], foot: 'right', accent: 'red', idol: '', gym: 3 })
  const [games, setGames] = useState<Game[]>(initialGames)
  const [step, setStep] = useState<number>(welcome ? -1 : 0)
  const set = (patch: Partial<Profile>) => setP((x) => ({ ...x, ...patch }))
  const hero = usePersonalImage('hero.jpg')

  if (step === -1) {
    return (
      <Screen className="relative flex flex-col justify-end pb-8">
        {hero && <>
          <img src={hero} alt="" data-fade className="drift fixed inset-0 z-0 h-full w-full object-cover object-[50%_25%]" />
          <div className="fixed inset-0 z-0 bg-[linear-gradient(to_bottom,rgba(11,11,12,.1)_0%,rgba(11,11,12,.55)_30%,rgba(11,11,12,.85)_42%,var(--bg)_60%)]" />
        </>}
        <InstallButton className="absolute top-[max(16px,var(--sat))] right-4 z-20" />
        <div className="stagger relative z-10">
          <p style={{ '--i': 0 } as React.CSSProperties} className="m-0 font-display text-sm uppercase text-red-ink">Football Personal Coach</p>
          <h1 style={{ '--i': 1 } as React.CSSProperties} className="m-0 mt-2 font-display text-[clamp(2.6rem,12vw,3.8rem)] uppercase leading-[.88]">Train like<br />a pro.</h1>
          <ul style={{ '--i': 2 } as React.CSSProperties} className="m-0 mt-6 mb-8 flex list-none flex-col gap-3 p-0">
            {[
              ['🗓', 'A week planned around your games'],
              ['🔊', 'Drills coached out loud, LEFT / RIGHT cues'],
              ['🏋️', 'Gym sessions for speed and power'],
            ].map(([icon, text]) => (
              <li key={text} className="flex items-center gap-3">
                <span aria-hidden="true" className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white/10 text-lg backdrop-blur-md">{icon}</span>
                <span className="font-semibold">{text}</span>
              </li>
            ))}
          </ul>
          <div style={{ '--i': 3 } as React.CSSProperties}>
            <Button block onClick={() => setStep(0)}>Make my player</Button>
            <Button block variant="ghost" className="mt-3" onClick={onSample}>Try a sample player</Button>
            <p className="m-0 mt-4 text-center text-xs text-muted">30 seconds. No account needed.</p>
            {onSignIn && (
              <div className="mt-1 flex justify-center gap-5 text-sm font-semibold">
                {onSignUp && <button onClick={onSignUp} className="min-h-11 text-red-ink">Create account</button>}
                <button onClick={onSignIn} className="min-h-11">Sign in</button>
              </div>
            )}
          </div>
        </div>
      </Screen>
    )
  }

  const id = STEPS[step]
  const valid = id === 'player' ? p.name.trim().length > 0 : id === 'play' ? p.positions.length > 0 : true
  const next = () => (step === STEPS.length - 1 ? onDone({ ...p, name: p.name.trim(), sample: false }, games) : setStep(step + 1))
  const back = () => (step === 0 ? (welcome ? setStep(-1) : onCancel?.()) : setStep(step - 1))

  return (
    <Screen className="flex flex-col">
      <div className="mb-5 flex items-center gap-3">
        {(step > 0 || welcome || onCancel) && <Button variant="ghost" size="sm" onClick={back}>Back</Button>}
        <div className="flex flex-1 gap-1" aria-label={`Step ${step + 1} of ${STEPS.length}`}>
          {STEPS.map((_, i) => <div key={i} className={`h-1.5 flex-1 rounded-full transition-colors duration-300 ${i <= step ? 'bg-red' : 'bg-surface-2'}`} />)}
        </div>
      </div>

      <CardPreview p={p} />

      <div key={id} className="rise mt-6 flex-1">
        {id === 'player' && (
          <Step title="Your player">
            <div className="flex gap-3">
              <div className="min-w-0 flex-1">
                <label htmlFor="ob-name" className="text-sm text-muted">Name</label>
                <input id="ob-name" value={p.name} onChange={(e) => set({ name: e.target.value.slice(0, 20) })} autoComplete="given-name" autoFocus
                  className="mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 font-display text-xl uppercase text-ink" />
              </div>
              <div className="w-24">
                <label htmlFor="ob-number" className="text-sm text-muted">Number</label>
                <input id="ob-number" inputMode="numeric" value={p.number || ''} onChange={(e) => set({ number: Math.min(99, Number(e.target.value.replace(/\D/g, '')) || 0) })}
                  className="mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 text-center font-display text-2xl text-ink" />
              </div>
            </div>
          </Step>
        )}
        {id === 'play' && (
          <Step title="How you play" hint="Pick every position you play.">
            <div className="grid grid-cols-2 gap-2.5">
              {POSITIONS.map((x) => {
                const on = p.positions.includes(x.id)
                return (
                  <Choice key={x.id} on={on} onClick={() => set({ positions: on ? p.positions.filter((y) => y !== x.id) : [...p.positions, x.id] })}>
                    <span className="block font-display text-xl">{x.short}</span>
                    <span className="text-sm">{x.label}</span>
                  </Choice>
                )
              })}
            </div>
            <h2 className="mt-6 mb-2.5 font-display text-base uppercase">Stronger foot</h2>
            <div className="grid grid-cols-3 gap-2.5">
              {(['left', 'right', 'both'] as Foot[]).map((f) => (
                <Choice key={f} on={p.foot === f} onClick={() => set({ foot: f })}><span className="font-display text-base uppercase">{f}</span></Choice>
              ))}
            </div>
          </Step>
        )}
        {id === 'week' && (
          <Step title="Your week" hint="Tap the days you play.">
            <div className="grid grid-cols-7 gap-1.5">
              {DAYS.map((d, day) => {
                const on = games.some((g) => g.day === day)
                return (
                  <Choice key={d} on={on} label={`${d}${on ? ', game' : ''}`}
                    onClick={() => setGames(on ? games.filter((g) => g.day !== day) : [...games, { day, kind: 'competitive' as const }].sort((a, b) => a.day - b.day))}>
                    <span className="font-display text-sm">{d.slice(0, 2)}</span>
                  </Choice>
                )
              })}
            </div>
            <h2 className="mt-6 mb-2.5 font-display text-base uppercase">Gym sessions a week</h2>
            <div className="grid grid-cols-6 gap-1.5">
              {[0, 1, 2, 3, 4, 5].map((n) => (
                <Choice key={n} on={(p.gym ?? 3) === n} label={`${n} gym sessions`} onClick={() => set({ gym: n })}>
                  <span className="font-display text-lg">{n}</span>
                </Choice>
              ))}
            </div>
            <p className="m-0 mt-3 text-sm text-muted">{games.length ? `${games.length} game${games.length > 1 ? 's' : ''} a week.` : 'No games yet: more training instead.'} Heavy weeks keep the gym but make it lighter.</p>
          </Step>
        )}
        {id === 'style' && (
          <Step title="Your colours">
            <div className="grid grid-cols-9 gap-1.5">
              {ACCENTS.map((a) => (
                <button key={a.id} aria-pressed={p.accent === a.id} aria-label={a.label}
                  onClick={() => { set({ accent: a.id }); applyAccent(a.id) }}
                  className={`aspect-square rounded-full transition-transform duration-150 ease-snap active:scale-90 ${p.accent === a.id ? 'scale-110 ring-2 ring-ink ring-offset-2 ring-offset-bg' : ''}`}
                  style={{ background: a.hex }}>
                  <span className="sr-only">{a.label}</span>
                </button>
              ))}
            </div>
            <label htmlFor="ob-idol" className="mt-6 block text-sm text-muted">Role model (optional)</label>
            <input id="ob-idol" value={p.idol} onChange={(e) => set({ idol: e.target.value.slice(0, 40) })} placeholder="Your favourite player"
              className="mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 text-lg text-ink placeholder:text-muted" />
            <p className="m-0 mt-2 text-sm text-muted">Signs your motivation quotes.</p>
          </Step>
        )}
      </div>

      <Button block className="mt-6" disabled={!valid} onClick={next}>{step === STEPS.length - 1 ? "Done · let's train" : 'Next'}</Button>
    </Screen>
  )
}

// The player card, updating live as the steps are filled in.
function CardPreview({ p }: { p: Profile }) {
  const tag = positionTag(p)
  return (
    <div aria-hidden="true" className="relative h-28 overflow-hidden rounded-[22px] bg-[linear-gradient(120deg,color-mix(in_srgb,var(--red)_55%,#000)_0%,#111113_70%)] transition-[background] duration-300">
      <span className="absolute -top-3 right-3 select-none font-display text-[120px] leading-none text-transparent [-webkit-text-stroke:2px_rgba(255,255,255,.22)]">{p.number || ''}</span>
      <div className="absolute inset-y-0 left-4 flex flex-col justify-center">
        <span className="font-display text-2xl uppercase leading-none text-white">{p.name.trim() || 'Your name'}</span>
        <span className="mt-2 flex flex-wrap gap-1.5">
          {tag && <span className="rounded-full bg-white/15 px-2.5 py-0.5 font-display text-[11px] uppercase text-white">{tag}</span>}
          <span className="rounded-full bg-white/15 px-2.5 py-0.5 font-display text-[11px] uppercase text-white">{p.foot === 'both' ? 'Both feet' : `${p.foot} foot`}</span>
        </span>
      </div>
    </div>
  )
}

function Step({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section>
      <h1 className="m-0 font-display text-3xl uppercase leading-none">{title}</h1>
      {hint ? <p className="mt-2 mb-5 text-muted">{hint}</p> : <div className="mb-5" />}
      {children}
    </section>
  )
}

function Choice({ on, onClick, children, label }: { on: boolean; onClick: () => void; children: ReactNode; label?: string }) {
  return (
    <button aria-pressed={on} aria-label={label} onClick={onClick}
      className={`min-h-14 rounded-3xl p-3 text-center transition-[transform,background-color] duration-150 ease-snap active:scale-95 ${on ? 'bg-red text-on-red' : 'bg-surface text-ink'}`}>
      {children}
    </button>
  )
}

