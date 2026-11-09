import { useState, type ReactNode } from 'react'
import { Button, Screen } from '../components/ui'
import { ACCENTS, POSITIONS, applyAccent, type Foot, type Profile } from '../lib/profile'
import { DAYS, type Game } from '../lib/weekplan'
import { usePersonalImage } from '../lib/personal'

// Make your player: a few one-tap steps. Used by demo visitors ("Try it") and by new accounts.
// `welcome` adds the first screen with the sample-player shortcut (demo only).
const STEPS = ['name', 'positions', 'foot', 'games', 'colour', 'idol'] as const

export function Onboarding({ initial, initialGames = [], welcome = false, onDone, onSample, onSignIn, onCancel }: {
  initial?: Profile
  initialGames?: Game[]
  welcome?: boolean
  onDone: (p: Profile, games: Game[]) => void
  onSample?: () => void
  onSignIn?: () => void
  onCancel?: () => void
}) {
  const [p, setP] = useState<Profile>(initial ?? { name: '', number: 10, positions: [], foot: 'right', accent: 'red', idol: '' })
  const [games, setGames] = useState<Game[]>(initialGames)
  const [step, setStep] = useState<number>(welcome ? -1 : 0)
  const set = (patch: Partial<Profile>) => setP((x) => ({ ...x, ...patch }))
  const hero = usePersonalImage('hero.jpg')

  if (step === -1) {
    return (
      <Screen className="relative flex flex-col justify-end pb-10">
        {hero && <>
          <img src={hero} alt="" className="drift fixed inset-0 -z-10 h-full w-full object-cover object-[50%_25%]" />
          <div className="fixed inset-0 -z-10 bg-[linear-gradient(to_bottom,rgba(11,11,12,.2)_0%,rgba(11,11,12,.35)_40%,var(--bg)_72%)]" />
        </>}
        <p className="m-0 font-display text-sm uppercase text-red-ink">Sport Coach</p>
        <h1 className="m-0 mt-2 font-display text-[clamp(2.4rem,11vw,3.6rem)] uppercase leading-[.9]">Train like<br />a pro.</h1>
        <p className="mt-4 mb-8 text-muted">Football sessions with a voice-cue timer, a week plan built around your games, and gym work for power. Try it with your own player, or look around with a sample one.</p>
        <Button block onClick={() => setStep(0)}>Make my player</Button>
        <Button block variant="ghost" className="mt-3" onClick={onSample}>Use a sample player</Button>
        <p className="mt-4 text-center text-xs text-muted">Everything stays in this browser. No account needed.</p>
        {onSignIn && <button onClick={onSignIn} className="mt-2 min-h-11 text-center text-sm font-semibold">Have an invite? <span className="text-red-ink">Sign in</span></button>}
      </Screen>
    )
  }

  const id = STEPS[step]
  const valid = id === 'name' ? p.name.trim().length > 0 : id === 'positions' ? p.positions.length > 0 : true
  const next = () => (step === STEPS.length - 1 ? onDone({ ...p, name: p.name.trim(), sample: false }, games) : setStep(step + 1))
  const back = () => (step === 0 ? (welcome ? setStep(-1) : onCancel?.()) : setStep(step - 1))

  return (
    <Screen className="flex flex-col">
      <div className="mb-6 flex items-center gap-3">
        {(step > 0 || welcome || onCancel) && <Button variant="ghost" size="sm" onClick={back}>Back</Button>}
        <div className="flex flex-1 gap-1" aria-label={`Step ${step + 1} of ${STEPS.length}`}>
          {STEPS.map((_, i) => <div key={i} className={`h-1.5 flex-1 rounded-full ${i <= step ? 'bg-red' : 'bg-surface-2'}`} />)}
        </div>
      </div>

      <div key={id} className="rise flex-1">
        {id === 'name' && (
          <Step title="Your player" hint="Name and shirt number for your player card.">
            <label htmlFor="ob-name" className="text-sm text-muted">Name</label>
            <input id="ob-name" value={p.name} onChange={(e) => set({ name: e.target.value.slice(0, 20) })} autoComplete="given-name"
              className="mt-1 mb-4 block min-h-14 w-full rounded-2xl bg-surface px-4 font-display text-xl uppercase text-ink" />
            <label htmlFor="ob-number" className="text-sm text-muted">Number</label>
            <input id="ob-number" inputMode="numeric" value={p.number || ''} onChange={(e) => set({ number: Math.min(99, Number(e.target.value.replace(/\D/g, '')) || 0) })}
              className="mt-1 block min-h-14 w-28 rounded-2xl bg-surface px-4 text-center font-display text-2xl text-ink" />
          </Step>
        )}
        {id === 'positions' && (
          <Step title="Where you play" hint="Pick every position you play. Sessions are recommended for them.">
            <div className="grid grid-cols-2 gap-3">
              {POSITIONS.map((x) => {
                const on = p.positions.includes(x.id)
                return (
                  <Choice key={x.id} on={on} onClick={() => set({ positions: on ? p.positions.filter((y) => y !== x.id) : [...p.positions, x.id] })}>
                    <span className="block font-display text-2xl">{x.short}</span>
                    <span className="text-sm">{x.label}</span>
                  </Choice>
                )
              })}
            </div>
          </Step>
        )}
        {id === 'foot' && (
          <Step title="Stronger foot" hint="Drills that say “weaker foot” mean the other one.">
            <div className="grid grid-cols-3 gap-3">
              {(['left', 'right', 'both'] as Foot[]).map((f) => (
                <Choice key={f} on={p.foot === f} onClick={() => set({ foot: f })}><span className="font-display text-lg uppercase">{f}</span></Choice>
              ))}
            </div>
          </Step>
        )}
        {id === 'games' && (
          <Step title="Game days" hint="Tap the days you usually play. The week plan works around them. You can change this any week.">
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
            <p className="mt-3 text-sm text-muted">{games.length ? `${games.length} game${games.length > 1 ? 's' : ''} a week` : 'No games: more training sessions instead.'}</p>
            <h2 className="mt-8 mb-3 font-display text-lg uppercase">Gym sessions a week</h2>
            <div className="grid grid-cols-6 gap-1.5">
              {[0, 1, 2, 3, 4, 5].map((n) => (
                <Choice key={n} on={(p.gym ?? 3) === n} label={`${n} gym sessions`} onClick={() => set({ gym: n })}>
                  <span className="font-display text-lg">{n}</span>
                </Choice>
              ))}
            </div>
            <p className="mt-3 text-sm text-muted">Heavy game weeks keep these but make them lighter.</p>
          </Step>
        )}
        {id === 'colour' && (
          <Step title="Team colour" hint="Buttons, highlights and your player card.">
            <div className="grid grid-cols-3 gap-3">
              {ACCENTS.map((a) => (
                <button key={a.id} aria-pressed={p.accent === a.id} aria-label={a.label}
                  onClick={() => { set({ accent: a.id }); applyAccent(a.id) }}
                  className={`grid aspect-square place-items-center rounded-3xl transition-transform duration-150 ease-snap active:scale-95 ${p.accent === a.id ? 'ring-4 ring-ink ring-offset-4 ring-offset-bg' : ''}`}
                  style={{ background: a.hex }}>
                  <span className="sr-only">{a.label}</span>
                </button>
              ))}
            </div>
          </Step>
        )}
        {id === 'idol' && (
          <Step title="Your role model" hint="Optional. Their name goes on your motivation quotes. Photos and clips can be added later in Settings.">
            <label htmlFor="ob-idol" className="text-sm text-muted">Name</label>
            <input id="ob-idol" value={p.idol} onChange={(e) => set({ idol: e.target.value.slice(0, 40) })} placeholder="e.g. your favourite player"
              className="mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 text-lg text-ink placeholder:text-muted" />
          </Step>
        )}
      </div>

      <Button block className="mt-8" disabled={!valid} onClick={next}>{step === STEPS.length - 1 ? 'Done' : 'Next'}</Button>
    </Screen>
  )
}

function Step({ title, hint, children }: { title: string; hint: string; children: ReactNode }) {
  return (
    <section>
      <h1 className="m-0 font-display text-3xl uppercase leading-none">{title}</h1>
      <p className="mt-3 mb-6 text-muted">{hint}</p>
      {children}
    </section>
  )
}

function Choice({ on, onClick, children, label }: { on: boolean; onClick: () => void; children: ReactNode; label?: string }) {
  return (
    <button aria-pressed={on} aria-label={label} onClick={onClick}
      className={`min-h-16 rounded-3xl p-3 text-center transition-[transform,background-color] duration-150 ease-snap active:scale-95 ${on ? 'bg-red text-on-red' : 'bg-surface text-ink'}`}>
      {children}
    </button>
  )
}

