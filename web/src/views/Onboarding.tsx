import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { LanguagePicker } from '../components/LanguagePicker'
import { Button, Screen } from '../components/ui'
import { FlameIcon } from '../components/icons'
import { Demo3D } from '../components/Demo3D'
import { ACCENTS, POSITIONS, applyAccent, positionTag, roleOf, type Foot, type Goal, type GymExp, type Injury, type KitItem, type Level, type Position, type Profile, type TrainAt } from '../lib/profile'
import { DAYS, DAY_ABBR, FOCUS_SHORT, buildWeek, kindLabel, todayIndex, type Game } from '../lib/weekplan'
import { usePersonalImage } from '../lib/personal'
import { InstallButton } from '../components/InstallButton'
import { injurySafe, partLabel, safestInstead, seriousInjury, type Part, type Severity } from '../lib/injuries'
import { BUILTIN_SESSIONS } from '../lib/library'
import { planToday, setPlanGoals } from '../lib/today'
import { fitSession } from '../lib/fit'
import { sessionMinutes } from '../lib/logic'
import { t } from '../i18n'
import { tr } from '../i18n/rich'

// Make your player. New players: a 3-screen tour, then one question per screen (what the planner needs), a short
// "Building your week" moment and the plan it made. Edit player: the same questions plus colours and role model.
// Answers are kept on the phone after every step, so a closed app picks up where it stopped.
const NEW_STEPS = ['tour', 'name', 'position', 'foot', 'level', 'goals', 'week', 'gym', 'train', 'injuries'] as const
const EDIT_STEPS = ['name', 'position', 'foot', 'level', 'goals', 'week', 'gym', 'train', 'style'] as const
type StepId = (typeof NEW_STEPS)[number] | (typeof EDIT_STEPS)[number]
type InjuryRow = { part: Part; sev: Severity; now: boolean }
const DRAFT = 'onboarding:draft'
type Draft = { p: Profile; games: Game[]; step: number; rows: InjuryRow[] }
const readDraft = (): Draft | null => { try { return JSON.parse(localStorage.getItem(DRAFT) ?? 'null') } catch { return null } }
const writeDraft = (d: Draft) => { try { localStorage.setItem(DRAFT, JSON.stringify(d)) } catch { /* private mode */ } }
const clearDraft = () => { try { localStorage.removeItem(DRAFT) } catch { /* ignore */ } }

const ROLE_LINE = (): Record<string, string> => ({
  gk: t('Goalkeeper. Reactions, lateral power, jumps and kicking with both feet.'),
  cb: t('Centre-back. Strength, headers and the turn to stay with a runner.'),
  fb: t('Full-back. Repeated sprints, 1v1 defending and crossing.'),
  dm: t('Defensive midfielder. The engine, scanning and passing under pressure, strong in duels.'),
  cm: t('Central midfielder. The engine to cover the pitch, receiving on the half-turn, passing range.'),
  am: t('Attacking midfielder. Quick feet in tight spaces, the final pass and finishing.'),
  winger: t('Winger. Pace, 1v1s and crossing. Every drill is built for the role.'),
  striker: t('Striker. The first two steps, holding the ball up and finishing.'),
})
// where each role sits on the pitch (attacking up), as % of the pitch box
const SPOTS: { role: Position; x: number; y: number }[] = [
  { role: 'striker', x: 50, y: 11 }, { role: 'winger', x: 15, y: 20 }, { role: 'winger', x: 85, y: 20 }, { role: 'am', x: 50, y: 30 },
  { role: 'cm', x: 32, y: 46 }, { role: 'cm', x: 68, y: 46 }, { role: 'dm', x: 50, y: 58 }, { role: 'fb', x: 12, y: 70 }, { role: 'fb', x: 88, y: 70 },
  { role: 'cb', x: 34, y: 76 }, { role: 'cb', x: 66, y: 76 }, { role: 'gk', x: 50, y: 91 },
]
const INJURY_PARTS: Part[] = ['hamstrings', 'groin', 'quads', 'calves', 'knees', 'ankles', 'hips', 'back', 'shoulder', 'other']
const JOINT_PARTS: Part[] = ['knees', 'ankles', 'hips', 'back', 'shoulder']
const today = () => new Date().toLocaleDateString('en-CA')

export function Onboarding({ initial, initialGames = [], welcome = false, fresh = welcome, onDone, onSample, onSignIn, onCancel }: {
  initial?: Profile
  initialGames?: Game[]
  welcome?: boolean
  fresh?: boolean // a new player: tour, injuries, then "Building your week" and the plan
  onDone: (p: Profile, games: Game[]) => void
  onSample?: () => void
  onSignIn?: () => void
  onCancel?: () => void
}) {
  const STEPS: readonly StepId[] = fresh ? NEW_STEPS : EDIT_STEPS
  const draft = useMemo(() => (fresh ? readDraft() : null), [fresh])
  const [p, setP] = useState<Profile>(draft?.p ?? initial ?? { name: '', number: 10, positions: [], foot: 'right', accent: 'red', idol: '', gym: 3, trainAt: ['gym', 'pitch'], kit: ['ball', 'wall', 'cones'] })
  const [games, setGames] = useState<Game[]>(draft?.games ?? initialGames)
  const [step, setStep] = useState<number>(draft ? draft.step : welcome ? -1 : 0)
  const [rows, setRows] = useState<InjuryRow[]>(draft?.rows ?? [])
  const [phase, setPhase] = useState<'questions' | 'building' | 'reveal'>('questions')
  const [lastRole, setLastRole] = useState<Position | null>(null)
  const set = (patch: Partial<Profile>) => setP((x) => ({ ...x, ...patch }))
  const hero = usePersonalImage('hero.jpg')
  useEffect(() => { if (fresh && step >= 0) writeDraft({ p, games, step, rows }) }, [fresh, p, games, step, rows])

  // the player as saved: injuries in the app's terms (right now: as bad as they said; last 2 weeks: light)
  const final = (): Profile => ({
    ...p, name: p.name.trim(), sample: false,
    ...(fresh ? { injuries: rows.map((r, i): Injury => ({ id: `${Date.now().toString(36)}${i}`, part: r.part, tissue: JOINT_PARTS.includes(r.part) ? 'joint' : 'muscle', kind: 'pain', severity: r.now ? r.sev : 'light', since: today() })) } : {}),
  })
  const finish = () => { clearDraft(); onDone(final(), games) }

  if (step === -1) {
    return (
      <Screen className="relative flex flex-col justify-end pb-8">
        {hero && <>
          <img src={hero} alt="" data-fade className="drift fixed inset-0 z-0 h-full w-full object-cover object-[50%_25%]" />
          <div className="fixed inset-0 z-0 bg-[linear-gradient(to_bottom,rgba(11,11,12,.1)_0%,rgba(11,11,12,.55)_30%,rgba(11,11,12,.85)_42%,var(--bg)_60%)]" />
        </>}
        <div className="absolute top-[max(16px,var(--sat))] left-4 z-20"><LanguagePicker compact /></div>
        <InstallButton className="absolute top-[max(16px,var(--sat))] right-4 z-20" />
        <div className="stagger relative z-10">
          <p style={{ '--i': 0 } as React.CSSProperties} className="m-0 font-display text-sm uppercase text-red-ink">{t('Football Personal Coach')}</p>
          <h1 style={{ '--i': 1 } as React.CSSProperties} className="m-0 mt-2 font-display text-[clamp(2.6rem,12vw,3.8rem)] uppercase leading-[.88]">{tr('Train like{br}a pro.', { br: <br /> })}</h1>
          <ul style={{ '--i': 2 } as React.CSSProperties} className="m-0 mt-6 mb-8 flex list-none flex-col gap-3 p-0">
            {[
              ['🗓', t('A week planned around your games')],
              ['🔊', t('Drills coached out loud, LEFT / RIGHT cues')],
              ['🏋️', t('Gym sessions with a 3D demo for every exercise')],
              ['💬', t('A coach that knows your training')],
            ].map(([icon, text]) => (
              <li key={text} className="flex items-center gap-3">
                <span aria-hidden="true" className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white/10 text-lg backdrop-blur-md">{icon}</span>
                <span className="font-semibold">{text}</span>
              </li>
            ))}
          </ul>
          <div style={{ '--i': 3 } as React.CSSProperties}>
            <Button block onClick={() => setStep(0)}>{t('Get started')}</Button>
            <Button block variant="ghost" className="mt-3" onClick={onSample}>{t('Try a sample player')}</Button>
            <p className="m-0 mt-4 text-center text-xs text-muted">{t('30 seconds. No account needed.')}</p>
            {onSignIn && <button onClick={onSignIn} className="mt-1 min-h-11 w-full text-center text-sm">{t('Already have an account?')} <span className="font-semibold text-red-ink">{t('Sign in')}</span></button>}
          </div>
        </div>
      </Screen>
    )
  }

  if (phase === 'building') return <Building p={final()} games={games} onDone={() => setPhase('reveal')} />
  if (phase === 'reveal') return <Reveal p={final()} games={games} onGo={finish} />

  const id = STEPS[step]
  const valid = id === 'name' ? p.name.trim().length > 0 : id === 'position' ? p.positions.length > 0 : true
  const last = step === STEPS.length - 1
  const next = () => (last ? (fresh ? setPhase('building') : finish()) : setStep(step + 1))
  const back = () => (step === 0 ? (welcome ? setStep(-1) : onCancel?.()) : setStep(step - 1))
  const toggle = <T,>(list: T[] | undefined, x: T, max = 99): T[] => { const l = list ?? []; return l.includes(x) ? l.filter((y) => y !== x) : [...l, x].slice(-max) }

  if (id === 'tour') return <Tour onDone={() => setStep(step + 1)} onBack={welcome ? () => setStep(-1) : undefined} />

  return (
    <Screen className="flex flex-col">
      <div className="mb-5 flex items-center gap-3">
        {(step > 0 || welcome || onCancel) && <Button variant="ghost" size="sm" onClick={back}>{t('Back')}</Button>}
        <div className="flex flex-1 gap-1" aria-label={t('Step {n} of {total}', { n: step + 1, total: STEPS.length })}>
          {STEPS.map((_, i) => <div key={i} className={`h-1.5 flex-1 rounded-full transition-colors duration-300 ${i <= step ? 'bg-red' : 'bg-surface-2'}`} />)}
        </div>
      </div>

      {(id === 'name' || id === 'position' || id === 'foot') && <CardPreview p={p} />}

      <div key={id} className="rise mt-6 flex-1">
        {id === 'name' && (
          <Step title={t('Your player')}>
            <div className="flex gap-3">
              <div className="min-w-0 flex-1">
                <label htmlFor="ob-name" className="text-sm text-muted">{t('Name')}</label>
                <input id="ob-name" value={p.name} onChange={(e) => set({ name: e.target.value.slice(0, 20) })} autoComplete="given-name" autoFocus
                  className="mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 font-display text-xl uppercase text-ink" />
              </div>
              <div className="w-24">
                <label htmlFor="ob-number" className="text-sm text-muted">{t('Number')}</label>
                <input id="ob-number" inputMode="numeric" value={p.number || ''} onChange={(e) => set({ number: Math.min(99, Number(e.target.value.replace(/\D/g, '')) || 0) })}
                  className="mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 text-center font-display text-2xl text-ink" />
              </div>
            </div>
          </Step>
        )}
        {id === 'position' && (
          <Step title={t('Where do you play?')} hint={t('Tap your position. Play more than one? Tap them all (up to 3).')}>
            <Pitch picked={p.positions.map(roleOf)} onTap={(r) => { setLastRole(p.positions.map(roleOf).includes(r) ? null : r); set({ positions: toggle(p.positions.map(roleOf), r, 3) }) }} />
            {(lastRole ?? p.positions[0]) && <p className="rise m-0 mt-3 rounded-[18px] bg-surface p-3.5 text-sm font-semibold">{ROLE_LINE()[roleOf(lastRole ?? p.positions[0])]}</p>}
          </Step>
        )}
        {id === 'foot' && (
          <Step title={t('Stronger foot')}>
            <div className="grid grid-cols-3 gap-2.5">
              {(['left', 'right', 'both'] as Foot[]).map((f) => (
                <Choice key={f} on={p.foot === f} onClick={() => set({ foot: f })}><span className="font-display text-base uppercase">{f === 'left' ? t('Left') : f === 'right' ? t('Right') : t('Both')}</span></Choice>
              ))}
            </div>
          </Step>
        )}
        {id === 'level' && (
          <Step title={t('Your level')} hint={t('It sets how much and how hard your sessions start.')}>
            <div className="flex flex-col gap-2">
              {([['casual', t('Casual'), t('Kickabouts and five-a-side')], ['amateur', t('Amateur club'), t('A team, a league, weekly games')], ['semi', t('Semi-pro'), t('Paid or high-level amateur')], ['academy', t('Academy or pro'), t('Training most days with a club')]] as [Level, string, string][]).map(([v, label, hint]) => (
                <Row key={v} on={p.level === v} onClick={() => set({ level: v })} title={label} hint={hint} />
              ))}
            </div>
          </Step>
        )}
        {id === 'goals' && (
          <Step title={t('Your goals')} hint={t('Pick up to 2. They decide what comes first in your plan.')}>
            <div className="flex flex-col gap-2">
              {([['speed', t('Faster and more explosive')], ['strength', t('Stronger')], ['stamina', t('More stamina')], ['skills', t('Better on the ball')], ['injury-free', t('Stay injury-free')]] as [Goal, string][]).map(([g, label]) => (
                <Row key={g} on={!!p.goals?.includes(g)} onClick={() => set({ goals: toggle(p.goals, g, 2) })} title={label} />
              ))}
            </div>
          </Step>
        )}
        {id === 'week' && (
          <Step title={t('Your games')} hint={t('Tap the days you play. Tap again for a casual game, a third time to clear it.')}>
            <div className="grid grid-cols-[repeat(7,minmax(0,1fr))] gap-1.5">
              {DAYS.map((d, day) => {
                const g = games.find((x) => x.day === day)
                const cycle = () => setGames((gs) => {
                  const rest = gs.filter((x) => x.day !== day)
                  return (!g ? [...rest, { day, kind: 'competitive' as const }] : g.kind === 'competitive' ? [...rest, { day, kind: 'casual' as const }] : rest).sort((a, b) => a.day - b.day)
                })
                return (
                  <button key={d} aria-pressed={!!g} aria-label={g ? `${d}, ${g.kind === 'casual' ? t('casual game') : t('competitive game')}` : d} onClick={cycle}
                    className={`flex min-h-16 flex-col items-center justify-center rounded-2xl transition-[transform,background-color] duration-150 ease-snap active:scale-95 ${g ? (g.kind === 'casual' ? 'bg-red/45 text-ink' : 'bg-red text-on-red') : 'bg-surface text-ink'}`}>
                    <span className="font-display text-sm">{DAY_ABBR[day]}</span>
                    <span className={`text-[10px] font-semibold uppercase ${g ? '' : 'text-muted'}`}>{g ? (g.kind === 'casual' ? t('Casual|strip') : t('Comp')) : '·'}</span>
                  </button>
                )
              })}
            </div>
            <p className="m-0 mt-3 text-sm font-semibold">{games.length ? (games.length === 1 ? t('1 game a week') : t('{n} games a week', { n: games.length })) : t('No games: more training instead')}</p>
          </Step>
        )}
        {id === 'gym' && (
          <Step title={t('The gym')}>
            <h2 className="m-0 mb-2.5 font-display text-base uppercase">{t('Gym sessions a week')}</h2>
            <div className="grid grid-cols-6 gap-1.5">
              {[0, 1, 2, 3, 4, 5].map((n) => (
                <Choice key={n} on={(p.gym ?? 3) === n} label={t('{n} gym sessions', { n })} onClick={() => set({ gym: n })}><span className="font-display text-lg">{n}</span></Choice>
              ))}
            </div>
            <h2 className="m-0 mt-6 mb-2.5 font-display text-base uppercase">{t('Gym experience')}</h2>
            <div className="flex flex-col gap-2">
              {([['new', t('New'), t('Technique first, no Olympic lifts yet')], ['some', t('Some'), t('I know the main lifts')], ['experienced', t('Experienced'), t('Heavy lifts and cleans are fine')]] as [GymExp, string, string][]).map(([v, label, hint]) => (
                <Row key={v} on={p.gymExp === v} onClick={() => set({ gymExp: v })} title={label} hint={hint} />
              ))}
            </div>
          </Step>
        )}
        {id === 'train' && (
          <Step title={t('Where you train')} hint={t('Sessions swap what you can\'t do there.')}>
            <div className="grid grid-cols-3 gap-2">
              {([['gym', t('Full gym')], ['home', t('Home')], ['pitch', t('Pitch')]] as [TrainAt, string][]).map(([v, label]) => (
                <Choice key={v} on={!!p.trainAt?.includes(v)} onClick={() => set({ trainAt: toggle(p.trainAt, v) })}><span className="font-semibold">{label}</span></Choice>
              ))}
            </div>
            <h2 className="m-0 mt-6 mb-2.5 font-display text-base uppercase">{t('Kit you have')}</h2>
            <div className="grid grid-cols-3 gap-2">
              {([['ball', t('Ball')], ['wall', t('A wall')], ['cones', t('Cones')]] as [KitItem, string][]).map(([v, label]) => (
                <Choice key={v} on={!!p.kit?.includes(v)} onClick={() => set({ kit: toggle(p.kit, v) })}><span className="font-semibold">{label}</span></Choice>
              ))}
            </div>
          </Step>
        )}
        {id === 'injuries' && (
          <Step title={t('Anything hurting?')} hint={t('Sore or injured right now, or injured in the last 2 weeks. Your plan works around it. Only you see this.')}>
            <div className="flex flex-wrap gap-1.5">
              {INJURY_PARTS.map((part) => {
                const on = rows.some((r) => r.part === part)
                return (
                  <button key={part} aria-pressed={on} onClick={() => setRows((rs) => (on ? rs.filter((r) => r.part !== part) : [...rs, { part, sev: 'light', now: true }]))}
                    className={`min-h-10 rounded-full px-4 text-sm font-semibold first-letter:uppercase active:scale-95 ${on ? 'bg-ink text-bg' : 'bg-surface text-muted'}`}>{partLabel(part)}</button>
                )
              })}
            </div>
            <div className="mt-4 flex flex-col gap-2">
              {rows.map((r) => (
                <div key={r.part} className="rounded-[18px] bg-surface p-3">
                  <p className="m-0 font-semibold first-letter:uppercase">{partLabel(r.part)}</p>
                  <Seg label={t('When')} value={r.now ? 'now' : 'past'} options={[['now', t('Right now')], ['past', t('Last 2 weeks')]]}
                    onChange={(v) => setRows((rs) => rs.map((x) => (x.part === r.part ? { ...x, now: v === 'now' } : x)))} />
                  {r.now && <Seg label={t('How bad')} value={r.sev} options={[['light', t('Niggle')], ['moderate', t('Limits training')], ['serious', t("Can't train")]]}
                    onChange={(v) => setRows((rs) => rs.map((x) => (x.part === r.part ? { ...x, sev: v as Severity } : x)))} />}
                </div>
              ))}
            </div>
            {!rows.length && <p className="m-0 mt-4 text-sm text-muted">{t('Nothing? Great: tap Next.')}</p>}
          </Step>
        )}
        {id === 'style' && (
          <Step title={t('Your colours')}>
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
            <label htmlFor="ob-idol" className="mt-6 block text-sm text-muted">{t('Role model (optional)')}</label>
            <input id="ob-idol" value={p.idol} onChange={(e) => set({ idol: e.target.value.slice(0, 40) })} placeholder={t('Your favourite player')}
              className="mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 text-lg text-ink placeholder:text-muted" />
            <p className="m-0 mt-2 text-sm text-muted">{t('Signs your motivation quotes.')}</p>
          </Step>
        )}
      </div>

      <Button block className="mt-6" disabled={!valid} onClick={next}>{last ? (fresh ? t('Build my week') : t("Done · let's train")) : t('Next')}</Button>
    </Screen>
  )
}

// Three screens before the questions: what the app does, shown with the app itself. Skippable.
function Tour({ onDone, onBack }: { onDone: () => void; onBack?: () => void }) {
  const [i, setI] = useState(0)
  const week = buildWeek([{ day: 2, kind: 'competitive' }], { gym: 3 }).slots
  const slides = [
    { title: t('Your week, planned around game day'), body: t('Tell it when you play. Hard days stay away from games, recovery comes after them.'),
      art: (
        <div className="grid grid-cols-7 gap-1.5" aria-hidden="true">
          {DAY_ABBR.map((d, day) => {
            const s = week.filter((x) => x.day === day)
            const game = s.some((x) => x.kind === 'game')
            return (
              <div key={d} className={`flex min-h-24 flex-col items-center justify-between rounded-2xl p-2 ${game ? 'bg-red text-on-red' : 'bg-surface'}`}>
                <span className="font-display text-xs">{d}</span>
                <span className="text-center text-[10px] font-semibold leading-tight">{s.length ? s.map((x) => (x.kind === 'gym' && x.focus ? FOCUS_SHORT[x.focus] : kindLabel(x.kind))).join(' + ') : t('Rest')}</span>
              </div>
            )
          })}
        </div>
      ) },
    { title: t('Correct form on every exercise'), body: t('A 3D demo for each one. Form check shows the common mistake in red and the fix in green.'), art: <Demo3D name="Back squat" /> },
    { title: t('Streaks, a player card, your friends'), body: t('Keep your streak, rate yourself every week, and share your sessions.'),
      art: (
        <div className="flex items-center justify-center gap-4 rounded-[22px] bg-surface py-8" aria-hidden="true">
          <FlameIcon className="h-20 w-20" /><span className="font-display text-6xl tabular-nums">12</span>
          <FlameIcon tone="wins" className="h-14 w-14" /><span className="font-display text-4xl tabular-nums">4</span>
        </div>
      ) },
  ]
  const s = slides[i]
  return (
    <Screen className="flex flex-col">
      <div className="mb-5 flex items-center justify-between">
        {onBack || i > 0 ? <Button variant="ghost" size="sm" onClick={() => (i ? setI(i - 1) : onBack?.())}>{t('Back')}</Button> : <span />}
        <button onClick={onDone} className="min-h-10 px-2 text-sm font-semibold text-muted">{t('Skip')}</button>
      </div>
      <div key={i} className="rise flex flex-1 flex-col">
        <div className="min-h-[18rem]">{s.art}</div>
        <h1 className="m-0 mt-6 font-display text-3xl uppercase leading-none">{s.title}</h1>
        <p className="m-0 mt-3 text-muted">{s.body}</p>
      </div>
      <div className="mt-6 flex justify-center gap-1.5" aria-hidden="true">{slides.map((_, k) => <span key={k} className={`h-1.5 rounded-full transition-all ${k === i ? 'w-6 bg-red' : 'w-1.5 bg-surface-2'}`} />)}</div>
      <Button block className="mt-5" onClick={() => (i < slides.length - 1 ? setI(i + 1) : onDone())}>{i < slides.length - 1 ? t('Next') : t("Let's set you up")}</Button>
    </Screen>
  )
}

// The pitch: tap where you play (attacking up).
function Pitch({ picked, onTap }: { picked: Position[]; onTap: (r: Position) => void }) {
  const short = (r: Position) => POSITIONS.find((x) => x.id === r)?.short ?? r
  return (
    <div className="relative mx-auto aspect-[3/4] w-full max-w-[19rem] overflow-hidden rounded-[22px] border-2 border-white/15 bg-[repeating-linear-gradient(0deg,#123d25_0_12.5%,#15452a_12.5%_25%)]">
      <div className="pointer-events-none absolute inset-x-0 top-1/2 h-0.5 -translate-y-1/2 bg-white/25" />
      <div className="pointer-events-none absolute top-1/2 left-1/2 h-20 w-20 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white/25" />
      <div className="pointer-events-none absolute top-0 left-1/2 h-[14%] w-[46%] -translate-x-1/2 border-2 border-t-0 border-white/25" />
      <div className="pointer-events-none absolute bottom-0 left-1/2 h-[14%] w-[46%] -translate-x-1/2 border-2 border-b-0 border-white/25" />
      {SPOTS.map((s, k) => {
        const on = picked.includes(s.role)
        return (
          <button key={k} onClick={() => onTap(s.role)} aria-pressed={on} aria-label={POSITIONS.find((x) => x.id === s.role)?.label}
            className={`absolute grid h-12 w-12 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full font-display text-xs shadow-[0_4px_12px_rgba(0,0,0,.35)] transition-[transform,background-color] duration-150 ease-snap active:scale-90 ${on ? 'scale-110 bg-red text-on-red' : 'bg-white/90 text-[#123d25]'}`}
            style={{ left: `${s.x}%`, top: `${s.y}%` }}>{short(s.role)}</button>
        )
      })}
    </div>
  )
}

// What the planner made of the answers, a line at a time (about 3 s), then the plan.
export function buildingLines(p: Profile, games: Game[]): string[] {
  const out: string[] = []
  for (const g of games) out.push(t('Game {day}, so nothing heavy the day before.', { day: DAYS[g.day] }))
  if (!games.length) out.push(t('No games: more training days instead.'))
  const goalLine: Record<Goal, string> = {
    speed: t('Faster: speed and jump work comes first.'), strength: t('Stronger: heavy strength days come first.'), stamina: t('More stamina: interval sessions come first.'),
    skills: t('Better on the ball: dribbling and first-touch sessions come first.'), 'injury-free': t('Injury-free: Nordics, Copenhagen planks and core come first.'),
  }
  for (const g of p.goals ?? []) out.push(goalLine[g])
  if (p.gymExp === 'new') out.push(t('New to the gym: technique first, no Olympic lifts yet.'))
  if (p.trainAt?.length && !p.trainAt.includes('gym')) out.push(t('No gym: home versions on gym days.'))
  for (const i of p.injuries ?? []) out.push(i.severity === 'serious' ? t('{part}: no sessions until it\'s healed.', { part: partLabel(i.part) }) : t('{part}: exercises that load it are swapped out.', { part: partLabel(i.part) }))
  return out.map((x) => x.charAt(0).toUpperCase() + x.slice(1))
}
function Building({ p, games, onDone }: { p: Profile; games: Game[]; onDone: () => void }) {
  const lines = useMemo(() => buildingLines(p, games), [p, games])
  const [n, setN] = useState(0)
  useEffect(() => {
    const reduce = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches
    const step = reduce ? 250 : Math.max(380, Math.min(650, 3000 / Math.max(1, lines.length)))
    const id = setInterval(() => setN((k) => k + 1), step)
    return () => clearInterval(id)
  }, [lines.length])
  useEffect(() => { if (n > lines.length + 1) onDone() }, [n, lines.length, onDone])
  return (
    <Screen className="flex flex-col justify-center">
      <h1 className="m-0 font-display text-4xl uppercase leading-[.9]">{tr('Building <r>your week</r>', { r: (x) => <><br /><span className="text-red-ink">{x}</span></> })}</h1>
      <ul className="m-0 mt-8 flex list-none flex-col gap-3 p-0" aria-live="polite">
        {lines.slice(0, n).map((l) => (
          <li key={l} className="rise flex items-start gap-3">
            <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-red text-on-red" aria-hidden="true">
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="m5 12.5 4.5 4.5L19 7" /></svg>
            </span>
            <span className="font-semibold">{l}</span>
          </li>
        ))}
      </ul>
    </Screen>
  )
}

// The plan it made: the week, today's session and why, one button.
function Reveal({ p, games, onGo }: { p: Profile; games: Game[]; onGo: () => void }) {
  const td = useMemo(() => { setPlanGoals(p.goals); return planToday(BUILTIN_SESSIONS, games, new Date(), p.gym ?? 3) }, [p, games])
  const injuries = (p.injuries ?? []).filter((i) => !i.healed)
  const session = injurySafe(fitSession(safestInstead(td.session, BUILTIN_SESSIONS, injuries) ?? td.session, p).session, injuries).session
  const off = seriousInjury(injuries)
  const now = todayIndex()
  return (
    <Screen className="flex flex-col">
      <h1 className="rise m-0 mt-4 font-display text-4xl uppercase leading-[.9]">{tr('Your <r>week</r>', { r: (x) => <span className="text-red-ink">{x}</span> })}</h1>
      <div className="rise mt-6 grid grid-cols-7 gap-1.5">
        {DAY_ABBR.map((d, day) => {
          const s = td.plan.slots.filter((x) => x.day === day)
          const game = s.some((x) => x.kind === 'game')
          return (
            <div key={d} className={`flex min-h-24 flex-col items-center justify-between rounded-2xl p-2 ${game ? 'bg-red text-on-red' : 'bg-surface'} ${day === now ? 'ring-2 ring-ink' : ''}`}>
              <span className="font-display text-xs">{d}</span>
              <span className="text-center text-[10px] font-semibold leading-tight">{s.length ? s.map((x) => (x.kind === 'gym' && x.focus ? FOCUS_SHORT[x.focus] : kindLabel(x.kind))).join(' + ') : t('Rest')}</span>
            </div>
          )
        })}
      </div>
      {off ? (
        <section className="rise mt-5 rounded-[22px] bg-surface p-4">
          <p className="m-0 font-semibold">{t('Your {part} injury is serious', { part: partLabel(off.part) })}</p>
          <p className="m-0 mt-1 text-sm text-muted">{t('No sessions until it\'s healed. Rest it and see a physio; when it\'s better, mark it healed in Record → Body and your plan starts.')}</p>
        </section>
      ) : <section className="rise mt-5 rounded-[22px] bg-surface p-4">
        <p className="m-0 font-display text-xs uppercase text-red-ink">{td.scheduled ? t('Today') : t('Next up')}</p>
        <p className="m-0 mt-1 font-display text-xl uppercase leading-tight">{session.title}</p>
        <p className="m-0 mt-1 text-sm text-muted">{t('{n} min · {d} drills', { n: sessionMinutes(session), d: session.blocks.filter((b) => b.kind === 'drill').length })}</p>
      </section>}
      {td.plan.reasons.length > 0 && (
        <ul className="rise m-0 mt-4 flex list-none flex-col gap-2 p-0 text-sm text-muted">
          {td.plan.reasons.slice(0, 3).map((r) => <li key={r}>• {r}</li>)}
        </ul>
      )}
      <Button block className="mt-auto" onClick={onGo}>{t("Let's train")}</Button>
    </Screen>
  )
}

function Row({ on, onClick, title, hint }: { on: boolean; onClick: () => void; title: string; hint?: string }) {
  return (
    <button aria-pressed={on} onClick={onClick}
      className={`flex min-h-14 w-full flex-col justify-center rounded-2xl px-4 py-2.5 text-left transition-[transform,background-color] duration-150 ease-snap active:scale-[0.98] ${on ? 'bg-red text-on-red' : 'bg-surface text-ink'}`}>
      <span className="font-semibold">{title}</span>
      {hint && <span className={`text-sm ${on ? 'opacity-85' : 'text-muted'}`}>{hint}</span>}
    </button>
  )
}

function Seg({ label, value, options, onChange }: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void }) {
  return (
    <div role="radiogroup" aria-label={label} className="mt-2 grid gap-1 rounded-full bg-surface-2 p-1" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      {options.map(([v, l]) => (
        <button key={v} role="radio" aria-checked={value === v} onClick={() => onChange(v)}
          className={`min-h-9 rounded-full px-1 text-xs font-semibold ${value === v ? 'bg-ink text-bg' : 'text-muted'}`}>{l}</button>
      ))}
    </div>
  )
}

// The player card, updating live as the steps are filled in.
function CardPreview({ p }: { p: Profile }) {
  const tag = positionTag(p)
  return (
    <div aria-hidden="true" className="relative h-28 overflow-hidden rounded-[22px] bg-[linear-gradient(120deg,color-mix(in_srgb,var(--red)_55%,#000)_0%,#111113_70%)] transition-[background] duration-300">
      <span className="absolute -top-3 right-3 select-none font-display text-[120px] leading-none text-transparent [-webkit-text-stroke:2px_rgba(255,255,255,.22)]">{p.number || ''}</span>
      <div className="absolute inset-y-0 left-4 flex flex-col justify-center">
        <span className="font-display text-2xl uppercase leading-none text-white">{p.name.trim() || t('Your name')}</span>
        <span className="mt-2 flex flex-wrap gap-1.5">
          {tag && <span className="rounded-full bg-white/15 px-2.5 py-0.5 font-display text-[11px] uppercase text-white">{tag}</span>}
          <span className="rounded-full bg-white/15 px-2.5 py-0.5 font-display text-[11px] uppercase text-white">{p.foot === 'both' ? t('Both feet') : p.foot === 'left' ? t('Left foot') : t('Right foot')}</span>
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

