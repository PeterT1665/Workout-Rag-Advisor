import { useState } from 'react'
import { Button, Pill, SectionHead } from '../components/ui'
import { BallIcon, BrushIcon, FlameIcon, GearIcon, KitGlyph } from '../components/icons'
import type { Streak } from '../lib/streak'
import { usePersonalImage } from '../lib/personal'
import { DEMO } from '../lib/demo'
import { useProfile } from '../lib/profile'
import { kitFor, loadPacked, togglePacked, type KitItem } from '../lib/kit'
import { fmtReps, sessionMinutes } from '../lib/logic'
import { InstallButton } from '../components/InstallButton'
import { DrillSheet } from '../components/DrillSheet'
import { canInstall } from '../lib/install'
import { CATEGORY_LABELS, drillPhoto } from '../lib/library'
import { load, save } from '../lib/storage'
import { DAYS, DAY_SHORT, type SlotKind } from '../lib/weekplan'
import type { Today } from '../lib/today'
import type { Block, Session } from '../lib/types'

type Props = {
  today: Today
  hasClips: boolean
  onKickOff: (s: Session) => void
  onOpen: (s: Session) => void
  onSettings: () => void
  onAllSessions: () => void
  onPlayClip: () => void
  onAddClips: () => void
  onPlayer: () => void
  onWeek?: () => void
  active?: { title: string; onResume: () => void; onDiscard: () => void } | null // unfinished session
  dayKind?: (day: number) => string | null | undefined // hand-changed day: session type, null = rest, undefined = unchanged
  streak?: Streak
  makeup?: { missed: string; by: string; picks: Session[] } | null // a missed day that can still be made up
}

const SLOT_LABEL: Record<SlotKind, string> = { game: 'Game day', football: 'Football', gym: 'Gym', recovery: 'Recovery', cheer: 'Cheer' }
const dateKey = () => new Date().toISOString().slice(0, 10)

// ---------- hero ----------
// The hero's text slides in once per app open, not every time Home is shown again.
let heroIntroDone = false

export function Hero({ today, onKickOff, onOpen, onSettings, onCustomise, streak, onStreak, desktop = false }: Pick<Props, 'today' | 'onKickOff' | 'onOpen' | 'onSettings' | 'streak'> & { onCustomise?: () => void; onStreak?: () => void; desktop?: boolean }) {
  const hero = usePersonalImage('hero.jpg')
  const [entrance] = useState(() => { const first = !heroIntroDone; heroIntroDone = true; return first })
  const s = today.session
  const drills = s.blocks.filter((b) => b.kind === 'drill').length
  const when = today.scheduled ? 'Today' : today.nextDay !== null ? `Next · ${DAYS[today.nextDay].slice(0, 3)}` : 'Anytime'
  const status = !today.scheduled && today.slot
    ? today.slot.kind === 'game' ? 'Game day. Keep the legs fresh.'
      : today.slot.kind === 'cheer' ? 'Cheer practice this morning.'
        : null
    : !today.scheduled ? 'Rest day.' : null
  const date = new Date().toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })
  const me = useProfile()
  const sample = DEMO && me.sample

  return (
    <section className={desktop
      ? 'relative h-full min-h-[460px] overflow-hidden rounded-[28px] bg-[#0b0d0c]'
      : 'relative -mx-4 h-[640px] overflow-hidden rounded-b-[28px] bg-[#0b0d0c]'}>
      {hero
        ? <img src={hero} alt="" data-fade className={`absolute inset-0 h-full w-full object-cover ${desktop ? 'object-[50%_30%]' : 'object-[50%_22%]'}`} />
        : <PitchBackdrop />}
      <div className={desktop
        ? 'absolute inset-0 bg-[linear-gradient(90deg,rgba(10,12,11,.95)_0%,rgba(10,12,11,.7)_38%,rgba(10,12,11,0)_70%)]'
        : 'absolute inset-0 bg-[linear-gradient(to_bottom,rgba(10,12,11,.6)_0%,rgba(10,12,11,0)_22%,rgba(10,12,11,0)_55%,rgba(10,12,11,.96)_86%)]'} />
      {!desktop && <div className="absolute inset-x-4 top-[max(16px,var(--sat))] flex items-center justify-between">
        {streak ? (
          <button onClick={onStreak} aria-label={`Plan streak: ${streak.days} ${streak.days === 1 ? 'day' : 'days'}${streak.today ? '' : ', train today to keep it going'}. Open your training log`}
            className="flex h-10 items-center gap-1.5 rounded-full bg-white/15 pr-3.5 pl-2.5 text-white backdrop-blur-md active:scale-95">
            <FlameIcon lit={streak.days > 0} className={`h-6 w-6 ${streak.today ? 'drop-shadow-[0_0_6px_rgba(255,90,31,.7)]' : ''}`} />
            <span className={`font-display text-lg tabular-nums leading-none ${streak.days > 0 ? 'text-[#ff9a1f]' : 'text-white/80'}`}>{streak.days}</span>
          </button>
        ) : <span />}
        <span className="flex items-center gap-2">
          {canInstall() ? <InstallButton /> : <Pill tone="glass">{sample ? 'Sample data' : date}</Pill>}
          {onCustomise && <button onClick={onCustomise} aria-label="Make it yours" className="grid h-10 w-10 place-items-center rounded-full bg-white/15 text-white backdrop-blur-md active:scale-95">
            <BrushIcon />
          </button>}
          <button onClick={onSettings} aria-label="Settings" className="grid h-10 w-10 place-items-center rounded-full bg-white/15 text-white backdrop-blur-md active:scale-95">
            <GearIcon />
          </button>
        </span>
      </div>}
      <div className={`${desktop ? 'absolute bottom-8 left-8 max-w-[64%]' : 'absolute inset-x-4 bottom-5'} ${entrance ? 'rise' : ''}`}>
        <div className="flex flex-wrap gap-2">
          {desktop && <Pill tone="glass">{sample ? 'Sample data' : date}</Pill>}
          {today.md && <Pill tone="red">{today.md}</Pill>}
          <Pill tone="glass">{when}{today.scheduled && today.slot ? ` · ${SLOT_LABEL[today.slot.kind]}` : ''}</Pill>
        </div>
        <h1 className={`m-0 mt-3 mb-2 font-display uppercase leading-[.95] text-white ${desktop ? 'text-[38px]' : 'text-[32px]'}`}>{s.title}</h1>
        {status && <p className="m-0 mb-3 text-sm text-white/75">{status}</p>}
        <div className="mt-3 flex gap-2.5">
          <Button onClick={() => onKickOff(s)}>Kick off →</Button>
          <Button variant="glass" className="font-sans normal-case tracking-normal text-[15px] font-semibold" onClick={() => onOpen(s)}>
            {sessionMinutes(s)} min · {s.blocks.length} drills
          </Button>
        </div>
        <span className="sr-only">{drills} drills</span>
      </div>
    </section>
  )
}

function PitchBackdrop() {
  return (
    <svg className="absolute inset-0 h-full w-full" viewBox="0 0 390 640" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <defs>
        <radialGradient id="flood" cx="50%" cy="0%" r="90%"><stop offset="0" stopColor="#2b3a30" /><stop offset="1" stopColor="#0b0d0c" /></radialGradient>
      </defs>
      <rect width="390" height="640" fill="url(#flood)" />
      <g fill="none" stroke="rgba(255,255,255,.18)" strokeWidth="2">
        <line x1="-20" y1="420" x2="410" y2="380" />
        <ellipse cx="200" cy="400" rx="140" ry="40" transform="rotate(-5 200 400)" />
      </g>
      <text x="380" y="330" textAnchor="end" fontFamily="Archivo Variable" fontWeight="900" fontSize="300" fill="none" stroke="rgba(255,255,255,.08)" strokeWidth="2">7</text>
    </svg>
  )
}

// ---------- kit check ----------
function KitTile({ item, packed, onToggle }: { item: KitItem; packed: boolean; onToggle: () => void }) {
  const photo = usePersonalImage(item.photo ?? '__none__')
  return (
    <button onClick={onToggle} aria-pressed={packed}
      className="relative overflow-hidden rounded-[20px] bg-surface pb-2.5 text-center transition-transform duration-150 ease-snap active:scale-[0.97]">
      <div className="grid aspect-square place-items-center overflow-hidden">
        {photo ? <img src={photo} alt="" className="h-full w-full object-cover" /> : <KitGlyph id={item.id} />}
      </div>
      <div className="mt-2 font-display text-[11px] uppercase tracking-wide">{item.label}</div>
      {packed && <span className="pop absolute top-2 right-2 grid h-6 w-6 place-items-center rounded-full bg-red text-xs font-black text-white shadow">✓</span>}
    </button>
  )
}

export function KitCheck({ session }: { session: Session }) {
  const items = kitFor(session)
  const [packed, setPacked] = useState(() => loadPacked(dateKey()))
  return (
    <>
      <SectionHead title="Kit check" right={`${items.filter((i) => packed.has(i.id)).length} of ${items.length} packed`} />
      <div className="grid grid-cols-4 gap-2">
        {items.map((it) => <KitTile key={it.id} item={it} packed={packed.has(it.id)} onToggle={() => setPacked(new Set(togglePacked(dateKey(), it.id)))} />)}
      </div>
    </>
  )
}

// ---------- today's drills ----------
export function DrillCard({ name, meta, cue, category, onOpen, fluid = false }: { i?: number; name: string; meta: string; cue: boolean; category?: string; onOpen: () => void; fluid?: boolean; football?: boolean }) {
  const label = category ? CATEGORY_LABELS[category] ?? category : null
  return (
    <button onClick={onOpen} className={`${fluid ? 'w-full' : 'w-[200px] shrink-0 snap-start'} relative block aspect-[3/4] overflow-hidden rounded-[22px] bg-surface text-left transition-transform duration-150 ease-snap active:scale-[0.98]`}>
      <img src={drillPhoto(category)} alt="" data-fade loading="lazy" className="absolute inset-0 h-full w-full object-cover" />
      <div className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(0,0,0,.35)_0%,rgba(0,0,0,0)_30%,rgba(0,0,0,.15)_50%,rgba(0,0,0,.9)_100%)]" />
      <div className="absolute top-2.5 left-2.5 flex flex-wrap gap-1.5">
        {label && <Pill tone="glass">{label}</Pill>}
        {cue && <Pill tone="red">Voice cues</Pill>}
      </div>
      <div className="absolute inset-x-3.5 bottom-3.5 text-white">
        <h3 className="m-0 mb-1 font-display text-sm uppercase leading-tight">{name}</h3>
        <p className="m-0 text-sm text-white/75">{meta}</p>
      </div>
    </button>
  )
}

// ---------- week strip ----------
export function WeekStrip({ today, kindFor }: { today: Today; kindFor?: (day: number) => string | null | undefined }) {
  return (
    <div className="grid grid-cols-7 gap-1.5">
      {DAY_SHORT.map((d, day) => {
        const slots = today.plan.slots.filter((s) => s.day === day)
        const planned = slots.find((s) => s.kind === 'game') ?? slots[0]
        // a day changed by hand shows what it holds now (a game stays a game)
        const override = planned?.kind === 'game' || !kindFor ? undefined : kindFor(day)
        const main = override === undefined ? planned : override === null ? undefined : { ...planned, day, kind: override as typeof planned.kind }
        const isToday = day === today.day
        const base = 'mt-1.5 grid h-12 place-items-center rounded-[14px] font-display text-[11px] uppercase'
        let cell
        if (!main) cell = <div className={`${base} border-[1.5px] border-dashed border-white/15 text-muted`}>Rest</div>
        else if (main.kind === 'game') cell = <div className={`${base} bg-red text-white`}>Game</div>
        else if (main.kind === 'football') cell = <div className={`${base} bg-surface-2`}><BallIcon className="h-7 w-7" /></div>
        else cell = <div className={`${base} bg-surface-2 text-muted`}>{main.kind === 'recovery' ? 'Rec' : main.kind === 'cheer' ? 'Chr' : main.kind}</div>
        return (
          <div key={day} className="text-center">
            <div className="font-display text-[11px] text-muted">{d}</div>
            <div className={isToday ? 'rounded-[16px] ring-2 ring-white ring-offset-2 ring-offset-bg' : ''}>{cell}</div>
          </div>
        )
      })}
    </div>
  )
}



export function Home(p: Props) {
  const s = p.today.session
  const drills = s.blocks.filter((b) => b.kind === 'drill')
  const games = p.today.plan.slots.filter((x) => x.kind === 'game').length
  const [drill, setDrill] = useState<Block | null>(null)
  return (
    <main className="mx-auto max-w-xl px-4 pb-32">
      <Hero today={p.today} onKickOff={p.onKickOff} onOpen={p.onOpen} onSettings={p.onSettings} onCustomise={p.onAddClips} streak={p.streak} onStreak={p.onPlayer} />
      {p.active && (
        <section aria-label="Unfinished session" className="rise mt-6 flex items-center gap-3 rounded-[22px] border border-red/50 bg-surface p-4">
          <div className="min-w-0 flex-1">
            <p className="m-0 font-display text-xs uppercase text-red-ink">Unfinished session</p>
            <p className="m-0 mt-1 truncate font-semibold">{p.active.title}</p>
          </div>
          <button onClick={p.active.onDiscard} className="min-h-11 shrink-0 px-2 text-sm font-semibold text-muted">Discard</button>
          <Button size="sm" onClick={p.active.onResume}>Resume</Button>
        </section>
      )}
      {p.makeup && p.streak && (
        <section aria-label="Make up a missed session" className="rise mt-6 rounded-[22px] border border-[#ff7a1f]/50 bg-surface p-4">
          <div className="flex items-center gap-2">
            <FlameIcon className="h-5 w-5" />
            <p className="m-0 font-display text-xs uppercase text-[#ff9a1f]">Make up for {p.makeup.missed}</p>
          </div>
          <p className="m-0 mt-1.5 font-semibold">An extra session (20+ min) by {p.makeup.by} keeps your {p.streak.days}-day streak.</p>
          <div className="mt-3 flex flex-col gap-2">
            {p.makeup.picks.map((s) => (
              <button key={s.id} onClick={() => p.onOpen(s)}
                className="flex min-h-12 items-center justify-between gap-3 rounded-2xl bg-surface-2 px-4 text-left transition-transform duration-150 ease-snap active:scale-[0.98]">
                <span className="font-semibold">{s.title}</span>
                <span className="shrink-0 font-display text-xs tabular-nums text-muted">{sessionMinutes(s)} min</span>
              </button>
            ))}
          </div>
        </section>
      )}
      <HowItWorks onWeek={p.onWeek} />

      <SectionHead title="Today's drills" right={<button onClick={p.onAllSessions} className="text-sm font-semibold text-muted">All sessions</button>} />
      <div className="no-scrollbar -mx-4 flex snap-x scroll-px-4 gap-3 overflow-x-auto px-4">
        {drills.slice(0, 5).map((b, i) => (
          <DrillCard key={b.name} i={i} category={b.category} name={b.name.replace(/\s*\([^)]*\)/g, '')} cue={!!b.cue}
            meta={`${b.rounds || 1} × ${b.reps ? fmtReps(b.reps) : `${b.work}s`}${b.rest ? ` · rest ${b.rest}s` : ''}`} football={s.type === 'football'} onOpen={() => setDrill(b)} />
        ))}
      </div>

      <SectionHead title="This week" right={`${games} game${games === 1 ? '' : 's'} week`} />
      <WeekStrip today={p.today} kindFor={p.dayKind} />
      {drill && <DrillSheet block={drill} onClose={() => setDrill(null)} onSession={() => { setDrill(null); p.onOpen(s) }} />}
    </main>
  )
}

// First visit only: three steps that explain the app, then it gets out of the way.
function HowItWorks({ onWeek }: { onWeek?: () => void }) {
  const [seen, setSeen] = useState(() => load<boolean>('intro:seen', false))
  if (seen) return null
  const close = () => { save('intro:seen', true); setSeen(true) }
  const steps: [string, string][] = [
    ['Set your week', 'Tap your game days.'],
    ['Kick off', 'Volume up: drills are called out loud.'],
    ['Check in', 'Tap how hard it was.'],
  ]
  return (
    <section aria-label="How it works" className="rise mt-6 rounded-[22px] bg-surface p-4">
      <div className="flex items-baseline justify-between">
        <h2 className="m-0 font-display text-base uppercase">How it works</h2>
        <button onClick={close} className="min-h-10 px-1 text-sm font-semibold text-muted">Got it</button>
      </div>
      <ol className="m-0 mt-2 flex list-none flex-col gap-3 p-0">
        {steps.map(([title, text], i) => (
          <li key={title} className="flex gap-3">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-red font-display text-sm text-on-red">{i + 1}</span>
            <span><strong className="block">{title}</strong><span className="text-sm text-muted">{text}</span></span>
          </li>
        ))}
      </ol>
      {onWeek && <button onClick={() => { close(); onWeek() }} className="mt-4 min-h-12 w-full rounded-full bg-surface-2 font-display text-sm uppercase">Set my week</button>}
    </section>
  )
}
