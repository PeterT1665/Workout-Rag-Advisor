import { useState } from 'react'
import { Button, Pill, SectionHead } from '../components/ui'
import { BallIcon, BrushIcon, GearIcon, KitGlyph } from '../components/icons'
import { usePersonalImage } from '../lib/personal'
import { DEMO } from '../lib/demo'
import { useProfile } from '../lib/profile'
import { kitFor, loadPacked, togglePacked, type KitItem } from '../lib/kit'
import { fmtReps, sessionMinutes } from '../lib/logic'
import { CATEGORY_LABELS, drillPhoto } from '../lib/library'
import { PlayerCard } from '../components/PlayerCard'
import { cardStatsFor } from '../lib/cardStats'
import { load, save } from '../lib/storage'
import { DAYS, DAY_SHORT, type SlotKind } from '../lib/weekplan'
import type { Today } from '../lib/today'
import type { Session } from '../lib/types'

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
}

const SLOT_LABEL: Record<SlotKind, string> = { game: 'Game day', football: 'Football', gym: 'Gym', recovery: 'Recovery', cheer: 'Cheer' }
const dateKey = () => new Date().toISOString().slice(0, 10)

// ---------- hero ----------
// The hero's text slides in once per app open, not every time Home is shown again.
let heroIntroDone = false

export function Hero({ today, onKickOff, onOpen, onSettings, onCustomise, desktop = false }: Pick<Props, 'today' | 'onKickOff' | 'onOpen' | 'onSettings'> & { onCustomise?: () => void; desktop?: boolean }) {
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
        <span className="font-display text-sm uppercase leading-[1.05] text-white [text-shadow:0_1px_8px_rgba(0,0,0,.6)]">Football Personal<br />Coach</span>
        <span className="flex items-center gap-2">
          <Pill tone="glass">{sample ? 'Sample data' : date}</Pill>
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
export function WeekStrip({ today }: { today: Today }) {
  return (
    <div className="grid grid-cols-7 gap-1.5">
      {DAY_SHORT.map((d, day) => {
        const slots = today.plan.slots.filter((s) => s.day === day)
        const main = slots.find((s) => s.kind === 'game') ?? slots[0]
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

// ---------- mindset ----------
export function MindsetCard({ i, quote, hasClips, onPlayClip, onAddClips, fluid = false }: { i: number; quote: string; hasClips: boolean; onPlayClip: () => void; onAddClips: () => void; fluid?: boolean }) {
  const by = useProfile().idol
  const photo = usePersonalImage(`mindset-${(i % 2) + 1}.jpg`)
  return (
    <div className={`${fluid ? 'w-full' : 'w-[300px] shrink-0 snap-start'} overflow-hidden rounded-[20px] bg-surface`}>
      <div className="relative h-[150px] bg-[linear-gradient(160deg,#3a0a12,#0b0d0c)]">
        {photo && <img src={photo} alt="" data-fade className="h-full w-full object-cover object-[50%_30%]" />}
        {i === 0 && (hasClips || !DEMO) && (
          <button onClick={hasClips ? onPlayClip : onAddClips}
            className="absolute top-2.5 left-2.5 rounded-full bg-red px-3 py-1 font-display text-[11px] uppercase tracking-wider text-white active:scale-95">
            {hasClips ? '▶ Clip' : '+ Add clips'}
          </button>
        )}
      </div>
      <div className="p-3.5">
        <p className="m-0 font-display text-sm normal-case leading-snug">“{quote}”</p>
        {by && <p className="m-0 mt-1 text-[13px] text-muted">{by}</p>}
      </div>
    </div>
  )
}

// ---------- player card tile ----------
export function CardTile({ onPlayer }: { onPlayer: () => void }) {
  const photo = usePersonalImage('card.jpg')
  const me = useProfile()
  return (
    <button onClick={onPlayer} className="flex w-full items-center gap-5 rounded-[20px] bg-surface p-4 text-left transition-transform duration-150 ease-snap active:scale-[0.99]">
      <div className="w-[118px] shrink-0"><PlayerCard p={me} stats={cardStatsFor(me)} photo={photo} /></div>
      <div className="min-w-0">
        <div className="font-display text-lg uppercase leading-tight">{me.name}</div>
        <p className="m-0 mt-1 text-sm text-muted">{me.sample ? 'Example ratings.' : me.ratings ? `Rated ${me.ratings.date}` : 'Not rated yet.'}</p>
        <span className="mt-3 inline-block font-display text-xs uppercase text-red-ink">Open player →</span>
      </div>
    </button>
  )
}

export function Home(p: Props) {
  const s = p.today.session
  const drills = s.blocks.filter((b) => b.kind === 'drill')
  const games = p.today.plan.slots.filter((x) => x.kind === 'game').length
  return (
    <main className="mx-auto max-w-xl px-4 pb-32">
      <Hero today={p.today} onKickOff={p.onKickOff} onOpen={p.onOpen} onSettings={p.onSettings} onCustomise={p.onAddClips} />
      <InstallGuide />
      <HowItWorks onWeek={p.onWeek} />

      <SectionHead title="Today's drills" right={<button onClick={p.onAllSessions} className="text-sm font-semibold text-muted">All sessions</button>} />
      <div className="no-scrollbar -mx-4 flex snap-x scroll-px-4 gap-3 overflow-x-auto px-4">
        {drills.slice(0, 5).map((b, i) => (
          <DrillCard key={b.name} i={i} category={b.category} name={b.name.replace(/\s*\([^)]*\)/g, '')} cue={!!b.cue}
            meta={`${b.rounds || 1} × ${b.reps ? fmtReps(b.reps) : `${b.work}s`}${b.rest ? ` · rest ${b.rest}s` : ''}`} football={s.type === 'football'} onOpen={() => p.onOpen(s)} />
        ))}
      </div>

      <SectionHead title="This week" right={`${games} game${games === 1 ? '' : 's'} week`} />
      <WeekStrip today={p.today} />
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

// iPhone Safari only: until the app is on the Home Screen, show how to add it (works offline, full screen, voice cues).
function InstallGuide() {
  const [hidden, setHidden] = useState(() => {
    if (typeof navigator === 'undefined') return true
    const ios = /iP(hone|od|ad)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
    if (!ios) return true
    const installed = (navigator as Navigator & { standalone?: boolean }).standalone === true || (typeof matchMedia === 'function' && matchMedia('(display-mode: standalone)').matches)
    return installed || load<number>('install:snooze', 0) > Date.now()
  })
  if (hidden) return null
  const steps: [string, string][] = [['1', 'Tap Share (the square with an arrow).'], ['2', 'Tap Add to Home Screen. If it is missing, tap More.'], ['3', 'Tap Add, then open it from your Home Screen.']]
  return (
    <section aria-label="Add to Home Screen" className="rise mt-6 rounded-[22px] border border-red/40 bg-surface p-4">
      <div className="flex items-baseline justify-between">
        <h2 className="m-0 font-display text-base uppercase">Add to Home Screen</h2>
        <button onClick={() => { save('install:snooze', Date.now() + 3 * 864e5); setHidden(true) }} className="min-h-10 px-1 text-sm font-semibold text-muted">Later</button>
      </div>
      <p className="m-0 mt-1 mb-3 text-sm text-muted">Full screen, works offline at the field.</p>
      <ol className="m-0 flex list-none flex-col gap-2.5 p-0">
        {steps.map(([n, t]) => (
          <li key={n} className="flex items-center gap-3">
            <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full border border-white/20 font-display text-xs">{n}</span>
            <span className="text-sm">{t}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
