import { useState } from 'react'
import { Button, Pill, SectionHead } from '../components/ui'
import { BallIcon, GearIcon, KitGlyph } from '../components/icons'
import { usePersonalImage } from '../lib/personal'
import { DEMO, QUOTE_BY } from '../lib/demo'
import { kitFor, loadPacked, togglePacked, type KitItem } from '../lib/kit'
import { sessionMinutes } from '../lib/logic'
import { loadQuotes } from '../lib/storage'
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
}

const SLOT_LABEL: Record<SlotKind, string> = { game: 'Game day', football: 'Football', gym: 'Gym', recovery: 'Recovery', cheer: 'Cheer' }
const dateKey = () => new Date().toISOString().slice(0, 10)

// ---------- hero ----------
export function Hero({ today, onKickOff, onOpen, onSettings, desktop = false }: Pick<Props, 'today' | 'onKickOff' | 'onOpen' | 'onSettings'> & { desktop?: boolean }) {
  const hero = usePersonalImage('hero.jpg')
  const s = today.session
  const drills = s.blocks.filter((b) => b.kind === 'drill').length
  const when = today.scheduled ? 'Today' : today.nextDay !== null ? `Next · ${DAYS[today.nextDay].slice(0, 3)}` : 'Anytime'
  const status = !today.scheduled && today.slot
    ? today.slot.kind === 'game' ? 'Game day. Keep the legs fresh, warm up properly.'
      : today.slot.kind === 'cheer' ? 'Cheer practice this morning.'
        : today.slot.kind === 'gym' ? 'Gym day. Gym sessions arrive with the planner.' : null
    : !today.scheduled ? 'Rest day. Train only if the legs feel good.' : null
  const date = new Date().toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })

  return (
    <section className={desktop
      ? 'relative h-full min-h-[460px] overflow-hidden rounded-[28px] bg-[#0b0d0c]'
      : 'relative -mx-4 -mt-[max(16px,env(safe-area-inset-top))] h-[640px] overflow-hidden rounded-b-[28px] bg-[#0b0d0c]'}>
      {hero
        ? <img src={hero} alt="" className={`absolute inset-0 h-full w-full object-cover ${desktop ? 'object-[50%_30%]' : 'object-[50%_22%]'}`} />
        : <PitchBackdrop />}
      <div className={desktop
        ? 'absolute inset-0 bg-[linear-gradient(90deg,rgba(10,12,11,.95)_0%,rgba(10,12,11,.7)_38%,rgba(10,12,11,0)_70%)]'
        : 'absolute inset-0 bg-[linear-gradient(to_bottom,rgba(10,12,11,.6)_0%,rgba(10,12,11,0)_22%,rgba(10,12,11,0)_55%,rgba(10,12,11,.96)_86%)]'} />
      {!desktop && <div className="absolute inset-x-4 top-[max(16px,env(safe-area-inset-top))] flex items-center justify-between">
        <span className="font-display text-sm uppercase leading-[1.05] text-white [text-shadow:0_1px_8px_rgba(0,0,0,.6)]">Sport Personal<br />Coach</span>
        <span className="flex items-center gap-2">
          <Pill tone="glass">{DEMO ? 'Sample data' : date}</Pill>
          <button onClick={onSettings} aria-label="Settings" className="grid h-10 w-10 place-items-center rounded-full bg-white/15 text-white backdrop-blur-md active:scale-95">
            <GearIcon />
          </button>
        </span>
      </div>}
      <div className={desktop ? 'absolute bottom-8 left-8 max-w-[64%] rise' : 'absolute inset-x-4 bottom-5 rise'}>
        <div className="flex flex-wrap gap-2">
          {desktop && <Pill tone="glass">{DEMO ? 'Sample data' : date}</Pill>}
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
export function DrillCard({ i, name, meta, cue, onOpen, fluid = false, football = true }: { i: number; name: string; meta: string; cue: boolean; onOpen: () => void; fluid?: boolean; football?: boolean }) {
  const found = usePersonalImage(`drill-${(i % 3) + 1}.jpg`)
  const photo = football ? found : null // pitch photos only on football drills, not on stretching
  return (
    <button onClick={onOpen} className={`${fluid ? 'w-full' : 'w-[250px] shrink-0 snap-start'} overflow-hidden rounded-[20px] bg-surface text-left transition-transform duration-150 ease-snap active:scale-[0.98]`}>
      <div className={`relative bg-[linear-gradient(160deg,#1f5533,#0f1d14)] ${photo ? 'h-[150px]' : 'h-[84px]'}`}>
        {photo && <img src={photo} alt="" className="h-full w-full object-cover" />}
        {cue && <Pill tone="red" className="absolute top-2.5 left-2.5">Voice cues</Pill>}
      </div>
      <div className="p-3.5">
        <h3 className="m-0 mb-1 font-display text-sm uppercase">{name}</h3>
        <p className="m-0 text-sm text-muted">{meta}</p>
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
  const photo = usePersonalImage(`mindset-${(i % 2) + 1}.jpg`)
  return (
    <div className={`${fluid ? 'w-full' : 'w-[300px] shrink-0 snap-start'} overflow-hidden rounded-[20px] bg-surface`}>
      <div className="relative h-[150px] bg-[linear-gradient(160deg,#3a0a12,#0b0d0c)]">
        {photo && <img src={photo} alt="" className="h-full w-full object-cover object-[50%_30%]" />}
        {i === 0 && (hasClips || !DEMO) && (
          <button onClick={hasClips ? onPlayClip : onAddClips}
            className="absolute top-2.5 left-2.5 rounded-full bg-red px-3 py-1 font-display text-[11px] uppercase tracking-wider text-white active:scale-95">
            {hasClips ? '▶ Clip' : '+ Add clips'}
          </button>
        )}
      </div>
      <div className="p-3.5">
        <p className="m-0 font-display text-sm normal-case leading-snug">“{quote}”</p>
        {QUOTE_BY && <p className="m-0 mt-1 text-[13px] text-muted">{QUOTE_BY}</p>}
      </div>
    </div>
  )
}

// ---------- player card tile ----------
export function CardTile({ onPlayer }: { onPlayer: () => void }) {
  const bg = usePersonalImage('card.jpg')
  const stats: [string, number][] = [['PAC', 82], ['DRI', 78], ['SHO', 71], ['WF', 58]]
  return (
    <button onClick={onPlayer} className="relative block h-[200px] w-full overflow-hidden rounded-[20px] bg-surface text-left active:scale-[0.99]">
      {bg && <img src={bg} alt="" className="absolute inset-0 h-full w-full object-cover object-[50%_30%]" />}
      <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(10,12,11,.97)_0%,rgba(10,12,11,.9)_50%,rgba(10,12,11,.25)_100%)]" />
      <div className="absolute inset-y-4 left-4 flex flex-col justify-between">
        <div>
          <div className="font-display text-[54px] leading-[.85] text-white">74</div>
          <Pill tone="red" className="mt-2">LW · Peter</Pill>
        </div>
        <div className="flex gap-3.5 font-display text-xs text-white">
          {stats.map(([k, v]) => <span key={k}>{k} <b className="text-red-ink">{v}</b></span>)}
        </div>
      </div>
      <span className="absolute top-4 right-4 rounded-full bg-black/50 px-2.5 py-1 text-[11px] text-white/80">Example until test day</span>
    </button>
  )
}

export function Home(p: Props) {
  const s = p.today.session
  const drills = s.blocks.filter((b) => b.kind === 'drill')
  const quotes = loadQuotes().slice(0, 3)
  const games = p.today.plan.slots.filter((x) => x.kind === 'game').length
  return (
    <main className="mx-auto max-w-xl px-4 pb-32">
      <Hero today={p.today} onKickOff={p.onKickOff} onOpen={p.onOpen} onSettings={p.onSettings} />
      <KitCheck session={s} />

      <SectionHead title="Today's drills" right={<button onClick={p.onAllSessions} className="text-sm font-semibold text-muted">All sessions</button>} />
      <div className="no-scrollbar -mx-4 flex snap-x scroll-px-4 gap-3 overflow-x-auto px-4">
        {drills.slice(0, 5).map((b, i) => (
          <DrillCard key={b.name} i={i} name={b.name.replace(/\s*\([^)]*\)/g, '')} cue={!!b.cue}
            meta={`${b.rounds || 1} × ${b.work}s${b.rest ? ` · rest ${b.rest}s` : ''}`} football={s.type === 'football'} onOpen={() => p.onOpen(s)} />
        ))}
      </div>

      <SectionHead title="This week" right={`${games} game${games === 1 ? '' : 's'} week`} />
      <WeekStrip today={p.today} />

      <SectionHead title="Mindset" right={<button onClick={p.onAddClips} className="text-sm font-semibold text-muted">Clips & quotes</button>} />
      <div className="no-scrollbar -mx-4 flex snap-x scroll-px-4 gap-3 overflow-x-auto px-4">
        {quotes.map((q, i) => <MindsetCard key={q} i={i} quote={q} hasClips={p.hasClips} onPlayClip={p.onPlayClip} onAddClips={p.onAddClips} />)}
      </div>

      <SectionHead title="Your number 7" right="Player card" />
      <CardTile onPlayer={p.onPlayer} />
    </main>
  )
}
