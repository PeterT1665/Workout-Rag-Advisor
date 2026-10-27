import { Panel, SectionHead } from '../components/ui'
import { CardTile, DrillCard, Hero, KitCheck, MindsetCard } from './Home'
import { DAYS } from '../lib/weekplan'
import { sessionForSlot } from '../lib/today'
import data from '../data/sessions.json'
import { loadHistory, loadQuotes, weekLoads } from '../lib/storage'
import { LoadChart } from '../components/LoadChart'
import { DEMO } from '../lib/demo'
import type { Today } from '../lib/today'
import type { Session } from '../lib/types'

// Laptop home: planning at a glance. The phone keeps the in-session experience.
export function Dashboard(p: {
  today: Today
  hasClips: boolean
  onKickOff: (s: Session) => void
  onOpen: (s: Session) => void
  onSettings: () => void
  onWeek: () => void
  onSessions: () => void
  onPlayer: () => void
  onPlayClip: () => void
  onAddClips: () => void
}) {
  const s = p.today.session
  const drills = s.blocks.filter((b) => b.kind === 'drill').slice(0, 3)
  const quotes = loadQuotes().slice(0, 3)
  const hour = new Date().getHours()
  const greet = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'
  return (
    <main className="px-10 pt-9 pb-12">
      <header className="mb-6 flex items-end justify-between">
        <div>
          <p className="m-0 text-muted">{new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' })}</p>
          <h1 className="m-0 mt-1 font-display text-4xl uppercase">{greet}, Peter</h1>
          {DEMO && <p className="m-0 mt-2 inline-block rounded-full bg-surface-2 px-3 py-1 text-sm">Public demo · sample data, not real training history</p>}
        </div>
      </header>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-8 h-[460px]">
          <Hero desktop today={p.today} onKickOff={p.onKickOff} onOpen={p.onOpen} onSettings={p.onSettings} />
        </div>
        <Panel className="col-span-4 flex flex-col gap-0 [&>button:last-child]:mt-4">
          <div className="flex items-baseline justify-between">
            <h2 className="m-0 font-display text-base uppercase">This week</h2>
            <button onClick={p.onWeek} className="text-sm font-semibold text-muted hover:text-ink">Plan →</button>
          </div>
          <p className="m-0 mt-2 text-sm text-muted">{p.today.plan.reasons[0]}</p>
          <ol className="m-0 mt-4 flex list-none flex-col gap-1.5 p-0">
            {DAYS.map((name, day) => {
              const slot = p.today.plan.slots.find((x) => x.day === day && x.kind === 'game') ?? p.today.plan.slots.find((x) => x.day === day)
              const sess = slot ? sessionForSlot(slot, p.today.plan, data.sessions as Session[]) : null
              const label = !slot ? 'Rest' : slot.kind === 'game' ? 'Game' : sess ? sess.title : slot.kind === 'cheer' ? 'Cheer' : 'Gym'
              const isToday = day === p.today.day
              return (
                <li key={day} className={`flex items-center gap-3 rounded-xl px-3 py-1.5 ${isToday ? 'bg-surface-2' : ''}`}>
                  <span className="w-9 font-display text-[11px] uppercase text-muted">{name.slice(0, 3)}</span>
                  <span className={`truncate text-sm ${slot?.kind === 'game' ? 'font-semibold text-red-ink' : !slot ? 'text-muted' : ''}`}>{label}</span>
                </li>
              )
            })}
          </ol>
                    <button onClick={p.onWeek}
            className="mt-auto pt-0 min-h-12 rounded-full bg-surface-2 font-display text-sm uppercase tracking-wide transition-transform duration-150 ease-snap active:scale-[0.97]">
            Set game days
          </button>
        </Panel>

        <section className="col-span-8">
          <SectionHead title="Today's drills" right={<button onClick={p.onSessions} className="text-sm font-semibold text-muted hover:text-ink">All sessions</button>} />
          <div className="grid grid-cols-3 gap-4">
            {drills.map((b, i) => (
              <DrillCard key={b.name} fluid i={i} name={b.name.replace(/\s*\([^)]*\)/g, '')} cue={!!b.cue}
                meta={`${b.rounds || 1} × ${b.work}s${b.rest ? ` · rest ${b.rest}s` : ''}`} football={s.type === 'football'} onOpen={() => p.onOpen(s)} />
            ))}
          </div>
        </section>
        <section className="col-span-4"><KitCheck session={s} /></section>

        <section className="col-span-5">
          <SectionHead title="Your number 7" right={<button onClick={p.onPlayer} className="text-sm font-semibold text-muted hover:text-ink">Player card</button>} />
          <CardTile onPlayer={p.onPlayer} />
        </section>
        <section className="col-span-7">
          <SectionHead title="Load" right={<button onClick={p.onPlayer} className="text-sm font-semibold text-muted hover:text-ink">History</button>} />
          <LoadChart loads={weekLoads(loadHistory())} />
        </section>
        <section className="col-span-12">
          <SectionHead title="Mindset" right={<button onClick={p.onAddClips} className="text-sm font-semibold text-muted hover:text-ink">Clips & quotes</button>} />
          <div className="grid grid-cols-3 gap-4">
            {quotes.map((q, i) => <MindsetCard key={q} fluid i={i} quote={q} hasClips={p.hasClips} onPlayClip={p.onPlayClip} onAddClips={p.onAddClips} />)}
          </div>
        </section>
      </div>
    </main>
  )
}
