import { motion } from 'motion/react'
import { GearIcon } from './icons'

export type DeskTab = 'home' | 'week' | 'sessions' | 'player' | 'coach'

const ITEMS: { id: DeskTab; label: string; d: string }[] = [
  { id: 'home', label: 'Home', d: 'M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z' },
  { id: 'week', label: 'Week', d: 'M4 6h16v14H4zM4 10h16M9 3v4M15 3v4' },
  { id: 'sessions', label: 'Sessions', d: 'M5 5h14M5 12h14M5 19h9' },
  { id: 'player', label: 'Player', d: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0' },
  { id: 'coach', label: 'Coach', d: 'M4 5h16v11H8l-4 4z' },
]

export function Sidebar({ tab, onTab, onSettings, onKickOff }: { tab: DeskTab | null; onTab: (t: DeskTab) => void; onSettings: () => void; onKickOff: () => void }) {
  return (
    <aside className="sticky top-0 flex h-dvh w-[248px] shrink-0 flex-col border-r border-white/5 bg-[#0d0f0e] px-4 py-7 text-white">
      <div className="px-3 font-display text-lg uppercase leading-none">Sport Personal<br /><span className="text-red-ink">Coach</span></div>
      <nav className="mt-10 flex flex-col gap-1">
        {ITEMS.map((it) => {
          const on = tab === it.id
          return (
            <button key={it.id} onClick={() => onTab(it.id)} aria-current={on ? 'page' : undefined}
              className={`relative flex min-h-11 items-center gap-3 rounded-full px-4 text-left font-semibold transition-colors duration-150 ${on ? 'text-white' : 'text-white/55 hover:bg-white/5 hover:text-white'}`}>
              {on && <motion.span layoutId="side-pill" className="absolute inset-0 -z-0 rounded-full bg-white/10" transition={{ type: 'spring', duration: 0.35, bounce: 0.15 }} />}
              <svg viewBox="0 0 24 24" className="relative h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={it.d} /></svg>
              <span className="relative">{it.label}</span>
            </button>
          )
        })}
      </nav>
      <button onClick={onKickOff}
        className="mt-8 min-h-12 rounded-full bg-red font-display text-sm uppercase tracking-wide text-white transition-transform duration-150 ease-snap active:scale-[0.97]">
        Kick off →
      </button>
      <div className="mt-auto">
        <div className="rounded-[20px] bg-white/5 p-4 text-sm text-white/70">
          <p className="m-0 font-semibold text-white">Training on the pitch?</p>
          <p className="m-0 mt-1">Open this same address on your phone. Sessions, timer and voice cues are built for it.</p>
        </div>
        <button onClick={onSettings} className="mt-3 flex min-h-11 w-full items-center gap-3 rounded-full px-4 font-semibold text-white/55 hover:bg-white/5 hover:text-white">
          <GearIcon /> Settings
        </button>
      </div>
    </aside>
  )
}
