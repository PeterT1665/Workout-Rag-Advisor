import { motion } from 'motion/react'
import { BallIcon } from './icons'

export type Tab = 'home' | 'week' | 'player' | 'coach'

export function TabBar({ tab, onTab, onKickOff }: { tab: Tab; onTab: (t: Tab) => void; onKickOff: () => void }) {
  const item = (t: Tab, label: string) => (
    <button onClick={() => onTab(t)} aria-current={tab === t ? 'page' : undefined}
      className={`relative min-h-12 font-display text-[11px] uppercase tracking-wider transition-colors duration-150 ${tab === t ? 'text-white' : 'text-white/50'}`}>
      {label}
      {tab === t && <motion.span layoutId="tab-dot" className="absolute bottom-1 left-1/2 h-1 w-5 -translate-x-1/2 rounded-full bg-red"
        transition={{ type: 'spring', duration: 0.35, bounce: 0.2 }} />}
    </button>
  )
  return (
    <nav className="fixed inset-x-3 bottom-[max(12px,env(safe-area-inset-bottom))] z-40 mx-auto grid h-[70px] max-w-[540px] grid-cols-[1fr_1fr_84px_1fr_1fr] items-center rounded-[26px] bg-[#161817]/90 shadow-[0_10px_30px_rgba(0,0,0,.4)] backdrop-blur-xl">
      {item('home', 'Home')}
      {item('week', 'Week')}
      <button onClick={onKickOff} aria-label="Kick off today's session"
        className="-mt-9 grid h-[70px] w-[70px] place-items-center justify-self-center rounded-full border-4 border-bg bg-red shadow-[0_8px_20px_-6px_rgba(0,0,0,.6)] transition-transform duration-150 ease-snap active:scale-95">
        <BallIcon className="h-11 w-11" />
      </button>
      {item('player', 'Player')}
      {item('coach', 'Coach')}
    </nav>
  )
}
