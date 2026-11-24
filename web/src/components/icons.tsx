import { usePersonalImage } from '../lib/personal'

// Small inline icons used when a personal photo isn't available.
// Real ball: Peter's ball photo when available, otherwise the system football emoji (photo-real on iPhone).
export function BallIcon({ className = '' }: { className?: string }) {
  const photo = usePersonalImage('kit-ball.jpg')
  return photo
    ? <img src={photo} alt="" aria-hidden="true" className={`rounded-full object-cover ${className}`} />
    : (
      <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
        <circle cx="12" cy="12" r="10.5" fill="#f5f5f2" />
        <path d="M12 7.6l3.4 2.5-1.3 4H9.9l-1.3-4z" fill="#17181b" />
        <path d="M12 1.5v6.1M22.4 9.4l-7 .7M18.6 20.4l-4.5-6.3M5.4 20.4l4.5-6.3M1.6 9.4l7 .7" stroke="#17181b" strokeWidth="1.2" fill="none" />
        <circle cx="12" cy="12" r="10.5" fill="none" stroke="#17181b" strokeOpacity=".25" />
      </svg>
    )
}
export function KitGlyph({ id }: { id: string }) {
  const common = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
  return (
    <svg viewBox="0 0 24 24" className="h-10 w-10 text-ink" aria-hidden="true">
      {id === 'water' && <g {...common}><path d="M9 3h6M10 3v3l-2 3v11a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V9l-2-3V3" /><path d="M8 13h8" stroke="var(--red)" /></g>}
      {id === 'mat' && <g {...common}><rect x="3" y="8" width="18" height="8" rx="4" /><path d="M7 8v8" /></g>}
      {id === 'roller' && <g {...common}><rect x="3" y="9" width="18" height="6" rx="3" /><path d="M7 9v6M11 9v6M15 9v6" /></g>}
      {id === 'discs' && <g {...common}><path d="M4 16c0-4 3.6-7 8-7s8 3 8 7z" stroke="var(--red)" /><path d="M4 18h16" /></g>}
      {id === 'boots' && <g {...common}><path d="M3 15V8h6l3 4 8 1a2 2 0 0 1 1 2v1H3z" /><path d="M5 19v-2M9 19v-2M14 19v-2M18 19v-2" /></g>}
      {id === 'ball' && <circle cx="12" cy="12" r="9" {...common} />}
    </svg>
  )
}
export function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z" />
    </svg>
  )
}
export function MoreIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
      <circle cx="5" cy="12" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="19" cy="12" r="2" />
    </svg>
  )
}
export function ListIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M8 6h12M8 12h12M8 18h12" /><circle cx="4" cy="6" r="1" fill="currentColor" /><circle cx="4" cy="12" r="1" fill="currentColor" /><circle cx="4" cy="18" r="1" fill="currentColor" />
    </svg>
  )
}
export function DumbbellIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M6.5 6.5v11M17.5 6.5v11M3.5 9v6M20.5 9v6M6.5 12h11" />
    </svg>
  )
}
export function BrushIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14.5 4.5l5 5L10 19l-5-5z" />
      <path d="M5 14c-1.5 0-2.5 1.2-2.5 2.6 0 1.6-.8 2.4-1.5 2.9 2.5.6 6 .3 7-2.5" />
    </svg>
  )
}

// Streak flame: orange-to-red while there's a streak, grey at 0. Home adds a glow once today counts.
export type FlameTone = 'plan' | 'unbeaten' | 'wins'
// Fire colours: the plan streak burns orange; game streaks stand apart: unbeaten burns blue, a win streak purple.
export const FIRE: Record<FlameTone | 'off', readonly [string, string, string]> = {
  plan: ['#ff3b1f', '#ff9a1f', '#ffd23f'], unbeaten: ['#1460ff', '#38a8ff', '#b4ecff'], wins: ['#6a16ff', '#b45cff', '#f1cbff'], off: ['#7a7d84', '#a3a6ad', '#c9cbd0'],
}
export function FlameIcon({ lit = true, tone = 'plan', className = 'h-6 w-6' }: { lit?: boolean; tone?: FlameTone; className?: string }) {
  const [a, b, c] = FIRE[lit ? tone : 'off']
  const id = `flame-${lit ? tone : 'off'}`
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="1" x2="0" y2="0">
          <stop offset="0" stopColor={a} /><stop offset="1" stopColor={b} />
        </linearGradient>
      </defs>
      <path fill={`url(#${id})`} d="M12 1.8c.6 3.1-1 4.9-2.6 6.6C7.8 10.1 6 12 6 15a6 6 0 0 0 12 0c0-2.4-1.1-4.3-2.3-5.8-.2 1.3-.8 2.3-1.8 2.9.3-3.9-.9-7.6-1.9-10.3Z" />
      <path fill={c} d="M12 12.2c-.3 1.4-1.1 2.1-1.8 2.8-.6.6-1 1.2-1 2.1a2.8 2.8 0 0 0 5.6 0c0-1.2-.6-2.1-1.3-2.8-.2.6-.6 1-1.1 1.2.1-1.2 0-2.3-.4-3.3Z" />
    </svg>
  )
}

// Match review (the centre button after a game): a clipboard with a tick.
export function ReviewIcon({ className = 'h-9 w-9' }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="5" y="4" width="14" height="17" rx="2.5" />
      <path d="M9 4.5V3.5h6v1" />
      <path d="M8.5 12.5l2.5 2.5 4.5-5" />
    </svg>
  )
}
