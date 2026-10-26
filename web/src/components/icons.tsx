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
