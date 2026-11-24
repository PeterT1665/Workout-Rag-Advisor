import { FINISH, SHIELD } from './PlayerCard'

// A small gold, silver or bronze card: the number and a word. Used where the rating is explained.
export function MiniCard({ tier, value, label, className = '' }: { tier: 'gold' | 'silver' | 'bronze'; value: number | string; label?: string; className?: string }) {
  const f = FINISH[tier]
  return (
    <div className={`@container relative aspect-[0.72] drop-shadow-[0_8px_16px_rgba(0,0,0,.4)] ${className}`}>
      <div className="absolute inset-0" style={{ clipPath: SHIELD, background: f.outer }} />
      <div className="absolute inset-[3%] flex flex-col items-center justify-center overflow-hidden text-center" style={{ clipPath: SHIELD, background: f.inner, color: f.ink }}>
        <div className="absolute inset-0 bg-[repeating-linear-gradient(55deg,rgba(255,255,255,.18)_0_2px,transparent_2px_14px)] [mask-image:linear-gradient(135deg,#000_0%,transparent_50%)]" />
        <span className="relative font-display text-[34cqw] leading-none tabular-nums">{value}</span>
        {label && <span className="relative mt-[4cqw] px-[8cqw] font-display text-[10cqw] uppercase leading-tight opacity-80">{label}</span>}
      </div>
    </div>
  )
}
