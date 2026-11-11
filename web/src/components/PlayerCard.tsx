import type { CSSProperties } from 'react'
import { POSITIONS, type Profile } from '../lib/profile'

// Our own game-style player card (inspired by football game cards, not a copy: no club badges or game branding).
// Finish follows the overall rating: gold 75+, silver 65-74, bronze below; unrated players get their team colour.
export const CARD_STATS = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'] as const
export type CardStats = Record<(typeof CARD_STATS)[number], number>

const SHIELD = 'polygon(0% 9%, 9% 9%, 15% 3%, 85% 3%, 91% 9%, 100% 9%, 100% 88%, 50% 100%, 0% 88%)'

const FINISH = {
  gold: { outer: 'linear-gradient(160deg,#8a6a1f,#e9c96a 45%,#8a6a1f)', inner: 'linear-gradient(160deg,#fff3c4 0%,#e8cf7a 40%,#d4b45a 70%,#f3dd8f 100%)', ink: '#2b230f', line: 'rgba(43,35,15,.18)' },
  silver: { outer: 'linear-gradient(160deg,#6f7377,#d9dde0 45%,#6f7377)', inner: 'linear-gradient(160deg,#ffffff 0%,#d7dbde 40%,#b9bec2 70%,#e9ecee 100%)', ink: '#1f2326', line: 'rgba(31,35,38,.16)' },
  bronze: { outer: 'linear-gradient(160deg,#6b3f1d,#d39a63 45%,#6b3f1d)', inner: 'linear-gradient(160deg,#f6d2ae 0%,#d6a072 40%,#b97c4c 70%,#e8b98c 100%)', ink: '#2e1a0b', line: 'rgba(46,26,11,.18)' },
  team: { outer: 'linear-gradient(160deg,color-mix(in srgb,var(--red) 55%,#000),var(--red) 45%,color-mix(in srgb,var(--red) 55%,#000))', inner: 'linear-gradient(160deg,color-mix(in srgb,var(--red) 70%,#fff) 0%,var(--red) 45%,color-mix(in srgb,var(--red) 60%,#000) 100%)', ink: '#ffffff', line: 'rgba(255,255,255,.22)' },
}

export const overall = (s: CardStats) => Math.round(CARD_STATS.reduce((a, k) => a + s[k], 0) / CARD_STATS.length)

export function PlayerCard({ p, stats, photo, className = '' }: { p: Profile; stats: CardStats | null; photo?: string | null; className?: string }) {
  const ovr = stats ? overall(stats) : null
  const f = FINISH[ovr === null ? 'team' : ovr >= 75 ? 'gold' : ovr >= 65 ? 'silver' : 'bronze']
  const pos = POSITIONS.find((x) => x.id === p.positions[0])?.short ?? ''
  const foot = p.foot === 'both' ? 'Both feet' : `${p.foot[0].toUpperCase()}${p.foot.slice(1)} foot`
  return (
    <div className={`@container relative aspect-[0.72] w-full select-none ${className}`} role="img"
      aria-label={`${p.name || 'Player'} card${ovr !== null ? `, overall ${ovr}` : ', unrated'}`}>
      <div className="absolute inset-0 drop-shadow-[0_12px_24px_rgba(0,0,0,.45)]">
        <div className="absolute inset-0" style={{ clipPath: SHIELD, background: f.outer }} />
        <div className="absolute inset-[2.2%] overflow-hidden" style={{ clipPath: SHIELD, background: f.inner, color: f.ink } as CSSProperties}>
          {/* diagonal sheen stripes, top right */}
          <div className="absolute inset-0 bg-[repeating-linear-gradient(-55deg,rgba(255,255,255,.22)_0_2px,transparent_2px_16px)] [mask-image:linear-gradient(225deg,#000_0%,transparent_45%)]" />

          {/* photo, or a big outlined number when there is none */}
          <div className="absolute top-[9%] right-[4%] h-[44%] w-[60%]">
            {photo
              ? <img src={photo} alt="" className="h-full w-full object-cover object-top [mask-image:linear-gradient(to_bottom,#000_70%,transparent)]" />
              : <span className="absolute right-0 bottom-0 font-display text-[44cqw] leading-[.8] text-transparent" style={{ WebkitTextStroke: `0.5cqw ${f.line.replace(/[\d.]+\)$/, '.45)')}` }}>{p.number || ''}</span>}
          </div>

          {/* rating, position, number */}
          <div className="absolute top-[12%] left-[9%] flex flex-col items-center leading-none">
            <span className="font-display text-[17cqw] leading-[.85]">{ovr ?? '--'}</span>
            <span className="mt-[1.5cqw] font-display text-[7.5cqw]">{pos}</span>
            <span className="mt-[2.5cqw] font-display text-[5.5cqw] opacity-75">#{p.number || '--'}</span>
          </div>

          {/* name band */}
          <div className="absolute inset-x-[6%] top-[53%] border-y py-[1.4cqw] text-center" style={{ borderColor: f.line }}>
            <span className="block truncate font-display text-[9cqw] uppercase leading-none">{p.name || 'Your name'}</span>
          </div>

          {/* stats */}
          <div className="absolute inset-x-[12%] top-[64.5%] grid grid-cols-2 gap-x-[8cqw]">
            {[0, 1].map((col) => (
              <div key={col}>
                {CARD_STATS.slice(col * 3, col * 3 + 3).map((k, i) => (
                  <div key={k} className="flex items-baseline gap-[2cqw] py-[0.5cqw]" style={{ borderTop: i ? `1px solid ${f.line}` : undefined }}>
                    <span className="w-[10cqw] text-right font-display text-[6.6cqw] tabular-nums">{stats ? stats[k] : '--'}</span>
                    <span className="font-display text-[5.6cqw] opacity-80">{k}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>

          <div className="absolute inset-x-0 bottom-[6.5%] text-center font-display text-[4.2cqw] uppercase opacity-70">{foot}</div>
        </div>
      </div>
    </div>
  )
}

// Example ratings for the demo's sample player (overall 75: a gold card).
export const SAMPLE_STATS: CardStats = { PAC: 86, SHO: 79, PAS: 74, DRI: 84, DEF: 58, PHY: 71 }
