import { useState } from 'react'
import { DAY_SHORT, todayIndex } from '../lib/weekplan'

// This week's training load per day (minutes x effort). One series, so no legend;
// the title names it. Tap or hover a bar for its value.
export function LoadChart({ loads }: { loads: number[] }) {
  const [hover, setHover] = useState<number | null>(null)
  const max = Math.max(...loads, 1)
  const total = loads.reduce((a, b) => a + b, 0)
  const today = todayIndex()
  return (
    <figure className="m-0 rounded-[20px] bg-surface p-4">
      <figcaption className="flex items-baseline justify-between">
        <span className="font-semibold">Training load this week</span>
        <span className="font-display text-xl">{total}</span>
      </figcaption>
      <div className="relative mt-8 grid h-36 grid-cols-7 items-end gap-2" role="img"
        aria-label={`Load by day: ${loads.map((l, i) => `${DAY_SHORT[i]} ${l}`).join(', ')}`}>
        {/* recessive baseline */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-[color-mix(in_oklab,var(--text)_14%,transparent)]" />
        {loads.map((l, i) => (
          <button key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}
            onClick={() => setHover(hover === i ? null : i)} aria-label={`${DAY_SHORT[i]}: ${l}`}
            className="relative flex h-full items-end justify-center">
            {(hover === i || (l > 0 && hover === null)) && (
              <span className="absolute -top-1 text-xs font-semibold tabular-nums text-muted" style={{ bottom: `calc(${(l / max) * 100}% + 6px)`, top: 'auto' }}>{l}</span>
            )}
            <span className={`w-full max-w-7 rounded-t-[4px] transition-[height] duration-300 ease-snap ${l ? (i === today ? 'bg-red' : 'bg-[color-mix(in_oklab,var(--red)_70%,var(--surface))]') : 'bg-surface-2'}`}
              style={{ height: l ? `${Math.max(6, (l / max) * 100)}%` : '4px' }} />
          </button>
        ))}
      </div>
      <div className="mt-2 grid grid-cols-7 gap-2 text-center font-display text-[11px] text-muted">
        {DAY_SHORT.map((d, i) => <span key={i} className={i === today ? 'text-ink' : ''}>{d}</span>)}
      </div>
      {total === 0 && <p className="m-0 mt-3 text-sm text-muted">Finish a session and do the 20-second check-in to see your load here.</p>}
    </figure>
  )
}
