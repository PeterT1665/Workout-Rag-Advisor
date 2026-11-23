import { useState, type CSSProperties } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { PlayerCard, SHIELD, finishFor, overall, type CardStats } from './PlayerCard'
import { TESTS, nextRatingDue } from '../lib/testday'
import type { Profile } from '../lib/profile'
import { locale, t } from '../i18n'

const fmtDay = (iso: string) => { const [y, m, d] = iso.split('-').map(Number); return new Date(y, m - 1, d).toLocaleDateString(locale(), { day: 'numeric', month: 'short' }) }
const hide: CSSProperties = { backfaceVisibility: 'hidden', WebkitBackfaceVisibility: 'hidden' }

// Your card: tap to turn it over and see the rating session behind the numbers.
export function FlipCard({ p, stats, photo }: { p: Profile; stats: CardStats | null; photo?: string | null }) {
  const [back, setBack] = useState(false)
  const reduce = useReducedMotion()
  return (
    <button type="button" onClick={() => setBack((b) => !b)} aria-pressed={back}
      aria-label={back ? t('Show the front of your card') : t('Turn your card over: your rating session results')}
      className="block w-full cursor-pointer touch-manipulation select-none [perspective:1400px] [-webkit-tap-highlight-color:transparent] focus-visible:outline-none">
      <motion.div className="relative w-full [transform-style:preserve-3d]" initial={false}
        animate={{ rotateY: back ? 180 : 0, scale: 1 }} whileTap={reduce ? undefined : { scale: 0.97 }}
        transition={reduce ? { duration: 0 } : { rotateY: { type: 'spring', stiffness: 140, damping: 18, mass: 0.9 }, scale: { type: 'spring', stiffness: 500, damping: 30 } }}>
        <div style={hide}><PlayerCard p={p} stats={stats} photo={photo} /></div>
        <div className="absolute inset-0" style={{ ...hide, transform: 'rotateY(180deg)' }} aria-hidden={!back}><CardBack p={p} stats={stats} /></div>
      </motion.div>
    </button>
  )
}

// The back: every test from the last rating session, the raw result, the rating it gave and the change since the one before.
export function CardBack({ p, stats }: { p: Profile; stats: CardStats | null }) {
  const f = finishFor(stats ? overall(stats) : null)
  const r = p.ratings, prev = p.prevRatings
  const due = nextRatingDue(r)
  return (
    <div className="@container relative aspect-[0.72] w-full text-left">
      <div className="absolute inset-0 drop-shadow-[0_12px_24px_rgba(0,0,0,.45)]">
        <div className="absolute inset-0" style={{ clipPath: SHIELD, background: f.outer }} />
        <div className="absolute inset-[2.2%] overflow-hidden" style={{ clipPath: SHIELD, background: f.inner, color: f.ink }}>
          <div className="absolute inset-0 bg-[repeating-linear-gradient(55deg,rgba(255,255,255,.18)_0_2px,transparent_2px_16px)] [mask-image:linear-gradient(135deg,#000_0%,transparent_45%)]" />
          <div className="absolute inset-x-[11%] top-[10%] flex flex-col items-center border-b pb-[1.2cqw] text-center" style={{ borderColor: f.line }}>
            <span className="w-full truncate font-display text-[5.2cqw] uppercase leading-none">{t('Rating session')}</span>
            <span className="mt-[0.8cqw] whitespace-nowrap font-display text-[3.8cqw] opacity-75">{r ? fmtDay(r.date) : '--'}</span>
          </div>
          {r && !p.sample ? (
            <div className="absolute inset-x-[11%] top-[21.5%]">
              {TESTS.map((test, i) => {
                const raw = r.raw[test.stat], score = r.stats[test.stat], before = prev?.stats[test.stat]
                const diff = score !== undefined && before !== undefined ? score - before : null
                return (
                  <div key={test.stat} className="flex items-center gap-[2.4cqw] py-[1.25cqw]" style={{ borderTop: i ? `1px solid ${f.line}` : undefined }}>
                    <span className="w-[11cqw] shrink-0 font-display text-[4.2cqw] opacity-75">{test.stat}</span>
                    <span className="min-w-0 flex-1 leading-tight">
                      <span className="block truncate text-[3.7cqw] font-semibold">{test.name}</span>
                      <span className="block font-display text-[4.6cqw] tabular-nums">{raw !== undefined ? `${raw} ${test.unit}` : t('Skipped')}</span>
                    </span>
                    {diff !== null && diff !== 0 && <span className="shrink-0 font-display text-[3.6cqw] tabular-nums opacity-75">{diff > 0 ? `▲${diff}` : `▼${-diff}`}</span>}
                    <span className="w-[9cqw] shrink-0 text-right font-display text-[6.6cqw] tabular-nums leading-none">{score ?? '--'}</span>
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="absolute inset-x-[10%] top-[30%] m-0 text-center text-[5cqw] font-semibold leading-snug">
              {p.sample ? t('Example card. Your own rating session results show here: sprint time, passes, jump and more.') : t('No rating session yet. Do one and every result shows here.')}
            </p>
          )}
          <div className="absolute inset-x-0 bottom-[8%] text-center font-display text-[4cqw] uppercase opacity-70">
            {due && !p.sample ? t('Next rating {date}', { date: fmtDay(due) }) : t('Tap to turn back')}
          </div>
        </div>
      </div>
    </div>
  )
}
