import { useEffect, useState } from 'react'
import { usePersonalImage } from '../lib/personal'
import type { Session } from '../lib/types'

// Kick-off moment: a short "KICK OFF" intro (rare event, so it gets the delight budget),
// then the mindset card and clip. Always dark and cinematic, whatever the system theme.
const INTRO_MS = 1100

const reducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches

export function Preroll({ session, quote, clipName, onGo }: { session: Session; quote: string | null; clipName: string | null; onGo: () => void }) {
  const bg = usePersonalImage('preroll.jpg')
  const [phase, setPhase] = useState<'intro' | 'leaving' | 'card'>('intro')
  const nothingToShow = !quote && !clipName

  useEffect(() => {
    const total = reducedMotion() ? 400 : INTRO_MS
    const t1 = setTimeout(() => setPhase('leaving'), total - 260)
    const t2 = setTimeout(() => (nothingToShow ? onGo() : setPhase('card')), total)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="fixed inset-0 overflow-hidden bg-[#0a0c0b] text-white">
      {bg && <img src={bg} alt="" className="drift absolute inset-0 h-full w-full object-cover opacity-60" />}
      <div className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(10,12,11,.35)_0%,rgba(10,12,11,.2)_35%,rgba(10,12,11,.95)_80%)]" />

      {phase !== 'card' && (
        <div className={`absolute inset-0 grid place-items-center ${phase === 'leaving' ? 'ko-out' : ''}`}>
          <div className="text-center">
            <div className="ko-word font-display text-[clamp(4rem,24vw,8rem)] uppercase leading-[.85]">Kick<br /><span className="text-[var(--red-ink)]">off</span></div>
            <div className="ko-sweep mx-auto mt-5 h-1.5 w-40 rounded-full bg-[var(--red)]" />
          </div>
        </div>
      )}

      {phase === 'card' && (
        <div className="stagger absolute inset-x-0 bottom-0 mx-auto flex max-w-xl flex-col gap-5 px-5 pb-[max(28px,env(safe-area-inset-bottom))]">
          <h1 style={{ '--i': 1 } as React.CSSProperties} className="m-0 font-display text-[clamp(2rem,9vw,2.8rem)] uppercase leading-[.92]">
            Lock in.<br /><span className="text-[var(--red-ink)]">{session.title}</span>
          </h1>
          {quote && (
            <figure style={{ '--i': 2 } as React.CSSProperties} className="m-0 rounded-[24px] bg-white/10 p-5 backdrop-blur-md">
              <blockquote className="m-0 font-display text-lg leading-snug normal-case">“{quote}”</blockquote>
              <figcaption className="mt-2 text-sm text-white/60">Cristiano Ronaldo</figcaption>
            </figure>
          )}
          {clipName && <p style={{ '--i': 3 } as React.CSSProperties} className="m-0 text-sm text-white/70">▶ Playing: {clipName}</p>}
          <button style={{ '--i': 4 } as React.CSSProperties} onClick={onGo}
            className="min-h-14 w-full rounded-full bg-[var(--red)] font-display text-sm uppercase tracking-wide text-white shadow-[0_6px_16px_-6px_rgba(0,0,0,.6)] transition-transform duration-150 ease-snap active:scale-[0.97]">
            {clipName ? "Skip, let's go" : "Let's go"}
          </button>
        </div>
      )}
    </div>
  )
}
