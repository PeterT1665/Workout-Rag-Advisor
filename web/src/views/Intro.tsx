import { useEffect, useRef, useState } from 'react'
import { BallIcon } from '../components/icons'
import { t } from '../i18n'

const LENGTH = 4000 // ms, then the welcome screen

// The first thing a new visitor sees: a few seconds on a pitch, then "Train like a pro". Tap anywhere to skip.
export function Intro({ onDone }: { onDone: () => void }) {
  const [leaving, setLeaving] = useState(false)
  const brand = t('Football Personal Coach').split(' ')
  const done = useRef(false)
  const finish = () => {
    if (done.current) return
    done.current = true
    setLeaving(true)
    setTimeout(onDone, 260)
  }
  useEffect(() => {
    const id = setTimeout(finish, LENGTH)
    return () => clearTimeout(id)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <button type="button" onClick={finish} aria-label={t('Skip intro')}
      className={`fixed inset-0 z-[60] flex flex-col items-center justify-center overflow-hidden bg-[#0a0c0b] text-white ${leaving ? 'ko-out' : ''}`}>
      {/* the pitch: halfway line and centre circle draw in */}
      <svg viewBox="0 0 400 800" preserveAspectRatio="xMidYMid slice" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
        <g fill="none" stroke="rgba(255,255,255,.14)" strokeWidth="2.5">
          <path className="intro-draw" d="M0 400 H400" style={{ '--len': 400 } as React.CSSProperties} />
          <circle className="intro-draw" cx="200" cy="400" r="78" style={{ '--len': 490, animationDelay: '150ms' } as React.CSSProperties} />
          <path className="intro-draw" d="M120 0 V90 H280 V0" style={{ '--len': 340, animationDelay: '300ms' } as React.CSSProperties} />
          <path className="intro-draw" d="M120 800 V710 H280 V800" style={{ '--len': 340, animationDelay: '300ms' } as React.CSSProperties} />
        </g>
        <circle cx="200" cy="400" r="4" fill="rgba(255,255,255,.3)" />
      </svg>
      <div className="intro-roll relative grid h-24 w-24 place-items-center rounded-full bg-[var(--red)] shadow-[0_18px_40px_-12px_rgba(224,27,53,.7)]">
        <BallIcon className="h-16 w-16" />
      </div>
      <p className="ko-word m-0 mt-8 text-center font-display text-[clamp(2rem,10vw,3rem)] uppercase leading-[.9]" style={{ animationDelay: '900ms' }}>
        {brand[0]}<br />{brand.slice(1).join(' ')}
      </p>
      <p className="rise m-0 mt-4 max-w-[28ch] px-6 text-center text-white/75" style={{ animationDelay: '1500ms', animationDuration: '500ms' }}>
        {t("Welcome. Let's make you a better player.")}
      </p>
      <span className="absolute inset-x-0 bottom-[max(28px,env(safe-area-inset-bottom))] flex flex-col items-center gap-3">
        <span className="h-1 w-24 overflow-hidden rounded-full bg-white/15"><span className="intro-bar block h-full bg-white/70" style={{ animationDuration: `${LENGTH}ms` }} /></span>
        <span className="text-xs text-white/45">{t('Tap to skip')}</span>
      </span>
    </button>
  )
}
