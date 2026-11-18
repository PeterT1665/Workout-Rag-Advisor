import { useLayoutEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { BallIcon, ReviewIcon } from './icons'
import { t } from '../i18n'

export type Tab = 'home' | 'week' | 'player' | 'team'

// review: after a game the centre button opens the match review instead of kicking off a session
export function TabBar({ tab, onTab, onKickOff, review = false }: { tab: Tab; onTab: (t: Tab) => void; onKickOff: () => void; review?: boolean }) {
  const item = (t: Tab, label: string) => (
    <button onClick={() => onTab(t)} data-tour={t} aria-current={tab === t ? 'page' : undefined}
      className={`relative min-h-12 min-w-0 px-0.5 font-display uppercase tracking-wide transition-colors duration-150 ${tab === t ? 'text-white' : 'text-white/50'}`}>
      <FitLabel text={label} />
      {tab === t && <motion.span layoutId="tab-dot" className="absolute bottom-1 left-1/2 h-1 w-5 -translate-x-1/2 rounded-full bg-red"
        transition={{ type: 'spring', duration: 0.35, bounce: 0.2 }} />}
    </button>
  )
  return (
    <nav className="fixed inset-x-3 bottom-[max(12px,env(safe-area-inset-bottom))] z-40 mx-auto grid h-[70px] max-w-[540px] grid-cols-[minmax(0,1fr)_minmax(0,1fr)_80px_minmax(0,1fr)_minmax(0,1fr)] items-center rounded-[26px] bg-[#161817]/90 shadow-[0_10px_30px_rgba(0,0,0,.4)] backdrop-blur-xl">
      {item('home', t('Home'))}
      {item('week', t('Week'))}
      <button onClick={onKickOff} data-tour="kickoff" aria-label={review ? t('Review your match') : t("Kick off today's session")}
        className="-mt-9 grid h-[70px] w-[70px] place-items-center justify-self-center rounded-full border-4 border-bg bg-red shadow-[0_8px_20px_-6px_rgba(0,0,0,.6)] transition-transform duration-150 ease-snap active:scale-95">
        {review ? <ReviewIcon className="h-9 w-9 text-on-red" /> : <BallIcon className="h-11 w-11" />}
      </button>
      {item('player', t('Performance'))}
      {item('team', t('Team'))}

    </nav>
  )
}

// A tab label at 10px, shrunk just enough to fit its column (long words in some languages, small phones).
function FitLabel({ text }: { text: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const [size, setSize] = useState(10)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el?.parentElement) return
    const fit = () => {
      el.style.fontSize = '10px'
      const room = el.parentElement!.clientWidth - 4
      const need = el.scrollWidth
      const next = need > room && need > 0 ? Math.max(6.5, Math.floor((10 * room) / need * 10) / 10) : 10
      el.style.fontSize = `${next}px` // set here too: React skips it when the size didn't change
      setSize(next)
    }
    fit()
    void document.fonts?.ready.then(fit) // the display font is wider than the fallback
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(fit)
    ro.observe(el.parentElement)
    return () => ro.disconnect()
  }, [text])
  return <span ref={ref} className="inline-block whitespace-nowrap" style={{ fontSize: size, letterSpacing: size < 10 ? '0' : undefined }}>{text}</span>
}
