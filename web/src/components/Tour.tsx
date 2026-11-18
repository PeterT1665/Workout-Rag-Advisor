import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { load, save } from '../lib/storage'
import { t } from '../i18n'

// First-run tour: the screen dims, one button at a time is lit up and a small box with an arrow says what it does.
// Buttons opt in with data-tour="<id>"; a step whose button isn't on screen is skipped.

export type TourStep = { target: string; title: string; text: string }

const SEEN = 'tour:seen'
// New players only: anyone who already closed the old "How it works" card has found their way around.
export const tourDue = () => !load<boolean>(SEEN, false) && !load<boolean>('intro:seen', false)
export const tourDone = () => save(SEEN, true)
// A new player (sample or your own) gets the tour again, even on a phone that has used the app before.
export const tourReset = () => { try { localStorage.removeItem(SEEN); localStorage.removeItem('intro:seen') } catch { /* ignore */ } }

export const homeTour = (): TourStep[] => [
  { target: 'streak', title: t('Your streak'), text: t('Train on your plan days to keep the flame going. Tap it to open your training log.') },
  { target: 'player', title: t('Your player card'), text: t('Your rating, stats and training log. Take the rating test to get your score.') },
  { target: 'week', title: t('Your week'), text: t('Tap your game days and the week plans itself around them. Change any day you like.') },
  { target: 'kickoff', title: t('Kick off'), text: t("Starts today's session. Turn the volume up: every drill is called out loud.") },
  { target: 'team', title: t('Team'), text: t("Add friends with your QR code and see who's training.") },
  { target: 'settings', title: t('Settings'), text: t('Language, voice, notifications and your player.') },
]

const PAD = 6 // spotlight margin around the button
const GAP = 14 // spotlight to box
const EDGE = 12 // box to screen edge

type Box = { x: number; y: number; w: number; h: number; r: number }

function measure(target: string): Box | null {
  const el = document.querySelector<HTMLElement>(`[data-tour="${target}"]`)
  if (!el) return null
  const b = el.getBoundingClientRect()
  if (!b.width || !b.height) return null
  const r = parseFloat(getComputedStyle(el).borderTopLeftRadius) || 12
  return { x: b.left - PAD, y: b.top - PAD, w: b.width + PAD * 2, h: b.height + PAD * 2, r: Math.min(r + PAD, (b.height + PAD * 2) / 2) }
}

export function Tour({ steps, onDone }: { steps: TourStep[]; onDone: () => void }) {
  const [i, setI] = useState(0)
  const [box, setBox] = useState<Box | null>(null)
  const [vw, setVw] = useState(() => window.innerWidth)
  const [vh, setVh] = useState(() => window.innerHeight)
  const reduce = useReducedMotion()
  const next = useRef<HTMLButtonElement>(null)
  const step = steps[i]

  const finish = useCallback(() => { tourDone(); onDone() }, [onDone])
  const advance = useCallback(() => (i + 1 < steps.length ? setI(i + 1) : finish()), [i, steps.length, finish])

  // find this step's button (bring it on screen if needed), skip it if it isn't there
  useLayoutEffect(() => {
    if (!step) return
    const el = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`)
    const b = el?.getBoundingClientRect()
    if (el && b && !el.closest('nav') && (b.top < 0 || b.bottom > window.innerHeight - 90)) el.scrollIntoView({ block: 'center' })
    const m = measure(step.target)
    if (m) setBox(m)
    else advance()
  }, [step]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!step) return
    const on = () => { setVw(window.innerWidth); setVh(window.innerHeight); const m = measure(step.target); if (m) setBox(m) }
    window.addEventListener('resize', on)
    window.addEventListener('scroll', on, { passive: true })
    const late = setTimeout(on, 400) // entrance animations settle
    return () => { window.removeEventListener('resize', on); window.removeEventListener('scroll', on); clearTimeout(late) }
  }, [step])

  useEffect(() => { next.current?.focus({ preventScroll: true }) }, [i, box === null])
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') finish(); else if (e.key === 'ArrowRight') advance() }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [finish, advance])

  if (!step || !box) return null

  // box below the button when it's in the top half of the screen, above otherwise; arrow points at its centre
  const below = box.y + box.h / 2 < vh / 2
  const width = Math.min(320, vw - EDGE * 2)
  const cx = box.x + box.w / 2
  const left = Math.min(Math.max(cx - width / 2, EDGE), vw - width - EDGE)
  const arrow = Math.min(Math.max(cx - left, 22), width - 22)
  const last = i === steps.length - 1
  const ease = [0.23, 1, 0.32, 1] as const

  return (
    <div className="fixed inset-0 z-[70] touch-none" role="dialog" aria-modal="true" aria-labelledby="tour-title" aria-describedby="tour-text">
      {/* dim everything except the lit button; taps outside the box do nothing */}
      <div aria-hidden="true" className="absolute rounded-full ring-2 ring-red shadow-[0_0_0_200vmax_rgba(4,5,5,.74)] transition-[left,top,width,height,border-radius] duration-[380ms] ease-[cubic-bezier(.23,1,.32,1)] motion-reduce:transition-none"
        style={{ left: box.x, top: box.y, width: box.w, height: box.h, borderRadius: box.r }} />
      <AnimatePresence mode="wait">
        <motion.div key={i}
          initial={{ opacity: 0, y: reduce ? 0 : below ? -6 : 6, scale: reduce ? 1 : 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, transition: { duration: 0.1 } }}
          transition={{ duration: 0.24, ease, delay: reduce ? 0 : 0.12 }}
          style={{ left, width, transformOrigin: `${arrow}px ${below ? '0%' : '100%'}`, ...(below ? { top: box.y + box.h + GAP } : { bottom: vh - box.y + GAP }) }}
          className="absolute rounded-[20px] bg-ink p-4 text-bg shadow-[0_18px_40px_-12px_rgba(0,0,0,.7)]">
          <span aria-hidden="true" className={`absolute h-3.5 w-3.5 rotate-45 rounded-[3px] bg-ink ${below ? '-top-1.5' : '-bottom-1.5'}`} style={{ left: arrow - 7 }} />
          <p className="m-0 font-display text-[11px] uppercase tracking-wide text-bg/60 tabular-nums">{t('{n} of {total}', { n: i + 1, total: steps.length })}</p>
          <h2 id="tour-title" className="m-0 mt-1 font-display text-lg uppercase leading-tight">{step.title}</h2>
          <p id="tour-text" className="m-0 mt-1 text-[15px] leading-snug text-bg/80">{step.text}</p>
          <div className="mt-3 flex items-center justify-between gap-3">
            {last ? <span /> : <button onClick={finish} className="-ml-2 min-h-11 px-2 text-sm font-semibold text-bg/60">{t('Skip tour')}</button>}
            <button ref={next} onClick={advance}
              className="min-h-11 rounded-full bg-red px-5 font-display text-sm uppercase tracking-wide text-on-red transition-transform duration-150 ease-snap active:scale-[0.97]">
              {last ? t("Let's go") : t('Next')}
            </button>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
