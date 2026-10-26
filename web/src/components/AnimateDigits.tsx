// Adapted from "Animate Digits" by unlumen (21st.dev). Each digit blur-slides on change; only changed digits move.
// Changes: no shadcn `cn`, reduced-motion support, tuned defaults for this app.
import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion, useSpring, useTransform } from 'motion/react'

type Tuning = { stiffness: number; damping: number; enterY: number; enterBlur: number; enterScale: number }

let nextId = 0

function DigitCell({ char, isDigit, className = '', t, reduce }: { char: string; isDigit: boolean; className?: string; t: Tuning; reduce: boolean }) {
  const [exits, setExits] = useState<{ id: number; char: string; y: number }[]>([])
  const prev = useRef(char)
  const first = useRef(true)
  const cfg = { stiffness: t.stiffness, damping: t.damping }
  const y = useSpring(0, cfg)
  const opacity = useSpring(1, cfg)
  const scale = useSpring(1, cfg)
  const blur = useSpring(0, cfg)
  const filter = useTransform(blur, (v) => `blur(${v}px)`)

  useEffect(() => {
    if (!isDigit) return
    const p = prev.current
    prev.current = char
    if (first.current) { first.current = false; return }
    if (char === p || !/\d/.test(p) || reduce) return
    const up = Number(char) > Number(p)
    const id = nextId++
    setExits((q) => [...q, { id, char: p, y: up ? -t.enterY : t.enterY }].slice(-3))
    y.jump(up ? t.enterY : -t.enterY); opacity.jump(0); scale.jump(t.enterScale); blur.jump(t.enterBlur)
    y.set(0); opacity.set(1); scale.set(1); blur.set(0)
  }, [char, isDigit, reduce, t.enterY, t.enterBlur, t.enterScale, y, opacity, scale, blur])

  if (!isDigit) return <span className={className}>{char}</span>
  return (
    <span className={`relative inline-grid place-items-center [&>*]:col-start-1 [&>*]:row-start-1 ${className}`}>
      <AnimatePresence>
        {exits.map((e) => (
          <motion.span key={e.id} aria-hidden
            initial={{ opacity: 1, scale: 1, filter: 'blur(0px)', y: 0 }}
            animate={{ opacity: 0, scale: 0.7, filter: 'blur(8px)', y: e.y }}
            transition={{ type: 'spring', stiffness: t.stiffness, damping: t.damping + 5 }}
            onAnimationComplete={() => setExits((q) => q.filter((x) => x.id !== e.id))}>
            {e.char}
          </motion.span>
        ))}
      </AnimatePresence>
      <motion.span style={{ opacity, scale, filter, y }}>{char}</motion.span>
    </span>
  )
}

// `calm` is for values that change every second (the session timer): short travel, light blur.
export function AnimateDigits({ value, className = '', calm = false }: { value: string; className?: string; calm?: boolean }) {
  const reduce = !!useReducedMotion()
  const t: Tuning = calm
    ? { stiffness: 320, damping: 30, enterY: 14, enterBlur: 6, enterScale: 0.9 }
    : { stiffness: 170, damping: 14, enterY: 32, enterBlur: 24, enterScale: 0.7 }
  return (
    <span className={`inline-flex items-center tabular-nums ${className}`} aria-label={value} role="text">
      {value.split('').map((c, i) => <DigitCell key={i} char={c} isDigit={/\d/.test(c)} t={t} reduce={reduce} />)}
    </span>
  )
}
