import { useEffect, useRef, useState } from 'react'
import { demoFor } from '../lib/demo3d/catalog'
import type { Demo, Palette } from '../lib/demo3d/scene'
import { t } from '../i18n'

// Colours from the app's theme (dark or light), so the demo sits in the page.
function palette(): Palette {
  const css = getComputedStyle(document.documentElement)
  const v = (k: string, d: string) => css.getPropertyValue(k).trim() || d
  const bg = v('--surface-2', '#222225')
  const light = parseInt(bg.slice(1, 3), 16) > 128
  return light
    ? { body: '#ffffff', joint: '#b9bdc2', accent: v('--red', '#d4002a'), floor: '#d6d6d3', bg, kit: '#7d858c', line: '#2a2d31' }
    : { body: '#d9dcdf', joint: '#7d8288', accent: v('--red', '#e01b35'), floor: '#2b2b2f', bg, kit: '#9aa3ab', line: '#3a3e43' }
}
const reducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches

// A faceless 3D figure doing the exercise, on a loop. Drag to turn it. Shown for every library exercise;
// three.js loads only when one is on screen. Pauses off screen and in the background.
export function Demo3D({ name, className = '' }: { name: string; className?: string }) {
  const spec = demoFor(name)
  const canvas = useRef<HTMLCanvasElement>(null)
  const demo = useRef<Demo | null>(null)
  const [state, setState] = useState<'loading' | 'ready' | 'failed'>('loading')
  const [playing, setPlaying] = useState(() => !reducedMotion())
  const playingRef = useRef(playing)
  playingRef.current = playing

  useEffect(() => {
    if (!spec || !canvas.current) return
    let gone = false
    let visible = true
    const c = canvas.current
    const sync = () => demo.current?.setPlaying(playingRef.current && visible && document.visibilityState === 'visible')
    const io = typeof IntersectionObserver !== 'undefined' ? new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync() }) : null
    const onVis = () => sync()
    const onResize = () => demo.current?.resize()
    Promise.all([import('../lib/demo3d/scene'), import('../lib/demo3d/moves')]).then(([{ createDemo }, { MOVES }]) => {
      if (gone) return
      const move = MOVES[spec.move]
      try {
        demo.current = createDemo(c, move, palette(), { speed: spec.speed, ...(playingRef.current ? {} : { at: 0.45 }) })
        setState('ready')
        io?.observe(c)
        sync()
      } catch { setState('failed') } // no WebGL: the sheet just shows the written how-to
    }).catch(() => setState('failed'))
    document.addEventListener('visibilitychange', onVis)
    window.addEventListener('resize', onResize)
    return () => {
      gone = true
      io?.disconnect()
      document.removeEventListener('visibilitychange', onVis)
      window.removeEventListener('resize', onResize)
      demo.current?.dispose()
      demo.current = null
    }
  }, [spec?.move, spec?.speed]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { demo.current?.setPlaying(playing) }, [playing])

  if (!spec || state === 'failed') return null
  return (
    <figure className={`relative m-0 overflow-hidden rounded-2xl bg-surface-2 ${className}`}>
      <canvas ref={canvas} role="img" aria-label={t('3D demo of {name}', { name: t(name) })}
        className={`block aspect-[4/3] w-full touch-pan-y transition-opacity duration-300 ${state === 'ready' ? 'opacity-100' : 'opacity-0'}`} />
      {state === 'loading' && <div className="absolute inset-0 animate-pulse bg-surface-2" aria-hidden="true" />}
      <figcaption className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 p-2.5">
        <span className="rounded-full bg-black/45 px-2.5 py-1 text-[11px] font-semibold text-white backdrop-blur-sm">{t('3D demo · drag to turn')}</span>
        <button type="button" onClick={() => setPlaying((p) => !p)} aria-label={playing ? t('Pause demo') : t('Play demo')}
          className="pointer-events-auto grid h-9 w-9 place-items-center rounded-full bg-black/45 text-white backdrop-blur-sm active:scale-95">
          {playing
            ? <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
            : <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.4-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z" /></svg>}
        </button>
      </figcaption>
    </figure>
  )
}
