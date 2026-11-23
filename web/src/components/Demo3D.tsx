import { useEffect, useMemo, useRef, useState } from 'react'
import { demoFor } from '../lib/demo3d/catalog'
import type { Demo, Palette } from '../lib/demo3d/scene'
import { formFor, type Mistake } from '../lib/demo3d/form'
import { say } from '../lib/audio'
import { loadSettings } from '../lib/storage'
import { t } from '../i18n'

const BAD = '#ef4444', GOOD = '#22c55e'
// Form check: the right way with the key cue, then each common mistake (red) and its fix (green), on a loop.
type Phase = { kind: 'do' | 'bad' | 'fix'; text: string; mistake?: Mistake }
const VOICE_KEY = 'formVoice'
const voiceOn = () => { try { return localStorage.getItem(VOICE_KEY) !== 'off' } catch { return true } }
const stopVoice = () => { try { speechSynthesis.cancel() } catch { /* no speech */ } }

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
  const demoMoveDur = useRef(0)
  const [state, setState] = useState<'loading' | 'ready' | 'failed'>('loading')
  const [playing, setPlaying] = useState(() => !reducedMotion())
  const playingRef = useRef(playing)
  playingRef.current = playing
  const form = spec ? formFor(spec.move) : null
  const [checking, setChecking] = useState(false)
  const [step, setStep] = useState(0)
  const [voice, setVoice] = useState(voiceOn)
  const phases = useMemo<Phase[]>(() => form
    ? [{ kind: 'do', text: form.cue }, ...form.mistakes.flatMap((m): Phase[] => [{ kind: 'bad', text: m.label, mistake: m }, { kind: 'fix', text: m.fix, mistake: m }])]
    : [], [form])
  const phase = checking ? phases[step % phases.length] : null

  useEffect(() => {
    if (!spec || !canvas.current) return
    let gone = false
    let visible = true
    const c = canvas.current
    const sync = () => demo.current?.setPlaying(playingRef.current && visible && document.visibilityState === 'visible')
    const io = typeof IntersectionObserver !== 'undefined' ? new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync() }) : null
    const onVis = () => sync()
    const onResize = () => demo.current?.resize()
    Promise.all([import('../lib/demo3d/scene'), import('../lib/demo3d/moves'), import('../lib/demo3d/form')]).then(([{ createDemo }, { MOVES }, { withMistake }]) => {
      if (gone) return
      const base = MOVES[spec.move]
      demoMoveDur.current = base.dur
      const move = phase?.kind === 'bad' && phase.mistake ? withMistake(base, phase.mistake) : base
      const marks = phase?.mistake ? { parts: phase.mistake.marks, color: phase.kind === 'bad' ? BAD : GOOD } : undefined
      try {
        demo.current?.dispose()
        demo.current = createDemo(c, move, palette(), { speed: spec.speed, marks, ...(playingRef.current ? {} : { at: 0.45 }) })
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
  }, [spec?.move, spec?.speed, phase]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { demo.current?.setPlaying(playing) }, [playing])

  // form check: each step plays two loops (at least 4.5 s), says its line, then moves on
  useEffect(() => {
    if (!phase || !spec || !playing) return
    if (voice) { const s = loadSettings(); say({ ...s, volume: Math.max(s.volume, 0.6) }, phase.text, true) }
    const loops = 2 * (demoMoveDur.current || 3) / (spec.speed ?? 1)
    const id = setTimeout(() => setStep((n) => n + 1), Math.max(4500, loops * 1000))
    return () => clearTimeout(id)
  }, [phase, playing, voice]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => stopVoice(), [])
  useEffect(() => { if (!checking) stopVoice() }, [checking])

  if (!spec || state === 'failed') return null
  return (
    <figure className={`relative m-0 overflow-hidden rounded-2xl bg-surface-2 ${className}`}>
      <canvas ref={canvas} role="img" aria-label={t('3D demo of {name}', { name: t(name) })}
        className={`block aspect-[4/3] w-full touch-pan-y transition-opacity duration-300 ${state === 'ready' ? 'opacity-100' : 'opacity-0'}`} />
      {state === 'loading' && <div className="absolute inset-0 animate-pulse bg-surface-2" aria-hidden="true" />}
      {phase && (
        <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start gap-2 p-2.5" aria-live="polite">
          <span className="flex min-w-0 items-center gap-2 rounded-2xl bg-black/60 py-1.5 pl-1.5 pr-3 text-[13px] font-semibold leading-snug text-white backdrop-blur-sm">
            <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full text-white" style={{ background: phase.kind === 'bad' ? BAD : GOOD }} aria-hidden="true">
              {phase.kind === 'bad'
                ? <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
                : <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"><path d="m5 12.5 4.5 4.5L19 7" /></svg>}
            </span>
            <span className="sr-only">{phase.kind === 'bad' ? t('Mistake:') : phase.kind === 'fix' ? t('Fix:') : t('Do:')}</span>
            <span className="min-w-0">{phase.text}</span>
          </span>
          <button type="button" onClick={() => { const v = !voice; setVoice(v); try { localStorage.setItem(VOICE_KEY, v ? 'on' : 'off') } catch { /* private mode */ } if (!v) stopVoice() }}
            aria-label={voice ? t('Mute the coaching voice') : t('Turn the coaching voice on')} aria-pressed={!voice}
            className="pointer-events-auto ml-auto grid h-9 w-9 shrink-0 place-items-center rounded-full bg-black/45 text-white backdrop-blur-sm active:scale-95">
            {voice
              ? <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M11 5 6 9H3v6h3l5 4V5Z" fill="currentColor" /><path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" /></svg>
              : <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M11 5 6 9H3v6h3l5 4V5Z" fill="currentColor" /><path d="m16 9 5 6M21 9l-5 6" /></svg>}
          </button>
        </div>
      )}
      <figcaption className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 p-2.5">
        {form
          ? <button type="button" onClick={() => { setChecking((c) => !c); setStep(0); setPlaying(true) }} aria-pressed={checking}
              className={`pointer-events-auto flex h-9 items-center gap-1.5 rounded-full px-3 text-[12px] font-bold backdrop-blur-sm active:scale-95 ${checking ? 'bg-white text-black' : 'bg-black/45 text-white'}`}>
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m4 12.5 4 4L16 8" /><path d="m14 16 6-6" opacity=".5" /></svg>
              {checking ? t('Form check on') : t('Form check')}
            </button>
          : <span className="rounded-full bg-black/45 px-2.5 py-1 text-[11px] font-semibold text-white backdrop-blur-sm">{t('3D demo · drag to turn')}</span>}
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
