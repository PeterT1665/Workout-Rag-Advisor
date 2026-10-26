import { useEffect, useMemo, useSyncExternalStore } from 'react'
import { Button, QuoteCard, Screen } from '../components/ui'
import { fmtTime } from '../lib/logic'
import { nextQuote } from '../lib/quotes'
import { FullTime } from './FullTime'
import { AnimateDigits } from '../components/AnimateDigits'
import type { Player } from '../lib/player'
import type { Session, Settings } from '../lib/types'

const R = 100
const C = 2 * Math.PI * R

function useWakeLock(active: boolean) {
  useEffect(() => {
    if (!active || !('wakeLock' in navigator)) return
    let lock: WakeLockSentinel | null = null
    const request = async () => { try { lock = await navigator.wakeLock.request('screen') } catch { /* ignore */ } }
    void request()
    const onVis = () => { if (document.visibilityState === 'visible') void request() }
    document.addEventListener('visibilitychange', onVis)
    return () => { document.removeEventListener('visibilitychange', onVis); void lock?.release().catch(() => {}) }
  }, [active])
}

export function PlayerView({ session, player, settings, onExit }: { session: Session; player: Player; settings: Settings; onExit: () => void }) {
  const st = useSyncExternalStore(player.subscribe, player.getState)
  useEffect(() => { player.start(); return () => player.stop() }, [player])
  useWakeLock(!st.finished)

  const step = st.steps[st.i]
  // a new card per rest step, not per tick
  const restQuote = useMemo(() => (step?.type === 'rest' && settings.cards ? nextQuote() : null), [st.i]) // eslint-disable-line react-hooks/exhaustive-deps

  if (st.finished) return <FullTime session={session} st={st} onExit={onExit} />
  if (!step) return null

  const isRest = step.type === 'rest'
  const accent = isRest ? 'var(--muted)' : 'var(--red)'
  const frac = Math.max(0, Math.min(1, st.remainingMs / (step.seconds * 1000)))
  const next = st.steps[st.i + 1]
  const blocks = session.blocks.map((_, bi) => {
    let start = 0, len = 0
    st.steps.forEach((x) => {
      if (x.blockIdx < bi) start += x.seconds
      else if (x.blockIdx === bi) len += x.seconds
    })
    return { start, len }
  })
  const phase = isRest ? 'Rest' : step.kind === 'warmup' ? 'Warm-up' : step.kind === 'cooldown' ? 'Cool-down' : 'Work'

  return (
    <Screen className="flex flex-col">
      <div className="flex items-center justify-between">
        <span className="font-display text-sm font-bold uppercase tracking-[0.2em]" style={{ color: accent }}>{isRest ? `Round ${step.round} of ${step.rounds} done` : phase}</span>
        <span className="font-display text-sm font-bold tabular-nums text-muted">{fmtTime(st.total - st.elapsed)} left</span>
      </div>
      <h1 key={st.i} className="slam m-0 mt-1 font-display text-3xl uppercase leading-none" data-testid="step-name">
        {isRest ? 'Rest' : step.name}
        {!isRest && step.rounds > 1 && <span className="text-muted"> {step.round}/{step.rounds}</span>}
      </h1>

      <div className="relative mx-auto my-6 aspect-square w-[min(78vw,320px)]">
        <svg viewBox="0 0 220 220" className="h-full w-full -rotate-90">
          <circle cx="110" cy="110" r={R} fill="none" strokeWidth="12" stroke="var(--surface-2)" />
          <circle cx="110" cy="110" r={R} fill="none" strokeWidth="12" stroke={accent} strokeLinecap="round"
            strokeDasharray={C} strokeDashoffset={C * (1 - frac)} />
        </svg>
        <div className="absolute inset-0 grid place-items-center font-display text-[clamp(2.6rem,14vw,4.2rem)] tabular-nums">
          <AnimateDigits calm value={fmtTime(st.remainingMs / 1000)} />
        </div>
      </div>

      <div className="min-h-24 rounded-[20px] bg-surface p-4">
        {isRest ? (restQuote ? <QuoteCard text={restQuote} /> : <p className="m-0">Breathe. Walk it off. Sip water.</p>) : <p className="m-0">{step.how}</p>}
      </div>
      {isRest && next
        ? <div className="mt-3 mb-4 flex items-center justify-between gap-3 rounded-[20px] bg-surface-2 px-4 py-3">
            <span className="shrink-0 whitespace-nowrap text-sm text-muted">Up next</span>
            <span className="text-right font-display text-sm uppercase">{next.name.replace(/\s*\([^)]*\)/g, '')}{next.rounds > 1 ? ` · ${next.round}/${next.rounds}` : ''}</span>
          </div>
        : <p className="mt-3 mb-4 text-sm text-muted">Up next: {next ? (next.type === 'rest' ? 'Rest' : next.name) : 'Last one.'}</p>}

      <div className="flex h-2 gap-[3px]" aria-label={`Block ${step.blockIdx + 1} of ${session.blocks.length}`}>
        {blocks.map((b, bi) => {
          const fill = Math.max(0, Math.min(1, (st.elapsed - b.start) / b.len))
          return (
            <div key={bi} className={`relative overflow-hidden rounded-full ${bi === step.blockIdx ? 'bg-surface-2' : 'bg-surface-2/70'}`} style={{ flex: b.len }}>
              <div className="absolute inset-y-0 left-0 rounded-full bg-red" style={{ width: `${fill * 100}%` }} />
            </div>
          )
        })}
      </div>

      <div className="mt-auto grid grid-cols-[1fr_2fr_1fr] gap-3 pt-6">
        <Button variant="ghost" onClick={player.back}>Back</Button>
        <Button onClick={player.togglePause}>{st.running ? 'Pause' : 'Resume'}</Button>
        <Button variant="ghost" onClick={player.skip}>Skip</Button>
      </div>
      <Button variant="ghost" size="sm" className="mt-3 self-center" onClick={() => { if (confirm('End the session?')) onExit() }}>End session</Button>

      {/* Full-screen cue: readable from a few metres away, phone on the ground */}
      <div aria-hidden={!st.cue} data-testid="cue"
        className={`fixed inset-0 z-50 flex flex-col bg-red px-4 pt-[max(20px,env(safe-area-inset-top))] pb-[max(24px,env(safe-area-inset-bottom))] text-white transition-opacity duration-100 ${st.cue ? 'opacity-100' : 'pointer-events-none opacity-0'}`}>
        <div className="flex justify-between font-display text-sm uppercase tracking-wider">
          <span>{step.name.replace(/\s*\([^)]*\)/g, '')} · {step.round}/{step.rounds}</span>
          <span className="tabular-nums">{fmtTime(st.remainingMs / 1000)}</span>
        </div>
        {st.cue && (
          <div key={st.cueKey} className="slam flex flex-1 flex-col items-center justify-center text-center">
            <svg viewBox="0 0 24 24" className="h-[min(70vw,300px)] w-[min(70vw,300px)]" fill="none" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"
              style={{ transform: st.cue === 'right' ? 'scaleX(-1)' : undefined }}>
              <path d="M20 12H5M11 5l-7 7 7 7" />
            </svg>
            <div className="font-display text-[clamp(3.5rem,21vw,7rem)] uppercase leading-[.8]">{st.cue}</div>
            <div className="mt-6 font-display text-sm uppercase tracking-wider opacity-90">Defender commits {st.cue} · go {st.cue === 'left' ? 'right' : 'left'}</div>
          </div>
        )}
      </div>
    </Screen>
  )
}
