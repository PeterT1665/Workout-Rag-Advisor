import { getProfile } from '../lib/profile'
import { isWordCue, wordFor } from '../lib/cues'
import { useEffect, useMemo, useState, useSyncExternalStore } from 'react'
import { Sheet } from '../components/Sheet'
import { PlanSheet } from '../components/PlanSheet'
import { VideoBox } from '../components/VideoBox'
import { Button, QuoteCard, Screen } from '../components/ui'
import { fmtReps, fmtTime, selfPaced, tempoFor } from '../lib/logic'
import { nextQuote } from '../lib/quotes'
import { FullTime } from './FullTime'
import { AnimateDigits } from '../components/AnimateDigits'
import type { Player } from '../lib/player'
import type { Session, Settings } from '../lib/types'
import { t } from '../i18n'
import { LiftsEditor } from '../components/LiftsEditor'
import { loggable, planLifts, type LiftLog } from '../lib/lifts'
import { liveLifts, saveLiveLifts, withLive } from '../lib/liveLifts'
import { loadHistory } from '../lib/storage'

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

export function PlayerView({ session: planned, player, settings, onExit }: { session: Session; player: Player; settings: Settings; onExit: () => void }) {
  const moves = getProfile().signatureMoves
  // the plan as it is now: exercises added, swapped, moved or paired in the Plan sheet count (step indexes point into it)
  const session = player.getSession() ?? planned
  const st = useSyncExternalStore(player.subscribe, player.getState)
  useEffect(() => { player.start(); return () => player.stop() }, [player])
  useWakeLock(!st.finished)
  const [plan, setPlan] = useState(false)
  const [info, setInfo] = useState<number | null>(null) // step whose how-to + video is open
  const [logging, setLogging] = useState<LiftLog[] | null>(null) // sets being logged after an exercise
  const [, setLogged] = useState(0)

  const step = st.steps[st.i]
  // a new card per rest step, not per tick
  const restQuote = useMemo(() => (step?.type === 'rest' && settings.cards ? nextQuote() : null), [st.i]) // eslint-disable-line react-hooks/exhaustive-deps

  if (st.finished) return <FullTime session={session} st={st} onExit={onExit} />
  if (!step) return null

  const isRest = step.type === 'rest'
  const paced = selfPaced(step) // a set with reps: their own pace, ends on Set done
  const accent = isRest ? 'var(--muted)' : 'var(--red)'
  const frac = paced ? 1 : Math.max(0, Math.min(1, st.remainingMs / (step.seconds * 1000)))
  const next = st.steps[st.i + 1]
  // the next exercise (past any rest): previewed during work too, not only during rests
  const upIdx = st.steps.findIndex((x, k) => k > st.i && x.type !== 'rest')
  const up = upIdx >= 0 ? st.steps[upIdx] : undefined
  // progress segments in the order they're done: one per exercise, one per superset (its exercises alternate)
  const blocks: { start: number; len: number; from: number; to: number }[] = []
  let at = 0
  st.steps.forEach((x, k) => {
    const key = x.pair ? x.pair[0] : x.blockIdx
    const prev = st.steps[k - 1]
    const last = blocks.at(-1)
    if (last && prev && (prev.pair ? prev.pair[0] : prev.blockIdx) === key) { last.len += x.seconds; last.to = k }
    else blocks.push({ start: at, len: x.seconds, from: k, to: k })
    at += x.seconds
  })
  const prevPair = isRest ? st.steps[st.i - 1]?.pair : undefined
  // The rest after an exercise's last set: log its sets now, while the numbers are fresh. In a superset each exercise is
  // asked for after its own last round (pull-ups 3 sets, bench 4: pull-ups after round 3, bench after round 4).
  const prevWork = isRest ? st.steps[st.i - 1] : undefined
  const finished = isRest && prevWork
    ? [...new Set(st.steps.filter((x) => x.type === 'work' && (prevWork.pair ? x.pair?.[0] === prevWork.pair[0] : x.blockIdx === prevWork.blockIdx)).map((x) => x.blockIdx))]
        .map((bi) => session.blocks[bi]).filter((b) => b && (b.rounds || 1) === step.round && loggable(b))
    : []
  const live = finished.length ? liveLifts(st.startedAt) : []
  const doneLogging = finished.length > 0 && finished.every((b) => live.some((l) => l.name === b.name))
  const openLog = () => setLogging(withLive(planLifts({ ...session, blocks: finished }, loadHistory()), live))
  const phase = isRest ? t('Rest') : step.kind === 'warmup' ? t('Warm-up') : step.kind === 'cooldown' ? t('Cool-down') : step.pair ? t('Superset {x}', { x: step.pair }) : t('Work')

  return (
    <Screen className="flex flex-col">
      <div className="flex items-center justify-between gap-3">
        <span className="min-w-0 font-display text-sm font-bold uppercase tracking-[0.2em]" style={{ color: accent }}>{isRest ? (prevPair ? t('Superset {x} round {n} done', { x: prevPair[0], n: step.round }) : t('Round {n} of {total} done', { n: step.round, total: step.rounds })) : phase}</span>
        <span className="shrink-0 font-display text-sm font-bold tabular-nums text-muted">{t('{time} left', { time: fmtTime(st.total - st.elapsed) })}</span>
      </div>
      <h1 key={st.i} className="slam m-0 mt-1 font-display text-3xl uppercase leading-none" data-testid="step-name">
        {isRest ? t('Rest') : step.name}
        {!isRest && step.rounds > 1 && <span className="text-muted"> {step.round}/{step.rounds}</span>}
      </h1>
      {!isRest && step.reps && <p className="m-0 mt-2 font-display text-lg uppercase text-[var(--red-ink)]">{fmtReps(step.reps)}</p>}
      {paced && <p className="m-0 mt-1 text-sm text-muted">{tempoFor(step.category)}</p>}

      <div className="relative mx-auto my-6 aspect-square w-[min(78vw,320px)]">
        <svg viewBox="0 0 220 220" className="h-full w-full -rotate-90">
          <circle cx="110" cy="110" r={R} fill="none" strokeWidth="12" stroke="var(--surface-2)" />
          <circle cx="110" cy="110" r={R} fill="none" strokeWidth="12" stroke={accent} strokeLinecap="round"
            strokeDasharray={C} strokeDashoffset={C * (1 - frac)} />
        </svg>
        <div className="absolute inset-0 grid place-items-center font-display text-[clamp(2.6rem,14vw,4.2rem)] tabular-nums">
          <AnimateDigits calm value={fmtTime((paced ? st.setMs : st.remainingMs) / 1000)} />
        </div>
      </div>

      {finished.length > 0 && (
        <button onClick={openLog} aria-haspopup="dialog"
          className={`mb-3 flex w-full items-center justify-between gap-3 rounded-[20px] px-4 py-3 text-left active:scale-[0.98] ${doneLogging ? 'bg-surface-2' : 'bg-red text-on-red'}`}>
          <span className="min-w-0">
            <span className="block font-display text-sm uppercase">{doneLogging ? t('Sets logged') : t('Log your sets')}</span>
            <span className={`block truncate text-sm ${doneLogging ? 'text-muted' : 'opacity-85'}`}>{finished.map((b) => t(b.name)).join(' + ')}</span>
          </span>
          <span aria-hidden className="shrink-0 font-display">{doneLogging ? '✓' : '›'}</span>
        </button>
      )}
      {isRest ? (
        <div className="min-h-24 rounded-[20px] bg-surface p-4">
          {restQuote ? <QuoteCard text={restQuote} /> : <p className="m-0">{t('Breathe. Walk it off. Sip water.')}</p>}
        </div>
      ) : (
        // the how-to opens this exercise's video instructions (3D demo and your own link)
        <button aria-haspopup="dialog" aria-label={t('{name}: how-to and video', { name: step.name.replace(/\s*\([^)]*\)/g, '') })} onClick={() => setInfo(st.i)}
          className="block min-h-24 w-full rounded-[20px] bg-surface p-4 text-left transition-transform duration-150 active:scale-[0.98]">
          {step.how && <span className="block">{step.how}</span>}
          <span className="mt-2 flex items-center gap-1.5 text-sm font-semibold text-[var(--red-ink)]">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.4-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z" /></svg>
            {t('Watch how')}
          </span>
        </button>
      )}
      {up
        ? <button aria-haspopup="dialog" onClick={() => setInfo(upIdx)} className="mt-3 mb-4 flex w-full items-center justify-between gap-3 rounded-[20px] bg-surface-2 px-4 py-3 text-left transition-transform duration-150 active:scale-[0.98]">
            <span className="shrink-0 whitespace-nowrap text-sm text-muted">{t('Up next')}</span>
            <span className="flex items-center gap-2 text-right font-display text-sm uppercase">{up.name.replace(/\s*\([^)]*\)/g, '')}{up.rounds > 1 ? ` · ${up.round}/${up.rounds}` : ''}<span aria-hidden className="text-muted">›</span></span>
          </button>
        : <p className="mt-3 mb-4 text-sm text-muted">{t('Up next')}: {t('Last one.')}</p>}

      <div className="flex h-2 gap-[3px]" aria-label={t('Part {n} of {total}', { n: blocks.findIndex((b) => st.i >= b.from && st.i <= b.to) + 1, total: blocks.length })}>
        {blocks.map((b, bi) => {
          const fill = Math.max(0, Math.min(1, (st.elapsed - b.start) / b.len))
          return (
            <div key={bi} className={`relative overflow-hidden rounded-full ${st.i >= b.from && st.i <= b.to ? 'bg-surface-2' : 'bg-surface-2/70'}`} style={{ flex: b.len }}>
              <div className="absolute inset-y-0 left-0 rounded-full bg-red" style={{ width: `${fill * 100}%` }} />
            </div>
          )
        })}
      </div>

      <div className="mt-auto grid grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_minmax(0,1fr)] gap-3 pt-6">
        <Button variant="ghost" className="min-w-0 px-1! tracking-normal!" onClick={player.back}>{t('Back')}</Button>
        <Button className="min-w-0 px-2!" onClick={player.press}>{paced && st.running ? t('Set done') : st.running ? t('Pause') : t('Resume')}</Button>
        <Button variant="ghost" className="min-w-0 px-1! tracking-normal!" onClick={player.skip}>{t('Skip')}</Button>
      </div>
      <div className="mt-3 flex flex-wrap justify-center gap-2">
        <Button variant="ghost" size="sm" aria-haspopup="dialog" onClick={() => setInfo(isRest && next ? st.i + 1 : st.i)}>{t('Video')}</Button>
        <Button variant="ghost" size="sm" aria-haspopup="dialog" onClick={() => setPlan(true)}>{t('Plan')}</Button>
        <Button variant="ghost" size="sm" onClick={() => { if (!confirm(t('End the session now?'))) return; if (player.getState().elapsed >= 60) player.endEarly(); else onExit() }}>{t('End session')}</Button>
      </div>
      {plan && <PlanSheet player={player} onClose={() => setPlan(false)} />}
      {logging && (
        <Sheet label={t('Log your sets')} onClose={() => setLogging(null)}>
          <div className="flex items-baseline justify-between gap-3">
            <h2 className="m-0 font-display text-xl uppercase">{t('Log your sets')}</h2>
            <span className="shrink-0 font-display text-sm tabular-nums text-muted">{isRest ? t('Rest {time}', { time: fmtTime(st.remainingMs / 1000) }) : phase}</span>
          </div>
          <p className="m-0 mt-1 mb-3 text-sm text-muted">{t("Last time's numbers are filled in, change what's different.")}</p>
          <LiftsEditor lifts={logging} onChange={setLogging} />
          <Button block className="mt-4" onClick={() => { saveLiveLifts(st.startedAt, logging); setLogging(null); setLogged((n) => n + 1) }}>{t('Save sets')}</Button>
        </Sheet>
      )}
      {info !== null && st.steps[info] && (() => {
        const x = st.steps[info]
        return (
          <Sheet label={t('{name} how-to', { name: x.name })} onClose={() => setInfo(null)}>
            <h2 className="m-0 font-display text-xl uppercase leading-tight">{x.name.replace(/\s*\([^)]*\)/g, '')}</h2>
            {x.reps && <p className="m-0 mt-1 font-display text-base uppercase text-[var(--red-ink)]">{fmtReps(x.reps)}{x.rounds > 1 ? ` · ${t('set {n}/{total}', { n: x.round, total: x.rounds })}` : ''}</p>}
            {x.how && <p className="m-0 mt-3">{x.how}</p>}
            <VideoBox name={x.name} />
            <Button block className="mt-4" onClick={() => setInfo(null)}>{t('Back to the session')}</Button>
          </Sheet>
        )
      })()}

      {/* Full-screen cue: readable from a few metres away, phone on the ground */}
      <div aria-hidden={!st.cue} data-testid="cue"
        className={`fixed inset-0 z-50 flex flex-col bg-red px-4 pt-[max(20px,var(--sat))] pb-[max(24px,env(safe-area-inset-bottom))] text-white transition-opacity duration-100 ${st.cue ? 'opacity-100' : 'pointer-events-none opacity-0'}`}>
        <div className="flex justify-between gap-3 font-display text-sm uppercase tracking-wider">
          <span className="min-w-0 truncate">{step.name.replace(/\s*\([^)]*\)/g, '')} · {step.round}/{step.rounds}</span>
          <span className="tabular-nums">{fmtTime(st.remainingMs / 1000)}</span>
        </div>
        {typeof st.cue === 'number' && (
          <div key={st.cueKey} className="slam flex flex-1 flex-col items-center justify-center text-center">
            <div className="font-display text-[clamp(10rem,60vw,18rem)] leading-[.8] tabular-nums">{st.cue}</div>
            <div className="mt-6 font-display text-sm uppercase tracking-wider opacity-90">{st.cue % 2 === 0 ? t('Even · pass it back first time') : t('Odd · turn and go')}</div>
          </div>
        )}
        {typeof st.cue === 'string' && st.cue !== 'left' && st.cue !== 'right' && isWordCue(step.cueKind) && (() => {
          const w = wordFor(st.cue, step.cueKind, moves)
          return w ? (
            <div key={st.cueKey} className="slam flex flex-1 flex-col items-center justify-center text-center">
              <div className="font-display text-[clamp(3.5rem,19vw,7rem)] uppercase leading-[.9]">{w.word}</div>
              {w.hint && <div className="mt-6 font-display text-sm uppercase tracking-wider opacity-90">{w.hint}</div>}
            </div>
          ) : null
        })()}
        {(st.cue === 'left' || st.cue === 'right') && (
          <div key={st.cueKey} className="slam flex flex-1 flex-col items-center justify-center text-center">
            <svg viewBox="0 0 24 24" className="h-[min(70vw,300px)] w-[min(70vw,300px)]" fill="none" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"
              style={{ transform: st.cue === 'right' ? 'scaleX(-1)' : undefined }}>
              <path d="M20 12H5M11 5l-7 7 7 7" />
            </svg>
            <div className="font-display text-[clamp(3.5rem,21vw,7rem)] uppercase leading-[.8]">{st.cue === 'left' ? t('left') : t('right')}</div>
            <div className="mt-6 font-display text-sm uppercase tracking-wider opacity-90">{st.cue === 'left' ? t('Defender commits left · go right') : t('Defender commits right · go left')}</div>
          </div>
        )}
      </div>
    </Screen>
  )
}
