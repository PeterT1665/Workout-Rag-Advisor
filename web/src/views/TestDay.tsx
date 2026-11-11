import { useEffect, useRef, useState } from 'react'
import { Button, Screen } from '../components/ui'
import { PlayerCard, overall } from '../components/PlayerCard'
import { TESTS, ratingsFrom, toCardStats, type Stat, type Test } from '../lib/testday'
import { getProfile, setProfile, type Profile } from '../lib/profile'
import { beep, say, unlockAudio } from '../lib/audio'
import { loadSettings } from '../lib/storage'
import { usePersonalImage } from '../lib/personal'

// Rating session: one test per screen (setup, how, timer, one number), then the card reveal.
// The only place the app asks for numbers mid-session: rating day is a test, not a workout.
export function TestDay({ onDone }: { onDone: () => void }) {
  const [i, setI] = useState(-1) // -1 intro, TESTS.length = reveal
  const [raw, setRaw] = useState<Partial<Record<Stat, number>>>({})
  const [saved, setSaved] = useState<Profile | null>(null)
  const photo = usePersonalImage('card.jpg')
  const me = getProfile()

  const finish = () => {
    const r = ratingsFrom(raw)
    const next = { ...me, prevRatings: me.ratings, ratings: r }
    setProfile(next)
    setSaved(next)
    setI(TESTS.length)
  }

  if (i === -1) {
    return (
      <Screen className="flex flex-col">
        <div className="mb-4 flex justify-between"><Button variant="ghost" size="sm" onClick={onDone}>Close</Button></div>
        <h1 className="m-0 font-display text-[clamp(2.2rem,10vw,3rem)] uppercase leading-[.9]">Rating<br /><span className="text-red-ink">session</span></h1>
        <p className="mt-4 text-muted">Six tests, about 30 minutes. Each sets a card stat.</p>
        <ul className="m-0 mt-4 flex list-none flex-col gap-2 p-0">
          {TESTS.map((t) => (
            <li key={t.stat} className="flex items-center gap-3 rounded-2xl bg-surface px-4 py-3">
              <span className="w-12 font-display text-sm text-red-ink">{t.stat}</span><span className="font-semibold">{t.name}</span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-sm text-muted">Bring a ball, 6 markers and a wall. Warm up first.</p>
        <Button block className="mt-auto" onClick={() => { unlockAudio(); setI(0) }}>Start</Button>
      </Screen>
    )
  }

  if (i === TESTS.length && saved) {
    const stats = toCardStats(saved.ratings, saved.prevRatings)
    const ovr = stats ? overall(stats) : null
    const tier = ovr === null ? null : ovr >= 75 ? 'Gold' : ovr >= 65 ? 'Silver' : 'Bronze'
    return (
      <Screen className="flex flex-col items-center text-center">
        <p className="m-0 mt-2 font-display text-sm uppercase text-red-ink">Rating session done</p>
        <h1 className="m-0 mt-2 font-display text-4xl uppercase leading-none">{tier ? `${tier} card` : 'Almost there'}</h1>
        <div className="pop mt-6 w-full max-w-[280px]"><PlayerCard p={saved} stats={stats} photo={photo} /></div>
        <p className="mt-6 max-w-[34ch] text-muted">
          {stats ? 'Redo it every 6-8 weeks.' : 'Do at least 3 tests to get rated.'}
        </p>
        <Button block className="mt-auto" onClick={onDone}>Done</Button>
      </Screen>
    )
  }

  const t = TESTS[i]
  return (
    <Screen className="flex flex-col">
      <div className="mb-5 flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => setI(i - 1)}>Back</Button>
        <div className="flex flex-1 gap-1" aria-label={`Test ${i + 1} of ${TESTS.length}`}>
          {TESTS.map((_, k) => <div key={k} className={`h-1.5 flex-1 rounded-full ${k <= i ? 'bg-red' : 'bg-surface-2'}`} />)}
        </div>
      </div>
      <TestStep key={t.stat} t={t} value={raw[t.stat]} onValue={(v) => setRaw((r) => ({ ...r, [t.stat]: v }))} />
      <div className="mt-6 flex gap-3">
        <Button variant="ghost" className="flex-1" onClick={() => { setRaw((r) => { const n = { ...r }; delete n[t.stat]; return n }); i === TESTS.length - 1 ? finish() : setI(i + 1) }}>Skip</Button>
        <Button className="flex-[2]" disabled={!raw[t.stat]} onClick={() => (i === TESTS.length - 1 ? finish() : setI(i + 1))}>
          {i === TESTS.length - 1 ? 'See my card' : 'Next test'}
        </Button>
      </div>
    </Screen>
  )
}

function TestStep({ t, value, onValue }: { t: Test; value?: number; onValue: (v: number) => void }) {
  return (
    <div className="rise flex-1">
      <p className="m-0 font-display text-sm uppercase text-red-ink">{t.stat}</p>
      <h1 className="m-0 mt-1 font-display text-3xl uppercase leading-none">{t.name}</h1>
      <p className="mt-4 mb-1 font-semibold">{t.setup}</p>
      <p className="m-0 text-muted">{t.how}</p>
      {t.timer === 'stopwatch' && <Stopwatch onStop={(s) => onValue(Math.round(s * 100) / 100)} />}
      {t.timer === 'countdown' && <Countdown seconds={t.seconds ?? 30} />}
      <label htmlFor={`result-${t.stat}`} className="mt-6 block text-sm text-muted">Your result</label>
      <div className="mt-1 flex items-center gap-3">
        <input id={`result-${t.stat}`} inputMode="decimal" value={value ?? ''} placeholder="0"
          onChange={(e) => { const v = Number(e.target.value.replace(',', '.')); if (!Number.isNaN(v)) onValue(v) }}
          className="block min-h-14 w-32 rounded-2xl bg-surface px-4 text-center font-display text-2xl text-ink placeholder:text-muted" />
        <span className="font-display text-lg text-muted">{t.unit}</span>
      </div>
    </div>
  )
}

const fmt = (ms: number) => `${Math.floor(ms / 1000)}.${String(Math.floor((ms % 1000) / 10)).padStart(2, '0')}`

function Stopwatch({ onStop }: { onStop: (seconds: number) => void }) {
  const [start, setStart] = useState<number | null>(null)
  const [ms, setMs] = useState(0)
  const raf = useRef(0)
  useEffect(() => {
    if (start === null) return
    const tick = () => { setMs(performance.now() - start); raf.current = requestAnimationFrame(tick) }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [start])
  const toggle = () => {
    if (start === null) { setMs(0); setStart(performance.now()); beep(loadSettings(), 1800, 120) }
    else { const s = (performance.now() - start) / 1000; setStart(null); setMs(s * 1000); onStop(s) }
  }
  return (
    <button onClick={toggle} aria-label={start === null ? 'Start stopwatch' : 'Stop stopwatch'}
      className={`mt-5 flex min-h-24 w-full items-center justify-between rounded-[24px] px-6 transition-colors duration-150 ${start === null ? 'bg-surface' : 'bg-red text-on-red'}`}>
      <span className="font-display text-sm uppercase">{start === null ? (ms ? 'Again' : 'Start') : 'Stop'}</span>
      <span className="font-display text-5xl tabular-nums">{fmt(ms)}</span>
    </button>
  )
}

function Countdown({ seconds }: { seconds: number }) {
  const [left, setLeft] = useState<number | null>(null)
  useEffect(() => {
    if (left === null || left <= 0) return
    const id = setTimeout(() => setLeft(left - 1), 1000)
    return () => clearTimeout(id)
  }, [left])
  useEffect(() => {
    if (left === 0) { const s = loadSettings(); beep(s, 2400, 400); say(s, 'Time') }
  }, [left])
  return (
    <button onClick={() => { const s = loadSettings(); beep(s, 1800, 120); setLeft(seconds) }} disabled={left !== null && left > 0}
      aria-label={`Start ${seconds} second countdown`}
      className={`mt-5 flex min-h-24 w-full items-center justify-between rounded-[24px] px-6 ${left !== null && left > 0 ? 'bg-red text-on-red' : 'bg-surface'}`}>
      <span className="font-display text-sm uppercase">{left === null ? 'Start' : left > 0 ? 'Go' : 'Time · again'}</span>
      <span className="font-display text-5xl tabular-nums">{left ?? seconds}s</span>
    </button>
  )
}
