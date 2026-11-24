import { useEffect, useRef, useState } from 'react'
import { Button, Screen } from '../components/ui'
import { PlayerCard, overall } from '../components/PlayerCard'
import { TESTS, WARMUP, checkResult, looksOff, ratingsFrom, startingPoint, toCardStats, type Stat, type Test } from '../lib/testday'
import { Demo3D } from '../components/Demo3D'
import { ACCENTS, getProfile, setProfile, type Profile } from '../lib/profile'
import { setDayPhoto, useDayPhotos } from '../lib/dayPhotos'
import { ShareSheet } from './CalendarView'
import { ratingShare } from '../lib/shareStats'
import { beep, say, unlockAudio } from '../lib/audio'
import { loadSettings } from '../lib/storage'
import { usePersonalImage } from '../lib/personal'
import { t } from '../i18n'
import { tr } from '../i18n/rich'

// Rating session: one test per screen (setup, how, timer, one number), then the card reveal.
// The only place the app asks for numbers mid-session: rating day is a test, not a workout.
export function TestDay({ onDone }: { onDone: () => void }) {
  const [i, setI] = useState(-1) // -1 intro, TESTS.length = reveal
  const [warm, setWarm] = useState<number | null>(null) // warm-up step, before the tests
  const [raw, setRaw] = useState<Partial<Record<Stat, number>>>({})
  const [saved, setSaved] = useState<Profile | null>(null)
  const [verify, setVerify] = useState(false)
  const photos = useDayPhotos()
  const photo = usePersonalImage('card.jpg')
  const me = getProfile()

  const finish = () => {
    const r = ratingsFrom(raw)
    // the first session is the starting point; later ones are measured from it
    const next = { ...me, baseRatings: startingPoint(me) ?? me.ratings, prevRatings: undefined, ratings: r }
    setProfile(next)
    setSaved(next)
    setI(TESTS.length)
  }

  if (i === -1 && warm !== null) {
    return <WarmUp step={warm} onStep={setWarm} onClose={() => setWarm(null)} onReady={() => { setWarm(null); setI(0) }} />
  }

  if (i === -1) {
    return (
      <Screen className="flex flex-col">
        <div className="mb-4 flex justify-between"><Button variant="ghost" size="sm" onClick={onDone}>{t('Close')}</Button></div>
        <h1 className="m-0 font-display text-[clamp(2.2rem,10vw,3rem)] uppercase leading-[.9]">{tr('Rating <r>session</r>', { r: (x) => <><br /><span className="text-red-ink">{x}</span></> })}</h1>
        <p className="mt-4 text-muted">{t('Six tests, about 30 minutes. Each sets a card stat.')}</p>
        <ul className="m-0 mt-4 flex list-none flex-col gap-2 p-0">
          {TESTS.map((test) => (
            <li key={test.stat} className="flex items-center gap-3 rounded-2xl bg-surface px-4 py-3">
              <span className="w-12 font-display text-sm text-red-ink">{test.stat}</span><span className="font-semibold">{test.name}</span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-sm text-muted">{t('Bring a ball, 6 markers and a wall. A guided {n}-minute warm-up comes first: warm legs give real, comparable results.', { n: WARMUP.reduce((a, w) => a + w.minutes, 0) })}</p>
        <Button block className="mt-auto" onClick={() => { unlockAudio(); setWarm(0) }}>{t('Start warm-up')}</Button>
        <Button block variant="ghost" className="mt-2" onClick={() => { unlockAudio(); setI(0) }}>{t('Already warm: go to the tests')}</Button>
      </Screen>
    )
  }

  if (i === TESTS.length && saved) {
    const stats = toCardStats(saved.ratings)
    const odd = saved.ratings ? looksOff(saved.ratings) : []
    const ovr = stats ? overall(stats, saved.positions) : null
    const tier = ovr === null ? null : ovr >= 75 ? t('Gold') : ovr >= 65 ? t('Silver') : 'Bronze'
    return (
      <Screen className="flex flex-col items-center text-center">
        <p className="m-0 mt-2 font-display text-sm uppercase text-red-ink">{t('Rating session done')}</p>
        <h1 className="m-0 mt-2 font-display text-4xl uppercase leading-none">{tier ? `${tier} card` : t('Almost there')}</h1>
        <div className="pop mt-6 w-full max-w-[280px]"><PlayerCard p={saved} stats={stats} photo={photo} /></div>
        <p className="mt-6 max-w-[34ch] text-muted">
          {stats ? t('Redo it every week, same day and same warm-up, so the card follows your real progress.') : t('Do at least 3 tests to get rated.')}
        </p>
        {stats && saved.ratings && <Proof date={saved.ratings.date} odd={odd.length > 0} added={!!saved.ratings.proof}
          onAdded={() => { const next = { ...saved, ratings: { ...saved.ratings!, proof: true } }; setProfile(next); setSaved(next) }}
          onVerify={() => setVerify(true)} />}
        {verify && saved.ratings && <ShareSheet item={ratingShare(saved.ratings)} photo={photos[saved.ratings.date]} streak={0} accent={ACCENTS.find((a) => a.id === saved.accent)?.hex ?? '#e01b35'} onClose={() => setVerify(false)} />}
        <Button block className="mt-6" onClick={onDone}>{t('Done')}</Button>
      </Screen>
    )
  }

  const test = TESTS[i]
  return (
    <Screen className="flex flex-col">
      <div className="mb-5 flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => setI(i - 1)}>{t('Back')}</Button>
        <div className="flex flex-1 gap-1" aria-label={t('Test {n} of {total}', { n: i + 1, total: TESTS.length })}>
          {TESTS.map((_, k) => <div key={k} className={`h-1.5 flex-1 rounded-full ${k <= i ? 'bg-red' : 'bg-surface-2'}`} />)}
        </div>
      </div>
      <TestStep key={test.stat} test={test} value={raw[test.stat]} onValue={(v) => setRaw((r) => ({ ...r, [test.stat]: v }))} />
      <div className="mt-6 flex gap-3">
        <Button variant="ghost" className="flex-1" onClick={() => { setRaw((r) => { const n = { ...r }; delete n[test.stat]; return n }); i === TESTS.length - 1 ? finish() : setI(i + 1) }}>{t('Skip')}</Button>
        <Button className="flex-[2]" disabled={!raw[test.stat] || checkResult(test.stat, raw[test.stat]!) === 'impossible'} onClick={() => (i === TESTS.length - 1 ? finish() : setI(i + 1))}>
          {i === TESTS.length - 1 ? t('See my card') : t('Next test')}
        </Button>
      </div>
    </Screen>
  )
}

function TestStep({ test, value, onValue }: { test: Test; value?: number; onValue: (v: number) => void }) {
  const check = value ? checkResult(test.stat, value) : null
  return (
    <div className="rise flex-1">
      <p className="m-0 font-display text-sm uppercase text-red-ink">{test.stat}</p>
      <h1 className="m-0 mt-1 font-display text-3xl uppercase leading-none">{test.name}</h1>
      <p className="mt-4 mb-1 font-semibold">{test.setup}</p>
      <p className="m-0 text-muted">{test.how}</p>
      {test.timer === 'stopwatch' && <Stopwatch onStop={(s) => onValue(Math.round(s * 100) / 100)} />}
      {test.timer === 'countdown' && <Countdown seconds={test.seconds ?? 30} />}
      <label htmlFor={`result-${test.stat}`} className="mt-6 block text-sm text-muted">{t('Your result')}</label>
      <div className="mt-1 flex items-center gap-3">
        <input id={`result-${test.stat}`} inputMode="decimal" value={value ?? ''} placeholder="0"
          onChange={(e) => { const v = Number(e.target.value.replace(',', '.')); if (!Number.isNaN(v)) onValue(v) }}
          className="block min-h-14 w-32 rounded-2xl bg-surface px-4 text-center font-display text-2xl text-ink placeholder:text-muted" />
        <span className="font-display text-lg text-muted">{test.unit}</span>
      </div>
      {check === 'impossible' && <p role="alert" className="m-0 mt-3 text-sm font-semibold text-red-ink">{t("That result isn't possible. Check it and enter it again.")}</p>}
      {check === 'proof' && <p className="m-0 mt-3 text-sm text-muted">{t('A world-class result. At the end you will be asked for a photo and a friend to check it.')}</p>}
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
    <button onClick={toggle} aria-label={start === null ? t('Start stopwatch') : t('Stop stopwatch')}
      className={`mt-5 flex min-h-24 w-full items-center justify-between rounded-[24px] px-6 transition-colors duration-150 ${start === null ? 'bg-surface' : 'bg-red text-on-red'}`}>
      <span className="font-display text-sm uppercase">{start === null ? (ms ? t('Again') : t('Start')) : t('Stop')}</span>
      <span className="font-display text-5xl tabular-nums">{fmt(ms)}</span>
    </button>
  )
}

const clock = (n: number) => (n >= 60 ? `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}` : `${n}s`)

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
      aria-label={t('Start {n} second countdown', { n: seconds })}
      className={`mt-5 flex min-h-24 w-full items-center justify-between rounded-[24px] px-6 ${left !== null && left > 0 ? 'bg-red text-on-red' : 'bg-surface'}`}>
      <span className="font-display text-sm uppercase">{left === null ? t('Start') : left > 0 ? 'Go' : 'Time · again'}</span>
      <span className="font-display text-5xl tabular-nums">{clock(left ?? seconds)}</span>
    </button>
  )
}

// The warm-up: one step at a time with its 3D demo and a timer (a beep when the time is up), then the tests.
function WarmUp({ step, onStep, onClose, onReady }: { step: number; onStep: (n: number) => void; onClose: () => void; onReady: () => void }) {
  const w = WARMUP[step]
  const last = step === WARMUP.length - 1
  return (
    <Screen className="flex flex-col">
      <div className="mb-5 flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => (step ? onStep(step - 1) : onClose())}>{t('Back')}</Button>
        <div className="flex flex-1 gap-1" aria-label={t('Warm-up step {n} of {total}', { n: step + 1, total: WARMUP.length })}>
          {WARMUP.map((_, k) => <div key={k} className={`h-1.5 flex-1 rounded-full ${k <= step ? 'bg-ink/60' : 'bg-surface-2'}`} />)}
        </div>
      </div>
      <div key={step} className="rise flex-1">
        <p className="m-0 font-display text-sm uppercase text-muted">{t('Warm-up')} · {t('{n} min', { n: w.minutes })}</p>
        <h1 className="m-0 mt-1 font-display text-3xl uppercase leading-none">{w.name}</h1>
        <p className="mt-3 mb-0 text-muted">{w.how}</p>
        <Demo3D name={w.demo} className="mt-4" />
        <Countdown key={step} seconds={w.minutes * 60} />
      </div>
      <Button block className="mt-6" onClick={() => (last ? onReady() : onStep(step + 1))}>{last ? t('Ready: start the tests') : t('Next')}</Button>
    </Screen>
  )
}

// After a rating session: a photo of the setup or the result, and a friend to check it. Odd-looking results ask for it
// more firmly. The photo stays on the phone; "Ask a friend" opens the share sheet with the results picture.
function Proof({ date, odd, added, onAdded, onVerify }: { date: string; odd: boolean; added: boolean; onAdded: () => void; onVerify: () => void }) {
  return (
    <section className="mt-6 w-full max-w-[340px] rounded-[22px] bg-surface p-4 text-left" aria-label={t('Make it count')}>
      <h2 className="m-0 font-display text-sm uppercase">{odd ? t('Some results look unusual') : t('Make it count')}</h2>
      <p className="m-0 mt-1.5 text-sm text-muted">{odd ? t('Add a photo and ask a friend to check them, so your card is believed.') : t('A photo of your test and a friend who checks it make your card count with your friends.')}</p>
      <div className="mt-3 flex gap-2">
        <label className="flex min-h-11 flex-1 cursor-pointer items-center justify-center rounded-full bg-surface-2 px-3 text-center text-xs font-semibold active:scale-95">
          {added ? t('Photo added ✓') : t('Add a photo')}
          <input type="file" accept="image/*" capture="environment" className="sr-only" aria-label={t('Add a photo of your rating session')}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void setDayPhoto(date, f).then(onAdded); e.target.value = '' }} />
        </label>
        <button onClick={onVerify} className="min-h-11 flex-1 rounded-full bg-red px-3 text-xs font-semibold text-on-red active:scale-95">{t('Ask a friend to verify')}</button>
      </div>
    </section>
  )
}
