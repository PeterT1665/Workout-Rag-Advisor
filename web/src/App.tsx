import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { allSessions, BUILTIN_SESSIONS, dayKindOf, deleteMySession, saveMySession, templateFor, toSession, type MySession, type Template } from './lib/library'
import { Home } from './views/Home'
import { Library } from './views/Library'
import { Builder } from './views/Builder'
import { Onboarding } from './views/Onboarding'
import { getProfile, reloadProfile, setProfile, useProfile } from './lib/profile'
import { Detail } from './views/Detail'
import { Preroll } from './views/Preroll'
import { PlayerView } from './views/PlayerView'
import { SettingsView } from './views/SettingsView'
import { MediaView } from './views/MediaView'
import { WeekView } from './views/WeekView'
import { PlayerTab } from './views/PlayerTab'
import { TestDay } from './views/TestDay'
import { Invite } from './views/Invite'
import { CoachView } from './views/CoachView'
import { TeamView } from './views/TeamView'
import { publishStats } from './lib/team'
import { cancelRest, refreshPush, scheduleRest } from './lib/push'
import { syncEnabled } from './lib/sync'
import { cardStatsFor } from './lib/cardStats'
import { overall } from './components/PlayerCard'
import { TabBar, type Tab } from './components/TabBar'
import { Tour, homeTour, tourDue } from './components/Tour'
import { removeOldClips } from './lib/clips'
import { openPhotos } from './lib/photos'
import { setBusy } from './lib/updates'
import { attachRemote, beginRemote, endRemote } from './lib/remote'
import { clearActive, loadActive, saveActive } from './lib/activeSession'
import { beep, realAudio, unlockAudio } from './lib/audio'
import { createPlayer, trainedMinutes, type Player, type Snapshot } from './lib/player'
import { nextQuote } from './lib/quotes'
import { addHistory, loadHistory, loadSettings, save } from './lib/storage'
import { loadGames, planToday, saveGames, withChange } from './lib/today'
import { planStreak } from './lib/streak'
import { DEFAULT_GYM } from './lib/weekplan'
import { changeFor, changedSession, loadWeekChanges, saveWeekChange, type DayChange } from './lib/weekChanges'
import type { Game } from './lib/weekplan'
import type { Session, Settings } from './lib/types'
import { langBack, locale, t } from './i18n'

const sessions = BUILTIN_SESSIONS // the week planner picks from the recommended sessions

// Full-screen views shown on top of the tabs.
type Overlay =
  | { name: 'library' }
  | { name: 'detail'; session: Session; fromLibrary?: boolean }
  | { name: 'preroll'; session: Session; quote: string | null }
  | { name: 'player'; session: Session; player: Player }
  | { name: 'settings' }
  | { name: 'media'; from: 'settings' | 'home' }
  | { name: 'builder'; from: Template | null }
  | { name: 'profile' }
  | { name: 'testday' }
  | { name: 'coach' }
  | { name: 'invite'; partner?: boolean; back?: Overlay }

// Make-up sessions for a missed day, by what the plan had: about 30 min at home, or recovery.
const MAKEUP_PICKS: Record<string, string[]> = {
  football: ['home-ball-small-space', 'recovery-hips'], gym: ['home-upper-core', 'recovery-hips'],
  recovery: ['recovery-hips', 'home-full-body'], session: ['home-full-body', 'recovery-hips'],
}
const ymd = (d: Date) => d.toLocaleDateString('en-CA')
const dayName = (date: string) => { const d = new Date(`${date}T12:00:00`).toLocaleDateString(locale(), { weekday: 'long' }); return d.charAt(0).toUpperCase() + d.slice(1) }

// onSignUp: guests can start making an account from inside the app (Team needs one)
export default function App({ onSignUp, onSignIn }: { onSignUp?: () => void; onSignIn?: () => void } = {}) {
  // a notification can open a tab directly ("plan next week" opens the week)
  const [tab, setTab] = useState<Tab>(() => (typeof location !== 'undefined' && new URLSearchParams(location.search).get('tab') === 'week' ? 'week' : 'home'))
  // just switched language: reopen Settings where you were, fading in from the switch
  const [back] = useState(langBack)
  const [overlay, setOverlay] = useState<Overlay | null>(() => (back?.screen === 'settings' ? { name: 'settings' } : null))
  const [cover, setCover] = useState(!!back)
  const restored = useRef(overlay) // that Settings screen comes in under the fade, without its slide
  useEffect(() => {
    if (!back) return
    requestAnimationFrame(() => { window.scrollTo(0, back.y); requestAnimationFrame(() => setCover(false)) })
  }, [back])
  const [tour, setTour] = useState(false) // first-run tour of the buttons
  useEffect(() => {
    if (!tourDue() || new URLSearchParams(location.search).has('preview')) return // not in the laptop page's showcase phones
    const go = setTimeout(() => setTour(true), 900) // after Home has come in
    return () => clearTimeout(go)
  }, [])
  const [settings, setSettings] = useState<Settings>(loadSettings)
  const [games, setGames] = useState<Game[]>(loadGames)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const started = useRef(false)
  const stopPersist = useRef<() => void>(() => {})
  const [active, setActive] = useState(loadActive) // a session left unfinished (app closed or reloaded)
  const gymPerWeek = useProfile().gym
  const [changes, setChanges] = useState(loadWeekChanges)
  const planned = useMemo(() => planToday(sessions, games, new Date(), gymPerWeek), [games, gymPerWeek])
  const [library, setLibrary] = useState(allSessions) // recommended + the user's own
  const today = useMemo(() => withChange(planned, changeFor(planned.day, changes), library), [planned, changes, library])
  // week 1 = next week (planned on Sunday)
  const changeDay = (day: number, change: DayChange | null, week: 0 | 1 = 0) =>
    setChanges(saveWeekChange(day, change, week ? new Date(Date.now() + 7 * 864e5) : new Date()))

  useEffect(() => { removeOldClips(); void openPhotos() }, [])
  // another device changed something: re-read synced state
  const [, setSyncTick] = useState(0)
  useEffect(() => {
    const on = () => { setGames(loadGames()); setSettings(loadSettings()); setLibrary(allSessions()); setChanges(loadWeekChanges()); reloadProfile(); setSyncTick((n) => n + 1) }
    window.addEventListener('ta-sync', on)
    return () => window.removeEventListener('ta-sync', on)
  }, [])
  // reset scroll before the new screen paints, so there is no visible jump
  useLayoutEffect(() => { window.scrollTo(0, 0) }, [tab, overlay?.name])
  // an app update may reload the page: hold it while a session, rating test or edit is open
  useEffect(() => { void refreshPush() }, []) // this phone still registered for notifications, in the app's language
  useEffect(() => { setBusy(tour || ['preroll', 'player', 'testday', 'builder', 'profile', 'media'].includes(overlay?.name ?? '')) }, [overlay?.name, tour]) // the tour isn't cut off by an update either

  const updateSettings = (p: Partial<Settings>) => {
    const next = { ...settingsRef.current, ...p }
    setSettings(next)
    save('settings', next)
  }
  const updateGames = (g: Game[]) => { setGames(g); saveGames(g) }

  function beginSession(session: Session, resume?: Snapshot) {
    if (started.current) return
    started.current = true
    const player = createPlayer(session, {
      audio: realAudio(() => settingsRef.current),
      getSettings: () => settingsRef.current,
      resume,
      onFinish: (st) => {
        clearActive()
        setActive(null)
        // training time actually done (pauses and skipped drills don't count)
        addHistory({
          id: session.id, title: session.title, type: session.type, date: new Date().toLocaleDateString('en-CA'),
          minutes: trainedMinutes(st),
          drills: st.drillsDone, cues: st.cuesCalled,
        })
      },
    })
    // keep the session safe if the app is closed or reloaded mid-way
    let lastSave = 0
    const persist = () => { if (!player.getState().finished) saveActive(player.getSession(), player.snapshot()) } // keeps mid-session edits
    // Rest call-back: booked on the server when a rest starts (the app may be gone by the time it ends),
    // dropped when the rest is paused, skipped, or ends with you still in the app.
    let booked = '' // which rest is booked
    let dropped = false // ...and already dropped because you stayed
    const nextName = (s: ReturnType<typeof player.getState>) => s.steps[s.i + 1]?.name.replace(/\s*\([^)]*\)/g, '')
    const syncRest = () => {
      const s = player.getState()
      const key = !s.finished && s.running && s.steps[s.i]?.type === 'rest' ? `${s.startedAt}-${s.i}` : ''
      if (key !== booked) {
        if (key) void scheduleRest(s.remainingMs / 1000, nextName(s)) // replaces the old one on the server
        else if (booked && !dropped) void cancelRest()
        booked = key
        dropped = false
      } else if (key && !dropped && document.visibilityState === 'visible' && s.remainingMs < 3000) {
        dropped = true
        void cancelRest()
      }
    }
    player.subscribe(() => { syncRest(); if (Date.now() - lastSave > 1500) { lastSave = Date.now(); persist() } })
    const onHide = () => {
      const s = player.getState()
      if (document.visibilityState === 'hidden') {
        persist()
        // re-book with the exact time left, in case the first booking didn't get through
        if (booked) { dropped = false; void scheduleRest(s.remainingMs / 1000, nextName(s)) }
      }
    }
    document.addEventListener('visibilitychange', onHide)
    stopPersist.current = () => document.removeEventListener('visibilitychange', onHide)
    // earbud presses (started in the Kick off tap) now drive this player
    attachRemote({ press: player.press, next: player.skip, prev: player.back })
    setOverlay({ name: 'player', session, player })
  }

  // Runs inside the tap, so iOS allows audio (whistle, voice, earbud controls).
  function onStart(session: Session) {
    unlockAudio()
    started.current = false
    const s = settingsRef.current
    if (s.earbudControls) beginRemote(session.title)
    // referee whistle: two short blasts
    beep(s, 2900, 90); setTimeout(() => beep(s, 2900, 260), 160)
    setOverlay({ name: 'preroll', session, quote: s.cards ? nextQuote() : null })
  }

  const close = () => setOverlay(null)
  const open = (session: Session) => setOverlay({ name: 'detail', session })
  const openFromLibrary = (session: Session) => setOverlay({ name: 'detail', session, fromLibrary: true })
  const build = (from: Template | null) => setOverlay({ name: 'builder', from })
  const saveBuilt = (s: MySession) => { saveMySession(s); setLibrary(allSessions()); setOverlay({ name: 'detail', session: toSession(s), fromLibrary: true }) }
  const deleteBuilt = (id: string) => { deleteMySession(id); setLibrary(allSessions()); setOverlay({ name: 'library' }) }

  // plan streak, and a missed day that can still be made up (Home shows ready-made sessions for it)
  const streak = planStreak(loadHistory(), games, gymPerWeek ?? DEFAULT_GYM, changes)
  // signed in: friends see your streak, rating and when you were last on (on open, on changes, every 3 min while open)
  const me = useProfile()
  useEffect(() => {
    if (!syncEnabled()) return
    const publish = () => {
      const card = cardStatsFor(me)
      void publishStats(me, { card, rating: card ? overall(card) : null, streak: streak.days, trainedToday: streak.today })
    }
    publish()
    const every = setInterval(() => { if (document.visibilityState === 'visible') publish() }, 180000)
    return () => clearInterval(every)
  }, [me, streak.days, streak.today])
  const makeup = streak.makeup ? {
    missed: dayName(streak.makeup.date),
    by: streak.makeup.by === ymd(new Date()) ? t('tonight') : streak.makeup.by === ymd(new Date(Date.now() + 864e5)) ? t('tomorrow') : dayName(streak.makeup.by),
    picks: (MAKEUP_PICKS[streak.makeup.kind] ?? MAKEUP_PICKS.session).map((id) => library.find((s) => s.id === id)).filter((s): s is Session => !!s),
  } : null

  const fade = back && <div aria-hidden="true" className={`pointer-events-none fixed inset-0 z-[100] bg-bg transition-opacity duration-200 ease-out ${cover ? 'opacity-100' : 'opacity-0'}`} />
  if (overlay) {
    // full-screen moments (kick-off, player) bring their own entrance
    if (overlay.name === 'preroll' || overlay.name === 'player') return renderOverlay(overlay)
    return <div key={overlay.name} className={overlay === restored.current ? '' : 'screen-in'}>{renderOverlay(overlay)}{fade}</div>
  }

  function renderOverlay(overlay: Overlay) {
    switch (overlay.name) {
      case 'coach':
        return <CoachView onBack={() => { close(); setTab('week') }} />
      case 'library':
        return <Library sessions={library} onOpen={openFromLibrary} onNew={() => build(null)} onBack={close} />
      case 'detail':
        return <Detail session={overlay.session} onSchedule={(day: number) => { changeDay(day, { sessionId: overlay.session.id }); setTab(day === today.day ? 'home' : 'week'); close() }} onBack={() => (overlay.fromLibrary ? setOverlay({ name: 'library' }) : close())} onStart={() => onStart(overlay.session)} onAlternative={(alt) => setOverlay({ ...overlay, session: alt })}
          onEdit={() => { const t = templateFor(overlay.session.id); if (t) build(t) }}
          onInvite={() => setOverlay({ name: 'invite', partner: true, back: overlay })} />
      case 'builder':
        return <Builder from={overlay.from} onSave={saveBuilt} onDelete={deleteBuilt}
          onCancel={() => setOverlay(overlay.from ? { name: 'detail', session: toSession(overlay.from), fromLibrary: true } : { name: 'library' })} />
      case 'preroll':
        return <Preroll session={overlay.session} quote={overlay.quote} onGo={() => beginSession(overlay.session)} />
      case 'player':
        return <PlayerView key={overlay.player.getState().startedAt} session={overlay.session} player={overlay.player} settings={settings} onExit={() => { endRemote(); stopPersist.current(); void cancelRest(); clearActive(); setActive(null); close() }} />
      case 'settings':
        return <SettingsView settings={settings} onChange={updateSettings} onMedia={() => setOverlay({ name: 'media', from: 'settings' })} onProfile={() => setOverlay({ name: 'profile' })} onInvite={() => setOverlay({ name: 'invite', back: { name: 'settings' } })} onTour={() => { close(); setTab('home'); setTour(true) }} onDone={close} />
      case 'invite':
        return <Invite partner={overlay.partner} onDone={() => (overlay.back ? setOverlay(overlay.back) : close())} />
      case 'testday':
        return <TestDay onDone={close} />
      case 'profile':
        return <Onboarding initial={getProfile()} initialGames={games} onCancel={() => setOverlay({ name: 'settings' })}
          onDone={(p, g) => { setProfile({ ...p, sample: getProfile().sample }); updateGames(g); setOverlay({ name: 'settings' }) }} />
      case 'media':
        return <MediaView onDone={() => (overlay.from === 'settings' ? setOverlay({ name: 'settings' }) : close())} />
    }
  }

  return (
    <>
      {/* min-h-lvh: a page shorter than the screen lets iOS home-screen apps lift the fixed tab bar */}
      <div key={tab} className="tab-in min-h-lvh">
      {tab === 'home' && (
        <Home
          today={today}
          onKickOff={onStart}
          onOpen={open}
          onSettings={() => setOverlay({ name: 'settings' })}
          onAllSessions={() => setOverlay({ name: 'library' })}
          onAddClips={() => setOverlay({ name: 'media', from: 'home' })}
          onPlayer={() => setTab('player')}
          streak={streak}
          makeup={makeup}
          active={active ? { title: active.session.title, onResume: () => { unlockAudio(); if (settingsRef.current.earbudControls) beginRemote(active.session.title); started.current = false; beginSession(active.session, active.snap) }, onDiscard: () => { clearActive(); setActive(null) } } : null}
          dayKind={(day) => {
            const c = changeFor(day, changes)
            if (!c) return undefined
            const s = changedSession(c, library)
            return s ? dayKindOf(s) : null
          }}
        />
      )}
      {tab === 'week' && <WeekView today={today} games={games} sessions={sessions} library={library} changes={changes} onChange={changeDay} onGames={updateGames} onOpen={open} onCoach={() => setOverlay({ name: 'coach' })} />}
      {tab === 'player' && <PlayerTab onTestDay={() => setOverlay({ name: 'testday' })} />}
      {tab === 'team' && <TeamView streak={streak} onSignUp={onSignUp} onSignIn={onSignIn} onInvite={() => setOverlay({ name: 'invite' })} />}
      </div>
      <TabBar tab={tab} onTab={setTab} onKickOff={() => onStart(today.session)} />
      {tour && tab === 'home' && <Tour steps={homeTour()} onDone={() => setTour(false)} />}
    </>
  )
}
