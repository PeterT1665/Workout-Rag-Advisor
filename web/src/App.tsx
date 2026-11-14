import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
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
import { TabBar, type Tab } from './components/TabBar'
import { allClips, openClips, subscribeClips } from './lib/clips'
import { openPhotos } from './lib/photos'
import { setBusy } from './lib/updates'
import { attachRemote, beginRemote, endRemote } from './lib/remote'
import { clearActive, loadActive, saveActive } from './lib/activeSession'
import { beep, playClip, realAudio, stopClip, unlockAudio } from './lib/audio'
import { createPlayer, type Player, type Snapshot } from './lib/player'
import { nextQuote } from './lib/quotes'
import { addHistory, loadHistory, loadSettings, save } from './lib/storage'
import { loadGames, planToday, saveGames, withChange } from './lib/today'
import { planStreak } from './lib/streak'
import { DEFAULT_GYM } from './lib/weekplan'
import { changeFor, changedSession, loadWeekChanges, saveWeekChange, type DayChange } from './lib/weekChanges'
import type { Game } from './lib/weekplan'
import type { Session, Settings } from './lib/types'

const sessions = BUILTIN_SESSIONS // the week planner picks from the recommended sessions

// Full-screen views shown on top of the tabs.
type Overlay =
  | { name: 'library' }
  | { name: 'detail'; session: Session; fromLibrary?: boolean }
  | { name: 'preroll'; session: Session; quote: string | null; clipName: string | null }
  | { name: 'player'; session: Session; player: Player }
  | { name: 'settings' }
  | { name: 'media'; from: 'settings' | 'home' }
  | { name: 'builder'; from: Template | null }
  | { name: 'profile' }
  | { name: 'testday' }
  | { name: 'invite'; partner?: boolean; back?: Overlay }

export default function App() {
  const [tab, setTab] = useState<Tab>('home')
  const [overlay, setOverlay] = useState<Overlay | null>(null)
  const [settings, setSettings] = useState<Settings>(loadSettings)
  const [games, setGames] = useState<Game[]>(loadGames)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const clips = useSyncExternalStore(subscribeClips, allClips)
  const started = useRef(false)
  const stopPersist = useRef<() => void>(() => {})
  const [active, setActive] = useState(loadActive) // a session left unfinished (app closed or reloaded)
  const gymPerWeek = useProfile().gym
  const [changes, setChanges] = useState(loadWeekChanges)
  const planned = useMemo(() => planToday(sessions, games, new Date(), gymPerWeek), [games, gymPerWeek])
  const [library, setLibrary] = useState(allSessions) // recommended + the user's own
  const today = useMemo(() => withChange(planned, changeFor(planned.day, changes), library), [planned, changes, library])
  const changeDay = (day: number, change: DayChange | null) => setChanges(saveWeekChange(day, change))

  useEffect(() => { void openClips(); void openPhotos() }, [])
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
  useEffect(() => { setBusy(['preroll', 'player', 'testday', 'builder', 'profile', 'media'].includes(overlay?.name ?? '')) }, [overlay?.name])

  const updateSettings = (p: Partial<Settings>) => {
    const next = { ...settingsRef.current, ...p }
    setSettings(next)
    save('settings', next)
  }
  const updateGames = (g: Game[]) => { setGames(g); saveGames(g) }

  function beginSession(session: Session, resume?: Snapshot) {
    if (started.current) return
    started.current = true
    stopClip()
    const player = createPlayer(session, {
      audio: realAudio(() => settingsRef.current),
      getSettings: () => settingsRef.current,
      resume,
      onFinish: (st) => {
        clearActive()
        setActive(null)
        // training time actually covered (pauses and breaks don't count)
        addHistory({
          id: session.id, title: session.title, date: new Date().toLocaleDateString('en-CA'),
          minutes: Math.max(1, Math.round(st.elapsed / 60)),
          drills: st.drillsDone, cues: st.cuesCalled,
        })
      },
    })
    // keep the session safe if the app is closed or reloaded mid-way
    let lastSave = 0
    const persist = () => { if (!player.getState().finished) saveActive(player.getSession(), player.snapshot()) } // keeps mid-session edits
    player.subscribe(() => { if (Date.now() - lastSave > 1500) { lastSave = Date.now(); persist() } })
    const onHide = () => { if (document.visibilityState === 'hidden') persist() }
    document.addEventListener('visibilitychange', onHide)
    stopPersist.current = () => document.removeEventListener('visibilitychange', onHide)
    // earbud presses (started in the Kick off tap) now drive this player
    attachRemote({ press: player.press, next: player.skip, prev: player.back })
    setOverlay({ name: 'player', session, player })
  }

  // Runs inside the tap, so iOS allows the clip to play.
  function onStart(session: Session) {
    unlockAudio()
    started.current = false
    const s = settingsRef.current
    if (s.earbuds) beginRemote(session.title)
    // referee whistle: two short blasts
    beep(s, 2900, 90); setTimeout(() => beep(s, 2900, 260), 160)
    const wantClip = s.clipStart && clips.some((c) => c.kind === 'mindset')
    const clip = wantClip ? playClip(s, 'mindset', () => beginSession(session)) : null
    setOverlay({ name: 'preroll', session, quote: s.cards ? nextQuote() : null, clipName: clip?.name ?? null })
  }

  const close = () => setOverlay(null)
  const open = (session: Session) => setOverlay({ name: 'detail', session })
  const openFromLibrary = (session: Session) => setOverlay({ name: 'detail', session, fromLibrary: true })
  const build = (from: Template | null) => setOverlay({ name: 'builder', from })
  const saveBuilt = (s: MySession) => { saveMySession(s); setLibrary(allSessions()); setOverlay({ name: 'detail', session: toSession(s), fromLibrary: true }) }
  const deleteBuilt = (id: string) => { deleteMySession(id); setLibrary(allSessions()); setOverlay({ name: 'library' }) }

  if (overlay) {
    // full-screen moments (kick-off, player) bring their own entrance
    if (overlay.name === 'preroll' || overlay.name === 'player') return renderOverlay(overlay)
    return <div key={overlay.name} className="screen-in">{renderOverlay(overlay)}</div>
  }

  function renderOverlay(overlay: Overlay) {
    switch (overlay.name) {
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
        return <Preroll session={overlay.session} quote={overlay.quote} clipName={overlay.clipName} onGo={() => beginSession(overlay.session)} />
      case 'player':
        return <PlayerView key={overlay.player.getState().startedAt} session={overlay.session} player={overlay.player} settings={settings} onExit={() => { endRemote(); stopPersist.current(); clearActive(); setActive(null); close() }} />
      case 'settings':
        return <SettingsView settings={settings} clipCount={clips.length} onChange={updateSettings} onMedia={() => setOverlay({ name: 'media', from: 'settings' })} onProfile={() => setOverlay({ name: 'profile' })} onInvite={() => setOverlay({ name: 'invite', back: { name: 'settings' } })} onDone={close} />
      case 'invite':
        return <Invite partner={overlay.partner} onDone={() => (overlay.back ? setOverlay(overlay.back) : close())} />
      case 'testday':
        return <TestDay onDone={close} />
      case 'profile':
        return <Onboarding initial={getProfile()} initialGames={games} onCancel={() => setOverlay({ name: 'settings' })}
          onDone={(p, g) => { setProfile({ ...p, sample: getProfile().sample }); updateGames(g); setOverlay({ name: 'settings' }) }} />
      case 'media':
        return <MediaView settings={settings} onDone={() => (overlay.from === 'settings' ? setOverlay({ name: 'settings' }) : close())} />
    }
  }

  return (
    <>
      {/* min-h-lvh: a page shorter than the screen lets iOS home-screen apps lift the fixed tab bar */}
      <div key={tab} className="tab-in min-h-lvh">
      {tab === 'home' && (
        <Home
          today={today}
          hasClips={clips.some((c) => c.kind === 'mindset')}
          onKickOff={onStart}
          onOpen={open}
          onSettings={() => setOverlay({ name: 'settings' })}
          onAllSessions={() => setOverlay({ name: 'library' })}
          onPlayClip={() => { unlockAudio(); playClip(settingsRef.current, 'mindset') }}
          onAddClips={() => setOverlay({ name: 'media', from: 'home' })}
          onPlayer={() => setTab('player')}
          streak={planStreak(loadHistory(), games, gymPerWeek ?? DEFAULT_GYM, changes)}
          onWeek={() => setTab('week')}
          active={active ? { title: active.session.title, onResume: () => { unlockAudio(); if (settingsRef.current.earbuds) beginRemote(active.session.title); started.current = false; beginSession(active.session, active.snap) }, onDiscard: () => { clearActive(); setActive(null) } } : null}
          dayKind={(day) => {
            const c = changeFor(day, changes)
            if (!c) return undefined
            const s = changedSession(c, library)
            return s ? dayKindOf(s) : null
          }}
        />
      )}
      {tab === 'week' && <WeekView today={today} games={games} sessions={sessions} library={library} changes={changes} onChange={changeDay} onGames={updateGames} onOpen={open} />}
      {tab === 'player' && <PlayerTab onTestDay={() => setOverlay({ name: 'testday' })} />}
      {tab === 'coach' && <CoachView />}
      </div>
      <TabBar tab={tab} onTab={setTab} onKickOff={() => onStart(today.session)} />
    </>
  )
}
