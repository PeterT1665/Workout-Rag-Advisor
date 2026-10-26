import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import data from './data/sessions.json'
import { Home } from './views/Home'
import { Library } from './views/Library'
import { Detail } from './views/Detail'
import { Preroll } from './views/Preroll'
import { PlayerView } from './views/PlayerView'
import { SettingsView } from './views/SettingsView'
import { MediaView } from './views/MediaView'
import { WeekView } from './views/WeekView'
import { PlayerTab } from './views/PlayerTab'
import { CoachView } from './views/CoachView'
import { TabBar, type Tab } from './components/TabBar'
import { Sidebar, type DeskTab } from './components/Sidebar'
import { Dashboard } from './views/Dashboard'
import { DESKTOP_QUERY, useMedia } from './lib/useMedia'
import { allClips, openClips, subscribeClips } from './lib/clips'
import { beep, playClip, realAudio, stopClip, unlockAudio } from './lib/audio'
import { createPlayer, type Player } from './lib/player'
import { nextQuote } from './lib/quotes'
import { addHistory, loadSettings, save } from './lib/storage'
import { loadGames, planToday, saveGames } from './lib/today'
import type { Game } from './lib/weekplan'
import type { Session, Settings } from './lib/types'

const sessions = data.sessions as Session[]

// Full-screen views shown on top of the tabs.
type Overlay =
  | { name: 'library' }
  | { name: 'detail'; session: Session }
  | { name: 'preroll'; session: Session; quote: string | null; clipName: string | null }
  | { name: 'player'; session: Session; player: Player }
  | { name: 'settings' }
  | { name: 'media'; from: 'settings' | 'home' }

export default function App() {
  const [tab, setTab] = useState<Tab>('home')
  const [deskTab, setDeskTab] = useState<DeskTab>('home')
  const desktop = useMedia(DESKTOP_QUERY)
  const [overlay, setOverlay] = useState<Overlay | null>(null)
  const [settings, setSettings] = useState<Settings>(loadSettings)
  const [games, setGames] = useState<Game[]>(loadGames)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const clips = useSyncExternalStore(subscribeClips, allClips)
  const started = useRef(false)
  const today = useMemo(() => planToday(sessions, games), [games])

  useEffect(() => { void openClips() }, [])
  useEffect(() => { window.scrollTo(0, 0) }, [tab, deskTab, overlay?.name])

  const updateSettings = (p: Partial<Settings>) => {
    const next = { ...settingsRef.current, ...p }
    setSettings(next)
    save('settings', next)
  }
  const updateGames = (g: Game[]) => { setGames(g); saveGames(g) }

  function beginSession(session: Session) {
    if (started.current) return
    started.current = true
    stopClip()
    const player = createPlayer(session, {
      audio: realAudio(() => settingsRef.current),
      getSettings: () => settingsRef.current,
      onFinish: (st) => addHistory({
        id: session.id, title: session.title, date: new Date().toLocaleDateString('en-CA'),
        minutes: Math.max(1, Math.round((Date.now() - st.startedAt) / 60000)),
        drills: st.drillsDone, cues: st.cuesCalled,
      }),
    })
    setOverlay({ name: 'player', session, player })
  }

  // Runs inside the tap, so iOS allows the clip to play.
  function onStart(session: Session) {
    unlockAudio()
    started.current = false
    const s = settingsRef.current
    // referee whistle: two short blasts
    beep(s, 2900, 90); setTimeout(() => beep(s, 2900, 260), 160)
    const wantClip = s.clipStart && clips.some((c) => c.kind === 'mindset')
    const clip = wantClip ? playClip(s, 'mindset', () => beginSession(session)) : null
    setOverlay({ name: 'preroll', session, quote: s.cards ? nextQuote() : null, clipName: clip?.name ?? null })
  }

  const close = () => setOverlay(null)
  const open = (session: Session) => setOverlay({ name: 'detail', session })

  if (desktop) {
    // full-screen moments stay full screen
    if (overlay && (overlay.name === 'preroll' || overlay.name === 'player')) return renderOverlay(overlay)
    const goTab = (t: DeskTab) => { setOverlay(null); setDeskTab(t) }
    const hasMindset = clips.some((c) => c.kind === 'mindset')
    return (
      <div className="flex min-h-dvh">
        <Sidebar tab={overlay ? null : deskTab} onTab={goTab} onSettings={() => setOverlay({ name: 'settings' })} onKickOff={() => onStart(today.session)} />
        <div className="min-w-0 flex-1">
          {overlay
            ? <div key={overlay.name} className="screen-in py-6">{renderOverlay(overlay)}</div>
            : <div key={deskTab} className="screen-in">
                {deskTab === 'home' && (
                  <Dashboard today={today} hasClips={hasMindset} onKickOff={onStart} onOpen={open}
                    onSettings={() => setOverlay({ name: 'settings' })} onWeek={() => goTab('week')} onSessions={() => goTab('sessions')}
                    onPlayer={() => goTab('player')} onPlayClip={() => { unlockAudio(); playClip(settingsRef.current, 'mindset') }}
                    onAddClips={() => setOverlay({ name: 'media', from: 'home' })} />
                )}
                {deskTab === 'week' && <WeekView wide today={today} games={games} sessions={sessions} onGames={updateGames} onOpen={open} />}
                {deskTab === 'sessions' && <Library wide sessions={sessions} onOpen={open} onBack={() => goTab('home')} />}
                {deskTab === 'player' && <PlayerTab wide />}
                {deskTab === 'coach' && <CoachView wide />}
              </div>}
        </div>
      </div>
    )
  }

  if (overlay) {
    // full-screen moments (kick-off, player) bring their own entrance
    if (overlay.name === 'preroll' || overlay.name === 'player') return renderOverlay(overlay)
    return <div key={overlay.name} className="screen-in">{renderOverlay(overlay)}</div>
  }

  function renderOverlay(overlay: Overlay) {
    switch (overlay.name) {
      case 'library':
        return <Library sessions={sessions} onOpen={open} onBack={close} />
      case 'detail':
        return <Detail session={overlay.session} desktop={desktop} onBack={close} onStart={() => onStart(overlay.session)} />
      case 'preroll':
        return <Preroll session={overlay.session} quote={overlay.quote} clipName={overlay.clipName} onGo={() => beginSession(overlay.session)} />
      case 'player':
        return <PlayerView key={overlay.player.getState().startedAt} session={overlay.session} player={overlay.player} settings={settings} onExit={close} />
      case 'settings':
        return <SettingsView settings={settings} clipCount={clips.length} onChange={updateSettings} onMedia={() => setOverlay({ name: 'media', from: 'settings' })} onDone={close} />
      case 'media':
        return <MediaView settings={settings} onDone={() => (overlay.from === 'settings' ? setOverlay({ name: 'settings' }) : close())} />
    }
  }

  return (
    <>
      <div key={tab} className="screen-in">
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
        />
      )}
      {tab === 'week' && <WeekView today={today} games={games} sessions={sessions} onGames={updateGames} onOpen={open} />}
      {tab === 'player' && <PlayerTab />}
      {tab === 'coach' && <CoachView />}
      </div>
      <TabBar tab={tab} onTab={setTab} onKickOff={() => onStart(today.session)} />
    </>
  )
}
