import { useEffect, useState } from 'react'
import App from './App'
import { Login } from './views/Login'
import { Onboarding } from './views/Onboarding'
import { checkAuth, pull, startSync, type AuthState } from './lib/sync'
import { DEMO, seedDemo, startOwnDemo } from './lib/demo'
import { SAMPLE_PROFILE, applyAccent, getProfile, hasProfile, setProfile } from './lib/profile'
import { saveGames } from './lib/today'

applyAccent(getProfile().accent)

// Decides welcome/login vs app. With no backend (dev, or first load offline) the app runs locally without sync.
export default function Root() {
  return DEMO ? <DemoRoot /> : <PrivateRoot />
}

// Public demo: no login, no sync. Visitors make their own player or use the sample one (daily-refreshed sample log).
function DemoRoot() {
  const [ready, setReady] = useState(hasProfile)
  if (!ready) {
    return (
      <Onboarding welcome
        onSample={() => { setProfile(SAMPLE_PROFILE); seedDemo(new Date(), true); setReady(true) }}
        onDone={(p, games) => { startOwnDemo(games); setProfile(p); setReady(true) }} />
    )
  }
  if (getProfile().sample) seedDemo()
  return <App />
}

function PrivateRoot() {
  const [state, setState] = useState<AuthState | 'checking'>('checking')
  const [profiled, setProfiled] = useState(hasProfile)
  const enter = async () => {
    startSync()
    await Promise.race([pull(), new Promise((r) => setTimeout(r, 2500))]) // don't block on a slow network
    setProfiled(hasProfile())
    setState('ok')
  }
  useEffect(() => {
    void checkAuth().then((s) => (s === 'ok' ? enter() : setState(s)))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (state === 'checking') return <div className="fixed inset-0 bg-bg" aria-busy="true" />
  if (state === 'login') return <Login onDone={() => void enter()} />
  if (!profiled) return <Onboarding onDone={(p, games) => { saveGames(games); setProfile(p); setProfiled(true) }} />
  return <App />
}
