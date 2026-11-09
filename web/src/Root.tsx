import { useEffect, useState } from 'react'
import App from './App'
import { Login, SetPassword } from './views/Login'
import { Onboarding } from './views/Onboarding'
import { clearLocalData, currentUser, pull, startSync } from './lib/sync'
import { clearPhotos, syncPhotos } from './lib/photos'
import { accountsEnabled, arrivedToSetPassword } from './lib/supabase'
import { seedDemo, startOwnDemo } from './lib/demo'
import { SAMPLE_PROFILE, applyAccent, getProfile, hasProfile, reloadProfile, setProfile } from './lib/profile'
import { saveGames } from './lib/today'

applyAccent(getProfile().accent)

type Mode = 'checking' | 'guest' | 'signin' | 'setpw' | 'account'

// One app, two ways in: guests ("Try it") keep everything in this browser; invited users sign in and sync.
export default function Root() {
  const [mode, setMode] = useState<Mode>(accountsEnabled ? 'checking' : 'guest')
  const [profiled, setProfiled] = useState(hasProfile)

  async function enterAccount(next: Mode = 'account') {
    // A guest's sample player or leftovers must not end up in the account. A guest's own player is kept and uploaded.
    const p = hasProfile() ? getProfile() : null
    if (!p || p.sample) { clearLocalData(); await clearPhotos() }
    startSync()
    await Promise.race([pull(), new Promise((r) => setTimeout(r, 2500))]) // don't block on a slow network
    reloadProfile()
    const me = await currentUser()
    if (me) void syncPhotos(me.id) // photos can arrive after the app is showing
    setProfiled(hasProfile())
    setMode(next)
  }

  useEffect(() => {
    if (!accountsEnabled) return
    void currentUser().then((me) => {
      if (me) void enterAccount(arrivedToSetPassword ? 'setpw' : 'account')
      else setMode('guest')
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (mode === 'checking') return <div className="fixed inset-0 bg-bg" aria-busy="true" />
  if (mode === 'signin') return <Login onDone={() => void enterAccount()} onBack={() => setMode('guest')} />
  if (mode === 'setpw') return <SetPassword onDone={() => setMode('account')} />
  if (mode === 'guest') return <Guest onSignIn={accountsEnabled ? () => setMode('signin') : undefined} />
  if (!profiled) return <Onboarding onDone={(p, games) => { saveGames(games); setProfile(p); setProfiled(true) }} />
  return <App />
}

// No account: visitors make their own player or use the sample one (daily-refreshed sample log).
function Guest({ onSignIn }: { onSignIn?: () => void }) {
  const [ready, setReady] = useState(hasProfile)
  if (!ready) {
    return (
      <Onboarding welcome onSignIn={onSignIn}
        onSample={() => { setProfile(SAMPLE_PROFILE); seedDemo(new Date(), true); setReady(true) }}
        onDone={(p, games) => { startOwnDemo(games); setProfile(p); setReady(true) }} />
    )
  }
  if (getProfile().sample) seedDemo()
  return <App />
}
