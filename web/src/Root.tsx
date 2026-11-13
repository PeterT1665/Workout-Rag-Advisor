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

// Laptop preview frames (see views/PhoneFrame): ?preview=welcome shows the welcome screen, ?preview=home the app.
const preview = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('preview') : null

// One app, two ways in: guests ("Try it") keep everything in this browser; invited users sign in and sync.
export default function Root() {
  return preview === 'welcome' ? <WelcomePreview /> : <Main />
}

function Main() {
  const [mode, setMode] = useState<Mode>(accountsEnabled ? 'checking' : 'guest')
  const [profiled, setProfiled] = useState(hasProfile)
  const [authStart, setAuthStart] = useState<'signin' | 'signup'>('signin')

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
  if (mode === 'signin') return <Login start={authStart} onDone={() => void enterAccount()} onBack={() => setMode('guest')} />
  if (mode === 'setpw') return <SetPassword onDone={() => setMode('account')} />
  const auth = (start: 'signin' | 'signup') => () => { setAuthStart(start); setMode('signin') }
  if (mode === 'guest') return <Guest onSignIn={accountsEnabled ? auth('signin') : undefined} onSignUp={accountsEnabled ? auth('signup') : undefined} />
  if (!profiled) return <Onboarding onDone={(p, games) => { saveGames(games); setProfile(p); setProfiled(true) }} />
  return <App />
}

// No account: visitors make their own player or use the sample one (daily-refreshed sample log).
function Guest({ onSignIn, onSignUp }: { onSignIn?: () => void; onSignUp?: () => void }) {
  const [ready, setReady] = useState(() => {
    // the laptop's home-screen preview opens straight on the sample player
    if (preview === 'home' && !hasProfile()) { setProfile(SAMPLE_PROFILE); seedDemo(new Date(), true) }
    return hasProfile()
  })
  if (!ready) {
    return (
      <Onboarding welcome onSignIn={onSignIn} onSignUp={onSignUp}
        onSample={() => { setProfile(SAMPLE_PROFILE); seedDemo(new Date(), true); setReady(true) }}
        onDone={(p, games) => { startOwnDemo(games); setProfile(p); setReady(true) }} />
    )
  }
  if (getProfile().sample) seedDemo()
  return <App />
}

// Laptop preview of the welcome screen, whatever is saved. Choosing a player turns this phone into the real app.
function WelcomePreview() {
  const enter = () => { location.href = '/?app=1' }
  return (
    <Onboarding welcome onSignIn={accountsEnabled ? enter : undefined} onSignUp={accountsEnabled ? enter : undefined}
      onSample={() => { setProfile(SAMPLE_PROFILE); seedDemo(new Date(), true); enter() }}
      onDone={(p, games) => { startOwnDemo(games); setProfile(p); enter() }} />
  )
}
