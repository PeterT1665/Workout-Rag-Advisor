import { useEffect, useState } from 'react'
import App from './App'
import { Login, SetPassword } from './views/Login'
import { Onboarding } from './views/Onboarding'
import { NextSteps } from './views/NextSteps'
import { canInstall } from './lib/install'
import { clearLocalData, currentUser, pull, startSync } from './lib/sync'
import { clearPhotos, syncPhotos } from './lib/photos'
import { accountsEnabled, arrivedToSetPassword } from './lib/supabase'
import { seedDemo, startOwnDemo } from './lib/demo'
import { SAMPLE_PROFILE, applyAccent, getProfile, hasProfile, reloadProfile, setProfile } from './lib/profile'
import { saveGames } from './lib/today'
import { acceptPendingInvite, inviteSeen, pendingInvite } from './lib/invite'
import { FriendInvite } from './views/FriendInvite'
import { tourReset } from './components/Tour'
import { PickUsername } from './views/PickUsername'
import { Intro } from './views/Intro'
import { load, save } from './lib/storage'

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
  const [invite, setInvite] = useState(() => !!pendingInvite() && !inviteSeen())
  const [needsName, setNeedsName] = useState(false) // signed in, no username yet: asked once, right away
  async function checkUsername() {
    if (load<boolean>('team:askedName', false)) return
    try {
      const { myTeamProfile } = await import('./lib/team')
      const mine = await Promise.race([myTeamProfile(), new Promise<'slow'>((r) => setTimeout(() => r('slow'), 4000))])
      if (mine === null) setNeedsName(true) // 'slow' or offline: ask another time
    } catch { /* offline: ask another time */ }
  }
  const nameDone = () => { save('team:askedName', true); setNeedsName(false) }

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
    void acceptPendingInvite() // came from a friend's QR: you're friends now
    void checkUsername()
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
  // a friend's QR, not signed in yet: who invited you, then sign up or sign in
  if (mode === 'guest' && invite && accountsEnabled) return <FriendInvite onSignUp={auth('signup')} onSignIn={auth('signin')} onSkip={() => setInvite(false)} />
  if (mode === 'guest') return <Guest onSignIn={accountsEnabled ? auth('signin') : undefined} onSignUp={accountsEnabled ? auth('signup') : undefined} />
  if (!profiled) return <Onboarding onDone={(p, games) => { tourReset(); saveGames(games); setProfile(p); setProfiled(true) }} />
  if (needsName) return <PickUsername onDone={nameDone} />
  return <App />
}

// No account: visitors make their own player or use the sample one (daily-refreshed sample log).
let introShown = false // once per visit
function Guest({ onSignIn, onSignUp }: { onSignIn?: () => void; onSignUp?: () => void }) {
  const [ready, setReady] = useState(() => {
    // the laptop's home-screen preview opens straight on the sample player
    if (preview === 'home' && !hasProfile()) { setProfile(SAMPLE_PROFILE); seedDemo(new Date(), true) }
    return hasProfile()
  })
  const [next, setNext] = useState(false) // the "You're in" screen after making a player
  // new visitors: a few seconds of welcome before "Train like a pro"
  const [intro, setIntro] = useState(() => !ready && !introShown && preview !== 'home')
  if (intro) return <Intro onDone={() => { introShown = true; setIntro(false) }} />
  if (!ready) {
    return (
      <Onboarding welcome onSignIn={onSignIn}
        onSample={() => { tourReset(); setProfile(SAMPLE_PROFILE); seedDemo(new Date(), true); setReady(true) }}
        onDone={(p, games) => { tourReset(); startOwnDemo(games); setProfile(p); setNext(!!onSignUp || canInstall()); setReady(true) }} />
    )
  }
  if (next) return <NextSteps name={getProfile().name} onSignUp={onSignUp} onDone={() => setNext(false)} />
  if (getProfile().sample) seedDemo()
  return <App onSignUp={onSignUp} onSignIn={onSignIn} />
}

// Laptop preview of the welcome screen, whatever is saved. Choosing a player turns this phone into the real app.
function WelcomePreview() {
  const enter = () => { location.href = '/?app=1' }
  return (
    <Onboarding welcome onSignIn={accountsEnabled ? enter : undefined}
      onSample={() => { tourReset(); setProfile(SAMPLE_PROFILE); seedDemo(new Date(), true); enter() }}
      onDone={(p, games) => { tourReset(); startOwnDemo(games); setProfile(p); enter() }} />
  )
}
