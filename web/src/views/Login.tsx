import { useState } from 'react'
import { Button, Screen } from '../components/ui'
import { sendPasswordReset, setNewPassword, signIn, signUp, type SignInResult, type SignUpResult } from '../lib/sync'

const MESSAGES: Record<Exclude<SignInResult, 'ok'>, string> = {
  wrong: 'Email or password is wrong. Try again.',
  throttled: 'Too many tries. Wait a few minutes.',
  offline: "Can't reach the server. Check your connection.",
}
const SIGNUP_MESSAGES: Record<Exclude<SignUpResult, 'signed-in' | 'confirm-email'>, string> = {
  exists: 'That email already has an account. Sign in instead.',
  closed: "Sign-up isn't open yet. Ask for an invite.",
  weak: 'Use a stronger password (8+ characters).',
  throttled: 'Too many tries. Wait a few minutes.',
  offline: "Can't reach the server. Check your connection.",
}

const field = 'mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 text-ink outline-none focus:ring-2 focus:ring-[var(--red-ink)]'
type Mode = 'signin' | 'signup' | 'forgot' | 'sent' | 'confirm'

// Sign in, create an account (email + password), or reset a forgotten password.
export function Login({ onDone, onBack, start = 'signin' }: { onDone: () => void; onBack?: () => void; start?: 'signin' | 'signup' }) {
  const [email, setEmail] = useState('')
  const [pw, setPw] = useState('')
  const [pw2, setPw2] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>(start)
  const go = (m: Mode) => { setMode(m); setErr(null) }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (busy) return
    if (mode === 'signup') {
      if (pw.length < 8) { setErr('Use at least 8 characters.'); return }
      if (pw !== pw2) { setErr("Passwords don't match."); return }
    }
    setBusy(true); setErr(null)
    if (mode === 'forgot') {
      const ok = await sendPasswordReset(email)
      setBusy(false)
      if (ok) setMode('sent'); else setErr(MESSAGES.offline)
      return
    }
    if (mode === 'signup') {
      const r = await signUp(email, pw)
      setBusy(false)
      if (r === 'signed-in') onDone()
      else if (r === 'confirm-email') setMode('confirm')
      else setErr(SIGNUP_MESSAGES[r])
      return
    }
    const r = await signIn(email, pw)
    setBusy(false)
    if (r === 'ok') onDone(); else setErr(MESSAGES[r])
  }

  const title = mode === 'signin' ? <>Sign <span className="text-red-ink">in</span></>
    : mode === 'signup' ? <>Create <span className="text-red-ink">account</span></>
    : mode === 'confirm' ? <>Check your <span className="text-red-ink">email</span></>
    : <>Reset <span className="text-red-ink">password</span></>

  return (
    <Screen className="flex flex-col justify-center">
      <form onSubmit={submit} className="rise w-full">
        <img src="/icon-192.png" alt="" className="h-14 w-14 rounded-2xl" />
        <h1 className="m-0 mt-6 font-display text-4xl uppercase leading-[.9]">{title}</h1>
        {mode === 'sent' && <p className="mt-6 text-muted">Check your email for a reset link.</p>}
        {mode === 'confirm' && <p className="mt-6 text-muted">Tap the link we sent to {email.trim()}, then you're in.</p>}
        {(mode === 'signin' || mode === 'signup' || mode === 'forgot') && <>
          <label htmlFor="email" className="mt-8 block text-sm text-muted">Email</label>
          <input id="email" type="email" autoComplete="email" inputMode="email" required value={email} onChange={(e) => setEmail(e.target.value)} className={field} />
          {mode !== 'forgot' && <>
            <label htmlFor="pw" className="mt-4 block text-sm text-muted">Password{mode === 'signup' ? ' (8+ characters)' : ''}</label>
            <input id="pw" type="password" autoComplete={mode === 'signup' ? 'new-password' : 'current-password'} required value={pw} onChange={(e) => setPw(e.target.value)}
              aria-invalid={!!err} aria-describedby={err ? 'login-err' : undefined} className={field} />
          </>}
          {mode === 'signup' && <>
            <label htmlFor="pw2" className="mt-4 block text-sm text-muted">Repeat password</label>
            <input id="pw2" type="password" autoComplete="new-password" required value={pw2} onChange={(e) => setPw2(e.target.value)} className={field} />
          </>}
          {err && <p id="login-err" role="alert" className="m-0 mt-3 text-sm text-red-ink">{err}</p>}
          <Button block type="submit" className="mt-6" disabled={busy || !email || (mode !== 'forgot' && !pw) || (mode === 'signup' && !pw2)}>
            {busy ? 'One moment…' : mode === 'signin' ? "Let's go" : mode === 'signup' ? 'Create account' : 'Send reset link'}
          </Button>
        </>}
        <div className="mt-4 flex justify-between text-sm">
          {onBack ? <button type="button" onClick={onBack} className="min-h-11 text-muted">← Back</button> : <span />}
          {mode === 'signin' && <button type="button" onClick={() => go('forgot')} className="min-h-11 text-muted">Forgot password?</button>}
          {mode !== 'signin' && <button type="button" onClick={() => go('signin')} className="min-h-11 text-muted">Back to sign in</button>}
        </div>
        {mode === 'signin' && <button type="button" onClick={() => go('signup')} className="mt-4 min-h-11 w-full text-center text-sm font-semibold">New here? <span className="text-red-ink">Create an account</span></button>}
        {mode === 'signup' && <p className="m-0 mt-4 text-center text-xs text-muted">Your training data stays private to your account.</p>}
      </form>
    </Screen>
  )
}

// Landing from an invite or reset email: choose a password, then straight into the app.
export function SetPassword({ onDone }: { onDone: () => void }) {
  const [pw, setPw] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (pw.length < 8) { setErr('Use at least 8 characters.'); return }
    setBusy(true); setErr(null)
    const r = await setNewPassword(pw)
    setBusy(false)
    if (r) setErr(r); else onDone()
  }
  return (
    <Screen className="flex flex-col justify-center">
      <form onSubmit={submit} className="rise w-full">
        <h1 className="m-0 font-display text-4xl uppercase leading-[.9]">Choose a <span className="text-red-ink">password</span></h1>
        <label htmlFor="new-pw" className="mt-8 block text-sm text-muted">New password (8+ characters)</label>
        <input id="new-pw" type="password" autoComplete="new-password" value={pw} onChange={(e) => setPw(e.target.value)}
          aria-invalid={!!err} aria-describedby={err ? 'pw-err' : undefined} className={field} />
        {err && <p id="pw-err" role="alert" className="m-0 mt-3 text-sm text-red-ink">{err}</p>}
        <Button block type="submit" className="mt-6" disabled={busy || !pw}>{busy ? 'Saving…' : 'Save and continue'}</Button>
      </form>
    </Screen>
  )
}
