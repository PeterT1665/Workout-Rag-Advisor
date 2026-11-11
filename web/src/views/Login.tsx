import { useState } from 'react'
import { Button, Screen } from '../components/ui'
import { sendPasswordReset, setNewPassword, signIn, type SignInResult } from '../lib/sync'

const MESSAGES: Record<Exclude<SignInResult, 'ok'>, string> = {
  wrong: 'Email or password is wrong. Try again.',
  throttled: 'Too many tries. Wait a few minutes.',
  offline: "Can't reach the server. Check your connection.",
}

const field = 'mt-1 block min-h-14 w-full rounded-2xl bg-surface px-4 text-ink outline-none focus:ring-2 focus:ring-[var(--red-ink)]'

// Sign in to an account (invite-only). "Forgot password" sends a reset link to the email.
export function Login({ onDone, onBack }: { onDone: () => void; onBack?: () => void }) {
  const [email, setEmail] = useState('')
  const [pw, setPw] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [mode, setMode] = useState<'signin' | 'forgot' | 'sent'>('signin')

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (busy) return
    setBusy(true); setErr(null)
    if (mode === 'forgot') {
      const ok = await sendPasswordReset(email)
      setBusy(false)
      if (ok) setMode('sent'); else setErr(MESSAGES.offline)
      return
    }
    const r = await signIn(email, pw)
    setBusy(false)
    if (r === 'ok') onDone(); else setErr(MESSAGES[r])
  }

  return (
    <Screen className="flex flex-col justify-center">
      <form onSubmit={submit} className="rise w-full">
        <img src="/icon-192.png" alt="" className="h-14 w-14 rounded-2xl" />
        <h1 className="m-0 mt-6 font-display text-4xl uppercase leading-[.9]">
          {mode === 'signin' ? <>Sign <span className="text-red-ink">in</span></> : <>Reset <span className="text-red-ink">password</span></>}
        </h1>
        {mode === 'sent'
          ? <p className="mt-6 text-muted">Check your email for a reset link.</p>
          : <>
              <label htmlFor="email" className="mt-8 block text-sm text-muted">Email</label>
              <input id="email" type="email" autoComplete="email" inputMode="email" required value={email} onChange={(e) => setEmail(e.target.value)} className={field} />
              {mode === 'signin' && <>
                <label htmlFor="pw" className="mt-4 block text-sm text-muted">Password</label>
                <input id="pw" type="password" autoComplete="current-password" required value={pw} onChange={(e) => setPw(e.target.value)}
                  aria-invalid={!!err} aria-describedby={err ? 'login-err' : undefined} className={field} />
              </>}
              {err && <p id="login-err" role="alert" className="m-0 mt-3 text-sm text-red-ink">{err}</p>}
              <Button block type="submit" className="mt-6" disabled={busy || !email || (mode === 'signin' && !pw)}>
                {busy ? 'One moment…' : mode === 'signin' ? "Let's go" : 'Send reset link'}
              </Button>
            </>}
        <div className="mt-4 flex justify-between text-sm">
          {onBack ? <button type="button" onClick={onBack} className="min-h-11 text-muted">← Back</button> : <span />}
          {mode === 'signin'
            ? <button type="button" onClick={() => { setMode('forgot'); setErr(null) }} className="min-h-11 text-muted">Forgot password?</button>
            : <button type="button" onClick={() => { setMode('signin'); setErr(null) }} className="min-h-11 text-muted">Back to sign in</button>}
        </div>
        <p className="m-0 mt-6 text-center text-xs text-muted">Accounts are invite-only for now. You'll stay signed in on this device.</p>
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
