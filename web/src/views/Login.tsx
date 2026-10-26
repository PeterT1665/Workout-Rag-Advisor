import { useState } from 'react'
import { login } from '../lib/sync'

const MESSAGES = {
  wrong: 'Wrong password. Try again.',
  throttled: 'Too many tries. Wait 15 minutes, then try again.',
  offline: "Can't reach the server. Check your connection.",
}

export function Login({ onDone }: { onDone: () => void }) {
  const [pw, setPw] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<keyof typeof MESSAGES | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!pw || busy) return
    setBusy(true); setErr(null)
    const r = await login(pw)
    setBusy(false)
    if (r === 'ok') onDone(); else setErr(r)
  }
  return (
    <main className="fixed inset-0 grid place-items-center bg-[#0a0c0b] px-5 text-white">
      <form onSubmit={submit} className="rise w-full max-w-sm">
        <img src="/icon-192.png" alt="" className="h-16 w-16 rounded-2xl" />
        <h1 className="m-0 mt-6 font-display text-4xl uppercase leading-[.9]">Training<br /><span className="text-[var(--red-ink)]">Agent</span></h1>
        <label htmlFor="pw" className="mt-8 block text-sm text-white/70">Password</label>
        <input id="pw" type="password" autoComplete="current-password" autoFocus value={pw} onChange={(e) => setPw(e.target.value)}
          aria-invalid={!!err} aria-describedby={err ? 'pw-err' : undefined}
          className="mt-2 block min-h-14 w-full rounded-2xl bg-white/10 px-4 text-white outline-none focus:ring-2 focus:ring-[var(--red-ink)]" />
        {err && <p id="pw-err" role="alert" className="m-0 mt-3 text-sm text-[var(--red-ink)]">{MESSAGES[err]}</p>}
        <button type="submit" disabled={!pw || busy}
          className="mt-6 min-h-14 w-full rounded-full bg-[var(--red)] font-display text-sm uppercase tracking-wide text-white transition-[transform,opacity] duration-150 ease-snap active:scale-[0.97] disabled:opacity-40">
          {busy ? 'Checking…' : "Let's go"}
        </button>
        <p className="m-0 mt-4 text-center text-sm text-white/50">You'll stay logged in on this device.</p>
      </form>
    </main>
  )
}
