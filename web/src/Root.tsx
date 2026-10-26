import { useEffect, useState } from 'react'
import App from './App'
import { Login } from './views/Login'
import { checkAuth, pull, startSync, type AuthState } from './lib/sync'

// Decides login vs app. With no backend (dev, or first load offline) the app runs locally without sync.
export default function Root() {
  const [state, setState] = useState<AuthState | 'checking'>('checking')
  const enter = async () => {
    startSync()
    await Promise.race([pull(), new Promise((r) => setTimeout(r, 2500))]) // don't block on a slow network
    setState('ok')
  }
  useEffect(() => {
    void checkAuth().then((s) => (s === 'ok' ? enter() : setState(s)))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (state === 'checking') return <div className="fixed inset-0 bg-[#0a0c0b]" aria-busy="true" />
  if (state === 'login') return <Login onDone={() => void enter()} />
  return <App />
}
