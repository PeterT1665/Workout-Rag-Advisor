// App updates for the installed PWA. iOS keeps home-screen apps paused, so we check for a new version whenever
// the app comes back to the foreground (and hourly), and apply it straight away unless a session, rating test
// or edit is in progress; then it waits until that screen closes. No more delete-and-re-add to update.
import { registerSW } from 'virtual:pwa-register'

let pending = false
let busy = false
let apply: ((reload?: boolean) => Promise<void>) | null = null

function maybeApply() {
  if (pending && !busy && apply) void apply(true)
}

export function setBusy(b: boolean) {
  busy = b
  maybeApply()
}

export function startUpdates() {
  if (import.meta.env.MODE === 'test' || !('serviceWorker' in navigator)) return
  apply = registerSW({
    immediate: true,
    onNeedRefresh() { pending = true; maybeApply() },
    onRegisteredSW(_url, reg) {
      if (!reg) return
      const check = () => { void reg.update().catch(() => {}) }
      document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') check() })
      setInterval(check, 60 * 60 * 1000)
    },
  })
}
