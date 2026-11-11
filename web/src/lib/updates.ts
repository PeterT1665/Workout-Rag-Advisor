// App updates for the installed PWA. iOS keeps home-screen apps paused, so we check for a new version whenever
// the app comes back to the foreground (and hourly). The update is applied the next time the app goes to the
// background, never while a session, rating test or edit is open. No more delete-and-re-add to update.
import { registerSW } from 'virtual:pwa-register'

let pending = false
let busy = false
let apply: ((reload?: boolean) => Promise<void>) | null = null

// Reload only while the app is out of sight (switched away / screen off), so nobody sees it happen.
function maybeApply() {
  if (pending && !busy && apply && document.visibilityState === 'hidden') void apply(true)
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
      document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') check(); else maybeApply() })
      setInterval(check, 60 * 60 * 1000)
    },
  })
}
