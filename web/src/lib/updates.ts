// App updates. A new version installs and takes over in the background straight away (skipWaiting +
// clientsClaim in vite.config), so any reload or fresh open shows the latest version, in a Safari tab or the
// installed app. The page already on screen keeps running; iOS keeps home-screen apps alive for days, so once a
// newer version has taken over we reload the next time the app goes to the background, never mid-session.
import { registerSW } from 'virtual:pwa-register'

let updated = false
let busy = false

function maybeReload() {
  if (updated && !busy && document.visibilityState === 'hidden') location.reload()
}

export function setBusy(b: boolean) {
  busy = b
}

export function startUpdates() {
  if (import.meta.env.MODE === 'test' || !('serviceWorker' in navigator)) return
  const hadController = !!navigator.serviceWorker.controller
  // first install takes control without anything changing for the user; later ones mean a new version
  const openedAt = Date.now()
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!hadController) return
    updated = true
    // just opened and a newer version took over: switch to it now, before anyone starts using the screen
    if (Date.now() - openedAt < 6000 && !busy) location.reload()
  })
  registerSW({
    immediate: true,
    onRegisteredSW(_url, reg) {
      if (!reg) return
      const check = () => { void reg.update().catch(() => {}) }
      document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') check(); else maybeReload() })
      setInterval(check, 60 * 60 * 1000)
    },
  })
}
