// iOS 26.1+ bug: home-screen web apps draw under the status bar but can report env(safe-area-inset-top) as 0,
// so headers land under the clock (https://bugs.webkit.org/show_bug.cgi?id=301994). On an iPhone/iPad running
// full screen (installed app), --sat is at least the status bar height, whatever the browser reports.
export type SafeAreaInfo = { ios: boolean; standalone: boolean; fullScreen: boolean; reported: number; applied: number | null }
export let safeAreaInfo: SafeAreaInfo | null = null

function measure(): number {
  const probe = document.createElement('div')
  probe.style.cssText = 'position:fixed;top:0;height:env(safe-area-inset-top);visibility:hidden;pointer-events:none'
  document.body.appendChild(probe)
  const h = probe.getBoundingClientRect().height
  probe.remove()
  return h
}

export function fixTopSafeArea() {
  if (typeof window === 'undefined' || typeof document === 'undefined') return
  const ios = /iP(hone|od|ad)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
  const apply = () => {
    const standalone = (navigator as Navigator & { standalone?: boolean }).standalone === true || matchMedia('(display-mode: standalone)').matches
    // installed apps fill the whole screen; a browser tab has toolbars, so it is shorter than the screen
    const fullScreen = standalone || window.innerHeight >= (window.innerHeight > window.innerWidth ? screen.height : screen.width) - 1
    const reported = measure()
    let applied: number | null = null
    if (ios && fullScreen) {
      // phones with a notch or Dynamic Island are at least 812pt tall; older ones have a 20pt status bar
      const statusBar = Math.max(screen.height, screen.width) >= 812 ? 58 : 20
      applied = Math.max(reported, statusBar)
      document.documentElement.style.setProperty('--sat', `${applied}px`)
    }
    safeAreaInfo = { ios, standalone, fullScreen, reported, applied }
  }
  apply()
  window.addEventListener('orientationchange', () => setTimeout(apply, 300))
}
