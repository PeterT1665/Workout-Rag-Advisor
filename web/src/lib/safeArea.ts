// iOS 26.1+ bug: home-screen web apps still draw under the status bar but report env(safe-area-inset-top) as 0,
// so headers end up under the clock (https://bugs.webkit.org/show_bug.cgi?id=301994). When that happens we set
// --sat to the status bar height instead. Browsers that report the inset correctly are left alone.
export function fixTopSafeArea() {
  if (typeof window === 'undefined' || typeof document === 'undefined') return
  const ios = /iP(hone|od|ad)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
  const standalone = (navigator as Navigator & { standalone?: boolean }).standalone === true || matchMedia('(display-mode: standalone)').matches
  if (!ios || !standalone) return
  const probe = document.createElement('div')
  probe.style.cssText = 'position:fixed;top:0;height:env(safe-area-inset-top);visibility:hidden;pointer-events:none'
  document.body.appendChild(probe)
  const reported = probe.getBoundingClientRect().height
  probe.remove()
  if (reported > 0) return
  // Phones with a notch or Dynamic Island are at least 812pt tall in portrait; older ones have a 20pt status bar.
  const tall = Math.max(screen.height, screen.width) >= 812
  document.documentElement.style.setProperty('--sat', tall ? '54px' : '20px')
}
