// "Install the app" (Add to Home Screen). iPhone has no install prompt, so we show the steps; Android/Chrome
// fires beforeinstallprompt, which we keep so our button can open the system prompt.
type PromptEvent = Event & { prompt: () => Promise<void>; userChoice: Promise<{ outcome: string }> }
let deferred: PromptEvent | null = null

if (typeof window !== 'undefined') {
  window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); deferred = e as PromptEvent })
  window.addEventListener('appinstalled', () => { deferred = null })
}

export const isIOS = () => typeof navigator !== 'undefined' &&
  (/iP(hone|od|ad)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1))

export function isInstalled() {
  if (typeof navigator === 'undefined') return true
  return (navigator as Navigator & { standalone?: boolean }).standalone === true ||
    (typeof matchMedia === 'function' && matchMedia('(display-mode: standalone)').matches)
}

// The browser inside another app (TikTok, Instagram...): it can't add to the Home Screen, so the player has to open the page in
// Safari first. Returns the app's name, or null in a normal browser.
export function inAppBrowser(ua = typeof navigator === 'undefined' ? '' : navigator.userAgent): string | null {
  if (/musical_ly|BytedanceWebview|TikTok|trill/i.test(ua)) return 'TikTok'
  if (/Instagram/i.test(ua)) return 'Instagram'
  if (/FBAN|FBAV|FB_IAB/i.test(ua)) return 'Facebook'
  if (/Snapchat/i.test(ua)) return 'Snapchat'
  if (/LinkedInApp/i.test(ua)) return 'LinkedIn'
  if (/\bLine\//i.test(ua)) return 'LINE'
  if (/Zalo/i.test(ua)) return 'Zalo'
  return null
}

// Phones and tablets that are not running the installed app yet.
export const canInstall = () => !isInstalled() && (isIOS() || /Android/.test(navigator.userAgent) || deferred !== null)

export async function systemPrompt(): Promise<boolean> {
  if (!deferred) return false
  await deferred.prompt()
  const { outcome } = await deferred.userChoice
  deferred = null
  return outcome === 'accepted'
}
