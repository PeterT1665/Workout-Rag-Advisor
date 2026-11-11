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

// Phones and tablets that are not running the installed app yet.
export const canInstall = () => !isInstalled() && (isIOS() || /Android/.test(navigator.userAgent) || deferred !== null)

export async function systemPrompt(): Promise<boolean> {
  if (!deferred) return false
  await deferred.prompt()
  const { outcome } = await deferred.userChoice
  deferred = null
  return outcome === 'accepted'
}
