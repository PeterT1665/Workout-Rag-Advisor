// Push notifications: "rest is over" (so a rest doesn't turn into ten minutes on TikTok) and Sunday "plan next week".
// iPhone pauses the app in the background, so the reminder is scheduled on the server when the app is hidden
// mid-rest and cancelled when you come back (supabase/migrations/0006_push.sql, supabase/functions/send-push).
// Needs an account, and on iPhone the app added to the Home Screen (iOS 16.4+).
import { supabase } from './supabase'
import { currentUser, syncEnabled } from './sync'
import { currentLang, t } from '../i18n'

const KEY = import.meta.env.VITE_VAPID_PUBLIC_KEY as string | undefined
const FLAG = 'push:on' // this device subscribed (device-only, not synced)
const ENDPOINT = 'push:endpoint' // this device's address, so a rest call-back reaches only the phone in the session
const saved = () => { try { return localStorage.getItem(ENDPOINT) } catch { return null } }
// Devices that turned notifications on before this was saved look it up once.
async function myEndpoint(): Promise<string | null> {
  const e = saved()
  if (e) return e
  try {
    const sub = await (await swReady())?.pushManager.getSubscription()
    if (sub) try { localStorage.setItem(ENDPOINT, sub.endpoint) } catch { /* ignore */ }
    return sub?.endpoint ?? null
  } catch { return null }
}

const isIOS = () => typeof navigator !== 'undefined' && /iPad|iPhone|iPod/.test(navigator.userAgent)
const standalone = () => typeof matchMedia === 'function' && (matchMedia('(display-mode: standalone)').matches || (navigator as unknown as { standalone?: boolean }).standalone === true)

export type PushState = 'on' | 'off' | 'denied' | 'need-install' | 'unsupported' | 'need-account'
export function pushState(): PushState {
  if (!syncEnabled()) return 'need-account'
  if (typeof window === 'undefined' || !KEY) return 'unsupported'
  if (isIOS() && !standalone()) return 'need-install'
  if (!('serviceWorker' in navigator) || !('PushManager' in window) || !('Notification' in window)) return 'unsupported'
  if (Notification.permission === 'denied') return 'denied'
  return pushOn() ? 'on' : 'off'
}
export const pushOn = () => { try { return localStorage.getItem(FLAG) === '1' } catch { return false } }
const setFlag = (on: boolean) => { try { if (on) localStorage.setItem(FLAG, '1'); else localStorage.removeItem(FLAG) } catch { /* ignore */ } }

const b64 = (s: string) => {
  const raw = atob(s.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (s.length % 4)) % 4))
  return Uint8Array.from(raw, (c) => c.charCodeAt(0))
}
const swReady = () => Promise.race([navigator.serviceWorker.ready, new Promise<null>((r) => setTimeout(() => r(null), 6000))])

export type Prefs = { rest_alerts: boolean; weekly: boolean }

// Ask permission (must run inside a tap), subscribe this device, and save it to the account.
export async function enablePush(prefs: Prefs = { rest_alerts: true, weekly: true }): Promise<PushState> {
  const state = pushState()
  if (state !== 'off' && state !== 'on') return state
  try {
    if ((await Notification.requestPermission()) !== 'granted') return 'denied'
    const reg = await swReady()
    if (!reg) return 'unsupported'
    const sub = (await reg.pushManager.getSubscription()) ?? (await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: b64(KEY!) }))
    const j = sub.toJSON()
    const me = await currentUser()
    if (!me || !j.endpoint || !j.keys) return 'unsupported'
    const sb = await supabase()
    const row = {
      endpoint: j.endpoint, user_id: me.id, p256dh: j.keys.p256dh, auth: j.keys.auth,
      tz: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC', ...prefs,
    }
    let { error } = await sb.from('push_subscriptions').upsert({ ...row, lang: currentLang() })
    if (error && /lang/.test(error.message)) ({ error } = await sb.from('push_subscriptions').upsert(row)) // before migration 0008
    if (error) return 'unsupported'
    setFlag(true)
    try { localStorage.setItem(ENDPOINT, j.endpoint) } catch { /* ignore */ }
    return 'on'
  } catch { return 'unsupported' }
}

export async function disablePush() {
  setFlag(false)
  try { localStorage.removeItem(ENDPOINT) } catch { /* ignore */ }
  try {
    const reg = await swReady()
    const sub = await reg?.pushManager.getSubscription()
    if (sub) {
      const sb = await supabase()
      await sb.from('push_subscriptions').delete().eq('endpoint', sub.endpoint)
      await sub.unsubscribe()
    }
  } catch { /* offline: the server drops it when Apple says it's gone */ }
}

export async function getPrefs(): Promise<Prefs | null> {
  try {
    const reg = await swReady()
    const sub = await reg?.pushManager.getSubscription()
    if (!sub) return null
    const sb = await supabase()
    const { data } = await sb.from('push_subscriptions').select('rest_alerts, weekly').eq('endpoint', sub.endpoint).maybeSingle()
    return (data as Prefs | null) ?? null
  } catch { return null }
}
export async function setPrefs(p: Partial<Prefs>) {
  try {
    const reg = await swReady()
    const sub = await reg?.pushManager.getSubscription()
    if (!sub) return
    const sb = await supabase()
    await sb.from('push_subscriptions').update(p).eq('endpoint', sub.endpoint)
  } catch { /* ignore */ }
}

// The Sunday reminder is written by the server: keep this device's language there (changing it reloads the app).
export async function syncPushLang() {
  if (!pushOn()) return
  try {
    const reg = await swReady()
    const sub = await reg?.pushManager.getSubscription()
    if (!sub) return
    const sb = await supabase()
    await sb.from('push_subscriptions').update({ lang: currentLang() }).eq('endpoint', sub.endpoint)
  } catch { /* ignore */ }
}

// What "rest is over" says. One at random each time.
export const REST_LINES = [
  t("Come back to training. Lock in, let's go."),
  t("No more doom scroll. Let's roll."),
  t('Phone down. Your next set is waiting.'),
  t("Timer's done. The set won't do itself."),
]

// Calls run one after another, so a cancel can't overtake the booking it is meant to undo.
let chain: Promise<unknown> = Promise.resolve()
const queue = (f: () => Promise<unknown>) => (chain = chain.then(f, f))

// A rest started: book the call-back for its end, while the app is still open (iPhone suspends the app the
// moment you leave it, too fast for a request sent then). Replaces any earlier one.
export function scheduleRest(seconds: number, next?: string) {
  if (!pushOn() || seconds < 5) return chain
  const line = REST_LINES[Math.floor(Math.random() * REST_LINES.length)]
  return queue(async () => {
    try {
      const sb = await supabase()
      await sb.rpc('schedule_rest_push', { p_seconds: Math.round(seconds), p_title: next ? `${t('Rest is over')} · ${next}`.slice(0, 80) : t('Rest is over'), p_body: line, p_endpoint: await myEndpoint() })
    } catch { /* offline: no reminder this time */ }
  })
}
// Still in the app at the end of the rest, paused, skipped or ended: no reminder needed.
export function cancelRest() {
  if (!pushOn()) return chain
  return queue(async () => {
    try {
      const sb = await supabase()
      await sb.from('scheduled_pushes').delete().eq('kind', 'rest').is('sent_at', null)
    } catch { /* ignore */ }
  })
}
// Settings: a test notification in a few seconds, so you can lock the phone and see it arrive.
export async function sendTest(): Promise<boolean> {
  if (!pushOn()) return false
  try {
    const sb = await supabase()
    const { error } = await sb.rpc('schedule_rest_push', { p_seconds: 8, p_title: t('Test notification'), p_body: t("Notifications work. Lock in, let's go."), p_endpoint: await myEndpoint() })
    return !error
  } catch { return false }
}
