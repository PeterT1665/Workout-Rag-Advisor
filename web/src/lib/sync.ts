// Phone <-> laptop sync through Supabase. Local-first: localStorage is what the UI reads;
// changes are pushed in the background and pulled when the app opens or comes back to the foreground.
// Rules (newer wins, history merged) live in the database function put_state; see supabase/migrations.
import { setSaveHook } from './storage'
import { supabase } from './supabase'
import { PHOTO_SLOTS, clearPhotos } from './photos'

export const SYNC_KEYS = ['history', 'games', 'settings', 'quotes', 'mySessions', 'profile'] as const
type Key = (typeof SYNC_KEYS)[number]
type Row = { key: string; value: unknown; updated_at: number }

let enabled = false
const timers: Partial<Record<Key, ReturnType<typeof setTimeout>>> = {}

const readMeta = (): Record<string, number> => { try { return JSON.parse(localStorage.getItem('sync:meta') || '{}') } catch { return {} } }
const writeMeta = (m: Record<string, number>) => { try { localStorage.setItem('sync:meta', JSON.stringify(m)) } catch { /* ignore */ } }
const readRaw = (k: string) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : null } catch { return null } }
const writeRaw = (k: string, v: unknown) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* ignore */ } }
const isKey = (k: string): k is Key => (SYNC_KEYS as readonly string[]).includes(k)

export const syncEnabled = () => enabled

async function push(key: Key) {
  const value = readRaw(key)
  if (value === null) return
  try {
    const sb = await supabase()
    const { data, error } = await sb.rpc('put_state', { p_key: key, p_value: value, p_updated_at: readMeta()[key] ?? Date.now() })
    if (error || !data) return
    const row = data as Row
    if (JSON.stringify(row.value) !== JSON.stringify(value)) {
      writeRaw(key, row.value) // the server merged or had something newer
      window.dispatchEvent(new Event('ta-sync'))
    }
    writeMeta({ ...readMeta(), [key]: row.updated_at })
  } catch { /* offline: the next pull/save retries */ }
}

function changed(key: string) {
  if (!enabled || !isKey(key)) return
  writeMeta({ ...readMeta(), [key]: Date.now() })
  clearTimeout(timers[key])
  timers[key] = setTimeout(() => void push(key), 800)
}

// Pull the account's copy, keep whichever side is newer, and push local changes the server hasn't seen.
export async function pull(): Promise<boolean> {
  try {
    const sb = await supabase()
    const { data, error } = await sb.from('user_state').select('key, value, updated_at')
    if (error || !data) return false
    const server = Object.fromEntries((data as Row[]).map((r) => [r.key, r]))
    const meta = readMeta()
    let touched = false
    for (const key of SYNC_KEYS) {
      const s = server[key]
      const local = meta[key] ?? 0
      if (key === 'history') {
        if (readRaw('history') === null) {
          if (s) { writeRaw('history', s.value); meta.history = s.updated_at; touched = true } // new device: take the account's
        } else {
          void push('history') // the database merges both lists and returns the union
        }
        continue
      }
      if (s && s.updated_at > local) {
        writeRaw(key, s.value); meta[key] = s.updated_at; touched = true
      } else if (readRaw(key) !== null && (!s || local > s.updated_at)) {
        void push(key)
      }
    }
    writeMeta(meta)
    if (touched) window.dispatchEvent(new Event('ta-sync'))
    return true
  } catch { return false }
}

let listening = false
export function startSync() {
  enabled = true
  setSaveHook(changed)
  if (listening) return
  listening = true
  document.addEventListener('visibilitychange', () => { if (enabled && document.visibilityState === 'visible') void pull() })
}

// Everything the app stores for one person. Cleared on sign-out (shared devices) and before an account takes over.
export function clearLocalData() {
  try {
    const drop = Object.keys(localStorage).filter((k) => isKey(k) || k === 'sync:meta' || k.startsWith('demo:') || k.startsWith('kit:'))
    drop.forEach((k) => localStorage.removeItem(k))
  } catch { /* ignore */ }
}

// ---- auth (email + password; invite-only, so there is no sign-up form) ----
export async function currentUser(): Promise<{ id: string; email: string } | null> {
  try {
    const sb = await supabase()
    const { data } = await sb.auth.getSession()
    const u = data.session?.user
    return u ? { id: u.id, email: u.email ?? '' } : null
  } catch { return null }
}

export type SignInResult = 'ok' | 'wrong' | 'throttled' | 'offline'
export async function signIn(email: string, password: string): Promise<SignInResult> {
  try {
    const sb = await supabase()
    const { error } = await sb.auth.signInWithPassword({ email: email.trim(), password })
    if (!error) return 'ok'
    return error.status === 429 ? 'throttled' : error.status && error.status < 500 ? 'wrong' : 'offline'
  } catch { return 'offline' }
}

export type SignUpResult = 'signed-in' | 'confirm-email' | 'exists' | 'closed' | 'weak' | 'throttled' | 'offline'
// Email + password sign-up. With "Confirm email" on in Supabase, the user confirms by email first.
export async function signUp(email: string, password: string): Promise<SignUpResult> {
  try {
    const sb = await supabase()
    const { data, error } = await sb.auth.signUp({ email: email.trim(), password, options: { emailRedirectTo: location.origin } })
    if (error) {
      const m = error.message.toLowerCase()
      if (error.status === 429) return 'throttled'
      if (m.includes('signups not allowed') || m.includes('signup is disabled')) return 'closed'
      if (m.includes('already registered') || m.includes('already exists')) return 'exists'
      if (m.includes('password')) return 'weak'
      return error.status && error.status < 500 ? 'weak' : 'offline'
    }
    // Supabase hides "email taken" (no error, no identities) to avoid leaking who has an account
    if (data.user && data.user.identities?.length === 0) return 'exists'
    return data.session ? 'signed-in' : 'confirm-email'
  } catch { return 'offline' }
}

export async function sendPasswordReset(email: string): Promise<boolean> {
  try {
    const sb = await supabase()
    const { error } = await sb.auth.resetPasswordForEmail(email.trim(), { redirectTo: location.origin })
    return !error
  } catch { return false }
}

export async function setNewPassword(password: string): Promise<string | null> {
  try {
    const sb = await supabase()
    const { error } = await sb.auth.updateUser({ password })
    return error ? error.message : null
  } catch { return "Can't reach the server. Check your connection." }
}

// Privacy: removes the account, its synced data (database cascade) and its photos.
export async function deleteAccount(): Promise<boolean> {
  try {
    const me = await currentUser()
    if (!me) return false
    const sb = await supabase()
    await sb.storage.from('photos').remove(PHOTO_SLOTS.map((s) => `${me.id}/${s.id}.jpg`))
    const { error } = await sb.rpc('delete_account')
    if (error) return false
    await signOut()
    return true
  } catch { return false }
}

export async function signOut() {
  enabled = false
  try { const sb = await supabase(); await sb.auth.signOut() } catch { /* ignore */ }
  clearLocalData()
  await clearPhotos()
}
