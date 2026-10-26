// Phone <-> laptop sync through the backend. Local-first: localStorage is the source the UI reads;
// changes are pushed in the background and pulled when the app opens or comes back to the foreground.
import { setSaveHook } from './storage'

const KEYS = ['history', 'games', 'settings', 'quotes'] as const
type Key = (typeof KEYS)[number]
type Entry = { value: unknown; updated_at: number }

let enabled = false
const timers: Partial<Record<Key, ReturnType<typeof setTimeout>>> = {}

const readMeta = (): Record<string, number> => { try { return JSON.parse(localStorage.getItem('sync:meta') || '{}') } catch { return {} } }
const writeMeta = (m: Record<string, number>) => { try { localStorage.setItem('sync:meta', JSON.stringify(m)) } catch { /* ignore */ } }
const readRaw = (k: string) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : null } catch { return null } }
const writeRaw = (k: string, v: unknown) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* ignore */ } }
const isKey = (k: string): k is Key => (KEYS as readonly string[]).includes(k)

export const syncEnabled = () => enabled

async function push(key: Key) {
  const meta = readMeta()
  const value = readRaw(key)
  if (value === null) return
  try {
    const r = await fetch(`/api/state/${key}`, {
      method: 'PUT', credentials: 'include', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value, updated_at: meta[key] ?? Date.now() }),
    })
    if (!r.ok) return
    const e: Entry = await r.json()
    if (JSON.stringify(e.value) !== JSON.stringify(value)) {
      writeRaw(key, e.value) // the server merged or had something newer
      window.dispatchEvent(new Event('ta-sync'))
    }
    writeMeta({ ...readMeta(), [key]: e.updated_at })
  } catch { /* offline: the next pull/save retries */ }
}

function changed(key: string) {
  if (!enabled || !isKey(key)) return
  writeMeta({ ...readMeta(), [key]: Date.now() })
  clearTimeout(timers[key])
  timers[key] = setTimeout(() => void push(key), 800)
}

// Pull the server copy, keep whichever side is newer, and push local changes the server hasn't seen.
export async function pull(): Promise<boolean> {
  try {
    const r = await fetch('/api/state', { credentials: 'include' })
    if (!r.ok) return false
    const server: Record<string, Entry> = await r.json()
    const meta = readMeta()
    let touched = false
    for (const key of KEYS) {
      const s = server[key]
      const local = meta[key] ?? 0
      if (key === 'history') {
        if (readRaw('history') === null) {
          if (s) { writeRaw('history', s.value); meta.history = s.updated_at; touched = true } // new device: take the server's
        } else {
          void push('history') // server merges both lists and returns the union
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

export function startSync() {
  enabled = true
  setSaveHook(changed)
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') void pull() })
}

// ---- auth ----
export type AuthState = 'local' | 'login' | 'ok'

// 'local' = no backend reachable (dev, or offline on first load): the app runs without login or sync.
export async function checkAuth(): Promise<AuthState> {
  try {
    const r = await fetch('/api/auth/me', { credentials: 'include' })
    if (r.status === 401) return 'login'
    if (r.ok && (r.headers.get('content-type') || '').includes('json')) return 'ok'
    return 'local'
  } catch { return 'local' }
}

export async function login(password: string): Promise<'ok' | 'wrong' | 'throttled' | 'offline'> {
  try {
    const r = await fetch('/api/auth/login', {
      method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password }),
    })
    if (r.ok) return 'ok'
    return r.status === 429 ? 'throttled' : 'wrong'
  } catch { return 'offline' }
}

export async function logout() {
  try { await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' }) } catch { /* ignore */ }
}
