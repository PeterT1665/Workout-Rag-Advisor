// The session in progress, saved as it runs so a reload (iOS closes paused apps) never loses it.
import { load } from './storage'
import type { Snapshot } from './player'
import type { Session } from './types'

export type ActiveSession = { session: Session; snap: Snapshot; savedAt: number }
const KEY = 'activeSession'
const MAX_AGE = 12 * 3600e3 // older than this, it's not "the session I just left"

export function loadActive(): ActiveSession | null {
  const a = load<ActiveSession | null>(KEY, null)
  return a && Date.now() - a.savedAt < MAX_AGE ? a : null
}
// written straight to localStorage: device-only, not synced
export function saveActive(session: Session, snap: Snapshot) {
  try { localStorage.setItem(KEY, JSON.stringify({ session, snap, savedAt: Date.now() })) } catch { /* ignore */ }
}
export function clearActive() {
  try { localStorage.removeItem(KEY) } catch { /* ignore */ }
}
