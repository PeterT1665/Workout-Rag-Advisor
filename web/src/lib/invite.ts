// Friend invites by QR (supabase/migrations/0008): your QR opens the app at /?friend=<code>. The visitor sees who
// invited them; once they sign up or sign in, you're friends straight away.
import { supabase } from './supabase'

const KEY = 'invite:friend' // code waiting to be accepted (kept through sign-up and email confirmation)
const DONE = 'invite:done'
const SEEN = 'invite:seen' // the invite screen was shown; the code waits until they have an account // name of the friend just added, for a one-time "you're friends" note

// Run once at start: keep the code from the link and tidy the address bar.
export function captureInvite() {
  try {
    const url = new URL(location.href)
    const code = url.searchParams.get('friend')?.toLowerCase()
    if (code && /^[a-z0-9]{10}$/.test(code)) { localStorage.setItem(KEY, code); localStorage.removeItem(SEEN) } // a new scan shows the invite again
    if (url.searchParams.has('friend')) { url.searchParams.delete('friend'); history.replaceState(null, '', url.pathname + url.search + url.hash) }
  } catch { /* ignore */ }
}
export const pendingInvite = () => { try { return localStorage.getItem(KEY) } catch { return null } }
export const dropInvite = () => { try { localStorage.removeItem(KEY); localStorage.removeItem(SEEN) } catch { /* ignore */ } }
// "Not now": try the app first. The invite is kept and accepted whenever they make an account or sign in.
export const laterInvite = () => { try { localStorage.setItem(SEEN, '1') } catch { /* ignore */ } }
export const inviteSeen = () => { try { return localStorage.getItem(SEEN) === '1' } catch { return false } }

export type Inviter = { name: string; number: number | null }
export async function inviteInfo(code: string): Promise<Inviter | null> {
  try {
    const sb = await supabase()
    const { data } = await sb.rpc('invite_info', { p_code: code })
    const row = (data as Inviter[] | null)?.[0]
    return row ? { name: row.name, number: row.number } : null
  } catch { return null }
}

// Signed in with a code waiting: become friends, remember who for the Team tab's note.
export async function acceptPendingInvite() {
  const code = pendingInvite()
  if (!code) return
  try {
    const sb = await supabase()
    const who = await inviteInfo(code)
    const { data, error } = await sb.rpc('accept_invite', { p_code: code })
    if (error) return // offline: try again next time
    dropInvite()
    if (data === 'ok' && who) localStorage.setItem(DONE, who.name)
  } catch { /* offline: try again next time */ }
}
export function takeNewFriend() {
  try { const n = localStorage.getItem(DONE); if (n !== null) localStorage.removeItem(DONE); return n } catch { return null }
}

// My invite link (made on first use).
export async function myInviteLink(name: string, number: number): Promise<string | null> {
  try {
    const sb = await supabase()
    const { data, error } = await sb.rpc('my_invite_code', { p_name: name, p_number: number })
    return error || !data ? null : `${location.origin}/?friend=${data as string}`
  } catch { return null }
}
