// Team (phase 1): your public profile, friends and the friends leaderboard. Server rules: supabase/migrations/0005_team.sql
// (only you and accepted friends read a profile; search needs exact username + shirt number).
// The app publishes your streak and rating; the database keeps them sane.
import { supabase } from './supabase'
import { currentUser } from './sync'
import type { CardStats } from '../components/PlayerCard'
import type { Profile } from './profile'
import type { Socials } from './socials'
import { t } from '../i18n'

export type TeamProfile = {
  user_id: string; username: string; display_name: string; number: number
  positions: string[]; foot: string; accent: string; avatar_path: string | null
  card: CardStats | null; rating: number | null
  streak: number; best_streak: number; trained_today: boolean; last_active: string | null
  socials?: Socials | null // migration 0014; friends only, like the rest
  showcase?: { k: string; v: string }[] | null // migration 0015: the stats a player picked to show, friends only
}
export type FoundPlayer = { user_id: string; username: string; display_name: string; number: number; avatar_path: string | null; friendship: 'friends' | 'requested' | 'asked-you' | null }
export type FriendRequest = { user_id: string; username: string; display_name: string; number: number; avatar_path: string | null; incoming: boolean }

export const USERNAME = /^[a-z0-9_.]{3,20}$/
export const cleanUsername = (s: string) => s.trim().toLowerCase().replace(/^@/, '')

async function me() {
  const u = await currentUser()
  if (!u) throw new Error('signed-out')
  return u.id
}

export async function myTeamProfile(): Promise<TeamProfile | null> {
  const sb = await supabase()
  const id = await me()
  const { data } = await sb.from('profiles').select('*').eq('user_id', id).maybeSingle()
  return (data as TeamProfile | null) ?? null
}

export type ClaimResult = 'ok' | 'taken' | 'invalid' | 'too-soon' | 'offline'
// Create or rename your public profile.
export async function claimUsername(username: string, p: Profile): Promise<ClaimResult> {
  const name = cleanUsername(username)
  if (!USERNAME.test(name)) return 'invalid'
  try {
    const sb = await supabase()
    const id = await me()
    const row = { user_id: id, username: name, display_name: p.name.slice(0, 40), number: p.number, positions: p.positions, foot: p.foot, accent: p.accent }
    const { error } = await sb.from('profiles').upsert(row, { onConflict: 'user_id' })
    if (!error) return 'ok'
    if (error.code === '23505') return 'taken'
    if (error.message.includes('username_recently_changed')) return 'too-soon'
    if (error.code === '23514') return 'invalid'
    return 'offline'
  } catch { return 'offline' }
}

// What friends see of you: card, streak, rating, last active. Called on open, after sessions, and while the app is open.
export async function publishStats(p: Profile, stats: { card: CardStats | null; rating: number | null; streak: number; trainedToday: boolean }) {
  try {
    const sb = await supabase()
    const id = await me()
    await sb.from('profiles').update({
      display_name: p.name.slice(0, 40), number: p.number, positions: p.positions, foot: p.foot, accent: p.accent,
      card: stats.card, rating: stats.rating, streak: Math.min(3660, stats.streak), trained_today: stats.trainedToday,
      last_active: new Date().toISOString(),
    }).eq('user_id', id)
  } catch { /* offline: next time */ }
}

// You and your accepted friends (row-level security returns exactly those), best streak first.
export async function leaderboard(): Promise<TeamProfile[]> {
  const sb = await supabase()
  const { data } = await sb.from('profiles').select('*')
  return ((data as TeamProfile[] | null) ?? []).sort((a, b) => b.streak - a.streak || (b.rating ?? 0) - (a.rating ?? 0) || a.username.localeCompare(b.username))
}

// The stats you picked to show friends. False when it didn't save (offline, or before migration 0015): the card works without it.
export async function publishShowcase(items: { k: string; v: string }[]): Promise<boolean> {
  try {
    const sb = await supabase()
    const id = await me()
    const { error } = await sb.from('profiles').update({ showcase: items }).eq('user_id', id)
    return !error
  } catch { return false }
}

// Your social links (cleaned in lib/socials). False when it didn't save (offline, or before migration 0014).
export async function saveSocials(socials: Socials): Promise<boolean> {
  try {
    const sb = await supabase()
    const id = await me()
    const clean = Object.fromEntries(Object.entries(socials).filter(([, v]) => v))
    const { error } = await sb.from('profiles').update({ socials: clean }).eq('user_id', id)
    return !error
  } catch { return false }
}

export async function findPlayer(username: string, number: number): Promise<FoundPlayer | null> {
  const sb = await supabase()
  const { data } = await sb.rpc('find_player', { p_username: cleanUsername(username), p_number: number })
  return ((data as FoundPlayer[] | null) ?? [])[0] ?? null
}

export async function friendRequests(): Promise<FriendRequest[]> {
  const sb = await supabase()
  const { data } = await sb.rpc('friend_requests')
  return (data as FriendRequest[] | null) ?? []
}

export async function askFriend(userId: string) {
  const sb = await supabase()
  const { error } = await sb.from('friendships').insert({ requester: await me(), addressee: userId })
  return !error
}
export async function acceptFriend(userId: string) {
  const sb = await supabase()
  const { error } = await sb.from('friendships').update({ status: 'accepted' }).eq('requester', userId).eq('addressee', await me())
  return !error
}
// Decline a request, cancel yours, or unfriend: the friendship row goes either way.
export async function removeFriend(userId: string) {
  const sb = await supabase()
  const id = await me()
  const { error } = await sb.from('friendships').delete().or(`and(requester.eq.${id},addressee.eq.${userId}),and(requester.eq.${userId},addressee.eq.${id})`)
  return !error
}

// ---------- avatars ----------
// Square, small (256 px JPEG): only signed-in players can load them, via short-lived links.
async function square(file: Blob): Promise<Blob> {
  try {
    const img = await createImageBitmap(file)
    const side = Math.min(img.width, img.height)
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = 256
    canvas.getContext('2d')!.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, 256, 256)
    return await new Promise<Blob>((res) => canvas.toBlob((b) => res(b ?? file), 'image/jpeg', 0.85))
  } catch { return file }
}
export async function setAvatar(file: Blob): Promise<boolean> {
  try {
    const sb = await supabase()
    const id = await me()
    const path = `${id}/avatar.jpg`
    const { error } = await sb.storage.from('avatars').upload(path, await square(file), { upsert: true, contentType: 'image/jpeg' })
    if (error) return false
    // a new value each time so friends' apps load the new photo
    await sb.from('profiles').update({ avatar_path: `${path}?v=${Date.now()}` }).eq('user_id', id)
    return true
  } catch { return false }
}
const urlCache = new Map<string, string>()
export async function avatarUrls(paths: (string | null)[]): Promise<Record<string, string>> {
  const want = [...new Set(paths.filter((p): p is string => !!p && !urlCache.has(p)))]
  if (want.length) {
    try {
      const sb = await supabase()
      const { data } = await sb.storage.from('avatars').createSignedUrls(want.map((p) => p.split('?')[0]), 3600)
      data?.forEach((d, i) => { if (d.signedUrl) urlCache.set(want[i], d.signedUrl) })
    } catch { /* offline: initials instead */ }
  }
  return Object.fromEntries(paths.filter((p): p is string => !!p && urlCache.has(p)).map((p) => [p, urlCache.get(p)!]))
}

// Online: active in the last 5 minutes; otherwise "today" / "2 days ago".
export function activity(lastActive: string | null, now = Date.now()): { online: boolean; label: string } {
  if (!lastActive) return { online: false, label: t('Not seen yet') }
  const mins = (now - Date.parse(lastActive)) / 60000
  if (mins < 5) return { online: true, label: t('Online') }
  if (mins < 60) return { online: false, label: t('{n} min ago', { n: Math.round(mins) }) }
  if (mins < 60 * 24) return { online: false, label: `${Math.round(mins / 60)} h ago` }
  const days = Math.round(mins / 1440)
  return { online: false, label: days === 1 ? t('Yesterday') : t('{n} days ago', { n: days }) }
}
