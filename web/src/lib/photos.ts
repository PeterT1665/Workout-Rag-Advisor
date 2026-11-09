// The user's own photos (role model, gear...) for the app's image slots. Kept on the device (IndexedDB);
// signed-in users also get a private copy in Supabase Storage (photos/<user id>/<slot>.jpg) so other devices see them.
import { useSyncExternalStore } from 'react'
import { supabase } from './supabase'

export const PHOTO_SLOTS = [
  { id: 'hero', label: 'Home hero', hint: 'Big photo on the home screen' },
  { id: 'card', label: 'Player card', hint: 'Behind your player card' },
  { id: 'preroll', label: 'Kick-off', hint: 'Shown before each session' },
  { id: 'mindset-1', label: 'Mindset 1', hint: 'Motivation card' },
  { id: 'mindset-2', label: 'Mindset 2', hint: 'Motivation card' },
] as const
export type Slot = (typeof PHOTO_SLOTS)[number]['id']
type Photo = { slot: Slot; blob: Blob; updated: number }

const DB = 'training-photos'
const STORE = 'photos'
const BUCKET = 'photos'
let db: IDBDatabase | null = null
const mem = new Map<Slot, Photo>()
let urls: Partial<Record<Slot, string>> = {}
const listeners = new Set<() => void>()
let ready: Promise<void> | null = null
let accountId: string | null = null // set while signed in: uploads mirror to storage

function req<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T | null> {
  return new Promise((res) => {
    try {
      const r = fn(db!.transaction(STORE, mode).objectStore(STORE))
      r.onsuccess = () => res(r.result)
      r.onerror = () => res(null)
    } catch { res(null) }
  })
}

export function openPhotos(): Promise<void> {
  ready ??= new Promise<void>((res) => {
    try {
      const r = indexedDB.open(DB, 1)
      r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'slot' })
      r.onsuccess = () => { db = r.result; res() }
      r.onerror = () => res()
    } catch { res() }
  }).then(refresh)
  return ready
}

async function all(): Promise<Photo[]> {
  return db ? ((await req<Photo[]>('readonly', (s) => s.getAll())) ?? []) : [...mem.values()]
}

async function refresh() {
  const next: Partial<Record<Slot, string>> = {}
  for (const p of await all()) next[p.slot] = typeof URL.createObjectURL === 'function' ? URL.createObjectURL(p.blob) : `blob:${p.slot}`
  Object.values(urls).forEach((u) => { if (u?.startsWith('blob:http')) URL.revokeObjectURL(u) })
  urls = next
  listeners.forEach((l) => l())
}

async function putLocal(p: Photo) {
  await openPhotos()
  if (db) await req('readwrite', (s) => s.put(p)); else mem.set(p.slot, p)
}

// Big phone photos are scaled down (max 1600 px, JPEG) before storing: fast to load, small to sync.
async function shrink(file: Blob): Promise<Blob> {
  try {
    const img = await createImageBitmap(file)
    const scale = Math.min(1, 1600 / Math.max(img.width, img.height))
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(img.width * scale); canvas.height = Math.round(img.height * scale)
    canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height)
    return await new Promise<Blob>((res) => canvas.toBlob((b) => res(b ?? file), 'image/jpeg', 0.85))
  } catch { return file } // no canvas (tests, very old browsers): keep the original
}

export async function setPhoto(slot: Slot, file: Blob) {
  const blob = await shrink(file)
  await putLocal({ slot, blob, updated: Date.now() })
  await refresh()
  if (accountId) {
    try {
      const sb = await supabase()
      await sb.storage.from(BUCKET).upload(`${accountId}/${slot}.jpg`, blob, { upsert: true, contentType: 'image/jpeg' })
    } catch { /* offline: the next sign-in pull re-uploads */ }
  }
}

export async function removePhoto(slot: Slot) {
  await openPhotos()
  if (db) await req('readwrite', (s) => s.delete(slot)); else mem.delete(slot)
  await refresh()
  if (accountId) {
    try { const sb = await supabase(); await sb.storage.from(BUCKET).remove([`${accountId}/${slot}.jpg`]) } catch { /* ignore */ }
  }
}

// Signed in: download photos this device doesn't have, upload ones only this device has.
export async function syncPhotos(userId: string) {
  accountId = userId
  try {
    await openPhotos()
    const sb = await supabase()
    const { data } = await sb.storage.from(BUCKET).list(userId)
    const remote = new Map((data ?? []).map((f) => [f.name.replace(/\.jpg$/, ''), Date.parse(f.updated_at ?? '') || 0]))
    const local = new Map((await all()).map((p) => [p.slot, p]))
    for (const { id } of PHOTO_SLOTS) {
      const r = remote.get(id)
      const l = local.get(id)
      if (r !== undefined && (!l || r > l.updated + 1000)) {
        const { data: blob } = await sb.storage.from(BUCKET).download(`${userId}/${id}.jpg`)
        if (blob) await putLocal({ slot: id, blob, updated: r })
      } else if (l && r === undefined) {
        await sb.storage.from(BUCKET).upload(`${userId}/${id}.jpg`, l.blob, { upsert: true, contentType: 'image/jpeg' })
      }
    }
    await refresh()
  } catch { /* offline: try again next time */ }
}

export async function clearPhotos() {
  accountId = null
  await openPhotos()
  if (db) await req('readwrite', (s) => s.clear()); else mem.clear()
  await refresh()
}

const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } }
const snapshot = () => urls
export const usePhotos = () => useSyncExternalStore(subscribe, snapshot)
