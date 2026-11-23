// A photo for a training day (asked at Full time, or added in the calendar): the day's tile in the streak calendar and
// the background when you share it. On this phone only (IndexedDB), never uploaded: it leaves the phone only in a
// picture you choose to share.
import { useSyncExternalStore } from 'react'

type DayPhoto = { date: string; blob: Blob; updated: number }
const DB = 'training-day-photos'
const STORE = 'days'
let db: IDBDatabase | null = null
const mem = new Map<string, DayPhoto>()
let urls: Record<string, string> = {}
const listeners = new Set<() => void>()
let ready: Promise<void> | null = null

function req<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T | null> {
  return new Promise((res) => {
    try {
      const r = fn(db!.transaction(STORE, mode).objectStore(STORE))
      r.onsuccess = () => res(r.result)
      r.onerror = () => res(null)
    } catch { res(null) }
  })
}

export function openDayPhotos(): Promise<void> {
  ready ??= new Promise<void>((res) => {
    try {
      const r = indexedDB.open(DB, 1)
      r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'date' })
      r.onsuccess = () => { db = r.result; res() }
      r.onerror = () => res()
    } catch { res() }
  }).then(refresh)
  return ready
}

async function refresh() {
  const list = db ? ((await req<DayPhoto[]>('readonly', (s) => s.getAll())) ?? []) : [...mem.values()]
  const next: Record<string, string> = {}
  for (const p of list) next[p.date] = typeof URL.createObjectURL === 'function' ? URL.createObjectURL(p.blob) : `blob:${p.date}`
  Object.values(urls).forEach((u) => { if (u.startsWith('blob:http')) URL.revokeObjectURL(u) })
  urls = next
  listeners.forEach((l) => l())
}

// Phone photos are big: scaled to 1400 px on the long side, JPEG.
async function shrink(file: Blob): Promise<Blob> {
  try {
    const img = await createImageBitmap(file)
    const scale = Math.min(1, 1400 / Math.max(img.width, img.height))
    const c = document.createElement('canvas')
    c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale)
    c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height)
    return await new Promise<Blob>((res) => c.toBlob((b) => res(b ?? file), 'image/jpeg', 0.85))
  } catch { return file }
}

export async function setDayPhoto(date: string, file: Blob) {
  await openDayPhotos()
  const p = { date, blob: await shrink(file), updated: Date.now() }
  if (db) await req('readwrite', (s) => s.put(p)); else mem.set(date, p)
  await refresh()
}
export async function removeDayPhoto(date: string) {
  await openDayPhotos()
  if (db) await req('readwrite', (s) => s.delete(date)); else mem.delete(date)
  await refresh()
}

const subscribe = (l: () => void) => { void openDayPhotos(); listeners.add(l); return () => { listeners.delete(l) } }
const snapshot = () => urls
// date (YYYY-MM-DD) -> image URL, for every day with a photo
export const useDayPhotos = () => useSyncExternalStore(subscribe, snapshot, snapshot)

// The photo as an image, ready to draw on a share card.
export function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => res(null); i.src = src })
}
