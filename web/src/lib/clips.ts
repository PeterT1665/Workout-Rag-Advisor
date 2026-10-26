// Peter's own audio clips. Stored only on this device (IndexedDB); never uploaded.
import type { Clip, ClipKind } from './types'

const DB = 'training-media'
const STORE = 'clips'

let db: IDBDatabase | null = null
let cache: Clip[] = []
const mem: Clip[] = []
const listeners = new Set<() => void>()

function req<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T | null> {
  return new Promise((res) => {
    try {
      const r = fn(db!.transaction(STORE, mode).objectStore(STORE))
      r.onsuccess = () => res(r.result)
      r.onerror = () => res(null)
    } catch {
      res(null)
    }
  })
}

let ready: Promise<void> | null = null

// Opens once; every read/write waits for it, so clips added during startup are never lost.
export function openClips(): Promise<void> {
  ready ??= new Promise<void>((res) => {
    try {
      const r = indexedDB.open(DB, 1)
      r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'id' })
      r.onsuccess = () => { db = r.result; res() }
      r.onerror = () => res()
    } catch {
      res()
    }
  }).then(refresh)
  return ready
}

async function refresh() {
  cache = db ? ((await req<Clip[]>('readonly', (s) => s.getAll())) ?? []) : mem.slice()
  listeners.forEach((l) => l())
}

export async function addClip(clip: Clip) {
  await openClips()
  if (db) await req('readwrite', (s) => s.put(clip))
  else mem.push(clip)
  await refresh()
}

export async function removeClip(id: string) {
  await openClips()
  if (db) await req('readwrite', (s) => s.delete(id))
  else mem.splice(mem.findIndex((c) => c.id === id), 1)
  await refresh()
}

export const clipsOf = (kind: ClipKind) => cache.filter((c) => c.kind === kind)
export const allClips = () => cache
export function subscribeClips(fn: () => void) {
  listeners.add(fn)
  return () => { listeners.delete(fn) }
}
