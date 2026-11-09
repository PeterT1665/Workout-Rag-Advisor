import { beforeEach, describe, expect, test, vi } from 'vitest'

// Fake Supabase: an in-memory user_state table and a put_state that follows the database rules (newer wins).
let deleted = false
const removed: string[][] = []
const storage = { from: () => ({ remove: async (paths: string[]) => { removed.push(paths); return { error: null } } }) }
const rows = new Map<string, { key: string; value: unknown; updated_at: number }>()
const rpc = vi.fn(async (fn: string, a?: { p_key: string; p_value: unknown; p_updated_at: number }) => {
  if (fn === 'delete_account') { deleted = true; return { data: null, error: null } }
  if (!a) return { data: null, error: { message: 'bad call' } }
  const cur = rows.get(a.p_key)
  if (cur && cur.updated_at >= a.p_updated_at && a.p_key !== 'history') return { data: cur, error: null }
  const row = { key: a.p_key, value: a.p_value, updated_at: a.p_updated_at }
  rows.set(a.p_key, row)
  return { data: row, error: null }
})
const auth = {
  signInWithPassword: vi.fn(),
  signOut: vi.fn(async () => ({ error: null })),
  getSession: vi.fn(async () => ({ data: { session: null } })),
}
vi.mock('./supabase', () => ({
  accountsEnabled: true,
  arrivedToSetPassword: false,
  supabase: async () => ({ rpc, auth, storage, from: () => ({ select: async () => ({ data: [...rows.values()], error: null }) }) }),
}))

const { clearLocalData, deleteAccount, pull, signIn, signOut } = await import('./sync')
const get = (k: string) => JSON.parse(localStorage.getItem(k) || 'null')
const flush = () => new Promise((r) => setTimeout(r, 0))

beforeEach(() => {
  localStorage.clear()
  rows.clear()
  rpc.mockClear()
})

describe('sync', () => {
  test('a new device takes the account copy', async () => {
    rows.set('history', { key: 'history', value: [{ id: 'a' }], updated_at: 5 })
    rows.set('profile', { key: 'profile', value: { name: 'Peter' }, updated_at: 5 })
    expect(await pull()).toBe(true)
    expect(get('history')).toEqual([{ id: 'a' }])
    expect(get('profile')).toEqual({ name: 'Peter' })
  })

  test('the newer side wins: newer account copy replaces local, newer local copy is pushed', async () => {
    localStorage.setItem('games', JSON.stringify([{ day: 1 }]))
    localStorage.setItem('settings', JSON.stringify({ voice: false }))
    localStorage.setItem('sync:meta', JSON.stringify({ games: 10, settings: 30 }))
    rows.set('games', { key: 'games', value: [{ day: 4 }], updated_at: 20 })
    rows.set('settings', { key: 'settings', value: { voice: true }, updated_at: 20 })
    await pull()
    await flush()
    expect(get('games')).toEqual([{ day: 4 }])
    expect(rpc).toHaveBeenCalledWith('put_state', { p_key: 'settings', p_value: { voice: false }, p_updated_at: 30 })
    expect(rows.get('settings')?.value).toEqual({ voice: false })
  })

  test('local history is always sent up to be merged', async () => {
    localStorage.setItem('history', JSON.stringify([{ id: 'b' }]))
    await pull()
    await flush()
    expect(rpc).toHaveBeenCalledWith('put_state', expect.objectContaining({ p_key: 'history' }))
  })

  test('sign-out clears this person\'s data but nothing else', async () => {
    for (const k of ['history', 'profile', 'mySessions', 'sync:meta', 'kit:2026-11-09', 'demo:seedDay']) localStorage.setItem(k, '1')
    localStorage.setItem('unrelated', 'keep')
    await signOut()
    expect(Object.keys(localStorage)).toEqual(['unrelated'])
    expect(auth.signOut).toHaveBeenCalled()
  })

  test('clearLocalData keeps keys from other apps', () => {
    localStorage.setItem('games', '[]')
    localStorage.setItem('other', '1')
    clearLocalData()
    expect(Object.keys(localStorage)).toEqual(['other'])
  })

  test('sign-in results', async () => {
    auth.signInWithPassword.mockResolvedValueOnce({ error: null })
    expect(await signIn(' a@b.co ', 'pw')).toBe('ok')
    expect(auth.signInWithPassword).toHaveBeenLastCalledWith({ email: 'a@b.co', password: 'pw' })
    auth.signInWithPassword.mockResolvedValueOnce({ error: { status: 400 } })
    expect(await signIn('a@b.co', 'x')).toBe('wrong')
    auth.signInWithPassword.mockResolvedValueOnce({ error: { status: 429 } })
    expect(await signIn('a@b.co', 'x')).toBe('throttled')
    auth.signInWithPassword.mockRejectedValueOnce(new Error('network'))
    expect(await signIn('a@b.co', 'x')).toBe('offline')
  })
  test('delete account: removes photos, deletes the user, clears this device', async () => {
    auth.getSession.mockResolvedValueOnce({ data: { session: { user: { id: 'u1', email: 'a@b.co' } } } } as never)
    localStorage.setItem('history', '[]')
    expect(await deleteAccount()).toBe(true)
    expect(removed.at(-1)).toContain('u1/hero.jpg')
    expect(deleted).toBe(true)
    expect(localStorage.getItem('history')).toBeNull()
  })
})
