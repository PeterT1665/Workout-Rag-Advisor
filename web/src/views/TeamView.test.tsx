import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { activity } from '../lib/team'

// A tiny in-memory stand-in for the Supabase calls Team makes (the real rules are tested in supabase/tests/team_test.sql).
type Row = Record<string, unknown>
const db = { profiles: [] as Row[], friendships: [] as Row[] }
const ME = 'me-id'
const SAM = { user_id: 'sam-id', username: 'sam', display_name: 'Sam', number: 11, positions: ['winger'], foot: 'right', accent: 'red', avatar_path: null, card: null, rating: 70, streak: 12, best_streak: 12, trained_today: true, last_active: new Date().toISOString() }

function table(name: 'profiles' | 'friendships') {
  const filters: [string, unknown][] = []
  const rows = () => db[name].filter((r) => filters.every(([k, v]) => r[k] === v))
  const q = {
    select: () => q,
    eq: (k: string, v: unknown) => { filters.push([k, v]); return q },
    maybeSingle: async () => ({ data: rows()[0] ?? null }),
    upsert: async (row: Row) => { db[name] = [...db[name].filter((r) => r.user_id !== row.user_id), { streak: 0, best_streak: 0, trained_today: false, card: null, rating: null, avatar_path: null, last_active: null, ...row }]; return { error: null } },
    update: (patch: Row) => { const u = { eq: (k: string, v: unknown) => { filters.push([k, v]); rows().forEach((r) => Object.assign(r, patch)); return Object.assign(Promise.resolve({ error: null }), u) } }; return u },
    insert: async (row: Row) => { db[name].push({ status: 'pending', ...row }); return { error: null } },
    delete: () => ({ or: async () => ({ error: null }) }),
    then: (res: (v: { data: Row[] }) => void) => res({ data: name === 'profiles' ? visibleProfiles() : rows() }),
  }
  return q
}
const visibleProfiles = () => db.profiles.filter((p) => p.user_id === ME || db.friendships.some((f) => f.status === 'accepted' && [f.requester, f.addressee].includes(p.user_id)))
const fake = {
  from: (n: 'profiles' | 'friendships') => table(n),
  rpc: async (fn: string, args: Row) => {
    if (fn === 'find_player') return { data: args.p_username === 'sam' && args.p_number === 11 ? [{ ...SAM, friendship: db.friendships.length ? 'requested' : null }] : [] }
    return { data: [] }
  },
  storage: { from: () => ({ createSignedUrls: async () => ({ data: [] }), upload: async () => ({ error: null }) }) },
}
vi.mock('../lib/supabase', () => ({ supabase: async () => fake, accountsEnabled: true, arrivedToSetPassword: false }))
vi.mock('../lib/sync', async (orig) => ({ ...(await orig<object>()), syncEnabled: () => true, currentUser: async () => ({ id: ME, email: 'x@y.z' }) }))

const { TeamView } = await import('./TeamView')
const streak = { days: 3, today: true }

beforeEach(() => { db.profiles = []; db.friendships = []; localStorage.clear() })
afterEach(cleanup)

describe('team', () => {
  test('pick a username, find a friend by username + number, send a request', async () => {
    render(<TeamView streak={streak} />)
    fireEvent.change(await screen.findByLabelText('Username'), { target: { value: '@Peter.7' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(await screen.findByText(/@peter\.7/)).toBeTruthy()
    expect(db.profiles[0]).toMatchObject({ user_id: ME, username: 'peter.7' })

    const add = screen.getByRole('region', { name: 'Add a friend' })
    fireEvent.change(within(add).getByLabelText("Friend's username"), { target: { value: 'sam' } })
    fireEvent.change(within(add).getByLabelText("Friend's shirt number"), { target: { value: '10' } })
    fireEvent.click(within(add).getByRole('button', { name: 'Find' }))
    expect(await within(add).findByText(/No player with that username and number/)).toBeTruthy()

    fireEvent.change(within(add).getByLabelText("Friend's shirt number"), { target: { value: '11' } })
    fireEvent.click(within(add).getByRole('button', { name: 'Find' }))
    fireEvent.click(await within(add).findByRole('button', { name: 'Add' }))
    expect(await within(add).findByText('Requested')).toBeTruthy()
    expect(db.friendships[0]).toMatchObject({ requester: ME, addressee: 'sam-id', status: 'pending' })
  })

  test('friends are on the leaderboard by streak, with online status and their card', async () => {
    db.profiles = [{ ...SAM }, { ...SAM, user_id: ME, username: 'peter.7', display_name: 'Peter', number: 7, streak: 3, rating: 75, last_active: null }]
    db.friendships = [{ requester: ME, addressee: 'sam-id', status: 'accepted' }]
    render(<TeamView streak={streak} />)
    const board = await screen.findByRole('region', { name: 'Leaderboard' })
    await waitFor(() => expect(within(board).getAllByRole('listitem')).toHaveLength(2))
    const rows = within(board).getAllByRole('listitem')
    expect(rows[0].textContent).toMatch(/Sam.*#11.*Online.*12.*70/)
    expect(rows[1].textContent).toMatch(/Peter \(you\)/)
    fireEvent.click(within(rows[0]).getByRole('button'))
    expect(screen.getByRole('dialog', { name: "Sam's card" })).toBeTruthy()
  })

  test('activity labels', () => {
    const now = Date.parse('2026-11-16T12:00:00Z')
    expect(activity(new Date(now - 2 * 60000).toISOString(), now)).toEqual({ online: true, label: 'Online' })
    expect(activity(new Date(now - 3 * 3600000).toISOString(), now).label).toBe('3 h ago')
    expect(activity(new Date(now - 26 * 3600000).toISOString(), now).label).toBe('Yesterday')
    expect(activity(null, now).online).toBe(false)
  })
})
