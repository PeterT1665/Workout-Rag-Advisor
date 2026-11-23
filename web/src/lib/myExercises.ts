// Exercises a player adds themselves: a name, the body parts it works and a quick how-to. Usable in their own
// sessions straight away (no 3D demo yet), synced with their account ('myExercises'), and sent to the team as a
// request (exercise_requests, migration 0013) so the 3D demo can be built and it can join the shared library.
import { load, save } from './storage'
import { registerExercises, type Sport } from './library'
import { accountsEnabled, supabase } from './supabase'
import { currentLang } from '../i18n'

export type MyExercise = { id: string; name: string; parts: string[]; how: string; sport: Sport; created: string; sent?: boolean; status?: 'requested' | 'building' | 'added' | 'declined' }
const KEY = 'myExercises'
export const loadMyExercises = () => load<MyExercise[]>(KEY, [])

// As library exercises: their own category, the parts they picked count for the injury filter.
const asExercise = (m: MyExercise) => ({
  id: m.id, name: m.name, sport: m.sport, category: 'mine', kind: 'drill', how: m.how, works: m.parts, also: [],
  work: 45, rest: 60, rounds: 3, ...(m.sport === 'gym' || m.sport === 'home' ? { reps: '10' } : {}), mine: true,
})
export const registerMine = () => registerExercises(loadMyExercises().map(asExercise))

export function addMyExercise(x: Pick<MyExercise, 'name' | 'parts' | 'how' | 'sport'>): MyExercise {
  const m: MyExercise = { ...x, name: x.name.trim().slice(0, 60), how: x.how.trim().slice(0, 600), id: `my-ex-${Date.now().toString(36)}`, created: new Date().toISOString(), status: 'requested' }
  save(KEY, [...loadMyExercises(), m])
  registerMine()
  void sendRequests()
  return m
}

// Signed in: send the ones not sent yet, and read back how far the team got with each.
export async function sendRequests() {
  if (!accountsEnabled) return
  try {
    const sb = await supabase()
    const { data } = await sb.auth.getSession()
    if (!data.session) return
    const all = loadMyExercises()
    const todo = all.filter((m) => !m.sent)
    if (todo.length) {
      const { error } = await sb.from('exercise_requests').upsert(
        todo.map((m) => ({ local_id: m.id, name: m.name, parts: m.parts, how: m.how, sport: m.sport, lang: currentLang() })),
        { onConflict: 'user_id,local_id', ignoreDuplicates: true })
      if (error) return // before migration 0013, or offline: kept here and sent next time
    }
    const { data: rows } = await sb.from('exercise_requests').select('local_id, status')
    const status = new Map((rows ?? []).map((r: { local_id: string; status: MyExercise['status'] }) => [r.local_id, r.status]))
    save(KEY, loadMyExercises().map((m) => ({ ...m, sent: m.sent || todo.some((x) => x.id === m.id), status: status.get(m.id) ?? m.status })))
  } catch { /* offline: try again next time */ }
}
