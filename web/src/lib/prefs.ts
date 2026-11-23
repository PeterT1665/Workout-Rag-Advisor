// Your way of doing a session: what you change in the Plan sheet (swap an exercise, drop one, add one, change the sets,
// pair a superset) is learned when you finish, and the same session comes back that way next time. Football and gym.
// The match-day rules still win: nothing heavy the day before a game, whatever you added last time.
// Synced with the account ('prefs'), private.
import { load, save } from './storage'
import { exerciseBlock, exerciseByName, EXERCISES } from './library'
import type { Block, Session } from './types'
import { t } from '../i18n'

export type SessionPrefs = {
  swap?: Record<string, string> // exercise id -> the one you do instead
  drop?: string[] // exercise ids you leave out
  add?: { ex: string; after: string | null }[] // exercises you add, after which one (null: before the cool-down)
  rounds?: Record<string, number> // sets you do
  pair?: Record<string, boolean> // superset with the exercise before it, or not
}
export type Prefs = Record<string, SessionPrefs> // by session id
const KEY = 'prefs'
export const loadPrefs = () => load<Prefs>(KEY, {})
const savePrefs = (p: Prefs) => save(KEY, p)
export function forgetPrefs(sessionId: string) { const p = loadPrefs(); delete p[sessionId]; savePrefs(p) }

const idOf = (b: Pick<Block, 'name'>) => exerciseByName(b.name)?.id ?? b.name
const drills = (s: Session) => s.blocks.filter((b) => b.kind === 'drill')
const catOf = (b: Block) => b.category ?? exerciseByName(b.name)?.category

// What changed between the session as it started and as it was done: the player's own edits.
export function diff(base: Session, done: Session): SessionPrefs {
  const a = drills(base), b = drills(done)
  const aIds = a.map(idOf), bIds = b.map(idOf)
  const removed = a.filter((x) => !bIds.includes(idOf(x)))
  const added = b.filter((x) => !aIds.includes(idOf(x)))
  const out: SessionPrefs = {}
  // a removed and an added exercise of the same kind: a swap
  for (const r of [...removed]) {
    const k = added.findIndex((x) => catOf(x) === catOf(r))
    if (k < 0) continue
    ;(out.swap ??= {})[idOf(r)] = idOf(added[k])
    removed.splice(removed.indexOf(r), 1)
    added.splice(k, 1)
  }
  if (removed.length) out.drop = removed.map(idOf)
  if (added.length) out.add = added.map((x) => {
    const i = done.blocks.indexOf(x)
    const before = done.blocks.slice(0, i).reverse().find((y) => y.kind === 'drill' && aIds.includes(idOf(y)))
    return { ex: idOf(x), after: before ? idOf(before) : null }
  })
  for (const x of b) {
    const i = aIds.indexOf(idOf(x))
    const was = i >= 0 ? a[i] : null
    if (was && (x.rounds || 1) !== (was.rounds || 1)) (out.rounds ??= {})[idOf(x)] = x.rounds || 1
    if (was && !!x.superset !== !!was.superset) (out.pair ??= {})[idOf(x)] = !!x.superset
  }
  return out
}

// Learn from a finished session: the edits are added on top of what was learned before.
export function learn(base: Session, done: Session, store: Prefs = loadPrefs()): Prefs {
  const d = diff(base, done)
  if (!d.swap && !d.drop && !d.add && !d.rounds && !d.pair) return store
  const p: SessionPrefs = { ...(store[base.id] ?? {}) }
  const wasDropped = p.drop ?? [], wasAdded = p.add ?? []
  const addedNow = new Set([...(d.add ?? []).map((x) => x.ex), ...Object.values(d.swap ?? {})])
  if (d.swap) {
    // a swap of something learned earlier: follow it back to the original (A -> B, then B -> C is A -> C)
    const swap = { ...(p.swap ?? {}) }
    for (const [from, to] of Object.entries(d.swap)) {
      const orig = Object.keys(swap).find((k) => swap[k] === from)
      if (orig) { if (orig === to) delete swap[orig]; else swap[orig] = to } else swap[from] = to
    }
    p.swap = swap
  }
  // added back: no longer dropped; dropped now: no longer added
  p.drop = [...new Set([...wasDropped.filter((x) => !addedNow.has(x)), ...(d.drop ?? []).filter((x) => !wasAdded.some((a) => a.ex === x))])]
  p.add = [...wasAdded.filter((a) => !(d.drop ?? []).includes(a.ex)), ...(d.add ?? []).filter((a) => !wasDropped.includes(a.ex))]
  if (d.rounds) p.rounds = { ...(p.rounds ?? {}), ...d.rounds }
  if (d.pair) p.pair = { ...(p.pair ?? {}), ...d.pair }
  for (const k of ['drop', 'add'] as const) if (!p[k]?.length) delete p[k]
  const next = { ...store, [base.id]: p }
  savePrefs(next)
  return next
}

// The match-day rules an exercise has to pass to go into a session (lib/weekplan plans the jobs; this keeps
// what you add yourself inside them): no heavy legs or Olympic lifts within two days of a game, no conditioning the day
// before, only light work the day after.
const HEAVY = ['lower-strength', 'olympic']
export function fitsDay(category: string | undefined, md: string | null): boolean {
  if (!md || !category) return true
  if (md === 'MD') return false
  if (md === 'MD+1') return ['mobility', 'cooldown', 'warmup', 'core', 'isometrics', 'first-touch', 'passing'].includes(category)
  if (md === 'MD-1') return ![...HEAVY, 'single-limb', 'conditioning', 'gym-conditioning'].includes(category)
  if (md === 'MD-2') return !HEAVY.includes(category)
  return true
}

export type Applied = { text: string }[]
// A session the way you do it. `md` is today's match-day label (for the rules); `applied` lists what changed.
export function withPrefs(s: Session, md: string | null = null, store: Prefs = loadPrefs()): { session: Session; applied: Applied } {
  const p = store[s.id]
  if (!p) return { session: s, applied: [] }
  const applied: Applied = []
  const name = (id: string) => t(EXERCISES.find((e) => e.id === id)?.name ?? id)
  const swapTo = (id: string) => { let to = p.swap?.[id]; for (let i = 0; i < 3 && to && p.swap?.[to]; i++) to = p.swap[to]; return to }
  const present = new Set(s.blocks.map(idOf))
  let blocks: Block[] = []
  let unpairNext = false
  for (const orig of s.blocks) {
    const b = unpairNext && orig.superset ? { ...orig, superset: undefined } : orig
    unpairNext = false
    if (b.kind !== 'drill') { blocks.push(b); continue }
    const id = idOf(b)
    if (p.drop?.includes(id)) { applied.push({ text: t('{name}: left out', { name: t(b.name) }) }); if (!b.superset) unpairNext = true; continue }
    const to = swapTo(id)
    const blk = to && !present.has(to) ? exerciseBlock(to) : undefined
    if (blk && fitsDay(blk.category, md)) {
      blocks.push({ ...blk, rounds: b.rounds ?? blk.rounds, rest: b.rest ?? blk.rest, superset: b.superset })
      present.add(to!)
      applied.push({ text: `${t(b.name)} → ${t(blk.name)}` })
    } else blocks.push(b)
  }
  for (const a of p.add ?? []) {
    const blk = exerciseBlock(a.ex)
    if (!blk || present.has(a.ex) || !fitsDay(blk.category, md)) continue
    const at = a.after ? blocks.findIndex((x) => idOf(x) === a.after) : -1
    const cool = blocks.findIndex((x) => x.kind === 'cooldown')
    blocks.splice(at >= 0 ? at + 1 : cool >= 0 ? cool : blocks.length, 0, blk)
    present.add(a.ex)
    applied.push({ text: t('+ {name}', { name: name(a.ex) }) })
  }
  blocks = blocks.map((b) => {
    const id = idOf(b)
    const r = p.rounds?.[id], pr = p.pair?.[id]
    let out = b
    // more sets than planned: not the day before a game
    if (r && r !== (b.rounds || 1) && (r < (b.rounds || 1) || md !== 'MD-1')) { out = { ...out, rounds: r }; applied.push({ text: t('{name}: {n} sets', { name: t(b.name), n: r }) }) }
    if (pr !== undefined && pr !== !!b.superset) out = { ...out, superset: pr || undefined }
    return out
  })
  // a superset needs an exercise right before it
  blocks = blocks.map((b, k) => (b.superset && (k === 0 || blocks[k - 1].kind !== 'drill') ? { ...b, superset: undefined } : b))
  return { session: { ...s, blocks }, applied }
}
