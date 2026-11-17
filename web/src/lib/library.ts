// Exercise library (D30): every exercise lives once; sessions are lists of exercise ids with optional overrides.
// Recommended templates ship with the app; the user's own sessions are saved locally (and synced later).
import data from '../data/library.json'
import { load, save } from './storage'
import type { Block, Session } from './types'
import { mark, t } from '../i18n'

export type Sport = 'football' | 'gym' | 'home' | 'mobility'
export type Exercise = Block & { id: string; sport: Sport; category: string; positions?: string[] }
export type Item = { ex: string; work?: number; rest?: number; rounds?: number; how?: string; superset?: boolean }
export type Template = { id: string; title: string; type: string; summary: string; needs: string[]; items: Item[] }
export type MySession = Template & { mine: true; updated: string }

// Instructions and descriptive reps ("8 each leg") in the app's language; names stay English, and "needs"
// stay English data (the kit check reads them) and are translated where they're shown.
const tReps = (r?: string) => (r && !/^\d+(-\d+)?$/.test(r) ? t(r) : r)
export const EXERCISES = (data.exercises as Exercise[]).map((e) => ({ ...e, how: e.how && t(e.how), reps: tReps(e.reps) }))
export const TEMPLATES = data.templates as Template[]
const byId = new Map(EXERCISES.map((e) => [e.id, e]))
export const exercise = (id: string) => byId.get(id)

export const SPORTS: { id: Sport; label: string }[] = [
  { id: 'football', label: t('Football') }, { id: 'gym', label: t('Gym') }, { id: 'home', label: t('Home') }, { id: 'mobility', label: t('Mobility') },
]
export const CATEGORY_LABELS: Record<string, string> = {
  warmup: t('Warm-up'), cooldown: t('Cool-down'), 'ball-mastery': t('Dribbling'), 'first-touch': t('First touch'), passing: t('Passing'),
  finishing: t('Finishing'), '1v1': '1v1', speed: t('Speed & agility'), conditioning: t('Conditioning'),
  olympic: t('Olympic lifts'), explosive: t('Explosive'), plyometrics: t('Plyo & jumps'), sprints: t('Sprints'),
  'lower-strength': t('Lower body'), 'upper-strength': t('Upper body'), 'single-limb': t('Single leg / arm'), isometrics: t('Isometrics'),
  core: t('Core'), 'gym-conditioning': t('Conditioning'), mobility: t('Mobility'),
}
// Categories in display order for one sport, only those that have exercises.
export function categories(sport: Sport) {
  const seen: string[] = []
  for (const e of EXERCISES) if (e.sport === sport && !seen.includes(e.category)) seen.push(e.category)
  const order = Object.keys(CATEGORY_LABELS)
  const rank = (c: string) => (c === 'warmup' || c === 'cooldown' ? 100 : 0) + order.indexOf(c) // warm-up/cool-down last
  return seen.sort((a, b) => rank(a) - rank(b))
}

const NEED_LABELS: Record<string, string> = { markers: mark('4-6 markers (bottles/shoes/bags)') }

export function toSession(tpl: Template): Session {
  const blocks: Block[] = []
  for (const it of tpl.items) {
    const e = byId.get(it.ex)
    if (!e) continue // exercise removed from the library: skip rather than break the session
    const { id: _id, sport: _s, positions: _p, ...block } = e
    const { ex: _ex, ...overrides } = it
    blocks.push({ ...block, ...stripUndefined({ ...overrides, how: overrides.how && t(overrides.how) }) })
  }
  const needs = tpl.needs.length ? tpl.needs : [...new Set(blocks.flatMap((b) => b.needs || []))].map((n) => NEED_LABELS[n] || n)
  // your own sessions keep the title you gave them (t() leaves text it doesn't know as it is)
  return { id: tpl.id, title: t(tpl.title), type: tpl.type, summary: t(tpl.summary), needs, blocks }
}

const stripUndefined = (o: object) => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined))

// Session type follows its content: any football drill makes it football, then gym, else recovery.
export function inferType(items: Item[]) {
  const sports = items.map((i) => byId.get(i.ex)?.sport)
  return sports.includes('football') ? 'football' : sports.includes('gym') ? 'gym' : sports.includes('home') ? 'home' : 'recovery'
}

export const loadMySessions = () => load<MySession[]>('mySessions', [])
export function saveMySession(s: MySession) {
  const all = loadMySessions().filter((x) => x.id !== s.id)
  save('mySessions', [{ ...s, updated: new Date().toISOString() }, ...all])
}
export const deleteMySession = (id: string) => save('mySessions', loadMySessions().filter((x) => x.id !== id))

export const isMine = (id: string) => id.startsWith('my-')

// Drill-card photo per category group (licensed Unsplash photos in public/drill-img, credited in Settings).
const PHOTO_GROUP: Record<string, string> = {
  'ball-mastery': 'dribble', 'first-touch': 'dribble', '1v1': 'dribble', passing: 'pass', finishing: 'shoot',
  speed: 'speed', sprints: 'speed', plyometrics: 'speed', warmup: 'warmup', conditioning: 'run', 'gym-conditioning': 'run',
  'lower-strength': 'gym', 'upper-strength': 'gym', olympic: 'gym', explosive: 'gym', 'single-limb': 'gym', isometrics: 'gym', core: 'gym',
  mobility: 'stretch', cooldown: 'stretch',
}
export const drillPhoto = (category?: string) => `${import.meta.env.BASE_URL}drill-img/${PHOTO_GROUP[category ?? ''] ?? 'run'}.jpg`
export const newSessionId = () => `my-${Date.now().toString(36)}`
export const templateFor = (id: string): Template | undefined => TEMPLATES.find((t) => t.id === id) ?? loadMySessions().find((t) => t.id === id)

export const BUILTIN_SESSIONS: Session[] = TEMPLATES.map(toSession)
export const allSessions = (): Session[] => [...loadMySessions().map(toSession), ...BUILTIN_SESSIONS]

// Mid-session swaps: an exercise as a session block, and the others in the same category (the rack's taken, no box...).
export function exerciseBlock(id: string): Block | undefined {
  const e = byId.get(id)
  if (!e) return undefined
  const { id: _id, sport: _s, positions: _p, ...block } = e
  return block
}
export const alternativesFor = (b: Block) =>
  EXERCISES.filter((e) => e.category === b.category && e.name !== b.name && e.kind === b.kind)

// No gym today: the home session closest to a gym session (bodyweight and household kit only).
const HOME_VERSION: Record<string, string> = {
  'gym-lower-power': 'home-legs-power', 'gym-explosive': 'home-legs-power', 'gym-single-leg': 'home-legs-power', 'gym-contrast': 'home-legs-power',
  'gym-upper-core': 'home-upper-core', 'gym-full-light': 'home-full-body',
  'gym-upper-power': 'home-upper-core', 'gym-core-prevention': 'home-upper-core', 'gym-activation': 'home-full-body',
}
export function homeVersion(s: Session): Session | undefined {
  if (s.type !== 'gym') return undefined
  const id = HOME_VERSION[s.id] ?? 'home-full-body'
  return BUILTIN_SESSIONS.find((x) => x.id === id)
}

// Planner day kind for a session: a home workout stands in for a gym day.
export const dayKindOf = (s: Session) => (s.type === 'gym' || s.type === 'home' ? 'gym' : s.type === 'recovery' ? 'recovery' : 'football')
