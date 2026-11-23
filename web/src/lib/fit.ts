// The plan, fitted to the player (onboarding answers, Edit player). The week keeps its structure around the games;
// what changes is which session comes first and which exercises are in it:
//   goals        sessions that train the goal come first in each rotation
//   gym exp/new  no Olympic lifts or advanced jumps yet (a simpler one of the same kind), at most 3 sets
//   level casual at most 3 sets
//   train at     no gym: the home version of a gym session
//   kit          no wall / cones / ball: drills that need it swap for ones that don't (or are left out)
import { t } from '../i18n'
import { EXERCISES, exerciseBlock, exerciseByName, homeVersion } from './library'
import type { Goal, Profile } from './profile'
import type { Block, Session } from './types'

const GOAL_CATS: Record<Goal, string[]> = {
  speed: ['speed', 'sprints', 'plyometrics', 'explosive'],
  strength: ['lower-strength', 'upper-strength', 'olympic', 'single-limb'],
  stamina: ['conditioning', 'gym-conditioning'],
  skills: ['ball-mastery', 'first-touch', 'passing', 'finishing', '1v1'],
  'injury-free': ['core', 'mobility', 'isometrics'],
}
const catOf = (b: Block) => b.category ?? exerciseByName(b.name)?.category ?? ''
// How much of a session trains the player's goals (0..1).
export function goalScore(s: Session, goals: Goal[] = []): number {
  const drills = s.blocks.filter((b) => b.kind === 'drill')
  if (!goals.length || !drills.length) return 0
  const cats = goals.flatMap((g) => GOAL_CATS[g])
  const extra = goals.includes('injury-free') ? /nordic|copenhagen/ : null
  return drills.filter((b) => cats.includes(catOf(b)) || (extra && extra.test(exerciseByName(b.name)?.id ?? ''))).length / drills.length
}
// Sessions that train the goals first; ties keep their order (so the rotation still varies).
export const byGoals = <T extends Session>(pool: T[], goals: Goal[] = []): T[] =>
  goals.length ? pool.map((s, i) => ({ s, i, g: goalScore(s, goals) })).sort((a, b) => b.g - a.g || a.i - b.i).map((x) => x.s) : pool

// Lifts and jumps to leave until the gym is familiar.
const ADVANCED = /clean|snatch|jerk|depth-jump|skater-snap|single-leg-rdl-box-jump|single-leg-rdl-clean/
const KIT_NEEDS: Record<'wall' | 'cones' | 'ball', RegExp> = { wall: /wall/, cones: /marker|cone|hurdle/, ball: /^ball$/ }

export type Fitted = { session: Session; changes: string[] }
export function fitSession(s: Session, p: Pick<Profile, 'gymExp' | 'level' | 'trainAt' | 'kit'>): Fitted {
  const changes: string[] = []
  // no gym: the home version
  if (s.type === 'gym' && p.trainAt?.length && !p.trainAt.includes('gym')) {
    const home = homeVersion(s)
    if (home) { s = home; changes.push(t('No gym: the home version')) }
  }
  const missing = (['wall', 'cones', 'ball'] as const).filter((k) => p.kit && !p.kit.includes(k))
  const needsMissing = (needs?: string[]) => missing.find((k) => (needs ?? []).some((n) => KIT_NEEDS[k].test(n)))
  const newToGym = p.gymExp === 'new'
  const tooHard = (id: string, cat: string) => newToGym && (cat === 'olympic' || ADVANCED.test(id))
  const used = new Set(s.blocks.map((b) => b.name))
  const cap = newToGym || p.level === 'casual' ? 3 : 0
  const blocks: Block[] = []
  let unpairNext = false
  for (const orig of s.blocks) {
    const b = unpairNext && orig.superset ? { ...orig, superset: undefined } : orig
    unpairNext = false
    if (b.kind !== 'drill') { blocks.push(b); continue }
    const me = exerciseByName(b.name)
    const id = me?.id ?? '', cat = catOf(b)
    const lacking = needsMissing(b.needs ?? me?.needs)
    if (tooHard(id, cat) || lacking) {
      // same kind (Olympic lifts: an explosive one), that's fine for them, not in the session already
      const want = cat === 'olympic' ? 'explosive' : cat
      const pick = EXERCISES.find((e) => e.kind === 'drill' && e.category === want && e.sport === (me?.sport ?? e.sport) && !used.has(e.name) && !tooHard(e.id, e.category) && !needsMissing(e.needs))
      const why = lacking ? t('no {kit}', { kit: { wall: t('wall'), cones: t('cones'), ball: t('ball') }[lacking] }) : t('new to the gym')
      if (pick) {
        const blk = exerciseBlock(pick.id)!
        used.add(pick.name)
        blocks.push({ ...blk, rounds: b.rounds ?? blk.rounds, rest: b.rest ?? blk.rest, superset: b.superset })
        changes.push(`${t(b.name)} → ${t(pick.name)} (${why})`)
      } else {
        changes.push(t('{name}: left out ({why})', { name: t(b.name), why }))
        if (!b.superset) unpairNext = true
      }
      continue
    }
    blocks.push(b)
  }
  const out = blocks.map((b, k) => {
    let x = b
    if (cap && x.kind === 'drill' && (x.rounds ?? 1) > cap) x = { ...x, rounds: cap }
    if (x.superset && (k === 0 || blocks[k - 1].kind !== 'drill')) x = { ...x, superset: undefined }
    return x
  })
  if (cap && out.some((b, k) => b.rounds !== blocks[k].rounds)) changes.push(newToGym ? t('New to the gym: 3 sets at most') : t('3 sets at most'))
  return { session: { ...s, blocks: out }, changes }
}
