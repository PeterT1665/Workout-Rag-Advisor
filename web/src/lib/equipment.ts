// Barbell or dumbbells for today's gym session. No barbell free (or no landmine corner)? Every barbell and landmine lift has a dumbbell version.
// Chosen on the kick-off screen; the choice is remembered for the next session.
import { EXERCISES, exerciseBlock } from './library'
import { load, save } from './storage'
import type { Session } from './types'

export type Kit = 'barbell' | 'dumbbells'
export const loadKit = () => load<Kit>('equipment', 'barbell')
export const saveKit = (k: Kit) => save('equipment', k)

// barbell / landmine exercise -> its dumbbell version
export const DB_SWAP: Record<string, string> = {
  'back-squat': 'db-goblet-squat', 'front-squat': 'db-goblet-squat', 'half-squat': 'db-goblet-squat', 'quarter-squat': 'db-goblet-squat', 'landmine-squat': 'db-goblet-squat',
  'split-stance-quarter-squat': 'db-split-squat', 'landmine-diagonal-lunge': 'db-curtsy-lunge',
  deadlift: 'db-deadlift', 'trap-bar-deadlift': 'db-deadlift', 'romanian-deadlift': 'db-rdl', 'landmine-single-leg-rdl': 'single-leg-rdl',
  'hip-thrust': 'db-hip-thrust', 'landmine-lateral-lunge': 'home-cossack',
  'bench-press': 'db-bench-press', 'speed-bench': 'db-bench-press', 'bent-over-row': 'db-row', 'landmine-row': 'db-row', 'landmine-single-arm-row': 'single-arm-row', 'landmine-half-kneeling-press': 'db-overhead-press', 'barbell-overhead-press': 'db-overhead-press',
  'power-clean': 'db-hang-clean', 'hang-clean': 'db-hang-clean', 'hang-snatch': 'db-snatch',
  'push-jerk': 'db-push-press', 'split-jerk': 'db-push-press', 'landmine-push-press': 'db-push-press', 'landmine-squat-press': 'db-push-press', 'trap-bar-jump': 'db-jump-squat',
}

const idByName = new Map(EXERCISES.map((e) => [e.name.toLowerCase(), e.id]))
const idOf = (name: string) => idByName.get(name.replace(/\s*\([^)]*\)$/, '').trim().toLowerCase()) ?? idByName.get(name.trim().toLowerCase())

export const dumbbellFor = (name: string) => { const id = idOf(name); return id ? DB_SWAP[id] : undefined }
export const hasBarbellWork = (s: Session) => s.blocks.some((b) => !!dumbbellFor(b.name))

// The same session with dumbbells: swapped exercises take the dumbbell version's name, how-to, reps and set time,
// and keep the session's own sets, rest and superset pairing.
export function withDumbbells(s: Session): Session {
  if (!hasBarbellWork(s)) return s
  return {
    ...s,
    blocks: s.blocks.map((b) => {
      const alt = dumbbellFor(b.name)
      const blk = alt && exerciseBlock(alt)
      if (!blk) return b
      return { ...blk, rounds: b.rounds ?? blk.rounds, rest: b.rest ?? blk.rest, superset: b.superset }
    }),
  }
}
