// Drill families (data/drills.json, from the football drills research): one card per family with levels. Each level is a normal
// library exercise (id `${family}-${level}`, name "Family: Level") so the planner, the builder and the player work unchanged;
// the family card groups them on the Exercises page. English here: library.ts translates the text.
import data from '../data/drills.json'
import type { CueKind } from './cues'

export type DrillLevel = { name: string; how: string; mistake: string; fix: string; work: number; rest: number; rounds: number; reps?: string; cue?: true | CueKind; headers?: number }
export type DrillFamily = {
  id: string; name: string; category: string; demo: string; setup: 'solo' | 'partner' | 'group'; equipment: string[]
  gameDay: ('md-1' | 'recovery' | 'full')[]; levels: DrillLevel[]; positions?: string[]; players?: string; source?: { label: string }; note?: string; heading?: boolean
}
export const FAMILIES = (data as { families: DrillFamily[] }).families

// What a drill mainly works, by category (the muscle map and the injury filter read this).
const WORKS: Record<string, [string[], string[]]> = {
  'first-touch': [['groin', 'hips'], ['ankles', 'knees', 'quads']], passing: [['groin', 'hips'], ['ankles', 'knees', 'quads']],
  crossing: [['hips', 'quads'], ['groin', 'ankles', 'knees', 'hamstrings']], 'ball-mastery': [['ankles', 'calves'], ['groin', 'hips', 'knees']],
  finishing: [['quads', 'hips'], ['groin', 'hamstrings', 'knees', 'ankles']], '1v1': [['quads', 'calves'], ['ankles', 'knees', 'hips', 'groin']],
  speed: [['quads', 'hamstrings', 'calves'], ['ankles', 'knees', 'hips', 'glutes']], defending: [['quads', 'hips'], ['hamstrings', 'groin', 'ankles', 'knees']],
  warmup: [['hips', 'quads'], ['hamstrings', 'groin', 'core']], 'off-ball': [['quads', 'hamstrings'], ['calves', 'hips', 'glutes', 'knees', 'ankles']],
  conditioning: [['quads', 'calves'], ['hamstrings', 'hips', 'knees', 'ankles']],
}

export const drillId = (f: DrillFamily, i: number) => `${f.id}-${i + 1}`

export function drillExercises() {
  return FAMILIES.flatMap((f) => f.levels.map((l, i) => {
    const [works, also] = WORKS[f.category] ?? WORKS['first-touch']
    return {
      id: drillId(f, i), name: `${f.name}: ${l.name}`, sport: 'football' as const, category: f.category, kind: f.category === 'warmup' ? 'warmup' : 'drill',
      work: l.work, rest: l.rest || undefined, rounds: l.rounds, reps: l.reps, how: l.how, mistake: l.mistake, fix: l.fix,
      needs: f.equipment, works, also, positions: f.positions, family: f.id, level: i + 1, setup: f.setup, gameDay: f.gameDay,
      cue: l.cue, headers: l.headers, heading: f.heading || undefined,
    }
  }))
}
