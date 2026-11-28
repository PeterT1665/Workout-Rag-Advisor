// Voice cues: one engine, many drills. A drill's `cue` picks a word set; the player calls a random word from it every few seconds
// (never the same one twice in a row) and shows it full screen. LEFT/RIGHT (the 1v1 cue, `cue: true`) and the numbers drill
// (`cue: 'numbers'`) are older and handled in lib/player.ts; everything else is here.
import { t } from '../i18n'

export type CueKind = 'numbers' | 'touches' | 'scan' | 'target' | 'colours' | 'corner' | 'run' | 'turn' | 'go' | 'move'
export type CueWord = { id: string; word: string; hint?: string }
type Rng = () => number

// The words each cue can call (display text; the voice says the same word).
const SETS: Record<Exclude<CueKind, 'numbers' | 'move'>, () => CueWord[]> = {
  touches: () => [{ id: 'one', word: t('One'), hint: t('One touch') }, { id: 'two', word: t('Two'), hint: t('Two touches') }],
  scan: () => [{ id: 'space', word: t('Space behind'), hint: t('Dummy or half turn') }, { id: 'manon', word: t('Man on'), hint: t('Play it back, or turn the other way') }],
  target: () => [{ id: 't-left', word: t('Left'), hint: t('Pass to the left target') }, { id: 't-middle', word: t('Middle'), hint: t('Pass to the middle target') }, { id: 't-right', word: t('Right'), hint: t('Pass to the right target') }],
  colours: () => [{ id: 'red', word: t('Red'), hint: t('Touch out towards the red cone') }, { id: 'blue', word: t('Blue'), hint: t('Touch out towards the blue cone') }, { id: 'yellow', word: t('Yellow'), hint: t('Touch out towards the yellow cone') }, { id: 'white', word: t('White'), hint: t('Touch out towards the white cone') }],
  corner: () => [{ id: 'c-left', word: t('Left corner'), hint: t('Pick it late, strike low') }, { id: 'c-right', word: t('Right corner'), hint: t('Pick it late, strike low') }, { id: 'c-low', word: t('Low'), hint: t('Keep it down') }, { id: 'c-high', word: t('High'), hint: t('Over the keeper') }],
  run: () => [{ id: 'near', word: t('Near'), hint: t('Near-post run') }, { id: 'far', word: t('Far'), hint: t('Far-post run') }],
  turn: () => [{ id: 'turn', word: t('Turn'), hint: t('Turn and sprint back') }],
  go: () => [{ id: 'go', word: t('Go'), hint: t("React, don't anticipate") }],
}

export const isWordCue = (k: unknown): k is Exclude<CueKind, 'numbers'> => typeof k === 'string' && k !== 'numbers'

// A skill move from the player's signature moves (or three well-known ones before they pick any).
export const DEFAULT_MOVES = ['Step-over', 'Croqueta', 'Cruyff turn']

export function cueWords(kind: Exclude<CueKind, 'numbers'>, moves?: string[]): CueWord[] {
  if (kind === 'move') return (moves?.length ? moves : DEFAULT_MOVES).map((m) => ({ id: `m-${m}`, word: m, hint: t('Do this move at the cone') }))
  return SETS[kind]()
}

export function pickWord(kind: Exclude<CueKind, 'numbers'>, lastId: string | null, rng: Rng = Math.random, moves?: string[]): CueWord {
  const all = cueWords(kind, moves)
  const pool = all.length > 1 ? all.filter((w) => w.id !== lastId) : all
  return pool[Math.floor(rng() * pool.length)]
}

// The word behind a cue id, for the full-screen view (ids are what the player state keeps).
export function wordFor(id: string, kind: Exclude<CueKind, 'numbers'>, moves?: string[]): CueWord | undefined {
  return cueWords(kind, moves).find((w) => w.id === id)
}
