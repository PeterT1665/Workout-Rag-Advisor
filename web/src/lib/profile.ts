// The player profile: everything that used to be hard-coded for Peter. Saved locally (synced once accounts exist).
import { t } from '../i18n'
import { useSyncExternalStore } from 'react'
import { load, save } from './storage'
import type { Ratings } from './testday'

// Eight roles, picked on a pitch. 'midfielder' and 'defender' are the old broad ones (saved profiles, match reviews):
// read as CM and CB.
export type Position = 'gk' | 'cb' | 'fb' | 'dm' | 'cm' | 'am' | 'winger' | 'striker' | 'midfielder' | 'defender'
export type Family = 'goalkeeper' | 'defender' | 'midfielder' | 'winger' | 'striker' // what the library and match rating group by
export const familyOf = (p: Position): Family => (p === 'gk' ? 'goalkeeper' : p === 'cb' || p === 'fb' || p === 'defender' ? 'defender' : p === 'dm' || p === 'cm' || p === 'am' || p === 'midfielder' ? 'midfielder' : p)
export type Level = 'casual' | 'amateur' | 'semi' | 'academy'
export type Goal = 'speed' | 'strength' | 'stamina' | 'skills' | 'injury-free'
export type GymExp = 'new' | 'some' | 'experienced'
export type TrainAt = 'gym' | 'home' | 'pitch'
export type KitItem = 'ball' | 'wall' | 'cones'
export type Foot = 'right' | 'left' | 'both'
export type Accent = 'red' | 'blue' | 'green' | 'gold' | 'purple' | 'orange' | 'teal' | 'pink' | 'sky'

export type Profile = {
  name: string
  number: number
  positions: Position[]
  foot: Foot
  accent: Accent
  idol: string // role model (profile info; never shown under quotes: we can't verify who said what)
  gym?: number // gym sessions a week (default 3)
  sample?: boolean // the demo's sample player: history is reseeded daily
  ratings?: Ratings // latest rating session (test day)
  goal?: number // the overall rating the player is working toward (default 75: gold)
  baseRatings?: Ratings // the first rating session: the starting point the progress arrows measure from
  prevRatings?: Ratings // old profiles only: the session before the last (read as the starting point; no longer written)
  heightCm?: number
  body?: BodyEntry[] // body weight (and body fat, if known) over time, oldest first; private, never on Team
  injuries?: Injury[] // told to the coach; private, never on Team (lib/injuries)
  // from onboarding; private, never on Team. They shape which sessions and exercises come first (lib/fit)
  level?: Level
  goals?: Goal[] // up to 2
  gymExp?: GymExp
  trainAt?: TrainAt[]
  kit?: KitItem[]
}
export type BodyEntry = { date: string; kg: number; fat?: number }
// An injury in the app's own words (never the player's message): where, what tissue, what happened, how bad.
export type InjuryPart = 'hamstrings' | 'quads' | 'groin' | 'hips' | 'glutes' | 'knees' | 'calves' | 'ankles' | 'back' | 'core' | 'chest' | 'shoulder' | 'arms' | 'other'
export type Injury = {
  id: string; part: InjuryPart
  tissue: 'muscle' | 'joint' | 'ligament' | 'tendon' | 'bone'
  kind: 'strain' | 'tear' | 'sprain' | 'tendon' | 'bruise' | 'fracture' | 'pain'
  severity?: 'light' | 'moderate' | 'serious' // asked by the coach; unknown counts as moderate
  since: string; healed?: string
}

export const POSITIONS: { id: Position; label: string; short: string }[] = [
  { id: 'gk', label: t('Goalkeeper'), short: 'GK' }, { id: 'cb', label: t('Centre-back'), short: 'CB' }, { id: 'fb', label: t('Full-back'), short: 'FB' },
  { id: 'dm', label: t('Defensive midfielder'), short: 'DM' }, { id: 'cm', label: t('Central midfielder'), short: 'CM' }, { id: 'am', label: t('Attacking midfielder'), short: 'AM' },
  { id: 'winger', label: t('Winger'), short: 'W' }, { id: 'striker', label: t('Striker'), short: 'ST' },
]
// old broad positions, as the roles they now are
const LEGACY: Partial<Record<Position, Position>> = { midfielder: 'cm', defender: 'cb' }
export const roleOf = (p: Position): Position => LEGACY[p] ?? p
const normalise = (p: Profile | null): Profile | null => (p && p.positions.some((x) => LEGACY[x]) ? { ...p, positions: [...new Set(p.positions.map(roleOf))] } : p)
// Swatch colours match the dark-mode fills in index.css.
export const ACCENTS: { id: Accent; label: string; hex: string }[] = [
  { id: 'red', label: t('Red'), hex: '#e01b35' }, { id: 'blue', label: t('Blue'), hex: '#1f5fe0' }, { id: 'sky', label: t('Sky'), hex: '#1c9be6' },
  { id: 'green', label: t('Green'), hex: '#12854a' }, { id: 'teal', label: t('Teal'), hex: '#0b7470' }, { id: 'gold', label: t('Gold'), hex: '#f5b70a' },
  { id: 'orange', label: t('Orange'), hex: '#ff6a13' }, { id: 'purple', label: t('Purple'), hex: '#7c3aed' }, { id: 'pink', label: t('Pink'), hex: '#d61f80' },
]

export const SAMPLE_PROFILE: Profile = { name: 'Sam', number: 11, positions: ['winger', 'striker'], foot: 'right', accent: 'red', idol: '', sample: true }
const FALLBACK: Profile = { name: 'Player', number: 10, positions: ['winger'], foot: 'right', accent: 'red', idol: '' }

let current = normalise(load<Profile | null>('profile', null))
const listeners = new Set<() => void>()

export const hasProfile = () => current !== null
export const getProfile = (): Profile => current ?? FALLBACK
export function setProfile(p: Profile) {
  current = p
  save('profile', p)
  applyAccent(p.accent)
  listeners.forEach((l) => l())
}
// Re-read after a sync pulled a newer profile from another device.
export function reloadProfile() {
  current = normalise(load<Profile | null>('profile', null))
  applyAccent(getProfile().accent)
  listeners.forEach((l) => l())
}
const subscribe = (l: () => void) => { listeners.add(l); return () => listeners.delete(l) }
export const useProfile = () => useSyncExternalStore(subscribe, getProfile)

export function applyAccent(a: Accent) {
  if (typeof document === 'undefined') return
  if (a === 'red') delete document.documentElement.dataset.accent
  else document.documentElement.dataset.accent = a
}

export const weakFoot = (p: Profile) => (p.foot === 'right' ? 'left' : p.foot === 'left' ? 'right' : null)
export const positionTag = (p: Profile) => p.positions.map((x) => POSITIONS.find((y) => y.id === roleOf(x))?.short ?? x).join(' · ')
