// The player profile: everything that used to be hard-coded for Peter. Saved locally (synced once accounts exist).
import { useSyncExternalStore } from 'react'
import { load, save } from './storage'

export type Position = 'winger' | 'striker' | 'midfielder' | 'defender'
export type Foot = 'right' | 'left' | 'both'
export type Accent = 'red' | 'blue' | 'green' | 'gold' | 'purple' | 'orange' | 'teal' | 'pink' | 'sky'

export type Profile = {
  name: string
  number: number
  positions: Position[]
  foot: Foot
  accent: Accent
  idol: string // role model's name, shown on quotes; empty = no attribution
  sample?: boolean // the demo's sample player: history is reseeded daily
}

export const POSITIONS: { id: Position; label: string; short: string }[] = [
  { id: 'winger', label: 'Winger', short: 'W' }, { id: 'striker', label: 'Striker', short: 'ST' },
  { id: 'midfielder', label: 'Midfielder', short: 'CM' }, { id: 'defender', label: 'Defender', short: 'DEF' },
]
// Swatch colours match the dark-mode fills in index.css.
export const ACCENTS: { id: Accent; label: string; hex: string }[] = [
  { id: 'red', label: 'Red', hex: '#e01b35' }, { id: 'blue', label: 'Blue', hex: '#1f5fe0' }, { id: 'sky', label: 'Sky', hex: '#1c9be6' },
  { id: 'green', label: 'Green', hex: '#12854a' }, { id: 'teal', label: 'Teal', hex: '#0b7470' }, { id: 'gold', label: 'Gold', hex: '#f5b70a' },
  { id: 'orange', label: 'Orange', hex: '#ff6a13' }, { id: 'purple', label: 'Purple', hex: '#7c3aed' }, { id: 'pink', label: 'Pink', hex: '#d61f80' },
]

export const SAMPLE_PROFILE: Profile = { name: 'Sam', number: 11, positions: ['winger', 'striker'], foot: 'right', accent: 'red', idol: '', sample: true }
const FALLBACK: Profile = { name: 'Player', number: 10, positions: ['winger'], foot: 'right', accent: 'red', idol: '' }

let current = load<Profile | null>('profile', null)
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
  current = load<Profile | null>('profile', null)
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
export const positionTag = (p: Profile) => p.positions.map((x) => POSITIONS.find((y) => y.id === x)?.short ?? x).join(' · ')
