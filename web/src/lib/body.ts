// Body weight, height, BMI and body fat. Kept on the profile (synced to your account, never shown to friends).
import { getProfile, setProfile, type BodyEntry, type Profile } from './profile'

export const today = () => new Date().toLocaleDateString('en-CA')
export const latestBody = (p: Profile): BodyEntry | null => p.body?.[p.body.length - 1] ?? null
export const bodyWeight = (p: Profile = getProfile()) => latestBody(p)?.kg
// The latest body fat reading, even if newer weigh-ins didn't include one.
export const latestFat = (p: Profile) => [...(p.body ?? [])].reverse().find((b) => b.fat)?.fat

export const bmi = (kg: number, cm: number) => kg / (cm / 100) ** 2
// WHO adult ranges. BMI can't tell muscle from fat, so a strong player can read "above" while lean.
export type BmiBand = 'under' | 'healthy' | 'above' | 'well-above'
export const bmiBand = (v: number): BmiBand => (v < 18.5 ? 'under' : v < 25 ? 'healthy' : v < 30 ? 'above' : 'well-above')

// Change since the first weigh-in at least `days` ago (or the first one at all).
export function weightTrend(p: Profile, days = 28): { kg: number; since: string } | null {
  const log = p.body ?? []
  if (log.length < 2) return null
  const now = log[log.length - 1]
  const cut = new Date(Date.now() - days * 864e5).toLocaleDateString('en-CA')
  const from = [...log].reverse().find((b) => b.date <= cut) ?? log[0]
  if (from === now) return null
  return { kg: Math.round((now.kg - from.kg) * 10) / 10, since: from.date }
}

// A weigh-in: one per day (a second one the same day replaces it). Height lives on the profile.
export function logBody(e: { kg: number; fat?: number; heightCm?: number }, p: Profile = getProfile()) {
  const entry: BodyEntry = { date: today(), kg: Math.round(e.kg * 10) / 10, ...(e.fat ? { fat: Math.round(e.fat * 10) / 10 } : {}) }
  const body = [...(p.body ?? []).filter((b) => b.date !== entry.date), entry].sort((a, b) => a.date.localeCompare(b.date)).slice(-400)
  setProfile({ ...p, body, ...(e.heightCm ? { heightCm: Math.round(e.heightCm) } : {}) })
}
