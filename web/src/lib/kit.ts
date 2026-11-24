// Kit check: what to pack for a session, derived from its "needs". One tap per item, no logging.
import { load, save } from './storage'
import type { Session } from './types'
import { t } from '../i18n'

export type KitItem = { id: string; label: string; photo?: string }

export function kitFor(s: Session): KitItem[] {
  const needs = [...s.needs, ...s.blocks.flatMap((b) => b.needs ?? [])].map((n) => n.toLowerCase())
  const water: KitItem = { id: 'water', label: t('Water'), photo: 'kit-water.jpg' }
  if (s.type === 'recovery') return [{ id: 'mat', label: t('Mat') }, { id: 'roller', label: t('Roller') }, water]
  // the gym: training shoes, not boots; a medicine ball is not a football
  const items: KitItem[] = s.type === 'home' ? [{ id: 'mat', label: t('Mat') }]
    : s.type === 'gym' ? [{ id: 'shoes', label: t('Training shoes'), photo: 'kit-shoes.jpg' }]
    : [{ id: 'boots', label: t('Boots'), photo: 'kit-boots.jpg' }]
  if (needs.some((n) => /\bball\b/.test(n.replace('medicine ball', '')))) items.push({ id: 'ball', label: t('Ball'), photo: 'kit-ball.jpg' })
  if (needs.some((n) => n.includes('medicine ball'))) items.push({ id: 'medball', label: t('Med ball'), photo: 'kit-medball.jpg' })
  if (needs.some((n) => /marker|cone/.test(n))) items.push({ id: 'discs', label: t('Discs'), photo: 'kit-discs.jpg' })
  items.push(water)
  return items.slice(0, 4)
}

const key = (date: string) => `kit:${date}`
export const loadPacked = (date: string) => new Set(load<string[]>(key(date), []))
export function togglePacked(date: string, id: string) {
  const s = loadPacked(date)
  if (s.has(id)) s.delete(id); else s.add(id)
  save(key(date), [...s])
  return s
}
