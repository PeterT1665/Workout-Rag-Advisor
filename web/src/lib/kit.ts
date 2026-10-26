// Kit check: what to pack for a session, derived from its "needs". One tap per item, no logging.
import { load, save } from './storage'
import type { Session } from './types'

export type KitItem = { id: string; label: string; photo?: string }

export function kitFor(s: Session): KitItem[] {
  const needs = [...s.needs, ...s.blocks.flatMap((b) => b.needs ?? [])].join(' ').toLowerCase()
  if (s.type === 'recovery') return [{ id: 'mat', label: 'Mat' }, { id: 'roller', label: 'Roller' }, { id: 'water', label: 'Water', photo: 'kit-water.jpg' }]
  const items: KitItem[] = [{ id: 'boots', label: 'Boots', photo: 'kit-boots.jpg' }]
  if (needs.includes('ball')) items.push({ id: 'ball', label: 'Ball', photo: 'kit-ball.jpg' })
  if (/marker|cone/.test(needs)) items.push({ id: 'discs', label: 'Discs', photo: 'kit-discs.jpg' })
  items.push({ id: 'water', label: 'Water', photo: 'kit-water.jpg' })
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
