import { expect, test } from 'vitest'
import { BUILTIN_SESSIONS } from './library'
import { byGoals, fitSession, goalScore } from './fit'

const S = (id: string) => BUILTIN_SESSIONS.find((s) => s.id === id)!
const names = (s: { blocks: { name: string }[] }) => s.blocks.map((b) => b.name)

test('goals put the sessions that train them first', () => {
  const gym = ['gym-lower-power', 'gym-explosive', 'gym-single-leg'].map(S)
  expect(goalScore(S('gym-explosive'), ['speed'])).toBeGreaterThan(goalScore(S('gym-single-leg'), ['speed']))
  expect(byGoals(gym, ['speed'])[0].id).toBe('gym-explosive')
  expect(byGoals(gym, [])).toEqual(gym) // no goals: the rotation as it was
})
test('new to the gym: no Olympic lifts or advanced jumps, 3 sets at most', () => {
  const { session, changes } = fitSession(S('gym-explosive'), { gymExp: 'new' })
  expect(session.blocks.filter((b) => b.category === 'olympic')).toEqual([])
  expect(names(session)).not.toContain('Depth jump')
  expect(Math.max(...session.blocks.filter((b) => b.kind === 'drill').map((b) => b.rounds ?? 1))).toBeLessThanOrEqual(3)
  expect(changes.length).toBeGreaterThan(0)
  expect(fitSession(S('gym-explosive'), { gymExp: 'experienced' }).changes).toEqual([])
})
test('no gym: the home version; no wall: wall drills swap', () => {
  expect(fitSession(S('gym-lower-power'), { trainAt: ['home', 'pitch'] }).session.id).toBe('home-legs-power')
  const s = fitSession(S('fb-wall-weakfoot-shooting'), { kit: ['ball', 'cones'] }).session
  expect(s.blocks.filter((b) => (b.needs ?? []).some((n) => /wall/.test(n)))).toEqual([])
})
