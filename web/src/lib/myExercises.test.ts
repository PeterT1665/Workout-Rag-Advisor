import { beforeEach, expect, test } from 'vitest'
import { addMyExercise, loadMyExercises, registerMine } from './myExercises'
import { EXERCISES, exerciseByName, toSession } from './library'
import { loadsPart } from './injuries'
import { findExercise } from './coach/coach'

beforeEach(() => { localStorage.clear(); registerMine() })

test('your own exercise joins the library and can go in a session', () => {
  const m = addMyExercise({ name: 'Deficit lateral lunge', parts: ['groin', 'quads'], how: 'Standing foot on a plate, step out to the side.', sport: 'gym' })
  expect(loadMyExercises()).toHaveLength(1)
  expect(exerciseByName('Deficit lateral lunge')?.mine).toBe(true)
  const s = toSession({ id: 'my-x', title: 'Mine', type: 'gym', summary: '', needs: [], items: [{ ex: m.id }] })
  expect(s.blocks[0].name).toBe('Deficit lateral lunge')
  expect(loadsPart(s.blocks[0], 'groin')).toBe(true) // the injury filter uses the parts you picked
  expect(loadsPart(s.blocks[0], 'shoulder')).toBe(false)
  expect(findExercise('how do I do the deficit lateral lunge')?.id).toBe(m.id)
})
test('a name the library already has uses the library one', () => {
  const before = EXERCISES.length
  addMyExercise({ name: 'Back squat', parts: ['quads'], how: '', sport: 'gym' })
  expect(EXERCISES.length).toBe(before)
  expect(exerciseByName('Back squat')?.mine).toBeUndefined()
})
