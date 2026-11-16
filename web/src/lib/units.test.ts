import { describe, expect, test } from 'vitest'
import { moveUnit, unitsOf } from './units'

const x = (id: string, superset?: boolean) => ({ id, superset })
describe('drag units', () => {
  test('a superset is one unit', () => {
    expect(unitsOf([x('a'), x('b'), x('c', true), x('d')]).map((u) => u.items.map((i) => i.id).join(''))).toEqual(['a', 'bc', 'd'])
  })
  test('moving a superset moves both exercises, in order', () => {
    const items = [x('a'), x('b'), x('c', true), x('d')]
    expect(moveUnit(items, 1, 0).map((i) => i.id).join('')).toBe('bcad')
    expect(moveUnit(items, 1, 2).map((i) => i.id).join('')).toBe('adbc')
    expect(moveUnit(items, 0, 2).map((i) => i.id).join('')).toBe('bcda')
  })
})
