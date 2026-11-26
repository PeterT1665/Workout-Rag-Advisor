import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { Onboarding, buildingLines } from './Onboarding'

beforeEach(() => localStorage.clear())
afterEach(() => { cleanup(); vi.useRealTimers() })
const click = (name: RegExp | string) => fireEvent.click(screen.getAllByRole('button', { name })[0])

test('welcome: sample player shortcut', () => {
  const onSample = vi.fn()
  render(<Onboarding welcome onSample={onSample} onDone={vi.fn()} />)
  click(/sample player/)
  expect(onSample).toHaveBeenCalled()
})

test('new player: tour, one question per screen, building your week, the plan', () => {
  vi.useFakeTimers()
  const onDone = vi.fn()
  render(<Onboarding welcome onDone={onDone} />)
  click(/Get started/)
  click(/^Skip$/) // the tour
  expect((screen.getByRole('button', { name: /^Next$/ }) as HTMLButtonElement).disabled).toBe(true)
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: '  Alex ' } })
  fireEvent.change(screen.getByLabelText('Number'), { target: { value: '9' } })
  click(/^Next$/)
  click('Striker'); click('Central midfielder')
  expect(screen.getByText(/Central midfielder\. The engine/)).toBeTruthy() // the role card
  click(/^Next$/) // to foot
  click(/^Next$/) // foot: right
  click(/^Next$/) // level: optional
  click(/Faster/); click(/Stronger/); click(/Stay injury-free/) // up to 2: the oldest goes
  click(/^Next$/)
  expect(screen.getByText('Your card goal')).toBeTruthy()
  expect(screen.getByRole('button', { name: /Gold · 75/ }).getAttribute('aria-pressed')).toBe('true') // everyone works toward gold
  click(/Top · 85/)
  click(/^Next$/)
  click(/^Wednesday$/); click(/^Sunday$/); click(/^Sunday/) // second tap: casual
  click(/^Next$/)
  click(/Off season/) // season
  click(/^Next$/)
  click(/^No/) // football and gym on the same day?
  click(/^Next$/)
  click(/^New/) // gym experience
  click(/^Next$/)
  click(/^Next$/) // where you train: defaults
  click('hamstring'); fireEvent.click(screen.getByRole('radio', { name: 'Limits training' }))
  click(/Build my week/)
  expect(screen.getByText(/Building/)).toBeTruthy()
  act(() => { vi.advanceTimersByTime(6000) })
  click(/Let's train/)
  expect(onDone).toHaveBeenCalledWith(
    expect.objectContaining({ name: 'Alex', number: 9, positions: ['striker', 'cm'], foot: 'right', goals: ['strength', 'injury-free'], goal: 85, season: 'off', doubleDays: false, gymExp: 'new', sample: false,
      injuries: [expect.objectContaining({ part: 'hamstrings', severity: 'moderate', tissue: 'muscle' })] }),
    [{ day: 2, kind: 'competitive' }, { day: 6, kind: 'casual' }],
  )
  expect(localStorage.getItem('onboarding:draft')).toBeNull()
})

test('answers are kept after every step: a closed app picks up where it stopped', () => {
  const { unmount } = render(<Onboarding welcome onDone={vi.fn()} />)
  click(/Get started/); click(/^Skip$/)
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Sam' } })
  click(/^Next$/)
  unmount()
  render(<Onboarding welcome onDone={vi.fn()} />)
  expect(screen.getByText(/Where do you play/)).toBeTruthy()
  expect(screen.getByText('Sam')).toBeTruthy() // the card preview
})

test('edit player: no tour or injuries, colours at the end', () => {
  const onDone = vi.fn()
  render(<Onboarding initial={{ name: 'Peter', number: 7, positions: ['defender'], foot: 'left', accent: 'red', idol: '' }} onDone={onDone} onCancel={vi.fn()} />)
  for (let i = 0; i < 11; i++) click(/^Next$/)
  expect(screen.getByText('Your colours')).toBeTruthy()
  click(/^Done/)
  expect(onDone.mock.calls[0][0]).not.toHaveProperty('injuries')
})

test('building your week explains the answers', () => {
  const lines = buildingLines({ name: 'A', number: 1, positions: ['winger'], foot: 'right', accent: 'red', idol: '', goals: ['speed'], gymExp: 'new', trainAt: ['home'],
    injuries: [{ id: 'x', part: 'knees', tissue: 'joint', kind: 'pain', severity: 'serious', since: '2026-11-23' }] }, [{ day: 2, kind: 'competitive' }])
  expect(lines.join(' ')).toMatch(/Game Wednesday.*Faster.*New to the gym.*No gym.*Knee: no sessions/)
})
