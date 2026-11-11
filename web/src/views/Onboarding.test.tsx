import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { Onboarding } from './Onboarding'

afterEach(cleanup)
const click = (name: RegExp) => fireEvent.click(screen.getByRole('button', { name }))

test('welcome: sample player shortcut', () => {
  const onSample = vi.fn()
  render(<Onboarding welcome onSample={onSample} onDone={vi.fn()} />)
  click(/sample player/)
  expect(onSample).toHaveBeenCalled()
})

test('make my player: name and a position are required, several positions allowed', () => {
  const onDone = vi.fn()
  render(<Onboarding welcome onDone={onDone} />)
  click(/Make my player/)
  expect((screen.getByRole('button', { name: /^Next$/ }) as HTMLButtonElement).disabled).toBe(true)
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: '  Alex ' } })
  fireEvent.change(screen.getByLabelText('Number'), { target: { value: '9' } })
  click(/^Next$/)
  expect((screen.getByRole('button', { name: /^Next$/ }) as HTMLButtonElement).disabled).toBe(true)
  click(/Striker/)
  click(/Midfielder/)
  click(/^Next$/) // foot: default right
  click(/^Wednesday$/)
  click(/^Sunday$/)
  click(/^Next$/) // colour red and role model optional
  click(/^Done/)
  expect(onDone).toHaveBeenCalledWith(
    expect.objectContaining({ name: 'Alex', number: 9, positions: ['striker', 'midfielder'], foot: 'right', accent: 'red', idol: '', sample: false }),
    [{ day: 2, kind: 'competitive' }, { day: 6, kind: 'competitive' }],
  )
})
