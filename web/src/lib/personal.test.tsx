import { act, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { usePersonalImage } from './personal'

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

function Pic({ name }: { name: string }) {
  const src = usePersonalImage(name)
  return <span data-testid="src">{src ?? 'none'}</span>
}

test('a photo that fails to load (bad signal) is retried and then shown', async () => {
  vi.useFakeTimers()
  let calls = 0
  vi.stubGlobal('Image', class {
    onload: (() => void) | null = null
    onerror: (() => void) | null = null
    set src(_v: string) { calls++; const fail = calls === 1; setTimeout(() => (fail ? this.onerror?.() : this.onload?.()), 0) }
  })
  render(<Pic name="retry-test.jpg" />)
  await act(async () => { await vi.advanceTimersByTimeAsync(10) })
  expect(screen.getByTestId('src').textContent).toBe('none') // first try failed
  await act(async () => { await vi.advanceTimersByTimeAsync(1600) })
  expect(screen.getByTestId('src').textContent).toMatch(/retry-test\.jpg$/)
  expect(calls).toBe(2)
})
