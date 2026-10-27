import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import App from './App'

const spoken: string[] = []
const played: string[] = []
let lastAudio: { onended: (() => void) | null } | null = null

beforeEach(() => {
  spoken.length = 0
  played.length = 0
  localStorage.clear()
  vi.stubGlobal('speechSynthesis', { speak: (u: { text: string }) => spoken.push(u.text), cancel() {}, resume() {} })
  vi.stubGlobal('SpeechSynthesisUtterance', function (this: { text: string }, t: string) { this.text = t })
  vi.stubGlobal('confirm', () => true)
  URL.createObjectURL = (b: Blob) => 'blob:' + ((b as File).name ?? 'x')
  URL.revokeObjectURL = () => {}
  window.HTMLMediaElement.prototype.play = function (this: HTMLMediaElement) {
    played.push(this.src)
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    lastAudio = this as unknown as { onended: (() => void) | null }
    return Promise.resolve()
  }
  window.HTMLMediaElement.prototype.pause = () => {}
})
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals() })

const click = (name: RegExp) => fireEvent.click(screen.getByRole('button', { name }))
const stepName = () => screen.getByTestId('step-name').textContent ?? ''
const cues = () => spoken.filter((s) => s === 'left' || s === 'right')

describe('app', () => {
  test('session flow: card screen, steps, cues only in 1v1, finish and history', async () => {
    render(<App />)
    click(/All sessions/)
    click(/Close control & 1v1/)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toMatch(/Close control/)
    vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'clearTimeout', 'clearInterval', 'Date'] })
    click(/Start session/)
    expect(screen.getByText(/^off$/i)).toBeTruthy() // kick-off intro first
    act(() => { vi.advanceTimersByTime(1200) })
    expect(screen.getByRole('blockquote') ?? null).toBeTruthy() // then the motivational card
    click(/Let's go/)
    expect(stepName()).toMatch(/Jog/)
    expect(spoken[0]).toMatch(/Jog/)

    // skip to the 1v1 cue drill
    let guard = 0
    while (!/1v1 reaction/.test(stepName()) && guard++ < 100) click(/^Skip$/)
    expect(cues()).toHaveLength(0)
    act(() => { vi.advanceTimersByTime(38000) })
    expect(cues().length).toBeGreaterThanOrEqual(3)

    // pause freezes the clock
    click(/^Pause$/)
    const frozen = screen.getByText(/left$/).textContent
    act(() => { vi.advanceTimersByTime(5000) })
    expect(screen.getByText(/left$/).textContent).toBe(frozen)
    click(/^Resume$/)

    // skip to the end
    guard = 0
    while (!screen.queryByText(/Session/i, { selector: 'p' }) && guard++ < 200) {
      const skip = screen.queryByRole('button', { name: /^Skip$/ })
      if (!skip) break
      fireEvent.click(skip)
    }
    expect(screen.getByRole('heading', { name: /full\s*time/i })).toBeTruthy()
    // 20-second check-in: effort 7, legs OK, sore hips -> load saved
    fireEvent.click(screen.getByRole('button', { name: 'Effort 7' }))
    fireEvent.click(screen.getByRole('button', { name: 'OK' }))
    fireEvent.click(screen.getByRole('button', { name: 'Hips' }))
    fireEvent.click(screen.getByRole('button', { name: /Save check-in/ }))
    const saved = JSON.parse(localStorage.getItem('history')!).at(-1)
    expect(saved.rpe).toBe(7)
    expect(saved.legs).toBe('ok')
    expect(saved.sore).toEqual(['Hips'])
    expect(saved.load).toBe(saved.minutes * 7)
    expect(screen.getByText(/Session load/)).toBeTruthy()
    expect(spoken.at(-1)).toBe('Session complete. Nice work.')
    const hist = JSON.parse(localStorage.getItem('history')!)
    expect(hist[0].id).toBe('fb-close-control-1v1')
  })

  test('clips: upload, mindset clip before the session, SIUUU at the end', async () => {
    render(<App />)
    click(/^Settings$/)
    click(/Clips & quotes \(/)
    fireEvent.change(screen.getByLabelText(/Type/), { target: { value: 'mindset' } })
    fireEvent.change(screen.getByLabelText(/Audio files/), { target: { files: [new File(['x'], 'cr7-interview.mp3')] } })
    await waitFor(() => expect(screen.getByText('cr7-interview')).toBeTruthy())
    fireEvent.change(screen.getByLabelText(/Type/), { target: { value: 'siuu' } })
    fireEvent.change(screen.getByLabelText(/Audio files/), { target: { files: [new File(['x'], 'siuu.m4a')] } })
    await waitFor(() => expect(screen.getByText('siuu')).toBeTruthy())
    click(/^Done$/)
    expect(screen.getByRole('button', { name: /Clips & quotes \(2 clips\)/ })).toBeTruthy()
    click(/^Done$/)

    click(/All sessions/)
    click(/Wall passing/)
    click(/Start session/)
    expect(await screen.findByText(/Playing: cr7-interview/, {}, { timeout: 3000 })).toBeTruthy()
    expect(played).toHaveLength(1) // the mindset clip (name checked above; the test DB drops File names)
    act(() => { lastAudio?.onended?.() }) // clip ends -> session starts
    expect(stepName()).toMatch(/Jog/)

    let guard = 0
    while (screen.queryByRole('button', { name: /^Skip$/ }) && guard++ < 200) click(/^Skip$/)
    expect(screen.getByRole('heading', { name: /siuuu/i })).toBeTruthy()
    expect(played).toHaveLength(2) // + SIUUU (no rest clips: they need real time to pass)
    expect(spoken).not.toContain('Session complete. Nice work.')
  })

  test('home: kit check toggles and remembers packed items', () => {
    render(<App />)
    const boots = screen.getByRole('button', { name: /Water/ })
    expect(boots.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(boots)
    expect(screen.getByRole('button', { name: /Water/ }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText(/1 of \d packed/)).toBeTruthy()
  })

  test('week: tapping game days rebuilds the plan', () => {
    render(<App />)
    click(/^Week$/)
    expect(screen.getByText(/1 game this week/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Monday: no game/ }))
    expect(screen.getByText(/2 games this week/)).toBeTruthy()
    const saved = JSON.parse(localStorage.getItem('games')!)
    expect(saved.map((g: { day: number }) => g.day)).toEqual([0, 2])
  })

  test('player: the cue fills the screen', () => {
    render(<App />)
    click(/All sessions/)
    click(/Close control & 1v1/)
    vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'clearTimeout', 'clearInterval', 'Date'] })
    click(/Start session/)
    act(() => { vi.advanceTimersByTime(1200) })
    click(/let's go/i)
    let guard = 0
    while (!/1v1 reaction/.test(stepName()) && guard++ < 200) click(/^Skip$/)
    act(() => { vi.advanceTimersByTime(3200) })
    const cue = screen.getByTestId('cue')
    expect(cue.className).toMatch(/fixed inset-0/)
    expect(cue.className).toMatch(/opacity-100/)
    expect(cue.textContent).toMatch(/Defender commits (left|right)/)
  })

  test('clips opened from home: Done goes back home, not to settings', () => {
    render(<App />)
    click(/^Clips & quotes$/)
    expect(screen.getByRole('heading', { name: /Clips & quotes/i })).toBeTruthy()
    click(/^Done$/)
    expect(screen.queryByRole('heading', { name: /^Settings$/i })).toBeNull()
    expect(screen.getAllByRole('button', { name: /Kick off/ }).length).toBeGreaterThan(0)
  })

  test('laptop: sidebar planner instead of the phone tab bar', () => {
    vi.stubGlobal('matchMedia', (q: string) => ({ matches: q.includes('min-width: 1024px'), addEventListener() {}, removeEventListener() {} }))
    render(<App />)
    expect(screen.getByText(/Good (morning|afternoon|evening), Peter/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: /Kick off today's session/ })).toBeNull() // no phone tab bar
    fireEvent.click(screen.getByRole('button', { name: /^Week$/ }))
    expect(screen.getByText(/Tap the days you play/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /^Sessions$/ }))
    expect(screen.getByRole('heading', { name: /All sessions/i })).toBeTruthy()
  })
})

