import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import App from './App'
import { PhoneFrame, showPhoneFrame } from './views/PhoneFrame'
import { getProfile, reloadProfile } from './lib/profile'

const spoken: string[] = []
const played: string[] = []
let lastAudio: { onended: (() => void) | null } | null = null

beforeEach(() => {
  spoken.length = 0
  played.length = 0
  localStorage.clear()
  reloadProfile()
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
    click(/Make it yours: photos, colour, clips \(/)
    fireEvent.change(screen.getByLabelText(/Clip type/), { target: { value: 'mindset' } })
    fireEvent.change(screen.getByLabelText(/Audio files/), { target: { files: [new File(['x'], 'cr7-interview.mp3')] } })
    await waitFor(() => expect(screen.getByText('cr7-interview')).toBeTruthy())
    fireEvent.change(screen.getByLabelText(/Clip type/), { target: { value: 'siuu' } })
    fireEvent.change(screen.getByLabelText(/Audio files/), { target: { files: [new File(['x'], 'siuu.m4a')] } })
    await waitFor(() => expect(screen.getByText('siuu')).toBeTruthy())
    click(/^Done$/)
    expect(screen.getByRole('button', { name: /Make it yours: photos, colour, clips \(2\)/ })).toBeTruthy()
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

  test('session details: kit check toggles and remembers packed items', () => {
    render(<App />)
    click(/All sessions/)
    click(/Recovery: hips/)
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

  test('week: the gym stepper changes how many gym sessions are planned', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Week$/ }))
    expect(screen.getByText(/3 gym/)).toBeTruthy()
    click(/More gym sessions/)
    expect(screen.getByText(/4 gym/)).toBeTruthy()
    expect(getProfile().gym).toBe(4)
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
    click(/^Make it yours$/)
    expect(screen.getByRole('heading', { name: /^Make it yours$/i })).toBeTruthy()
    click(/^Green$/)
    expect(document.documentElement.dataset.accent).toBe('green')
    fireEvent.change(screen.getByLabelText('Role model'), { target: { value: 'Kaka' } })
    expect(getProfile().idol).toBe('Kaka')
    click(/^Done$/)
    expect(screen.queryByRole('heading', { name: /^Settings$/i })).toBeNull()
    expect(screen.getAllByRole('button', { name: /Kick off/ }).length).toBeGreaterThan(0)
  })

  test('builder: customise a recommended session, add dribbling, save as my own', () => {
    render(<App />)
    click(/All sessions/)
    click(/Wall passing, weak foot & shooting/)
    click(/^Customise$/)
    expect((screen.getByLabelText(/Session name/) as HTMLInputElement).value).toMatch(/\(mine\)/)
    click(/Add exercises/)
    fireEvent.click(screen.getByRole('button', { name: /^Dribbling$/ }))
    click(/Add Toe taps/)
    click(/Remove Toe taps/) // added by mistake: take it back
    expect(screen.queryByRole('button', { name: /Remove Toe taps/ })).toBeNull()
    click(/^Cone weave dribble$/) // tap the name: details sheet
    const sheet = screen.getByRole('dialog', { name: /Cone weave dribble/ })
    expect(sheet.textContent).toMatch(/Weave through/)
    fireEvent.click(within(sheet).getByRole('button', { name: /Add Cone weave dribble/ }))
    click(/Close details/)
    click(/^Done/)
    fireEvent.click(screen.getAllByRole('button', { name: /More Rounds/ })[0]) // first item: warm-up gets 2 rounds
    click(/^Save$/)
    // lands on the new session's detail, and the original is unchanged in the library
    expect(screen.getByRole('heading', { name: /\(mine\)/ })).toBeTruthy()
    expect(screen.getByText(/Cone weave dribble/)).toBeTruthy()
    const mine = JSON.parse(localStorage.getItem('mySessions') || '[]')
    expect(mine).toHaveLength(1)
    expect(mine[0].items.at(-2)).toEqual({ ex: 'cone-weave' }) // added before the cool-down
    expect(mine[0].items.at(-1).ex).toMatch(/cool-down/)
    expect(mine[0].items[0].rounds).toBe(2)
    click(/^Back$/)
    expect(screen.getByRole('heading', { name: /^Sessions$/i })).toBeTruthy()
    expect(screen.getAllByRole('button', { name: /Wall passing, weak foot & shooting/ })).toHaveLength(2)
  })

  test('gym sessions show reps', () => {
    render(<App />)
    click(/All sessions/)
    click(/Gym: lower body \+ power/)
    expect(screen.getByText(/4 × 5 reps · 150s rest/)).toBeTruthy()
  })

  test('settings: edit player changes name, positions and team colour', () => {
    render(<App />)
    fireEvent.click(screen.getAllByRole('button', { name: /settings/i })[0])
    click(/Edit player/)
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Jordan' } })
    click(/^Next$/)
    click(/Defender/)
    click(/^left$/i)
    click(/^Next$/)
    click(/^Saturday$/)
    click(/^Next$/)
    click(/^Blue$/)
    expect(document.documentElement.dataset.accent).toBe('blue')
    click(/^Done/)
    expect(getProfile()).toMatchObject({ name: 'Jordan', positions: ['winger', 'defender'], foot: 'left', accent: 'blue' })
    expect(JSON.parse(localStorage.getItem('games') || '[]').map((g: { day: number }) => g.day)).toContain(5)
    expect(screen.getByText(/Jordan/)).toBeTruthy()
  })

  test('photos: upload your own role model photo, it replaces the app photo, remove it again', async () => {
    render(<App />)
    click(/^Settings$/)
    click(/Make it yours: photos, colour, clips \(/)
    const input = screen.getByLabelText('Add Home hero photo')
    fireEvent.change(input, { target: { files: [new File(['x'], 'idol.jpg', { type: 'image/jpeg' })] } })
    const removeBtn = await screen.findByRole('button', { name: 'Remove Home hero photo' })
    expect(screen.getByLabelText('Replace Home hero photo')).toBeTruthy()
    fireEvent.click(removeBtn)
    await waitFor(() => expect(screen.getByLabelText('Add Home hero photo')).toBeTruthy())
  })

  test('home: the how-it-works card shows once, Set my week opens the week', () => {
    const { unmount } = render(<App />)
    expect(screen.getByRole('region', { name: 'How it works' })).toBeTruthy()
    click(/Set my week/)
    expect(screen.getByText(/Tap the days you play/)).toBeTruthy()
    unmount()
    render(<App />)
    expect(screen.queryByRole('region', { name: 'How it works' })).toBeNull()
  })

  test('rating session: three tests rate the player, the card is revealed and kept', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Player$/ }))
    click(/Start rating session/)
    click(/^Start$/)
    fireEvent.change(screen.getByLabelText('Your result'), { target: { value: '3.3' } }) // PAC
    click(/Next test/)
    click(/^Skip$/) // DRI
    fireEvent.change(screen.getByLabelText('Your result'), { target: { value: '8' } }) // SHO
    click(/Next test/)
    fireEvent.change(screen.getByLabelText('Your result'), { target: { value: '24' } }) // PAS
    click(/Next test/)
    click(/^Skip$/) // DEF
    click(/^Skip$/) // PHY: last one finishes
    expect(screen.getByRole('heading', { name: /Gold card/ })).toBeTruthy()
    expect(getProfile().ratings?.stats).toEqual({ PAC: 82, SHO: 83, PAS: 85 })
    click(/^Done$/)
    expect(screen.getByText(/New rating session/)).toBeTruthy()
  })

  test('invite: QR from Settings, and Train together on partner sessions', () => {
    render(<App />)
    click(/^Settings$/)
    click(/^Invite a friend$/)
    expect(screen.getByRole('img', { name: /QR code for .*invite=1/ })).toBeTruthy()
    click(/^Done$/)
    expect(screen.getByRole('heading', { name: /^Settings$/i })).toBeTruthy()
    click(/^Done$/)
    click(/All sessions/)
    click(/With a partner/)
    click(/Train together/)
    expect(screen.getByRole('heading', { name: /Train together/i })).toBeTruthy()
    click(/^Done$/)
    expect(screen.getByRole('heading', { name: /With a partner/i })).toBeTruthy()
  })

  test('website on iPhone: Install button opens the Add to Home Screen steps; hidden on desktop browsers', async () => {
    const { unmount } = render(<App />)
    expect(screen.queryByRole('button', { name: 'Install the app' })).toBeNull() // not a phone
    unmount()
    const ua = vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue('Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)')
    render(<App />)
    click(/Install the app/)
    const sheet = await screen.findByRole('dialog', { name: 'Install the app' })
    expect(sheet.textContent).toMatch(/Add to Home Screen/)
    fireEvent.click(within(sheet).getByRole('button', { name: 'Got it' }))
    expect(screen.queryByRole('dialog', { name: 'Install the app' })).toBeNull()
    ua.mockRestore()
  })

  test('home: tapping a drill card shows that drill, Full session opens the session', () => {
    render(<App />)
    const card = screen.getAllByRole('button').find((b) => /× .*rest/.test(b.textContent ?? '') && b.closest('main'))!
    fireEvent.click(card)
    const sheet = screen.getByRole('dialog')
    expect(within(sheet).getByText('Rest')).toBeTruthy()
    fireEvent.click(within(sheet).getByRole('button', { name: 'Full session' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getAllByRole('button', { name: /Start session|Kick off/i }).length).toBeGreaterThan(0)
  })

  test('week: Do it today makes a session today\'s; Change can set rest and go back to the plan', () => {
    render(<App />)
    click(/All sessions/)
    click(/Gym: explosive power/)
    click(/^Do it today$/)
    expect(screen.getByRole('heading', { name: /Gym: explosive power/i })).toBeTruthy() // home hero
    fireEvent.click(screen.getByRole('button', { name: /^Week$/ }))
    expect(screen.getAllByText('Changed').length).toBe(1)
    const today = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][(new Date().getDay() + 6) % 7]
    click(new RegExp(`^Change ${today}$`))
    const sheet = screen.getByRole('dialog', { name: `Change ${today}` })
    fireEvent.click(within(sheet).getByRole('button', { name: 'Back to plan' }))
    expect(screen.queryByText('Changed')).toBeNull()
    expect(JSON.parse(localStorage.getItem('weekPlan') || '{}')).toEqual({})
  })

  test('builder: an exercise can be moved up with the grip (keyboard), and the order is saved', () => {
    render(<App />)
    click(/All sessions/)
    click(/Build a session/)
    click(/Add Juggling/)
    click(/Add Toe taps/)
    click(/^Done/)
    const grip = screen.getByRole('button', { name: /Reorder Toe taps/ })
    fireEvent.keyDown(grip, { key: 'ArrowUp' })
    fireEvent.change(screen.getByLabelText(/Session name/), { target: { value: 'Ball work' } })
    click(/^Save$/)
    const mine = JSON.parse(localStorage.getItem('mySessions') || '[]')
    expect(mine[0].items.map((x: { ex: string }) => x.ex)).toEqual(['toe-taps', 'juggling'])
  })

  test('a session survives a reload: Resume continues from the same drill; ending early still saves it', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'clearTimeout', 'clearInterval', 'Date'] })
    const first = render(<App />)
    click(/All sessions/)
    click(/Close control & 1v1/)
    click(/Start session/)
    act(() => { vi.advanceTimersByTime(1200) })
    click(/let's go/i)
    click(/^Skip$/)
    click(/^Skip$/)
    act(() => { vi.advanceTimersByTime(70_000) }) // train a bit
    const where = stepName()
    first.unmount() // iOS closes the app
    render(<App />)
    expect(screen.getByRole('region', { name: 'Unfinished session' })).toBeTruthy()
    click(/^Resume$/)
    expect(stepName()).toBe(where)
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    click(/End session/)
    const history = JSON.parse(localStorage.getItem('history') || '[]')
    expect(history.at(-1).title).toBe('Close control & 1v1')
    expect(history.at(-1).minutes).toBeGreaterThanOrEqual(1)
    expect(localStorage.getItem('activeSession')).toBeNull()
  })

  test('laptop: the website shows the phone app in a frame with a QR code', () => {
    vi.stubGlobal('matchMedia', (q: string) => ({ matches: q.includes('min-width: 1024px'), addEventListener() {}, removeEventListener() {} }))
    expect(showPhoneFrame()).toBe(true)
    render(<PhoneFrame />)
    expect(screen.getByTitle('Football Personal Coach').getAttribute('src')).toMatch(/[?&]app=1/)
    expect(screen.getByRole('img', { name: /open the app on your phone/ })).toBeTruthy()
  })
})

