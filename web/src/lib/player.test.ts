import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { createPlayer } from './player'
import { DEFAULT_SETTINGS } from './storage'
import type { AudioDeps } from './audio'
import type { ClipKind, Session, Settings } from './types'

const session: Session = {
  id: 't', title: 'T', type: 'football', summary: '', needs: [],
  blocks: [
    { name: 'Warm', kind: 'warmup', work: 3 },
    { name: '1v1', kind: 'drill', work: 14, rest: 22, rounds: 2, cue: true },
    { name: 'End', kind: 'cooldown', work: 2 },
  ],
}

function fakeAudio(clips: Partial<Record<ClipKind, number>> = {}) {
  const log: string[] = []
  const deps: AudioDeps = {
    say: (t) => log.push('say:' + t),
    beep: () => {},
    playClip: (k) => { log.push('clip:' + k); return (clips[k] ?? 0) > 0 ? { id: k, kind: k, name: k, blob: new Blob() } : null },
    stopClip: () => log.push('stop'),
    hasClips: (k) => (clips[k] ?? 0) > 0,
  }
  return { deps, log }
}

const settings = (p: Partial<Settings> = {}) => () => ({ ...DEFAULT_SETTINGS, cueMin: 2, cueMax: 3, ...p })
const said = (log: string[]) => log.filter((l) => l.startsWith('say:')).map((l) => l.slice(4))

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => { vi.useRealTimers() })

describe('player engine', () => {
  test('runs the steps in order and finishes', () => {
    const { deps, log } = fakeAudio()
    const onFinish = vi.fn()
    const p = createPlayer(session, { audio: deps, getSettings: settings(), onFinish })
    p.start()
    expect(p.getState().steps[p.getState().i].name).toBe('Warm')
    vi.advanceTimersByTime(3200)
    expect(p.getState().steps[p.getState().i].name).toBe('1v1')
    vi.advanceTimersByTime(14000 + 22000 + 14000 + 22000 + 2500)
    expect(p.getState().finished).toBe(true)
    expect(onFinish).toHaveBeenCalledOnce()
    expect(said(log).at(-1)).toBe('Session complete. Nice work.')
  })

  test('cues only during cue drills', () => {
    const { deps, log } = fakeAudio()
    const p = createPlayer(session, { audio: deps, getSettings: settings() })
    p.start()
    vi.advanceTimersByTime(3200)
    expect(said(log).filter((s) => s === 'left' || s === 'right')).toHaveLength(0)
    vi.advanceTimersByTime(14000)
    const cues = said(log).filter((s) => s === 'left' || s === 'right')
    expect(cues.length).toBeGreaterThanOrEqual(3)
    vi.advanceTimersByTime(21000) // during the rest
    expect(said(log).filter((s) => s === 'left' || s === 'right')).toHaveLength(cues.length)
  })

  test('pause freezes the clock and stops cues; resume continues', () => {
    const { deps, log } = fakeAudio()
    const p = createPlayer(session, { audio: deps, getSettings: settings() })
    p.start()
    vi.advanceTimersByTime(3200 + 4000)
    p.togglePause()
    const rem = p.getState().remainingMs
    const before = said(log).length
    vi.advanceTimersByTime(10000)
    expect(p.getState().remainingMs).toBe(rem)
    expect(said(log).length).toBe(before)
    p.togglePause()
    vi.advanceTimersByTime(rem + 300)
    expect(p.getState().steps[p.getState().i].type).toBe('rest')
  })

  test('spoken step names drop bracketed notes', () => {
    const { deps, log } = fakeAudio()
    const p = createPlayer({ ...session, blocks: [{ name: '1v1 reaction (voice cues)', kind: 'drill', work: 5 }] },
      { audio: deps, getSettings: settings() })
    p.start()
    expect(said(log)[0]).toBe('1v1 reaction')
  })

  test('skip and back move between steps', () => {
    const { deps } = fakeAudio()
    const p = createPlayer(session, { audio: deps, getSettings: settings() })
    p.start()
    p.skip()
    expect(p.getState().i).toBe(1)
    p.back()
    expect(p.getState().i).toBe(0)
    p.back()
    expect(p.getState().i).toBe(0)
  })

  test('rest clip plays during a long rest and stops before it ends', () => {
    const { deps, log } = fakeAudio({ mindset: 2 })
    const p = createPlayer(session, { audio: deps, getSettings: settings({ restClipChance: 1 }), rng: () => 0.3 })
    p.start()
    vi.advanceTimersByTime(3200 + 14000 + 3000)
    expect(log).toContain('clip:mindset')
    const idx = log.lastIndexOf('clip:mindset')
    vi.advanceTimersByTime(17000) // 4s before rest end
    expect(log.slice(idx + 1)).toContain('stop')
  })

  test('SIUUU clip replaces the spoken ending when available', () => {
    const { deps, log } = fakeAudio({ siuu: 1 })
    const p = createPlayer(session, { audio: deps, getSettings: settings() })
    p.start()
    vi.advanceTimersByTime(90000)
    expect(p.getState().siuu).toBe(true)
    expect(log).toContain('clip:siuu')
    expect(said(log)).not.toContain('Session complete. Nice work.')
  })
})
