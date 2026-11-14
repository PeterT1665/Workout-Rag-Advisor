import { afterEach, expect, test, vi } from 'vitest'
import { attachRemote, beginRemote, endRemote } from './remote'

afterEach(() => { endRemote(); vi.unstubAllGlobals() })

test('earbud presses drive the session: any single press is the main action, double skips, triple goes back', () => {
  const handlers: Record<string, (() => void) | null> = {}
  const session = { playbackState: 'none', metadata: null as unknown, setActionHandler: (a: string, h: (() => void) | null) => { handlers[a] = h } }
  vi.stubGlobal('navigator', { ...navigator, mediaSession: session })
  vi.stubGlobal('MediaMetadata', class { init: object; constructor(init: object) { this.init = init } })
  vi.stubGlobal('Audio', class { loop = false; play() { return Promise.resolve() } pause() {} })
  URL.createObjectURL ??= () => 'blob:silence'
  const c = { press: vi.fn(), next: vi.fn(), prev: vi.fn() }
  beginRemote('Close control')
  attachRemote(c)
  handlers.pause!()
  handlers.play!()
  handlers.pause!()
  expect(c.press).toHaveBeenCalledTimes(3) // every press arrives, whatever iOS thinks is playing
  expect(session.playbackState).toBe('playing')
  handlers.nexttrack!()
  handlers.previoustrack!()
  expect(c.next).toHaveBeenCalledTimes(1)
  expect(c.prev).toHaveBeenCalledTimes(1)
  endRemote()
  expect(handlers.pause).toBeNull()
})
