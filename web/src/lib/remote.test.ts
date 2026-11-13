import { afterEach, expect, test, vi } from 'vitest'
import { attachRemote, beginRemote, endRemote } from './remote'

afterEach(() => { endRemote(); vi.unstubAllGlobals() })

test('earbud presses drive the session: pause/play toggle it, double press skips, triple goes back', () => {
  const handlers: Record<string, (() => void) | null> = {}
  vi.stubGlobal('navigator', { ...navigator, mediaSession: { playbackState: 'none', metadata: null, setActionHandler: (a: string, h: (() => void) | null) => { handlers[a] = h } } })
  vi.stubGlobal('MediaMetadata', class { init: object; constructor(init: object) { this.init = init } })
  vi.stubGlobal('Audio', class { loop = false; play() { return Promise.resolve() } pause() {} })
  URL.createObjectURL ??= () => 'blob:silence'
  let running = true
  const c = { toggle: vi.fn(() => { running = !running }), next: vi.fn(), prev: vi.fn(), running: () => running }
  beginRemote('Close control')
  attachRemote(c)
  handlers.pause!()
  expect(c.toggle).toHaveBeenCalledTimes(1)
  expect(running).toBe(false)
  handlers.pause!() // already paused: nothing more
  expect(c.toggle).toHaveBeenCalledTimes(1)
  handlers.play!()
  expect(running).toBe(true)
  handlers.nexttrack!()
  handlers.previoustrack!()
  expect(c.next).toHaveBeenCalledTimes(1)
  expect(c.prev).toHaveBeenCalledTimes(1)
  endRemote()
  expect(handlers.pause).toBeNull()
})
