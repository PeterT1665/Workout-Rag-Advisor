import { afterEach, describe, expect, test } from 'vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import { PlayerView } from './PlayerView'
import { createPlayer } from '../lib/player'
import { DEFAULT_SETTINGS } from '../lib/storage'
import type { Session } from '../lib/types'

afterEach(() => { cleanup(); localStorage.clear() })

const bench = { name: 'Bench press', kind: 'drill', work: 40, rest: 90, rounds: 1, reps: '6', category: 'upper-strength' }
const pullups = { name: 'Pull-ups', kind: 'drill', work: 40, rest: 90, rounds: 1, reps: '6-8', category: 'upper-strength' }
const plank = { name: 'Plank', kind: 'drill', work: 30, category: 'core' }

describe('log your sets', () => {
  test('a superset paired mid-session logs both exercises (bench + pull-ups)', () => {
    // planned with bench on its own; the player pairs pull-ups with it in the Plan sheet
    const planned: Session = { id: 'gym', title: 'Gym', type: 'gym', summary: '', needs: [], blocks: [{ name: 'Warm', kind: 'warmup', work: 60 }, bench, plank] }
    const player = createPlayer(planned, { audio: { say: () => {}, beep: () => {} }, getSettings: () => DEFAULT_SETTINGS })
    render(<PlayerView session={planned} player={player} settings={DEFAULT_SETTINGS} onExit={() => {}} />) // starts the session
    act(() => { player.editUpcoming([bench, { ...pullups, superset: true }, plank]) })
    act(() => { player.skip(); player.skip(); player.skip() }) // warm-up, bench, pull-ups: now the rest after the superset
    expect(player.getState().steps[player.getState().i].type).toBe('rest')
    expect(screen.getByRole('button', { name: /Log your sets/ }).textContent).toMatch(/Bench press \+ Pull-ups/)
    player.stop()
  })
  test('a planned superset with different set counts (bench 4, pull-ups 3): each is logged after its own last set', () => {
    const planned: Session = { id: 'gym', title: 'Gym', type: 'gym', summary: '', needs: [], blocks: [{ ...bench, rounds: 4 }, { ...pullups, rounds: 3, superset: true }, plank] }
    const player = createPlayer(planned, { audio: { say: () => {}, beep: () => {} }, getSettings: () => DEFAULT_SETTINGS })
    render(<PlayerView session={planned} player={player} settings={DEFAULT_SETTINGS} onExit={() => {}} />)
    const restsSeen: string[] = []
    for (let k = 0; k < 20 && !player.getState().finished; k++) {
      const st = player.getState(), step = st.steps[st.i]
      if (step.type === 'rest') restsSeen.push(screen.queryByRole('button', { name: /Log your sets/ })?.textContent ?? '-')
      if (step.name === 'Plank') break
      act(() => player.skip())
    }
    expect(restsSeen).toEqual(['-', '-', 'Log your setsPull-ups›', 'Log your setsBench press›'])
    player.stop()
  })
})

