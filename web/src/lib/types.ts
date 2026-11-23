export type Block = {
  name: string
  kind: 'warmup' | 'drill' | 'cooldown' | string
  work: number
  rest?: number
  rounds?: number
  cue?: boolean | 'numbers' // voice cues in the drill: left/right (1v1), or random numbers (even: pass back, odd: turn)
  how?: string
  needs?: string[]
  reps?: string // gym: reps per set; work is then an estimated set time
  category?: string // library category, picks the drill-card photo
  superset?: boolean // done back to back with the exercise before it (alternating rounds, no rest between)
  works?: string[] // muscles and joints it mainly works (library exercises)
  also?: string[] // ...and the ones it also loads
  safe?: string // how not to get hurt doing it (big lifts)
}

export type Session = {
  id: string
  title: string
  type: 'football' | 'recovery' | 'gym' | 'cheer' | string
  summary: string
  needs: string[]
  blocks: Block[]
}

export type Step = {
  type: 'work' | 'rest'
  blockIdx: number
  name: string
  kind: string
  round: number
  rounds: number
  seconds: number
  cue: boolean
  cueKind?: 'numbers'
  how: string
  reps?: string
  category?: string
  pair?: string // superset position, e.g. 'A1', 'A2'
  needs: string[]
}

export type Direction = 'left' | 'right'

export type Settings = {
  voice: boolean
  volume: number
  cueMin: number
  cueMax: number
  cards: boolean
  earbudControls: boolean // AirPods/lock-screen controls during a session (lib/remote.ts); off by default: it pauses your music
}
