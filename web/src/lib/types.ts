export type Block = {
  name: string
  kind: 'warmup' | 'drill' | 'cooldown' | string
  work: number
  rest?: number
  rounds?: number
  cue?: boolean
  how?: string
  needs?: string[]
  reps?: string // gym: reps per set; work is then an estimated set time
  category?: string // library category, picks the drill-card photo
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
  how: string
  reps?: string
  category?: string
  needs: string[]
}

export type Direction = 'left' | 'right'
export type ClipKind = 'mindset' | 'siuu'
export type Clip = { id: string; kind: ClipKind; name: string; blob: Blob }

export type Settings = {
  voice: boolean
  volume: number
  cueMin: number
  cueMax: number
  clipStart: boolean
  clipRest: boolean
  siuuEnd: boolean
  cards: boolean
  clipVolume: number
  restClipChance: number
  earbuds: boolean // AirPods/lock-screen controls during a session (lib/remote.ts)
}
