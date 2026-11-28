// Daily play: one hand-written tactical scenario a day, the same for everyone (like Wordle), so friends can compare answers.
// Top-down pitch, you highlighted, a few options, then the best one and why. Scenarios live in data/scenarios.json.
import data from '../data/scenarios.json'
import { load, save } from './storage'

export type Kind = 'pass' | 'shoot' | 'dribble' | 'run' | 'hold'
export type Option = { label: string; kind: Kind; to: [number, number]; why: string }
// Pitch units: 100 long (we attack towards x = 100, the goal is at y 28 to 36) by 64 wide.
export type Scenario = { id: string; for: string[]; title: string; setup: string; you: [number, number]; mates: [number, number][]; opps: [number, number][]; onBall: boolean; options: Option[]; best: number }
export const SCENARIOS = (data as { scenarios: Scenario[] }).scenarios

const EPOCH = Date.UTC(2026, 0, 1)
export const ymd = (d: Date) => d.toLocaleDateString('en-CA')
export const dayNumber = (d = new Date()) => Math.floor((Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) - EPOCH) / 86400000) + 1
export const scenarioFor = (d = new Date()): Scenario => SCENARIOS[((dayNumber(d) - 1) % SCENARIOS.length + SCENARIOS.length) % SCENARIOS.length]

// The options in a different order each day (the same for everyone), so the best one isn't always in the same place.
function seeded(seed: number) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 } }
export function optionOrder(day: number, n: number): number[] {
  const rng = seeded(day * 2654435761)
  const order = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]] }
  return order
}

export type Answer = { id: string; picked: number; correct: boolean }
const store = () => load<Record<string, Answer>>('daily', {})
export const answerFor = (d = new Date()) => store()[ymd(d)] ?? null
export function answerDaily(picked: number, d = new Date()): Answer {
  const s = scenarioFor(d)
  const a = { id: s.id, picked, correct: picked === s.best }
  save('daily', { ...store(), [ymd(d)]: a })
  return a
}

// Days in a row you played (right or wrong), counting back from today (or yesterday, if today is still to play).
export function dailyStreak(d = new Date()): number {
  const all = store()
  let n = 0
  const day = new Date(d)
  if (!all[ymd(day)]) day.setDate(day.getDate() - 1)
  while (all[ymd(day)]) { n++; day.setDate(day.getDate() - 1) }
  return n
}

export const shareText = (d: Date, a: Answer, streak: number) => `Daily play #${dayNumber(d)} ${a.correct ? '✅' : '❌'}${streak > 1 ? ` 🔥${streak}` : ''}`
