import { describe, expect, test } from 'vitest'
import { answer, findExercise, type CoachCtx } from './coach'
import { BUILTIN_SESSIONS } from '../library'
import { planToday } from '../today'
import type { HistoryEntry } from '../storage'

const now = new Date(2026, 10, 19, 10) // Thursday
const games = [{ day: 2, kind: 'competitive' as const }]
const bench = (kg: number, reps: number) => ({ ex: 'bench-press', name: 'Bench press', sets: [{ kg, reps }, { kg, reps }] })
const history: HistoryEntry[] = [
  { id: 'gym-upper-core', title: 'Gym: upper body + core', type: 'gym', date: '2026-11-10', minutes: 50, rpe: 8, effort: 8, load: 400, lifts: [bench(60, 6)] },
  { id: 'gym-upper-core', title: 'Gym: upper body + core', type: 'gym', date: '2026-11-17', minutes: 52, rpe: 8, effort: 8, load: 416, lifts: [bench(65, 6)], sore: ['Hamstring'] },
]
const ctx = (over: Partial<CoachCtx> = {}): CoachCtx => ({
  profile: { name: 'Peter', number: 7, positions: ['defender'], foot: 'right', accent: 'red', idol: '', heightCm: 178, body: [{ date: '2026-11-01', kg: 72 }, { date: '2026-11-18', kg: 73 }] },
  history, matches: [], today: planToday(BUILTIN_SESSIONS, games, now), sessions: BUILTIN_SESSIONS, streak: { days: 4, today: false }, now, ...over,
})
const all = (q: string, c = ctx()) => { const r = answer(q, c); return [...r.text, ...(r.cards ?? []).map((x) => ('name' in x ? x.name : 'title' in x ? x.title : x.label))].join(' | ') }

describe('the coach', () => {
  test('finds exercises by name, alias or description', () => {
    expect(findExercise('how do I do a romanian deadlift')?.id).toBe('romanian-deadlift')
    expect(findExercise('my rdl')?.id).toBe('romanian-deadlift')
    expect(findExercise('nordics')?.id).toBe('nordic-curl')
    expect(findExercise('single-leg romanian deadlift form')?.id).toBe('single-leg-rdl')
  })
  test('today: the planned session, with a card to start it', () => {
    const r = answer('what should I do today?', ctx())
    expect(r.cards?.[0]?.kind).toBe('session')
    expect(r.text.join(' ')).toMatch(/minutes/)
  })
  test('lifts: estimated max and progress for one lift', () => {
    const s = all('what is my best bench?')
    expect(s).toMatch(/Bench press: logged in 2 sessions/)
    expect(s).toMatch(/Heaviest set: 65 kg × 6/)
    expect(s).toMatch(/Up \d+% since your first/)
  })
  test('body: weight, trend and BMI', () => {
    const s = all('what is my bmi')
    expect(s).toMatch(/73 kg/)
    expect(s).toMatch(/BMI 23\.0/)
  })
  test('soreness: stretches for the area, and a safety line', () => {
    const r = answer('my hamstrings are sore', ctx())
    expect(r.cards?.map((c) => c.kind === 'exercise' && c.id)).toContain('hamstring-stretch')
    expect([...r.text, ...(r.after ?? [])].join(' ')).toMatch(/physio/)
  })
  test('position: why the training helps a defender', () => {
    expect(all('I play defence, how do the exercises help me?')).toMatch(/Defenders win duels/)
  })
  test('how the app works: retrieval from the knowledge base', () => {
    expect(all('how is the load calculated')).toMatch(/minutes trained × session effort/)
    expect(all('why is gym on thursday and not monday? how is my week planned')).toMatch(/The plan starts from your games/)
    expect(all('what does MD-2 mean')).toMatch(/MD is match day/)
  })
  test('exercise how-to and alternatives', () => {
    expect(all('how do I do a nordic curl')).toMatch(/Nordic hamstring curl/)
    const alt = answer('alternative to back squat, no barbell at home', ctx())
    expect(alt.cards?.length).toBeGreaterThan(0)
    expect(alt.cards?.every((c) => c.kind === 'exercise' && c.id !== 'back-squat')).toBe(true)
  })
  test('sessions: by type and length', () => {
    const r = answer('give me a 30 minute home session', ctx())
    expect(r.cards?.[0]?.kind).toBe('session')
    expect((r.cards ?? []).every((c) => c.kind === 'session' && c.id.startsWith('home-'))).toBe(true)
  })
  test('week: sessions, minutes and the load trend', () => {
    expect(all('how was my week?')).toMatch(/This week: \d+ of \d+ planned sessions/)
  })
  test('matches: none yet says how to start; with one, the last result', () => {
    expect(all('how did I play in my last match')).toMatch(/No matches reviewed yet/)
    const m = { date: '2026-11-18', us: 3, them: 1, minutes: 40, length: 40, position: 'defender' as const, goals: 1, assists: 0, dribbles: 2, keyPasses: 1, duels: 5, firstTouch: 7, tiring: 6, self: 7, moments: '', rating: 8.4, savedAt: 1 }
    expect(all('how did I play in my last match', ctx({ matches: [m] }))).toMatch(/3–1, a win\. You played 40 of 40 minutes and the app rated you 8\.4/)
  })
  test('unknown questions get an honest answer and ideas', () => {
    expect(all('what is the capital of france')).toMatch(/not sure/)
  })
  test('accents optional: Vietnamese without marks still finds soreness', () => {
    expect(answer('dau gan kheo', ctx()).cards?.length).toBeGreaterThan(0)
  })
})
