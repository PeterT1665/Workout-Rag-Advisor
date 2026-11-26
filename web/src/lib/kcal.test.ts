import { describe, expect, test } from 'vitest'
import { estimateKcal, kcalText, sessionMet } from './kcal'
import { BUILTIN_SESSIONS } from './library'

const by = (id: string) => BUILTIN_SESSIONS.find((s) => s.id === id)!

describe('calories', () => {
  test('whole-session MET by type and effort (research table)', () => {
    expect(sessionMet('gym', 4)).toBeCloseTo(4.2); expect(sessionMet('gym', 8, true)).toBeCloseTo(6.6)
    expect(sessionMet('football', 8)).toBeCloseTo(8); expect(sessionMet('recovery', 9)).toBe(2.3); expect(sessionMet('gym', undefined)).toBeCloseTo(4.8)
  })
  test('without the session: MET × kg × hours, shown as a range', () => {
    const k = estimateKcal({ type: 'gym', minutes: 60, effort: 8, kg: 67 }) // research: MET 5.4 -> 362, range ~330-450
    expect(k.low).toBeGreaterThanOrEqual(320); expect(k.high).toBeLessThanOrEqual(460); expect(k.low).toBeLessThan(k.mid); expect(k.mid).toBeLessThan(k.high)
    expect(kcalText(k)).toMatch(/^~\d+–\d+$/)
  })
  test('a match-eve activation day is summed per exercise: well under the whole-session figure, whatever the effort felt like', () => {
    const s = by('gym-activation')
    const felt4 = estimateKcal({ session: s, type: 'gym', minutes: 31, effort: 4, kg: 67 })
    const felt10 = estimateKcal({ session: s, type: 'gym', minutes: 31, effort: 10, kg: 67 })
    const whole = estimateKcal({ type: 'gym', minutes: 31, effort: 4, kg: 67 })
    expect(felt4.mid).toBeGreaterThanOrEqual(80); expect(felt4.high).toBeLessThanOrEqual(200)
    expect(felt4.mid).toBeLessThan(whole.mid + 40)
    // the max efforts cost the same however it felt afterwards: only the strength and football parts move with effort
    expect(felt10.mid - felt4.mid).toBeLessThan(40)
  })
  test('more minutes means more calories, and football burns more than recovery', () => {
    const s = by('gym-upper-core') ?? BUILTIN_SESSIONS.find((x) => x.type === 'gym')!
    expect(estimateKcal({ session: s, minutes: 70, effort: 7, kg: 67 }).mid).toBeGreaterThan(estimateKcal({ session: s, minutes: 45, effort: 7, kg: 67 }).mid)
    const fb = BUILTIN_SESSIONS.find((x) => x.type === 'football')!, rec = BUILTIN_SESSIONS.find((x) => x.type === 'recovery')!
    expect(estimateKcal({ session: fb, minutes: 45, effort: 6, kg: 67 }).mid).toBeGreaterThan(estimateKcal({ session: rec, minutes: 45, effort: 6, kg: 67 }).mid)
  })
})
