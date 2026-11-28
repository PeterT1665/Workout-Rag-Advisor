// Stress tests over everything a player can combine: every session x every injury x every severity, every profile fit,
// every calorie estimate, every week the planner can build. Nothing may crash, empty out, or break the rules.
import { describe, expect, test } from 'vitest'
import { BUILTIN_SESSIONS, EXERCISES } from './library'
import { PARTS, injurySafe } from './injuries'
import { fitSession } from './fit'
import { estimateKcal } from './kcal'
import { sessionMinutes, flattenSession } from './logic'
import { buildWeek, type Game, type PlanOpts } from './weekplan'
import { sessionForSlot, setPlanHeading, setPlanPositions, planToday } from './today'
import { hasHeading, hasMaxSpeed, headersIn, isHighVolumeShooting, MAX_HEADERS_PER_SESSION } from './dayrules'
import type { Injury } from './profile'

const sevs = ['light', 'moderate', 'serious'] as const
const inj = (part: Injury['part'], severity: Injury['severity']): Injury => ({ id: 'x', part, tissue: 'muscle', kind: 'pain', severity, since: '2026-11-27' })

describe('stress: sessions', () => {
  test('every session x injury part x severity: no crash, never empty, no heading drill swapped in, supersets stay valid', () => {
    for (const s of BUILTIN_SESSIONS) for (const { part } of PARTS) for (const sev of sevs) {
      const r = injurySafe(s, [inj(part, sev)])
      expect(r.session.blocks.length, `${s.id} ${part} ${sev}`).toBeGreaterThan(0)
      if (!hasHeading(s)) expect(hasHeading(r.session), `${s.id} ${part} gained heading`).toBe(false)
      r.session.blocks.forEach((b, k) => { if (b.superset) expect(k > 0 && r.session.blocks[k - 1].kind === 'drill', `${s.id} superset`).toBe(true) })
      expect(flattenSession(r.session).length).toBeGreaterThan(0)
    }
  })
  test('every session x profile fit (gym experience, level, kit, weak foot): no crash, blocks kept, no heading added', () => {
    for (const s of BUILTIN_SESSIONS) for (const gymExp of ['new', 'some', 'experienced'] as const) for (const kit of [undefined, [], ['ball'], ['ball', 'wall', 'cones']] as const) for (const weakFootFocus of [false, true]) {
      const f = fitSession(s, { gymExp, level: 'casual', trainAt: ['pitch'], kit: kit as never, weakFootFocus })
      expect(f.session.blocks.length, s.id).toBeGreaterThan(0)
      expect(hasHeading(f.session) && !hasHeading(s)).toBe(false)
    }
  })
  test('every session: a positive duration, and calories as a believable range at 50 to 110 kg', () => {
    for (const s of BUILTIN_SESSIONS) {
      const m = sessionMinutes(s)
      expect(m, s.id).toBeGreaterThan(5); expect(m, s.id).toBeLessThan(110)
      for (const kg of [50, 70, 110]) for (const effort of [1, 6, 10]) {
        const k = estimateKcal({ session: s, type: s.type, minutes: m, effort, kg })
        expect(k.low, `${s.id} ${kg} ${effort}`).toBeGreaterThan(0); expect(k.low).toBeLessThanOrEqual(k.mid); expect(k.mid).toBeLessThanOrEqual(k.high)
        // under 2 to 14 kcal a minute: a walk to a hard match, per 70 kg
        expect(k.mid / m / (kg / 70), `${s.id} kcal/min`).toBeGreaterThan(1.2); expect(k.mid / m / (kg / 70), `${s.id} kcal/min`).toBeLessThan(15)
      }
    }
  })
  test('every drill level has text a phone can show: not empty, not huge, a how-to under 330 characters', () => {
    for (const e of EXERCISES.filter((x) => x.family)) {
      expect(e.how!.length, e.id).toBeGreaterThan(20); expect(e.how!.length, e.id).toBeLessThan(330)
      expect(e.mistake!.length, e.id).toBeLessThan(90); expect(e.fix!.length, e.id).toBeLessThan(110)
      expect(e.work, e.id).toBeGreaterThan(0); expect(e.rounds ?? 1, e.id).toBeLessThanOrEqual(12)
    }
  })
  test('every session keeps within the heading cap and has no duplicate block names', () => {
    for (const s of BUILTIN_SESSIONS) {
      expect(headersIn(s), s.id).toBeLessThanOrEqual(MAX_HEADERS_PER_SESSION)
      const names = s.blocks.map((b) => b.name); expect(new Set(names).size, `${s.id} duplicates`).toBe(names.length)
    }
  })
})

describe('stress: weeks', () => {
  const gameSets: Game[][] = [[], [{ day: 2, kind: 'competitive' }], [{ day: 5, kind: 'competitive' }], [{ day: 2, kind: 'casual' }, { day: 5, kind: 'competitive' }], [{ day: 0, kind: 'competitive' }, { day: 2, kind: 'competitive' }, { day: 4, kind: 'casual' }], [{ day: 6, kind: 'competitive' }]]
  const opts: PlanOpts[] = [{}, { season: 'off' }, { doubles: false }, { season: 'off', doubles: false }, { football: 0, gym: 5 }, { football: 6, gym: 0 }, { football: 6, gym: 5, doubles: false }]
  const positions = [[], ['gk'], ['cb'], ['fb'], ['dm'], ['cm'], ['am'], ['winger'], ['striker'], ['winger', 'striker']]
  test('every games x settings x position: a valid week, picks are football sessions, the rules hold', () => {
    setPlanHeading(true)
    for (const games of gameSets) for (const o of opts) for (const pos of positions) {
      setPlanPositions(pos)
      const plan = buildWeek(games, o)
      const gd = new Set(games.map((g) => g.day))
      for (const slot of plan.slots) { expect(slot.day).toBeGreaterThanOrEqual(0); expect(slot.day).toBeLessThan(7) }
      // never football or gym on a game day
      for (const g of gd) expect(plan.slots.filter((s) => s.day === g).every((s) => s.kind === 'game' || s.kind === 'recovery')).toBe(true)
      // doubles off: no day has both
      if (o.doubles === false) for (let d = 0; d < 7; d++) { const k = plan.slots.filter((s) => s.day === d).map((s) => s.kind); expect(k.includes('football') && k.includes('gym'), `doubles ${d}`).toBe(false) }
      const picks = plan.slots.filter((s) => s.kind === 'football').map((s) => sessionForSlot(s, plan, BUILTIN_SESSIONS))
      expect(picks.every((p) => p && p.type === 'football')).toBe(true)
      expect(picks.filter((p) => p && hasHeading(p)).length).toBeLessThanOrEqual(1)
      expect(picks.filter((p) => p && isHighVolumeShooting(p)).length).toBeLessThanOrEqual(3)
      // a week where every football day is the eve of a game (three games in a row) has no room for one
      const open = plan.slots.filter((x) => x.kind === 'football' && !gd.has((x.day + 1) % 7))
      if (open.length) expect(picks.some((p) => p && hasMaxSpeed(p)), `max speed ${JSON.stringify(o)} ${pos} games ${games.map((g) => g.day)} football ${plan.slots.filter((s) => s.kind === 'football').map((s) => s.day)} picks ${picks.map((p) => p?.id)}`).toBe(true)
      // every slot of the week resolves to something, or null for a game
      for (const slot of plan.slots) { const s = sessionForSlot(slot, plan, BUILTIN_SESSIONS); if (slot.kind !== 'game') expect(s, `${slot.kind} ${slot.day}`).toBeTruthy() }
    }
    setPlanPositions([])
  })
  test('planToday on every weekday and every game set returns a session', () => {
    for (let d = 0; d < 7; d++) for (const games of gameSets) {
      const t = planToday(BUILTIN_SESSIONS, games, new Date(2026, 10, 23 + d, 9))
      expect(t.session.blocks.length).toBeGreaterThan(0)
    }
  })
})
