// What goes on a share card for a session, a match or a rating session: up to three stats and one graphic
// (the "route line" of a Strava card): every set as a bar for the gym, a shot map on a goal for finishing, the card for
// a rating session.
import { t } from '../i18n'
import { bestsFrom, e1rm, isBodyweight, moved } from './lifts'
import { gameLength, resultOf, type MatchReview } from './match'
import { TESTS, type Ratings } from './testday'
import type { HistoryEntry } from './storage'

export type ShareStat = { label: string; value: string; tag?: string }
export type Graphic =
  | { type: 'sets'; bars: number[] }
  | { type: 'shots'; shots: number; onTarget: number; goals: number }
  | { type: 'card'; rows: { label: string; value: number }[] }
  | { type: 'flame'; tone?: 'plan' | 'unbeaten' | 'wins' | 'off' }
export type ShareItem = { kind: 'gym' | 'football' | 'recovery' | 'match' | 'rating'; title: string; date: string; stats: ShareStat[]; graphic: Graphic }

// Calories, estimated (MET × body weight × hours): the effort of the session moves the MET within the sport's range.
// Shown with "est.": a heart-rate watch knows better.
export function kcal(type: string | undefined, minutes: number, effort: number | undefined, kg: number) {
  const e = effort ?? 6
  const met = type === 'recovery' ? 2.5 : type === 'football' ? 4 + 0.55 * e : 3 + 0.4 * e // gym and home: 3.4–7, football: 4.5–9.5
  return Math.round((met * kg * minutes) / 60 / 10) * 10
}

// The best set of a gym session, and whether it beat everything before it.
export function bestLift(h: HistoryEntry, before: HistoryEntry[]) {
  if (!h.lifts?.length) return null
  let top: { name: string; kg?: number; reps: number; bw: boolean; score: number } | null = null
  for (const l of h.lifts) for (const s of l.sets) {
    const load = moved(l, s, h.bw)
    if (!load || !s.reps) continue
    const score = e1rm(load, s.reps)
    if (!top || score > top.score) top = { name: l.name, kg: s.kg ?? undefined, reps: s.reps, bw: isBodyweight(l), score }
  }
  if (!top) return null
  const key = h.lifts.find((l) => l.name === top!.name)
  const prev = bestsFrom(before, h.bw).get(key?.ex ?? top.name) ?? 0
  return { ...top, newBest: before.some((x) => x.lifts?.length) && top.score > prev + 0.01 }
}

const fmtMin = (m: number) => t('{n} min', { n: m })

export function sessionShare(h: HistoryEntry, all: HistoryEntry[], kg: number): ShareItem {
  const type = h.type === 'gym' || h.type === 'home' ? 'gym' : h.type === 'football' ? 'football' : 'recovery'
  const cal = { label: t('Calories'), value: `~${kcal(h.type, h.minutes, h.effort ?? h.rpe, h.bw ?? kg)}` } // ~: an estimate
  const stats: ShareStat[] = []
  let graphic: Graphic = { type: 'flame' }
  if (type === 'gym') {
    // time, calories and the best set; the bars under it are every set
    stats.push({ label: t('Time'), value: fmtMin(h.minutes) }, cal)
    const before = all.filter((x) => x.date < h.date || (x.date === h.date && (x.at ?? 0) < (h.at ?? 0) && x !== h))
    const best = bestLift(h, before)
    if (best) stats.push({ label: best.newBest ? `${t(best.name)} · ${t('New best')}` : t(best.name), value: best.bw && !best.kg ? t('{n} reps', { n: best.reps }) : `${best.kg} kg × ${best.reps}` })
    const bars = (h.lifts ?? []).flatMap((l) => l.sets.map((s) => moved(l, s, h.bw ?? kg) ?? 0)).filter((x) => x > 0)
    if (bars.length) graphic = { type: 'sets', bars }
  } else if (type === 'football') {
    const k = h.tally
    stats.push({ label: t('Time'), value: fmtMin(h.minutes) })
    if (k?.shots) {
      stats.push({ label: t('On target'), value: `${k.onTarget ?? 0}/${k.shots}` }, { label: t('Goals'), value: String(k.goals ?? 0) })
      graphic = { type: 'shots', shots: k.shots, onTarget: k.onTarget ?? 0, goals: k.goals ?? 0 }
    } else stats.push(cal)
  } else stats.push({ label: t('Time'), value: fmtMin(h.minutes) }, cal)
  return { kind: type, title: t(h.title), date: h.date, stats, graphic }
}

export function matchShare(m: MatchReview): ShareItem {
  const r = resultOf(m)
  const stats: ShareStat[] = [
    { label: r === 'W' ? t('Win') : r === 'L' ? t('Loss') : t('Draw'), value: `${m.us}–${m.them}` },
    { label: t('Rating'), value: m.rating.toFixed(1) },
  ]
  stats.push(m.goals || m.assists ? { label: t('Goals · assists'), value: `${m.goals ?? 0} · ${m.assists ?? 0}` } : { label: t('Minutes'), value: `${m.minutes}/${gameLength(m)}` })
  return { kind: 'match', title: t('Match'), date: m.date, stats, graphic: { type: 'flame', tone: r === 'W' ? 'wins' : r === 'D' ? 'unbeaten' : 'off' } }
}

export function ratingShare(r: Ratings): ShareItem {
  const done = TESTS.filter((x) => r.raw[x.stat] !== undefined)
  return {
    kind: 'rating', title: t('Rating session'), date: r.date,
    stats: done.slice(0, 3).map((x) => ({ label: x.name, value: `${r.raw[x.stat]} ${x.unit}` })),
    graphic: { type: 'card', rows: TESTS.filter((x) => r.stats[x.stat] !== undefined).map((x) => ({ label: x.stat, value: r.stats[x.stat]! })) },
  }
}
