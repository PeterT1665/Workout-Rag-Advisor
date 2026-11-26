// What goes on a share card for a session, a match or a rating session: up to three stats and one graphic
// (the "route line" of a Strava card): the streak flame for most, a shot map on a goal for finishing, the card for
// a rating session.
import { t } from '../i18n'
import { estimateKcal, kcalText } from './kcal'
import { allSessions } from './library'
import { gameLength, resultOf, type MatchReview } from './match'
import { TESTS, rescored, type Ratings } from './testday'
import type { HistoryEntry } from './storage'

export type ShareStat = { label: string; value: string; tag?: string }
export type Graphic =
  | { type: 'shots'; shots: number; onTarget: number; goals: number }
  | { type: 'card'; rows: { label: string; value: number }[] }
  | { type: 'flame'; tone?: 'plan' | 'unbeaten' | 'wins' | 'off' }
export type ShareItem = { kind: 'gym' | 'football' | 'recovery' | 'match' | 'rating'; title: string; date: string; stats: ShareStat[]; graphic: Graphic }

const fmtMin = (m: number) => t('{n} min', { n: m })

export function sessionShare(h: HistoryEntry, _all: HistoryEntry[], kg: number): ShareItem {
  const type = h.type === 'gym' || h.type === 'home' ? 'gym' : h.type === 'football' ? 'football' : 'recovery'
  // a range, summed per exercise when the session is known (lib/kcal.ts): a watch reads lifting higher
  const cal = { label: t('Calories'), value: kcalText(estimateKcal({ session: allSessions().find((x) => x.id === h.id), type: h.type, minutes: h.minutes, effort: h.effort ?? h.rpe, kg: h.bw ?? kg })) }
  const stats: ShareStat[] = []
  let graphic: Graphic = { type: 'flame' }
  if (type === 'gym') {
    // time and calories; the streak is the picture (Peter 24/11: the best set and the bars of every set were confusing)
    stats.push({ label: t('Time'), value: fmtMin(h.minutes) }, cal)
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

export function ratingShare(saved: Ratings): ShareItem {
  const r = rescored(saved)!
  const done = TESTS.filter((x) => r.raw[x.stat] !== undefined)
  return {
    kind: 'rating', title: t('Rating session'), date: r.date,
    stats: done.slice(0, 3).map((x) => ({ label: x.name, value: `${r.raw[x.stat]} ${x.unit}` })),
    graphic: { type: 'card', rows: TESTS.filter((x) => r.stats[x.stat] !== undefined).map((x) => ({ label: x.stat, value: r.stats[x.stat]! })) },
  }
}
