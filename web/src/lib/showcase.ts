// Showcase stats: the 3 to 6 stats a player picks to show friends on their card. Only things the app logs itself, so they can't be
// made up: match goals and assists (from the reviews), the 20 m sprint (from the rating session), gym bests (from logged sets).
// The profile keeps the picked keys; the values are worked out here and published as { k, v } (supabase/migrations/0015_showcase.sql).
import { e1rm, isBodyweight, moved } from './lifts'
import { exercise } from './library'
import type { MatchReview } from './match'
import type { HistoryEntry } from './storage'
import type { Profile } from './profile'
import { CARD_STATS, overall } from '../components/PlayerCard'
import { toCardStats } from './testday'
import { t } from '../i18n'

export type ShowItem = { k: string; v: string }
export type ShowOption = { k: string; label: string; value: string }
export const SHOWCASE_MIN = 3, SHOWCASE_MAX = 6

const CARD_LABEL: Record<string, string> = { PAC: t('Pace'), SHO: t('Shooting'), PAS: t('Passing'), DRI: t('Dribbling'), DEF: t('Defending'), PHY: t('Physical') }

export function showcaseLabel(k: string): string {
  if (k === 'goals') return t('Goals')
  if (k === 'assists') return t('Assists')
  if (k === 'sprint') return t('20 m sprint')
  if (k === 'card:OVR') return t('Overall')
  if (k.startsWith('card:')) return CARD_LABEL[k.slice(5)] ?? k
  if (k.startsWith('pr:')) return exercise(k.slice(3))?.name ?? k.slice(3)
  return k
}

// A gym best per exercise: the set with the highest estimated max.
function bestSets(history: HistoryEntry[]) {
  const best = new Map<string, { text: string; score: number }>()
  for (const h of history) {
    if (h.deleted) continue
    for (const l of h.lifts ?? []) for (const s of l.sets) {
      const load = moved(l, s, h.bw)
      if (!load || !s.reps) continue
      if (!l.ex) continue // a lift that isn't a library exercise has no name friends would recognise
      const score = e1rm(load, s.reps), cur = best.get(l.ex)
      if (!cur || score > cur.score) best.set(l.ex, { score, text: isBodyweight(l) && !s.kg ? t('{n} reps', { n: s.reps }) : `${s.kg} kg × ${s.reps}` })
    }
  }
  return best
}

// Every stat this player has data for, so only real ones can be picked.
export function showcaseOptions(history: HistoryEntry[], matches: MatchReview[], p: Partial<Pick<Profile, 'ratings' | 'positions'>>): ShowOption[] {
  const out: ShowOption[] = []
  const goals = matches.reduce((a, m) => a + (m.goals ?? 0), 0), assists = matches.reduce((a, m) => a + (m.assists ?? 0), 0)
  if (matches.length) out.push({ k: 'goals', label: showcaseLabel('goals'), value: String(goals) }, { k: 'assists', label: showcaseLabel('assists'), value: String(assists) })
  const pac = p.ratings?.raw.PAC
  if (pac) out.push({ k: 'sprint', label: showcaseLabel('sprint'), value: `${pac.toFixed(2)} s` })
  // the card itself: overall and each rated stat (a skipped test stays off the list)
  const card = toCardStats(p.ratings)
  if (card) {
    out.push({ k: 'card:OVR', label: showcaseLabel('card:OVR'), value: String(overall(card, p.positions)) })
    for (const k of CARD_STATS) if (card[k] !== undefined) out.push({ k: `card:${k}`, label: showcaseLabel(`card:${k}`), value: String(card[k]) })
  }
  for (const [ex, b] of [...bestSets(history)].sort((a, b) => b[1].score - a[1].score)) out.push({ k: `pr:${ex}`, label: showcaseLabel(`pr:${ex}`), value: b.text })
  return out
}

// The picked keys as published items, in the order picked; a key without data any more is dropped.
export function resolveShowcase(keys: string[] | undefined, history: HistoryEntry[], matches: MatchReview[], p: Partial<Pick<Profile, 'ratings' | 'positions'>>): ShowItem[] {
  const have = new Map(showcaseOptions(history, matches, p).map((o) => [o.k, o.value]))
  return (keys ?? []).filter((k) => have.has(k)).slice(0, SHOWCASE_MAX).map((k) => ({ k, v: have.get(k)! }))
}
