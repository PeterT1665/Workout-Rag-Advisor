// The stats a player's card shows: the sample player's examples, or their latest rating session.
import { SAMPLE_STATS, type CardStats } from '../components/PlayerCard'
import { toCardStats } from './testday'
import type { Profile } from './profile'
import { loadMatches, matchForm, withForm } from './match'

// Recent match reviews move the stats a little (match form, lib/match.ts).
export function cardStatsFor(p: Profile): CardStats | null {
  if (p.sample) return SAMPLE_STATS
  const base = toCardStats(p.ratings, p.prevRatings)
  return base ? withForm(base, matchForm(loadMatches())) : null
}
