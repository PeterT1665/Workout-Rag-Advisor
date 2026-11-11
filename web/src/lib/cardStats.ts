// The stats a player's card shows: the sample player's examples, or their latest rating session.
import { SAMPLE_STATS, type CardStats } from '../components/PlayerCard'
import { toCardStats } from './testday'
import type { Profile } from './profile'

export const cardStatsFor = (p: Profile): CardStats | null => (p.sample ? SAMPLE_STATS : toCardStats(p.ratings, p.prevRatings))
