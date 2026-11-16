import { Title } from '../components/ui'
import { FlameIcon } from '../components/icons'
import { cardStatsFor } from '../lib/cardStats'
import { overall } from '../components/PlayerCard'
import { useProfile } from '../lib/profile'
import type { Streak } from '../lib/streak'

// Team: friends, their streaks and ratings, a leaderboard. Needs shared profiles on the server (see
// docs/system-design.md), so for now this shows you on the board and what's coming. Nothing pretends to work.
export function TeamView({ streak }: { streak: Streak }) {
  const me = useProfile()
  const stats = cardStatsFor(me)
  const ovr = stats ? overall(stats) : null
  return (
    <main className="mx-auto max-w-xl px-4 pt-[max(20px,var(--sat))] pb-32">
      <Title>Team</Title>
      <p className="m-0 mt-2 text-muted">Add your friends, see who's training, and compete on streaks and ratings.</p>

      <section className="mt-6 rounded-[24px] bg-surface p-4" aria-label="Leaderboard">
        <h2 className="m-0 mb-3 font-display text-base uppercase">Leaderboard</h2>
        <div className="grid grid-cols-[2rem_1fr_auto_auto] items-center gap-3 rounded-2xl bg-surface-2 px-3 py-3">
          <span className="font-display text-lg text-muted">1</span>
          <span className="min-w-0">
            <span className="block truncate font-semibold">{me.name || 'You'} <span className="text-muted">#{me.number}</span></span>
            <span className="text-xs text-muted">You</span>
          </span>
          <span className="flex items-center gap-1 font-display tabular-nums"><FlameIcon lit={streak.days > 0} className="h-5 w-5" />{streak.days}</span>
          <span className="w-10 text-right font-display text-lg tabular-nums">{ovr ?? '–'}</span>
        </div>
        <p className="m-0 mt-3 text-sm text-muted">Your friends show up here, ranked by streak and rating.</p>
      </section>

      <section className="mt-4 rounded-[24px] bg-surface p-4" aria-label="Add a friend">
        <h2 className="m-0 font-display text-base uppercase">Add a friend</h2>
        <p className="m-0 mt-1 text-sm text-muted">Search by username and shirt number, check their player card, add them.</p>
        <div className="mt-3 flex items-center gap-3 rounded-full bg-bg px-5 py-3 text-muted" aria-disabled="true">
          <span className="flex-1">Username or #number</span>
          <span className="rounded-full bg-surface-2 px-3 py-1 font-display text-[11px] uppercase">Soon</span>
        </div>
      </section>
    </main>
  )
}
