import { Panel, SectionHead, Title } from '../components/ui'
import { PlayerCard, SAMPLE_STATS } from '../components/PlayerCard'
import { usePersonalImage } from '../lib/personal'
import { useProfile } from '../lib/profile'
import { loadHistory, weekLoads } from '../lib/storage'
import { LoadChart } from '../components/LoadChart'

const STATS: [string, string][] = [
  ['PAC', '20 m sprint'], ['SHO', 'shots on target /10'], ['PAS', 'wall passes in 30 s'],
  ['DRI', 'cone weave time'], ['DEF', '1v1 jockey drill'], ['PHY', 'end-of-game tiredness'],
]

const fmtDate = (iso: string) => {
  const [y, m, d] = iso.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })
}

export function PlayerTab({ wide = false }: { wide?: boolean }) {
  const bg = usePersonalImage('card.jpg')
  const me = useProfile()
  const rated = !!me.sample // only the sample player shows example ratings until test days exist
  const history = loadHistory().slice().reverse()
  return (
    <main className={wide ? 'px-10 pt-10 pb-12' : 'mx-auto max-w-xl px-4 pt-[max(20px,var(--sat))] pb-32'}>
      <Title>Player</Title>
      <div className={wide ? 'grid grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] items-start gap-10' : ''}>
      <div>
      <div className="mx-auto mt-4 w-full max-w-[300px]">
        <PlayerCard p={me} stats={rated ? SAMPLE_STATS : null} photo={bg} />
      </div>
      <p className="mt-4 text-center text-sm text-muted">{rated ? 'Example ratings for the sample player.' : 'Unrated for now.'} Ratings will come from a test day, not daily logging.</p>
      <ul className="m-0 mt-3 grid list-none grid-cols-2 gap-x-6 gap-y-1.5 p-0 text-sm">
        {STATS.map(([k, label]) => <li key={k}><span className="font-display text-muted">{k}</span> <span className="text-muted">{label}</span></li>)}
      </ul>

      </div>
      <div>
      <SectionHead title="This week" />
      <LoadChart loads={weekLoads(loadHistory())} />
      <SectionHead title="Sessions" right={`${history.length} done`} />
      {history.length === 0
        ? <Panel><p className="m-0 text-muted">No sessions yet. Kick off your first one.</p></Panel>
        : <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {history.slice(0, 20).map((h, i) => (
              <li key={i} className="flex items-center justify-between gap-3 rounded-[16px] bg-surface px-4 py-3">
                <span className="min-w-0">
                  <span className="block truncate font-semibold">{h.title}</span>
                  <span className="text-sm text-muted">{fmtDate(h.date)} · {h.minutes} min{h.rpe ? ` · effort ${h.rpe}` : ''}{h.sore?.length ? ` · sore: ${h.sore.join(', ').toLowerCase()}` : ''}</span>
                </span>
                {h.load ? <span className="shrink-0 text-right"><span className="block font-display text-lg leading-none">{h.load}</span><span className="text-xs text-muted">load</span></span> : null}
              </li>
            ))}
          </ul>}
      </div>
      </div>
    </main>
  )
}
