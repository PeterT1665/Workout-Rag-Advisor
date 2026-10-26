import { Panel, Pill, SectionHead, Title } from '../components/ui'
import { usePersonalImage } from '../lib/personal'
import { loadHistory, weekLoads } from '../lib/storage'
import { LoadChart } from '../components/LoadChart'

const STATS: [string, string, number][] = [
  ['PAC', 'Pace: 20 m sprint', 82], ['DRI', 'Dribbling: cone weave', 78], ['SHO', 'Shooting: on target /10', 71],
  ['PAS', 'Passing: wall passes /30s', 69], ['WF', 'Weak foot: left on target /10', 58], ['PHY', 'Physical: end-of-game tiredness', 70],
]

const fmtDate = (iso: string) => {
  const [y, m, d] = iso.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })
}

export function PlayerTab({ wide = false }: { wide?: boolean }) {
  const bg = usePersonalImage('card.jpg')
  const history = loadHistory().slice().reverse()
  return (
    <main className={wide ? 'px-10 pt-10 pb-12' : 'mx-auto max-w-xl px-4 pt-[max(20px,env(safe-area-inset-top))] pb-32'}>
      <Title>Player</Title>
      <div className={wide ? 'grid grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] items-start gap-10' : ''}>
      <div>
      <div className="relative mt-4 overflow-hidden rounded-[24px] bg-surface p-5">
        {bg && <img src={bg} alt="" className="absolute inset-0 h-full w-full object-cover opacity-30" />}
        <div className="relative">
          <div className="flex items-end gap-4">
            <span className="font-display text-7xl leading-[.8]">74</span>
            <div><Pill tone="red">LW · ST · CM</Pill><div className="mt-2 font-display text-2xl uppercase">Peter</div></div>
          </div>
          <div className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3">
            {STATS.map(([k, label, v]) => (
              <div key={k}>
                <div className="flex justify-between font-display text-sm"><span className="text-muted">{k}</span><span>{v}</span></div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-red" style={{ width: `${v}%` }} /></div>
                <div className="mt-1 text-xs text-muted">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <p className="mt-3 text-sm text-muted">Example ratings. Your real ones come from a test day (sprint, juggles, weak-foot shots), not daily logging.</p>

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
