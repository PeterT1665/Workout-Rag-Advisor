import { Button, Screen, Tag, Title, TopBar } from '../components/ui'
import { sessionMinutes } from '../lib/logic'
import { loadHistory } from '../lib/storage'
import type { Session } from '../lib/types'

export function Library({ sessions, onOpen, onBack, wide = false }: { sessions: Session[]; onOpen: (s: Session) => void; onBack: () => void; wide?: boolean }) {
  const last = loadHistory().at(-1)
  return (
    <Screen className={wide ? 'max-w-none px-10 pt-10' : ''}>
      <TopBar
        left={<Title>All sessions</Title>}
        right={wide ? null : <Button variant="ghost" size="sm" onClick={onBack}>Back</Button>}
      />
      <p className="mb-6 text-muted">{last ? `Last session: ${last.title}, ${last.date}` : 'Pick a session.'}</p>
      <ul className={`m-0 list-none gap-3 p-0 ${wide ? 'grid grid-cols-2 xl:grid-cols-3' : 'flex flex-col'}`}>
        {sessions.map((s) => (
          <li key={s.id}>
            <button
              onClick={() => onOpen(s)}
              className="h-full w-full rounded-[20px] bg-surface p-4 text-left transition-[transform,background-color] duration-150 ease-snap hover:bg-surface-2 active:scale-[0.98]"
            >
              <div className="flex items-start justify-between gap-3">
                <h2 className="m-0 font-display text-lg uppercase leading-tight">{s.title}</h2>
                <Tag accent={s.type === 'football'}>{s.type}</Tag>
              </div>
              <p className="my-2 text-muted">{s.summary}</p>
              <p className="m-0 font-display text-lg font-bold tabular-nums">{sessionMinutes(s)} min</p>
            </button>
          </li>
        ))}
      </ul>
      <p className="mt-6 text-sm text-muted">Draft sessions. The planning agent will replace these once the position research is done.</p>
    </Screen>
  )
}
