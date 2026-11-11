import { Button, SectionHead, Tag, Title, TopBar, Screen } from '../components/ui'
import { isMine } from '../lib/library'
import { sessionMinutes } from '../lib/logic'
import { loadHistory } from '../lib/storage'
import type { Session } from '../lib/types'

export function Library({ sessions, onOpen, onNew, onBack, wide = false }: {
  sessions: Session[]; onOpen: (s: Session) => void; onNew: () => void; onBack: () => void; wide?: boolean
}) {
  const last = loadHistory().at(-1)
  const mine = sessions.filter((s) => isMine(s.id))
  const recommended = sessions.filter((s) => !isMine(s.id))
  const grid = `m-0 list-none gap-3 p-0 ${wide ? 'grid grid-cols-2 xl:grid-cols-3' : 'flex flex-col'}`
  const card = (s: Session) => (
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
  )
  return (
    <Screen className={wide ? 'max-w-none px-10 pt-10' : ''}>
      <TopBar
        left={<Title>Sessions</Title>}
        right={wide ? null : <Button variant="ghost" size="sm" onClick={onBack}>Back</Button>}
      />
      <p className="mb-4 text-muted">{last ? `Last session: ${last.title}, ${last.date}` : 'Pick a session, or build your own.'}</p>
      <Button block={!wide} onClick={onNew}>+ Build a session</Button>

      <SectionHead title="My sessions" right={mine.length ? `${mine.length}` : undefined} />
      {mine.length
        ? <ul className={grid}>{mine.map(card)}</ul>
        : <p className="m-0 text-sm text-muted">Nothing yet. Build one, or Customise a recommended session.</p>}

      <SectionHead title="Recommended" />
      <ul className={grid}>{recommended.map(card)}</ul>
    </Screen>
  )
}
