import { useState } from 'react'
import { Button, Panel, Screen, Tag, Title, TopBar } from '../components/ui'
import { flattenSession, makeICS, sessionMinutes } from '../lib/logic'
import type { Session } from '../lib/types'

function downloadICS(s: Session, date: string, time: string) {
  if (!date || !time) return
  const ics = makeICS([{
    title: s.title, start: new Date(`${date}T${time}`), minutes: Math.max(sessionMinutes(s) + 15, 30),
    description: `${s.summary}\nNeeds: ${s.needs.join(', ')}`, alertMinutes: 30,
  }])
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([ics], { type: 'text/calendar' }))
  a.download = `${s.id}.ics`
  document.body.append(a)
  a.click()
  a.remove()
}

export function Detail({ session: s, onBack, onStart, desktop = false }: { session: Session; onBack: () => void; onStart: () => void; desktop?: boolean }) {
  const hasCue = flattenSession(s).some((x) => x.cue)
  const [date, setDate] = useState(() => new Date(Date.now() + 864e5).toISOString().slice(0, 10))
  const [time, setTime] = useState('17:30')
  return (
    <Screen className="pb-0">
      <TopBar left={<Button variant="ghost" size="sm" onClick={onBack}>Back</Button>} right={<Tag accent>{s.type}</Tag>} />
      <Title>{s.title}</Title>
      <p className="mt-4">{s.summary}</p>
      <p className="text-sm text-muted">About {sessionMinutes(s)} min of timed work. Allow extra time for setup and water.</p>
      <div className="flex flex-wrap gap-2">{s.needs.map((n) => <Tag key={n}>{n}</Tag>)}</div>
      {hasCue && <p className="text-sm">Includes random LEFT/RIGHT voice cues on the 1v1 drills. Phone volume up.</p>}

      {/* Match timeline: blocks sized by time, voice-cue drills striped */}
      <div className="mt-6 flex h-3 gap-[3px]" aria-hidden="true">
        {s.blocks.map((b, i) => (
          <div key={i} className={`rounded-full ${b.kind === 'drill' ? 'bg-red' : 'bg-surface-2'}`}
            style={{ flex: (b.rounds || 1) * (b.work + (b.rest || 0)), backgroundImage: b.cue ? 'repeating-linear-gradient(-45deg,rgba(255,255,255,.45) 0 3px,transparent 3px 7px)' : undefined }} />
        ))}
      </div>
      <div className="mt-2 flex justify-between font-display text-[11px] uppercase text-muted"><span>Kick-off</span><span>Full-time · {sessionMinutes(s)}′</span></div>

      {(['warmup', 'drill', 'cooldown'] as const).map((kind) => {
        const items = s.blocks.filter((b) => (kind === 'drill' ? b.kind !== 'warmup' && b.kind !== 'cooldown' : b.kind === kind))
        if (!items.length) return null
        const mins = Math.round(items.reduce((a, b) => a + (b.rounds || 1) * (b.work + (b.rest || 0)), 0) / 60)
        return (
          <section key={kind}>
            <div className="mt-8 mb-3 flex items-baseline justify-between">
              <h2 className="m-0 font-display text-lg uppercase">{kind === 'warmup' ? 'Warm-up' : kind === 'drill' ? 'Drills' : 'Cool-down'}</h2>
              <span className="text-sm text-muted">{mins} min</span>
            </div>
            <ol className="m-0 flex list-none flex-col gap-2 p-0">
              {items.map((b) => (
                <li key={b.name} className="rounded-[18px] bg-surface p-4">
                  <strong className="block font-semibold">{b.name.replace(/\s*\([^)]*\)/g, '')}</strong>
                  <span className="mt-1 block font-display text-xs tabular-nums text-muted">{b.rounds || 1} × {b.work}s{b.rest ? ` · ${b.rest}s rest` : ''}</span>
                  {b.cue && <span className="mt-2 inline-block rounded-full bg-red px-2.5 py-0.5 font-display text-[11px] uppercase text-on-red">Voice cues</span>}
                  <p className="m-0 mt-2 text-sm text-muted">{b.how}</p>
                </li>
              ))}
            </ol>
          </section>
        )
      })}

      <Panel className="mt-8">
        <h3 className="m-0 mb-3 font-display text-base uppercase">Add to calendar</h3>
        <div className="flex gap-3">
          <label className="flex-1 text-sm text-muted">Date
            <input type="date" value={date} onChange={(e) => setDate(e.target.value)}
              className="mt-1 block min-h-11 w-full rounded-md bg-surface-2 px-3 text-ink" />
          </label>
          <label className="flex-1 text-sm text-muted">Time
            <input type="time" value={time} onChange={(e) => setTime(e.target.value)}
              className="mt-1 block min-h-11 w-full rounded-md bg-surface-2 px-3 text-ink" />
          </label>
        </div>
        <Button variant="ghost" block className="mt-3 bg-surface-2" onClick={() => downloadICS(s, date, time)}>Download .ics (30 min alert)</Button>
      </Panel>

      <div className="sticky bottom-0 mt-6 bg-bg pt-3 pb-[max(16px,env(safe-area-inset-bottom))]">
        {desktop && <p className="m-0 mb-3 text-center text-sm text-muted">On the pitch? Start it from your phone: the timer and voice cues are built for it.</p>}
        <Button block onClick={onStart}>Start session</Button>
      </div>
    </Screen>
  )
}
