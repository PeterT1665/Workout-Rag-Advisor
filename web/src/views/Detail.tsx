import { useState } from 'react'
import { Button, Panel, Screen, Tag, Title, TopBar } from '../components/ui'
import { flattenSession, fmtReps, makeICS, sessionMinutes } from '../lib/logic'
import { isMine } from '../lib/library'
import { KitCheck } from './Home'
import { DAYS, todayIndex } from '../lib/weekplan'
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

export function Detail({ session: s, onBack, onStart, onEdit, onInvite, onSchedule, desktop = false }: { session: Session; onBack: () => void; onStart: () => void; onEdit?: () => void; onInvite?: () => void; onSchedule?: (day: number) => void; desktop?: boolean }) {
  const [pickDay, setPickDay] = useState(false)
  const withPartner = s.blocks.some((b) => b.needs?.includes('partner'))
  const steps = flattenSession(s)
  const hasCue = steps.some((x) => x.cue)
  const pairOf = new Map(steps.filter((x) => x.pair).map((x) => [x.blockIdx, x.pair!]))
  const [date, setDate] = useState(() => new Date(Date.now() + 864e5).toISOString().slice(0, 10))
  const [time, setTime] = useState('17:30')
  return (
    <Screen className="pb-0">
      <TopBar left={<Button variant="ghost" size="sm" onClick={onBack}>Back</Button>} right={
        <div className="flex items-center gap-2">
          {onEdit && <Button variant="ghost" size="sm" onClick={onEdit}>{isMine(s.id) ? 'Edit' : 'Customise'}</Button>}
          <Tag accent>{s.type}</Tag>
        </div>} />
      <Title>{s.title}</Title>
      <p className="mt-4">{s.summary}</p>
      <p className="text-sm text-muted">About {sessionMinutes(s)} min.</p>
      <div className="flex flex-wrap gap-2">{s.needs.map((n) => <Tag key={n}>{n}</Tag>)}</div>
      {hasCue && <p className="text-sm">LEFT/RIGHT voice cues: volume up.</p>}
      {withPartner && onInvite && <Button block variant="ghost" className="mt-4 bg-surface-2" onClick={onInvite}>Train together: invite your partner</Button>}
      {onSchedule && (
        <div className="mt-4">
          <div className="flex gap-2">
            <Button className="flex-1" size="sm" onClick={() => onSchedule(todayIndex())}>Do it today</Button>
            <Button variant="ghost" size="sm" className="flex-1 bg-surface-2" aria-expanded={pickDay} onClick={() => setPickDay(!pickDay)}>Pick a day</Button>
          </div>
          {pickDay && (
            <div className="rise mt-2 grid grid-cols-7 gap-1.5">
              {DAYS.map((d, i) => (
                <button key={d} onClick={() => onSchedule(i)} aria-label={`Do it on ${d}`}
                  className={`min-h-11 rounded-2xl font-display text-xs uppercase ${i === todayIndex() ? 'bg-surface-2 text-ink ring-1 ring-red' : 'bg-surface-2 text-muted'}`}>{d.slice(0, 2)}</button>
              ))}
            </div>
          )}
        </div>
      )}
      <KitCheck session={s} />

      {/* Match timeline: blocks sized by time, voice-cue drills striped */}
      <div className="mt-6 flex h-3 gap-[3px]" aria-hidden="true">
        {s.blocks.map((b, i) => (
          <div key={i} className={`rounded-full ${b.kind === 'drill' ? 'bg-red' : 'bg-surface-2'}`}
            style={{ flex: (b.rounds || 1) * (b.work + (b.rest || 0)), backgroundImage: b.cue ? 'repeating-linear-gradient(-45deg,rgba(255,255,255,.45) 0 3px,transparent 3px 7px)' : undefined }} />
        ))}
      </div>
      <div className="mt-2 flex justify-between font-display text-[11px] uppercase text-muted"><span>Kick-off</span><span>Full-time · {sessionMinutes(s)}′</span></div>

      {(['warmup', 'drill', 'cooldown'] as const).map((kind) => {
        const items = s.blocks.map((b, i) => ({ b, i })).filter(({ b }) => (kind === 'drill' ? b.kind !== 'warmup' && b.kind !== 'cooldown' : b.kind === kind))
        if (!items.length) return null
        const ids = new Set(items.map(({ i }) => i))
        const mins = Math.round(steps.filter((x) => ids.has(x.blockIdx)).reduce((a, x) => a + x.seconds, 0) / 60)
        return (
          <section key={kind}>
            <div className="mt-8 mb-3 flex items-baseline justify-between">
              <h2 className="m-0 font-display text-lg uppercase">{kind === 'warmup' ? 'Warm-up' : kind === 'drill' ? 'Drills' : 'Cool-down'}</h2>
              <span className="text-sm text-muted">{mins} min</span>
            </div>
            <ol className="m-0 flex list-none flex-col gap-2 p-0">
              {items.map(({ b, i }) => {
                const pair = pairOf.get(i)
                const next = pairOf.get(i + 1)
                const joined = !!next && next[0] === pair?.[0] // the next row is in the same superset
                return (
                <li key={i} className={`bg-surface p-4 ${pair && pair[1] !== '1' ? '-mt-1.5 rounded-b-[18px] border-l-4 border-red' : 'rounded-t-[18px]'} ${joined ? 'rounded-t-[18px] border-l-4 border-red' : 'rounded-b-[18px]'}`}>
                  {pair?.endsWith('1') && <span className="mb-1 block font-display text-[11px] uppercase tracking-wide text-red-ink">Superset {pair[0]}: back to back, then rest</span>}
                  <strong className="block font-semibold">{pair && <span className="mr-1.5 font-display text-red-ink">{pair}</span>}{b.name.replace(/\s*\([^)]*\)/g, '')}</strong>
                  <span className="mt-1 block font-display text-xs tabular-nums text-muted">{b.rounds || 1} × {b.reps ? fmtReps(b.reps) : `${b.work}s`}{joined ? ' · straight into the next' : b.rest ? ` · ${b.rest}s rest` : ''}</span>
                  {b.cue && <span className="mt-2 inline-block rounded-full bg-red px-2.5 py-0.5 font-display text-[11px] uppercase text-on-red">Voice cues</span>}
                  <p className="m-0 mt-2 text-sm text-muted">{b.how}</p>
                </li>
                )
              })}
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
