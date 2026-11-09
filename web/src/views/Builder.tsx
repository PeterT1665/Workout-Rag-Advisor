import { useState } from 'react'
import { Button, Pill, Screen, Title, TopBar } from '../components/ui'
import { fmtReps, sessionMinutes } from '../lib/logic'
import {
  CATEGORY_LABELS, EXERCISES, SPORTS, categories, exercise, inferType, isMine, newSessionId, toSession,
  type Item, type MySession, type Sport, type Template,
} from '../lib/library'

// Phone-first session builder: pick exercises from the library, tune rounds/time/rest, reorder, save.
// Editing a recommended session saves a copy; the original stays untouched.
export function Builder({ from, onSave, onCancel, onDelete }: {
  from: Template | null
  onSave: (s: MySession) => void
  onCancel: () => void
  onDelete?: (id: string) => void
}) {
  const editing = from && isMine(from.id) ? from : null
  const [title, setTitle] = useState(from ? (editing ? from.title : `${from.title} (mine)`) : '')
  const [items, setItems] = useState<Item[]>(from?.items ?? [])
  const [picking, setPicking] = useState(!from)

  const draft: Template = {
    id: editing?.id ?? newSessionId(), title: title.trim() || 'My session', type: inferType(items),
    summary: from?.summary && editing ? from.summary : 'My session, built from the library.', needs: [], items,
  }
  const minutes = items.length ? sessionMinutes(toSession(draft)) : 0

  const update = (i: number, p: Partial<Item>) => setItems((xs) => xs.map((x, j) => (j === i ? { ...x, ...p } : x)))
  const move = (i: number, d: -1 | 1) => setItems((xs) => {
    const j = i + d
    if (j < 0 || j >= xs.length) return xs
    const next = [...xs];[next[i], next[j]] = [next[j], next[i]]
    return next
  })

  if (picking) return <Picker items={items} onAdd={(id) => setItems((xs) => insertBeforeCooldown(xs, { ex: id }))} onDone={() => setPicking(false)} />

  return (
    <Screen>
      <TopBar
        left={<Button variant="ghost" size="sm" onClick={onCancel}>Cancel</Button>}
        right={<Button size="sm" disabled={!items.length} onClick={() => onSave({ ...draft, mine: true, updated: '' })}>Save</Button>}
      />
      <label htmlFor="session-title" className="text-sm text-muted">Session name</label>
      <input id="session-title" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Dribbling + weak foot"
        className="mt-1 mb-2 block min-h-12 w-full rounded-2xl bg-surface px-4 font-display text-lg uppercase text-ink placeholder:text-muted placeholder:normal-case" />
      <p className="mb-5 font-display text-sm tabular-nums text-muted">{items.length} exercises · about {minutes} min</p>

      <ol className="m-0 flex list-none flex-col gap-2 p-0">
        {items.map((it, i) => {
          const e = exercise(it.ex)
          if (!e) return null
          const rounds = it.rounds ?? e.rounds ?? 1
          const work = it.work ?? e.work
          const rest = it.rest ?? e.rest ?? 0
          return (
            <li key={`${it.ex}-${i}`} className="rounded-[18px] bg-surface p-3">
              <div className="flex items-start gap-2">
                <div className="min-w-0 flex-1">
                  <strong className="block font-semibold leading-tight">{e.name}</strong>
                  <span className="text-xs text-muted">{CATEGORY_LABELS[e.category] ?? e.category}{e.reps ? ` · ${fmtReps(e.reps)}` : ''}</span>
                </div>
                <IconBtn label={`Move ${e.name} up`} onClick={() => move(i, -1)} disabled={i === 0}>↑</IconBtn>
                <IconBtn label={`Move ${e.name} down`} onClick={() => move(i, 1)} disabled={i === items.length - 1}>↓</IconBtn>
                <IconBtn label={`Remove ${e.name}`} onClick={() => setItems((xs) => xs.filter((_, j) => j !== i))}>✕</IconBtn>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2">
                <Stepper label={e.reps ? 'Sets' : 'Rounds'} value={rounds} text={`${rounds}`} onChange={(v) => update(i, { rounds: Math.max(1, Math.min(20, v)) })} step={1} />
                <Stepper label={e.reps ? 'Set time' : 'Work'} value={work} text={fmtSec(work)} onChange={(v) => update(i, { work: Math.max(5, Math.min(1200, v)) })} step={work >= 120 ? 30 : 5} />
                <Stepper label="Rest" value={rest} text={fmtSec(rest)} onChange={(v) => update(i, { rest: Math.max(0, Math.min(600, v)) })} step={rest >= 120 ? 30 : 15} />
              </div>
            </li>
          )
        })}
      </ol>

      <Button variant="ghost" block className="mt-4" onClick={() => setPicking(true)}>+ Add exercises</Button>
      {editing && onDelete && (
        <button onClick={() => onDelete(editing.id)} className="mt-8 min-h-11 w-full text-sm font-semibold text-red-ink">Delete this session</button>
      )}
    </Screen>
  )
}

// New exercises go before the closing cool-down, so the session still ends with it.
function insertBeforeCooldown(xs: Item[], it: Item) {
  if (exercise(it.ex)?.kind === 'cooldown') return [...xs, it]
  let at = xs.length
  while (at > 0 && exercise(xs[at - 1].ex)?.kind === 'cooldown') at--
  return [...xs.slice(0, at), it, ...xs.slice(at)]
}

const fmtSec = (s: number) => (s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}` : `${s}s`)

function IconBtn({ label, children, ...rest }: { label: string; children: React.ReactNode; onClick: () => void; disabled?: boolean }) {
  return (
    <button aria-label={label} {...rest}
      className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-surface-2 text-sm text-ink transition-transform duration-150 ease-snap active:scale-90 disabled:opacity-30">
      {children}
    </button>
  )
}

function Stepper({ label, value, text, step, onChange }: { label: string; value: number; text: string; step: number; onChange: (v: number) => void }) {
  return (
    <div className="rounded-xl bg-surface-2 px-2 py-1.5 text-center">
      <span className="block text-[11px] uppercase tracking-wide text-muted">{label}</span>
      <div className="flex items-center justify-between">
        <button aria-label={`Less ${label}`} onClick={() => onChange(value - step)} className="h-8 w-8 rounded-full text-lg active:bg-surface">−</button>
        <span className="font-display text-sm tabular-nums">{text}</span>
        <button aria-label={`More ${label}`} onClick={() => onChange(value + step)} className="h-8 w-8 rounded-full text-lg active:bg-surface">+</button>
      </div>
    </div>
  )
}

function Picker({ items, onAdd, onDone }: { items: Item[]; onAdd: (id: string) => void; onDone: () => void }) {
  const [sport, setSport] = useState<Sport>('football')
  const cats = categories(sport)
  const [cat, setCat] = useState<string>('all')
  const list = EXERCISES.filter((e) => e.sport === sport && (cat === 'all' || e.category === cat))
    .sort((a, b) => cats.indexOf(a.category) - cats.indexOf(b.category)) // 'All' reads in category order
  const count = (id: string) => items.filter((i) => i.ex === id).length

  return (
    <Screen>
      <TopBar left={<Title>Library</Title>} right={<Button size="sm" onClick={onDone}>Done · {items.length}</Button>} />
      <div role="tablist" className="mb-3 grid grid-cols-3 gap-1 rounded-full bg-surface p-1">
        {SPORTS.map((s) => (
          <button key={s.id} role="tab" aria-selected={sport === s.id} onClick={() => { setSport(s.id); setCat('all') }}
            className={`min-h-10 rounded-full font-display text-xs uppercase transition-colors duration-150 ${sport === s.id ? 'bg-red text-on-red' : 'text-muted'}`}>
            {s.label}
          </button>
        ))}
      </div>
      <div className="no-scrollbar -mx-4 mb-4 flex gap-2 overflow-x-auto px-4">
        {['all', ...cats].map((c) => (
          <button key={c} onClick={() => setCat(c)} aria-pressed={cat === c}
            className={`min-h-9 shrink-0 rounded-full px-4 text-sm font-semibold ${cat === c ? 'bg-ink text-bg' : 'bg-surface text-muted'}`}>
            {c === 'all' ? 'All' : CATEGORY_LABELS[c] ?? c}
          </button>
        ))}
      </div>
      <ul className="m-0 flex list-none flex-col gap-2 p-0">
        {list.map((e) => {
          const n = count(e.id)
          return (
            <li key={e.id} className="flex items-start gap-3 rounded-[18px] bg-surface p-3">
              <div className="min-w-0 flex-1">
                <strong className="block font-semibold leading-tight">{e.name}</strong>
                <span className="mt-0.5 block text-xs text-muted">
                  {CATEGORY_LABELS[e.category]} · {e.rounds ?? 1} × {e.reps ? fmtReps(e.reps) : fmtSec(e.work)}
                </span>
                <p className="m-0 mt-1 line-clamp-2 text-sm text-muted">{e.how}</p>
                {(e.cue || e.positions?.length) && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {e.cue && <Pill tone="red">Voice cues</Pill>}
                    {e.positions?.map((p) => <Pill key={p}>{p}</Pill>)}
                  </div>
                )}
              </div>
              <button onClick={() => onAdd(e.id)} aria-label={`Add ${e.name}`}
                className={`grid h-11 min-w-11 shrink-0 place-items-center rounded-full px-3 font-display text-sm transition-transform duration-150 ease-snap active:scale-90 ${n ? 'bg-red text-on-red' : 'bg-surface-2 text-ink'}`}>
                {n ? `+${n}` : '+'}
              </button>
            </li>
          )
        })}
      </ul>
    </Screen>
  )
}
