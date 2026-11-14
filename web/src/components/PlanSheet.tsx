import { useState, useSyncExternalStore } from 'react'
import { Reorder, useDragControls } from 'motion/react'
import { Button } from './ui'
import { alternativesFor, exerciseBlock } from '../lib/library'
import { editableFrom, fmtReps } from '../lib/logic'
import type { Player } from '../lib/player'
import type { Block } from '../lib/types'

const short = (name: string) => name.replace(/\s*\([^)]*\)/g, '')
const detail = (b: Block) => `${b.rounds || 1} × ${b.reps ? fmtReps(b.reps) : `${b.work}s`}`

// Upcoming rows keep a stable key while they're dragged, swapped or removed.
type Row = { key: string; block: Block }
let nextKey = 0

// The whole session while it runs: what's done, what's now, and the exercises still to come,
// which can be reordered, swapped for another from the same category (kit taken) or dropped.
export function PlanSheet({ player, onClose }: { player: Player; onClose: () => void }) {
  const st = useSyncExternalStore(player.subscribe, player.getState)
  const blocks = player.getSession().blocks
  const from = editableFrom(st.steps, st.i)
  const fresh = () => ({ from, rows: player.upcoming().map((block): Row => ({ key: `r${nextKey++}`, block })) })
  const [list, setList] = useState(fresh)
  // the session moved on while this was open (the next exercise started): it's locked now
  if (list.from !== from) setList(fresh())
  const rows = list.rows
  const [swapping, setSwapping] = useState<string | null>(null)

  const apply = (next: Row[]) => {
    // a superset can't start the list of what's left: the first row stands on its own
    const fixed = next.map((r, k) => (k === 0 && r.block.superset ? { ...r, block: { ...r.block, superset: undefined } } : r))
    setList({ from, rows: fixed })
    player.editUpcoming(fixed.map((r) => r.block))
  }

  const steps = st.steps
  const pairOf = (bi: number) => steps.find((s) => s.blockIdx === bi && s.pair)?.pair
  const nowLetter = steps[st.i]?.pair?.[0] ?? (steps[st.i]?.type === 'rest' ? steps[st.i - 1]?.pair?.[0] : undefined)

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label="Session plan">
      <button aria-label="Close" className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="rise relative flex max-h-[88dvh] w-full max-w-xl flex-col rounded-t-[28px] bg-surface pb-[max(16px,env(safe-area-inset-bottom))]">
        <div className="px-5 pt-5">
          <h2 className="m-0 font-display text-xl uppercase">Session plan</h2>
          <p className="m-0 mt-1 text-sm text-muted">Drag to reorder what's left, or swap an exercise if the kit is taken.</p>
        </div>
        <div className="mt-4 flex-1 overflow-y-auto px-5">
          <ol className="m-0 flex list-none flex-col gap-1.5 p-0">
            {blocks.slice(0, from).map((b, bi) => {
              const pair = pairOf(bi)
              const now = steps[st.i]?.blockIdx === bi || (!!pair && pair[0] === nowLetter)
              const done = !now
              return (
                <li key={bi} className={`flex items-center gap-3 rounded-2xl px-4 py-3 ${now ? 'bg-surface-2 ring-1 ring-red' : 'bg-surface-2/50'}`}>
                  <span className={`w-10 shrink-0 font-display text-[11px] uppercase ${now ? 'text-red-ink' : 'text-muted'}`}>{now ? 'Now' : done ? '✓' : ''}</span>
                  <span className={`min-w-0 flex-1 ${done ? 'text-muted' : ''}`}>
                    {pair && <span className="mr-1.5 font-display text-red-ink">{pair}</span>}
                    {short(b.name)}
                  </span>
                  <span className="shrink-0 font-display text-xs tabular-nums text-muted">{detail(b)}</span>
                </li>
              )
            })}
          </ol>

          <h3 className="m-0 mt-5 mb-2 font-display text-sm uppercase text-muted">Still to come</h3>
          {!rows.length && <p className="m-0 text-sm text-muted">Nothing after this one.</p>}
          <Reorder.Group as="ol" axis="y" values={rows} onReorder={apply} className="m-0 flex list-none flex-col gap-1.5 p-0">
            {rows.map((r, k) => (
              <PlanRow key={r.key} row={r} pair={pairOf(from + k)} first={k === 0} last={k === rows.length - 1}
                swapping={swapping === r.key} onSwapToggle={() => setSwapping(swapping === r.key ? null : r.key)}
                onMove={(d) => { const j = k + d; const next = [...rows];[next[k], next[j]] = [next[j], next[k]]; apply(next) }}
                onSwap={(id) => {
                  const b = exerciseBlock(id)
                  if (!b) return
                  apply(rows.map((x) => (x.key === r.key ? { key: x.key, block: { ...b, superset: x.block.superset } } : x)))
                  setSwapping(null)
                }}
                onRemove={() => apply(rows.filter((x) => x.key !== r.key))} />
            ))}
          </Reorder.Group>
        </div>
        <div className="px-5 pt-3">
          <Button block onClick={onClose}>Back to the session</Button>
        </div>
      </div>
    </div>
  )
}

function PlanRow({ row: { block: b }, row, pair, first, last, swapping, onSwapToggle, onMove, onSwap, onRemove }: {
  row: Row; pair?: string; first: boolean; last: boolean; swapping: boolean
  onSwapToggle: () => void; onMove: (d: -1 | 1) => void; onSwap: (id: string) => void; onRemove: () => void
}) {
  const drag = useDragControls()
  const alts = swapping ? alternativesFor(b) : []
  const name = short(b.name)
  return (
    <Reorder.Item value={row} dragListener={false} dragControls={drag} as="li"
      whileDrag={{ scale: 1.02, boxShadow: '0 16px 40px rgba(0,0,0,.5)', zIndex: 10 }}
      className="relative rounded-2xl bg-surface-2 p-2 pl-1">
      <div className="flex items-center gap-1">
        <button aria-label={`Reorder ${name}: drag, or use the arrow keys`} onPointerDown={(ev) => drag.start(ev)}
          onKeyDown={(ev) => { if (ev.key === 'ArrowUp' && !first) { ev.preventDefault(); onMove(-1) } if (ev.key === 'ArrowDown' && !last) { ev.preventDefault(); onMove(1) } }}
          className="grid h-11 w-9 shrink-0 cursor-grab touch-none place-items-center rounded-xl text-muted active:cursor-grabbing active:bg-surface">
          <svg viewBox="0 0 12 20" className="h-5 w-3" fill="currentColor" aria-hidden="true">
            {[3, 10, 17].map((y) => <g key={y}><circle cx="3" cy={y} r="1.6" /><circle cx="9" cy={y} r="1.6" /></g>)}
          </svg>
        </button>
        <div className="min-w-0 flex-1">
          <span className="block leading-tight">{pair && <span className="mr-1.5 font-display text-red-ink">{pair}</span>}{name}</span>
          <span className="font-display text-xs tabular-nums text-muted">{detail(b)}</span>
        </div>
        {b.kind === 'drill' && (
          <button onClick={onSwapToggle} aria-expanded={swapping}
            className={`min-h-9 shrink-0 rounded-full px-3 font-display text-[11px] uppercase ${swapping ? 'bg-red text-on-red' : 'bg-surface text-ink'}`}>Swap</button>
        )}
        <button aria-label={`Remove ${name}`} onClick={onRemove}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-surface text-sm active:scale-90">✕</button>
      </div>
      {swapping && (
        <div className="rise mt-2 flex flex-col gap-1 pl-9">
          {!alts.length && <p className="m-0 py-2 text-sm text-muted">No other exercises in this category yet.</p>}
          {alts.map((e) => (
            <button key={e.id} onClick={() => onSwap(e.id)}
              className="flex min-h-11 items-center justify-between gap-2 rounded-xl bg-surface px-3 text-left text-sm active:bg-bg">
              <span>{short(e.name)}</span>
              <span className="shrink-0 font-display text-[11px] tabular-nums text-muted">{e.rounds ?? 1} × {e.reps ? fmtReps(e.reps) : `${e.work}s`}</span>
            </button>
          ))}
        </div>
      )}
    </Reorder.Item>
  )
}
