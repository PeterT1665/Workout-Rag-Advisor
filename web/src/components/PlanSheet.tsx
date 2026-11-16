import { useRef, useState, useSyncExternalStore } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Button } from './ui'
import { DragList, type GripProps } from './DragList'
import { moveUnit, unitsOf } from '../lib/units'
import { useSheet } from '../lib/useSheet'
import { EXERCISES, alternativesFor, exerciseBlock } from '../lib/library'
import { editableFrom, fmtDose } from '../lib/logic'
import { Picker } from '../views/Builder'
import type { Player } from '../lib/player'
import type { Block } from '../lib/types'

const short = (name: string) => name.replace(/\s*\([^)]*\)/g, '')

// Upcoming rows keep a stable key while they're dragged, swapped or removed. `superset` lives on the block.
type Row = { key: string; block: Block }
let nextKey = 0
const rowOf = (block: Block): Row => ({ key: `r${nextKey++}`, block })
// Remove a row; if it led a superset, its partner becomes a normal exercise instead of joining the one above.
function withoutRow(rows: Row[], key: string): Row[] {
  const at = rows.findIndex((r) => r.key === key)
  const next = rows.filter((r) => r.key !== key)
  const follower = next[at]
  if (follower?.block.superset && !rows[at].block.superset) next[at] = { ...follower, block: { ...follower.block, superset: undefined } }
  return next
}

// The whole session while it runs: what's done, what's now, and what's still to come. At the gym you can
// reorder (a superset moves as one), pair two exercises into a superset, swap one (kit taken), add or remove.
export function PlanSheet({ player, onClose }: { player: Player; onClose: () => void }) {
  // only re-render when the step changes: the timer ticks several times a second and would jolt a drag
  useSyncExternalStore(player.subscribe, () => player.getState().i)
  const st = player.getState()
  const [adding, setAdding] = useState(false)
  useSheet(adding ? undefined : onClose)
  const scroller = useRef<HTMLDivElement>(null)
  const blocks = player.getSession().blocks
  const from = editableFrom(st.steps, st.i)
  const fresh = () => ({ from, rows: player.upcoming().map(rowOf) })
  const [list, setList] = useState(fresh)
  // the session moved on while this was open (the next exercise started): it's locked now
  if (list.from !== from) setList(fresh())
  const rows = list.rows
  const [swapping, setSwapping] = useState<string | null>(null)

  const apply = (next: Row[]) => {
    // the first row still to come can't pair with an exercise that's already under way
    const fixed = next.map((r, k) => (k === 0 && r.block.superset ? { ...r, block: { ...r.block, superset: undefined } } : r))
    setList({ from, rows: fixed })
    player.editUpcoming(fixed.map((r) => r.block))
  }
  const patch = (key: string, p: Partial<Block>) => apply(rows.map((r) => (r.key === key ? { ...r, block: { ...r.block, ...p } } : r)))
  // new exercises go before the closing cool-down
  const insert = (b: Block) => {
    let at = rows.length
    while (at > 0 && rows[at - 1].block.kind === 'cooldown') at--
    apply([...rows.slice(0, at), rowOf(b), ...rows.slice(at)])
  }

  const steps = st.steps
  const pairOf = (bi: number) => steps.find((s) => s.blockIdx === bi && s.pair)?.pair
  const nowLetter = steps[st.i]?.pair?.[0] ?? (steps[st.i]?.type === 'rest' ? steps[st.i - 1]?.pair?.[0] : undefined)
  const units = unitsOf(rows.map((r) => ({ ...r, superset: r.block.superset })))
  const idOf = (b: Block) => EXERCISES.find((e) => e.name === b.name)?.id ?? b.name

  if (adding) {
    return (
      <div className="fixed inset-0 z-[60] overflow-y-auto overscroll-contain bg-bg">
        <Picker items={rows.map((r) => ({ ex: idOf(r.block) }))} onDone={() => setAdding(false)}
          onAdd={(id) => { const b = exerciseBlock(id); if (b) insert(b) }}
          onRemove={(id) => { const at = rows.map((r) => idOf(r.block)).lastIndexOf(id); if (at >= 0) apply(rows.filter((_, j) => j !== at)) }} />
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label="Session plan">
      <button aria-label="Close" className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="rise relative flex max-h-[90dvh] w-full max-w-xl flex-col rounded-t-[28px] bg-surface pb-[max(16px,env(safe-area-inset-bottom))]">
        <div className="px-5 pt-5">
          <h2 className="m-0 font-display text-xl uppercase">Session plan</h2>
          <p className="m-0 mt-1 text-sm text-muted">Hold ⋮⋮ to move an exercise. Pair two into a superset, swap one if the kit is taken, or add more.</p>
        </div>
        <div ref={scroller} className="mt-4 flex-1 overflow-y-auto overscroll-contain px-5 pb-2">
          <ol className="m-0 flex list-none flex-col gap-1.5 p-0">
            {blocks.slice(0, from).map((b, bi) => {
              const pair = pairOf(bi)
              const now = steps[st.i]?.blockIdx === bi || (!!pair && pair[0] === nowLetter)
              return (
                <li key={bi} className={`flex items-center gap-3 rounded-2xl px-4 py-3 ${now ? 'bg-surface-2 ring-1 ring-red' : 'bg-surface-2/50'}`}>
                  <span className={`w-10 shrink-0 font-display text-[11px] uppercase ${now ? 'text-red-ink' : 'text-muted'}`}>{now ? 'Now' : '✓'}</span>
                  <span className={`min-w-0 flex-1 ${now ? '' : 'text-muted'}`}>
                    {pair && <span className="mr-1.5 font-display text-red-ink">{pair}</span>}
                    {short(b.name)}
                  </span>
                  <span className="shrink-0 font-display text-xs tabular-nums text-muted">{fmtDose(b)}</span>
                </li>
              )
            })}
          </ol>

          <h3 className="m-0 mt-5 mb-2 font-display text-sm uppercase text-muted">Still to come</h3>
          {!rows.length && <p className="m-0 mb-2 text-sm text-muted">Nothing after this one.</p>}
          <DragList units={units} keyOf={(u) => u.items[0].key} scroller={scroller} className="flex flex-col gap-1.5"
            onMove={(a, b) => apply(moveUnit(units.flatMap((u) => u.items), a, b).map(({ key, block }) => ({ key, block })))}
            render={(u, _ui, grip) => (
              <div className={u.items.length > 1 ? 'rounded-2xl border-l-4 border-red' : ''}>
                {u.items.map((r, k) => {
                  const i = u.start + k
                  return (
                    <PlanRow key={r.key} row={r} pair={pairOf(from + i)} grip={grip} joined={k > 0} inSet={u.items.length > 1}
                      canPair={i > 0 && r.block.kind === 'drill' && rows[i - 1].block.kind === 'drill'}
                      swapping={swapping === r.key} onSwapToggle={() => setSwapping(swapping === r.key ? null : r.key)}
                      onPair={() => patch(r.key, { superset: r.block.superset ? undefined : true })}
                      onSwap={(id) => {
                        const b = exerciseBlock(id)
                        if (!b) return
                        apply(rows.map((x) => (x.key === r.key ? { key: x.key, block: { ...b, superset: x.block.superset } } : x)))
                        setSwapping(null)
                      }}
                      onRemove={() => apply(withoutRow(rows, r.key))} />
                  )
                })}
              </div>
            )} />
          <button onClick={() => setAdding(true)} className="mt-2 flex min-h-12 w-full items-center justify-center rounded-2xl border border-dashed border-muted/40 font-semibold text-muted active:bg-surface-2">
            + Add exercise
          </button>
        </div>
        <div className="px-5 pt-3">
          <Button block onClick={onClose}>Back to the session</Button>
        </div>
      </div>
    </div>
  )
}

function PlanRow({ row: { block: b }, pair, grip, joined, inSet, canPair, swapping, onSwapToggle, onPair, onSwap, onRemove }: {
  row: Row; pair?: string; grip: GripProps; joined: boolean; inSet: boolean; canPair: boolean; swapping: boolean
  onSwapToggle: () => void; onPair: () => void; onSwap: (id: string) => void; onRemove: () => void
}) {
  const alts = swapping ? alternativesFor(b) : []
  const name = short(b.name)
  return (
    <div className={`relative bg-surface-2 p-2 pl-1 ${inSet ? (joined ? 'mt-0.5 rounded-r-2xl rounded-bl-2xl' : 'rounded-r-2xl rounded-tl-2xl') : 'rounded-2xl'}`}>
      <div className="flex items-center gap-1">
        <button aria-label={`Move ${name}${inSet ? ' (moves with its superset)' : ''}: hold and drag, or use the arrow keys`} {...grip}
          className="grid h-11 w-9 shrink-0 select-none place-items-center rounded-xl text-muted active:bg-surface">
          <svg viewBox="0 0 12 20" className="h-5 w-3" fill="currentColor" aria-hidden="true">
            {[3, 10, 17].map((y) => <g key={y}><circle cx="3" cy={y} r="1.6" /><circle cx="9" cy={y} r="1.6" /></g>)}
          </svg>
        </button>
        <div className="min-w-0 flex-1">
          <span className="block leading-tight">{pair && <span className="mr-1.5 font-display text-red-ink">{pair}</span>}{name}</span>
          <span className="font-display text-xs tabular-nums text-muted">{fmtDose(b)}</span>
        </div>
        {b.kind === 'drill' && (
          <button onClick={onSwapToggle} aria-expanded={swapping}
            className={`min-h-9 shrink-0 rounded-full px-3 font-display text-[11px] uppercase transition-colors duration-150 ${swapping ? 'bg-red text-on-red' : 'bg-surface text-ink'}`}>Swap</button>
        )}
        <button aria-label={`Remove ${name}`} onClick={onRemove}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-surface text-sm active:scale-90">✕</button>
      </div>
      {canPair && (
        <button aria-pressed={!!b.superset} onClick={onPair}
          className={`mt-1.5 ml-10 min-h-8 rounded-full px-3 font-display text-[11px] uppercase ${b.superset ? 'bg-red text-on-red' : 'bg-surface text-muted'}`}>
          {b.superset ? 'Superset with above ✓' : '+ Superset with above'}
        </button>
      )}
      <AnimatePresence initial={false}>
        {swapping && (
          <motion.div key="alts" initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }} className="overflow-hidden">
            <div className="mt-2 flex flex-col gap-1 pl-9">
              {!alts.length && <p className="m-0 py-2 text-sm text-muted">No other exercises in this category yet.</p>}
              {alts.map((e) => (
                <button key={e.id} onClick={() => onSwap(e.id)}
                  className="flex min-h-11 items-center justify-between gap-2 rounded-xl bg-surface px-3 text-left text-sm active:bg-bg">
                  <span>{short(e.name)}</span>
                  <span className="shrink-0 font-display text-[11px] tabular-nums text-muted">{fmtDose(e)}</span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
