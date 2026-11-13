import { Button, Pill } from './ui'
import { CATEGORY_LABELS, drillPhoto } from '../lib/library'
import { fmtReps } from '../lib/logic'
import type { Block } from '../lib/types'

const fmtSec = (s: number) => (s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}` : `${s}s`)

// One drill's details, opened from a drill card. Video instructions will live here later.
export function DrillSheet({ block: b, onClose, onSession }: { block: Block; onClose: () => void; onSession: () => void }) {
  const label = b.category ? CATEGORY_LABELS[b.category] ?? b.category : null
  const name = b.name.replace(/\s*\([^)]*\)/g, '')
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label={name}>
      <button aria-label="Close" className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="rise relative max-h-[88dvh] w-full max-w-xl overflow-y-auto rounded-t-[28px] bg-surface pb-[max(20px,env(safe-area-inset-bottom))]">
        <div className="relative h-52 overflow-hidden rounded-t-[28px]">
          <img src={drillPhoto(b.category)} alt="" data-fade className="h-full w-full object-cover" />
          <div className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(0,0,0,0)_40%,var(--surface)_100%)]" />
          <div className="absolute top-3 left-4 flex gap-1.5">
            {label && <Pill tone="glass">{label}</Pill>}
            {b.cue && <Pill tone="red">Voice cues</Pill>}
          </div>
        </div>
        <div className="px-5">
          <h2 className="m-0 -mt-4 font-display text-2xl uppercase leading-tight">{name}</h2>
          <div className="mt-4 grid grid-cols-3 gap-2 text-center">
            <Stat label={b.reps ? 'Sets' : 'Rounds'} value={`${b.rounds || 1}`} />
            <Stat label={b.reps ? 'Reps' : 'Work'} value={b.reps ? fmtReps(b.reps).replace(/ reps$/, '') : fmtSec(b.work)} />
            <Stat label="Rest" value={b.rest ? fmtSec(b.rest) : '-'} />
          </div>
          {b.how && <p className="m-0 mt-5">{b.how}</p>}
          {!!b.needs?.length && (
            <div className="mt-4 flex flex-wrap gap-1.5">{b.needs.map((n) => <Pill key={n}>{n}</Pill>)}</div>
          )}
          <p className="m-0 mt-4 text-sm text-muted">Video: coming soon</p>
          <div className="mt-5 flex gap-3">
            <Button variant="ghost" className="flex-1 bg-surface-2" onClick={onClose}>Close</Button>
            <Button className="flex-[1.4]" onClick={onSession}>Full session</Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-surface-2 py-2.5">
      <div className="font-display text-xl tabular-nums">{value}</div>
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
    </div>
  )
}
