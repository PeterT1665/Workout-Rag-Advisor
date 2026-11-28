import { Sheet } from './Sheet'
import { Button, Pill } from './ui'
import { CATEGORY_LABELS, EXERCISES } from '../lib/library'
import { FAMILIES } from '../lib/drills'
import { POSITIONS } from '../lib/profile'
import { t } from '../i18n'

export const setupLabel = (s?: string) => (s === 'partner' ? t('Partner') : s === 'group' ? t('Group') : t('Solo'))

// One drill family: its levels, easiest first. Tap a level for its how-to and 3D demo.
export function FamilySheet({ family, onLevel, onClose }: { family: string; onLevel: (exerciseId: string) => void; onClose: () => void }) {
  const f = FAMILIES.find((x) => x.id === family)
  if (!f) return null
  const levels = EXERCISES.filter((e) => e.family === family).sort((a, b) => (a.level ?? 0) - (b.level ?? 0))
  return (
    <Sheet label={f.name} onClose={onClose}>
      <h2 className="m-0 font-display text-xl uppercase leading-tight">{f.name}</h2>
      <p className="m-0 mt-2 flex flex-wrap gap-1.5">
        <Pill>{CATEGORY_LABELS[f.category] ?? f.category}</Pill>
        <Pill>{setupLabel(f.setup)}{f.players ? ` · ${f.players}` : ''}</Pill>
        {f.positions?.length ? <Pill>{f.positions.map((x) => POSITIONS.find((y) => y.id === x)?.short ?? x).join(' · ')}</Pill> : null}
      </p>
      <ol className="m-0 mt-4 flex list-none flex-col gap-2 p-0">
        {levels.map((e, i) => (
          <li key={e.id}>
            <button onClick={() => onLevel(e.id)} className="flex min-h-14 w-full items-center gap-3 rounded-2xl bg-surface-2 px-3.5 py-2.5 text-left active:scale-[0.98]">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-surface font-display text-sm tabular-nums">{i + 1}</span>
              <span className="min-w-0 font-semibold">{e.name.slice(f.name.length + 2)}</span>
            </button>
          </li>
        ))}
      </ol>
      <Button block variant="ghost" className="mt-4 bg-surface-2" onClick={onClose}>{t('Close')}</Button>
    </Sheet>
  )
}
