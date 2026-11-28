import { Sheet } from './Sheet'
import { Button, Pill } from './ui'
import { CATEGORY_LABELS, EXERCISES } from '../lib/library'
import { FAMILIES } from '../lib/drills'
import { POSITIONS, setProfile, useProfile } from '../lib/profile'
import { load, save } from '../lib/storage'
import { useState } from 'react'
import { t } from '../i18n'

export const setupLabel = (s?: string) => (s === 'partner' ? t('Partner') : s === 'group' ? t('Group') : t('Solo'))

// One drill family: its levels, easiest first. Tap a level for its how-to and 3D demo.
export function FamilySheet({ family, onLevel, onClose }: { family: string; onLevel: (exerciseId: string) => void; onClose: () => void }) {
  const f = FAMILIES.find((x) => x.id === family)
  if (!f) return null
  const levels = EXERCISES.filter((e) => e.family === family).sort((a, b) => (a.level ?? 0) - (b.level ?? 0))
  const me = useProfile()
  const [pbs, setPbs] = useState(() => load<Record<string, number>>('pbs', {}))
  const setBest = (id: string, v: number) => { const next = { ...pbs, [id]: v }; setPbs(next); save('pbs', next) }
  const moveOf = (name: string) => name.slice(f.name.length + 2).replace(/\s*\([^)]*\)/, '') // "Puskas (defender reaches in)" -> "Puskas"
  const sig = me.signatureMoves ?? []
  const toggleMove = (m: string) => setProfile({ ...me, signatureMoves: sig.includes(m) ? sig.filter((x) => x !== m) : sig.length < 3 ? [...sig, m] : sig })
  return (
    <Sheet label={f.name} onClose={onClose}>
      <h2 className="m-0 font-display text-xl uppercase leading-tight">{f.name}</h2>
      <p className="m-0 mt-2 flex flex-wrap gap-1.5">
        <Pill>{CATEGORY_LABELS[f.category] ?? f.category}</Pill>
        <Pill>{setupLabel(f.setup)}{f.players ? ` · ${f.players}` : ''}</Pill>
        {f.positions?.length ? <Pill>{f.positions.map((x) => POSITIONS.find((y) => y.id === x)?.short ?? x).join(' · ')}</Pill> : null}
      </p>
      {f.note && <p className="m-0 mt-3 text-sm text-muted">{t(f.note)}</p>}
      {f.id === 'skill-moves' && <p className="m-0 mt-3 text-sm text-muted">{t('Pick 2 or 3 moves you can do at speed (the star). The app calls them in the Called move drill.')}</p>}
      <ol className="m-0 mt-4 flex list-none flex-col gap-2 p-0">
        {levels.map((e, i) => (
          <li key={e.id} className="flex items-center gap-2 rounded-2xl bg-surface-2 pr-2.5">
            <button onClick={() => onLevel(e.id)} className="flex min-h-14 min-w-0 flex-1 items-center gap-3 rounded-2xl px-3.5 py-2.5 text-left active:scale-[0.98]">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-surface font-display text-sm tabular-nums">{i + 1}</span>
              <span className="min-w-0 font-semibold">{e.name.slice(f.name.length + 2)}</span>
            </button>
            {f.id === 'juggling-ladder' && (
              <label className="flex shrink-0 items-center gap-1.5 text-xs text-muted">{t('Best')}
                <input inputMode="numeric" aria-label={`${t('Best')}: ${e.name}`} value={pbs[e.id] ?? ''} placeholder="–" onChange={(ev) => { const v = Number(ev.target.value.replace(/\D/g, '')); setBest(e.id, v) }}
                  className="h-9 w-14 rounded-xl bg-surface text-center font-display text-sm text-ink" />
              </label>
            )}
            {f.id === 'skill-moves' && (
              <button onClick={() => toggleMove(moveOf(e.name))} aria-pressed={sig.includes(moveOf(e.name))} aria-label={`${t('Signature move')}: ${moveOf(e.name)}`}
                className={`grid h-10 w-10 shrink-0 place-items-center rounded-full text-lg active:scale-95 ${sig.includes(moveOf(e.name)) ? 'bg-red text-on-red' : 'bg-surface text-muted'}`}>★</button>
            )}
          </li>
        ))}
      </ol>
      <Button block variant="ghost" className="mt-4 bg-surface-2" onClick={onClose}>{t('Close')}</Button>
    </Sheet>
  )
}
