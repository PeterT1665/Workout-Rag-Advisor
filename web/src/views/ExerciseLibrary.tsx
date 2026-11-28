import { useState } from 'react'
import { DrillSheet } from '../components/DrillSheet'
import { FamilySheet, setupLabel } from '../components/FamilySheet'
import { FAMILIES } from '../lib/drills'
import { Button, Screen, Title, TopBar } from '../components/ui'
import { CATEGORY_LABELS, categories, drillPhoto, drillStill, exerciseBlock, EXERCISES, SPORTS, type Sport } from '../lib/library'
import type { Block } from '../lib/types'
import { t } from '../i18n'

// Every exercise in the app, by place and category: tap one for its how-to and 3D demo. Build a session from here too.
export function ExerciseLibrary({ onSessions, onNew, onBack }: { onSessions: () => void; onNew: () => void; onBack: () => void }) {
  const [sport, setSport] = useState<Sport>('football')
  const [cat, setCat] = useState<string | null>(null)
  const [open, setOpen] = useState<Block | null>(null)
  const [family, setFamily] = useState<string | null>(null)
  const cats = categories(sport)
  // a drill family is one row (its first level stands for it); its other levels are inside it
  const list = EXERCISES.filter((e) => e.sport === sport && (!cat || e.category === cat) && (!e.family || e.level === 1))
  return (
    <Screen>
      <TopBar left={<Title>{t('Exercises')}</Title>} right={<Button variant="ghost" size="sm" onClick={onBack}>{t('Back')}</Button>} />
      <div className="grid grid-cols-2 gap-2">
        <Button variant="ghost" className="bg-surface-2" onClick={onSessions}>{t('Sessions')}</Button>
        <Button onClick={onNew}>{t('+ Build a session')}</Button>
      </div>
      <div role="radiogroup" aria-label={t('Where do you do it?')} className="mt-5 grid grid-cols-4 gap-1 rounded-full bg-surface p-1">
        {SPORTS.map((s) => (
          <button key={s.id} role="radio" aria-checked={sport === s.id} onClick={() => { setSport(s.id); setCat(null) }}
            className={`min-h-10 rounded-full font-display text-[11px] uppercase ${sport === s.id ? 'bg-red text-on-red' : 'text-muted'}`}>{s.label}</button>
        ))}
      </div>
      <div className="no-scrollbar -mx-4 mt-3 flex gap-1.5 overflow-x-auto px-4">
        {[null, ...cats].map((c) => (
          <button key={c ?? 'all'} aria-pressed={cat === c} onClick={() => setCat(c)}
            className={`min-h-9 shrink-0 rounded-full px-3.5 text-sm font-semibold active:scale-95 ${cat === c ? 'bg-ink text-bg' : 'bg-surface text-muted'}`}>{c ? CATEGORY_LABELS[c] ?? c : t('All')}</button>
        ))}
      </div>
      <ul className="m-0 mt-4 flex list-none flex-col gap-2 p-0">
        {list.map((e) => (
          <li key={e.id}>
            <button onClick={() => { if (e.family) { setFamily(e.family); return } const b = exerciseBlock(e.id); if (b) setOpen(b) }}
              className="flex w-full items-center gap-3 rounded-[18px] bg-surface p-2.5 text-left active:scale-[0.98]">
              <img src={drillStill(e.name) ?? drillPhoto(e.category)} alt="" loading="lazy" data-fade className="h-16 w-12 shrink-0 rounded-xl bg-surface-2 object-cover"
                onError={(ev) => { const fb = drillPhoto(e.category); if (!ev.currentTarget.src.endsWith(fb)) ev.currentTarget.src = fb }} />
              <span className="min-w-0">
                <span className="block font-display text-sm uppercase leading-tight">{e.family ? FAMILIES.find((f) => f.id === e.family)?.name : e.name}</span>
                <span className="block text-sm text-muted">{CATEGORY_LABELS[e.category] ?? e.category}{e.family ? ` · ${t('{n} levels', { n: FAMILIES.find((f) => f.id === e.family)?.levels.length ?? 0 })} · ${setupLabel(e.setup)}` : ''}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      {family && <FamilySheet family={family} onClose={() => setFamily(null)} onLevel={(id) => { const b = exerciseBlock(id); if (b) setOpen(b) }} />}
      {open && <DrillSheet block={open} onClose={() => setOpen(null)} />}
    </Screen>
  )
}
