import { useState } from 'react'
import { Button } from './ui'
import { useSheet } from '../lib/useSheet'
import { addMyExercise, type MyExercise } from '../lib/myExercises'
import { partLabel, type Part } from '../lib/injuries'
import { SPORTS, type Sport } from '../lib/library'
import { t } from '../i18n'

const PART_CHOICES: Part[] = ['quads', 'hamstrings', 'glutes', 'groin', 'hips', 'calves', 'knees', 'ankles', 'core', 'back', 'chest', 'shoulder', 'arms']

// Add your own exercise: name, the parts it works, a quick how-to. You can use it straight away; the team gets it
// as a request and builds its 3D demo.
export function AddExerciseSheet({ sport: first, onAdded, onClose }: { sport: Sport; onAdded: (m: MyExercise) => void; onClose: () => void }) {
  useSheet(onClose)
  const [name, setName] = useState('')
  const [parts, setParts] = useState<Part[]>([])
  const [how, setHow] = useState('')
  const [sport, setSport] = useState<Sport>(first)
  const ok = name.trim().length >= 2 && parts.length > 0
  const toggle = (p: Part) => setParts((x) => (x.includes(p) ? x.filter((y) => y !== p) : [...x, p]))
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label={t('Add your own exercise')}>
      <button aria-label={t('Close')} className="absolute inset-0 bg-black/60" onClick={onClose} />
      <form onSubmit={(e) => { e.preventDefault(); if (ok) onAdded(addMyExercise({ name, parts, how, sport })) }}
        className="rise relative max-h-[92dvh] w-full max-w-xl overflow-y-auto overscroll-contain rounded-t-[28px] bg-surface px-5 pt-3 pb-[max(20px,env(safe-area-inset-bottom))]">
        <div className="mx-auto mb-4 h-1.5 w-10 rounded-full bg-surface-2" aria-hidden="true" />
        <h2 className="m-0 font-display text-xl uppercase leading-tight">{t('Add your own exercise')}</h2>
        <p className="m-0 mt-1 text-sm text-muted">{t('Use it in your sessions straight away. It also goes to the team as a request: we build its 3D demo and add it to the library for everyone.')}</p>

        <label htmlFor="ex-name" className="mt-5 block text-sm font-semibold">{t('Name')}</label>
        <input id="ex-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} autoComplete="off" placeholder={t('e.g. Deficit lateral lunge')}
          className="mt-1.5 min-h-12 w-full rounded-2xl bg-surface-2 px-4 text-[16px] text-ink placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-[var(--red)]" />

        <fieldset className="m-0 mt-5 border-0 p-0">
          <legend className="text-sm font-semibold">{t('What does it work?')}</legend>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {PART_CHOICES.map((p) => (
              <button type="button" key={p} onClick={() => toggle(p)} aria-pressed={parts.includes(p)}
                className={`min-h-10 rounded-full px-4 text-sm font-semibold first-letter:uppercase active:scale-95 ${parts.includes(p) ? 'bg-ink text-bg' : 'bg-surface-2 text-muted'}`}>{partLabel(p)}</button>
            ))}
          </div>
        </fieldset>

        <fieldset className="m-0 mt-5 border-0 p-0">
          <legend className="text-sm font-semibold">{t('Where do you do it?')}</legend>
          <div role="radiogroup" className="mt-2 grid grid-cols-4 gap-1 rounded-full bg-surface-2 p-1">
            {SPORTS.map((s) => (
              <button type="button" key={s.id} role="radio" aria-checked={sport === s.id} onClick={() => setSport(s.id)}
                className={`min-h-10 rounded-full font-display text-[11px] uppercase ${sport === s.id ? 'bg-red text-on-red' : 'text-muted'}`}>{s.label}</button>
            ))}
          </div>
        </fieldset>

        <label htmlFor="ex-how" className="mt-5 block text-sm font-semibold">{t('How do you do it?')} <span className="font-normal text-muted">{t('(optional)')}</span></label>
        <textarea id="ex-how" value={how} onChange={(e) => setHow(e.target.value)} maxLength={600} rows={3} placeholder={t('A few words: the start position, the move, what to feel.')}
          className="mt-1.5 w-full resize-none rounded-2xl bg-surface-2 px-4 py-3 text-[16px] text-ink placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-[var(--red)]" />

        <div className="mt-5 flex gap-3">
          <Button type="button" variant="ghost" className="flex-1 bg-surface-2" onClick={onClose}>{t('Cancel')}</Button>
          <Button type="submit" className="flex-[1.4]" disabled={!ok}>{t('Add exercise')}</Button>
        </div>
      </form>
    </div>
  )
}
