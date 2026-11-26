import { Demo3D } from './Demo3D'
import { MuscleMap } from './MuscleMap'
import { exerciseByName } from '../lib/library'
import { partLabel, type Part } from '../lib/injuries'
import { t } from '../i18n'

// The exercise's demo (a 3D figure with Form check), what it works, and how not to get hurt doing it.
export function VideoBox({ name }: { name: string }) {
  const e = exerciseByName(name)
  const list = (xs?: string[]) => (xs ?? []).map((x) => partLabel(x as Part)).join(', ')
  return (
    <>
      <Demo3D name={name} className="mt-4" />
      {e?.mine && (
        <p className="m-0 mt-4 rounded-[16px] bg-surface-2 p-3 text-sm text-muted">{t('Your own exercise. The team has your request: its 3D demo is on the way.')}</p>
      )}
      {!!e?.works?.length && (
        <div className="mt-4 flex items-center gap-3">
          <dl className="m-0 grid min-w-0 flex-1 grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
            <dt className="font-semibold">{t('Works')}</dt><dd className="m-0 text-muted first-letter:uppercase">{list(e.works)}</dd>
            {!!e.also?.length && <><dt className="font-semibold">{t('Also loads')}</dt><dd className="m-0 text-muted first-letter:uppercase">{list(e.also)}</dd></>}
          </dl>
          <MuscleMap works={e.works} also={e.also} className="w-[104px] shrink-0" />
        </div>
      )}
      {e?.mistake && e.fix && (
        <div className="mt-3 flex flex-col gap-1.5 text-sm">
          <p className="m-0 flex gap-2.5 rounded-[16px] bg-surface-2 p-3"><span aria-hidden="true" className="font-display text-[var(--red-ink)]">✕</span><span><span className="font-semibold">{t('Common mistake:')}</span> {e.mistake}</span></p>
          <p className="m-0 flex gap-2.5 rounded-[16px] bg-surface-2 p-3"><span aria-hidden="true" className="font-display text-green-400">✓</span><span><span className="font-semibold">{t('Fix:')}</span> {e.fix}</span></p>
        </div>
      )}
      {e?.safe && (
        <p className="m-0 mt-3 flex gap-2.5 rounded-[16px] bg-surface-2 p-3 text-sm leading-relaxed">
          <svg viewBox="0 0 24 24" className="mt-0.5 h-4 w-4 shrink-0 text-[var(--red-ink)]" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 3 4 6v6c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V6l-8-3Z" /></svg>
          <span><span className="font-semibold">{t('Stay safe:')}</span> {e.safe}</span>
        </p>
      )}
    </>
  )
}
