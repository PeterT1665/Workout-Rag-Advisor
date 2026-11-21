import { isBodyweight, type LiftScore } from '../lib/lifts'
import { t } from '../i18n'

export const EFFORT_WORDS = () => ['', t('Very easy'), t('Easy'), t('Easy'), t('Moderate'), t('Moderate'), t('Hard'), t('Hard'), t('Very hard'), t('Very hard'), t('Max')]
export const effortWord = (e: number) => EFFORT_WORDS()[Math.min(10, Math.max(1, Math.round(e)))]

const fmtKg = (n: number) => String(Math.round(n * 10) / 10)
// "60 kg × 6", "BW × 15", "BW + 10 kg × 8"
export function topSet(l: Pick<LiftScore, 'ex' | 'top'>) {
  const { kg, reps } = l.top
  if (isBodyweight(l)) return kg ? t('BW + {kg} kg × {reps}', { kg: fmtKg(kg), reps: reps ?? 0 }) : t('BW × {reps}', { reps: reps ?? 0 })
  return t('{kg} kg × {reps}', { kg: fmtKg(kg ?? 0), reps: reps ?? 0 })
}

// Per exercise: the top set, how close to your max it was, and how hard that made it for you.
export function LiftSummary({ lifts, dark = false }: { lifts: LiftScore[]; dark?: boolean }) {
  const muted = dark ? 'text-white/60' : 'text-muted'
  return (
    <ul className="m-0 flex list-none flex-col gap-2 p-0">
      {lifts.map((l) => (
        <li key={l.name} className={`flex items-center gap-3 rounded-[16px] px-4 py-3 ${dark ? 'bg-white/[.06]' : 'bg-surface-2'}`}>
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-2">
              <span className="truncate font-semibold">{t(l.name)}</span>
              {l.best && <span className="shrink-0 rounded-full bg-[var(--red)] px-2 py-0.5 text-[11px] font-bold uppercase text-white">{t('New best')}</span>}
            </span>
            <span className={`block text-sm ${muted}`}>
              {l.sets === 1 ? t('1 set') : t('{n} sets', { n: l.sets })} · {t('top {set}', { set: topSet(l) })} · {t('{p}% of your max', { p: Math.round(l.intensity * 100) })}
            </span>
          </span>
          <span className="shrink-0 text-right">
            <span className="block font-display text-xl leading-none tabular-nums">{l.effort.toFixed(1)}</span>
            <span className={`text-xs ${muted}`}>{effortWord(l.effort)}</span>
          </span>
        </li>
      ))}
    </ul>
  )
}
