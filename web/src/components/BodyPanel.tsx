import { useState } from 'react'
import { Sheet } from './Sheet'
import { Button } from './ui'
import { bmi, bmiBand, latestBody, latestFat, logBody, weightTrend, type BmiBand } from '../lib/body'
import { useProfile } from '../lib/profile'
import { activeInjuries, healInjury, injuryKind, partLabel, rateInjury, severityLabel, type Severity } from '../lib/injuries'
import { locale, t } from '../i18n'

const BAND = (b: BmiBand) => ({ under: t('Below range'), healthy: t('Healthy range'), above: t('Above range'), 'well-above': t('Well above range') })[b]
const fmtShort = (iso: string) => { const [y, m, d] = iso.split('-').map(Number); return new Date(y, m - 1, d).toLocaleDateString(locale(), { day: 'numeric', month: 'short' }) }

function Tile({ value, unit, label, sub }: { value: string; unit?: string; label: string; sub?: string }) {
  return (
    <div className="min-w-0 rounded-[16px] bg-surface p-3">
      <p className="m-0 font-display text-2xl tabular-nums">{value}{unit && <span className="ml-0.5 text-sm text-muted">{unit}</span>}</p>
      <p className="m-0 text-xs text-muted">{label}</p>
      {sub && <p className="m-0 mt-0.5 truncate text-xs font-semibold">{sub}</p>}
    </div>
  )
}

// Record → Body: weight (and its trend), BMI, body fat. Private: never shown to friends.
export function BodyPanel() {
  const p = useProfile()
  const [open, setOpen] = useState(false)
  const last = latestBody(p)
  const fat = latestFat(p)
  const trend = weightTrend(p)
  const b = last && p.heightCm ? bmi(last.kg, p.heightCm) : null
  return (
    <>
      {last ? (
        <>
          <div className="grid grid-cols-3 gap-2">
            <Tile value={String(last.kg)} unit="kg" label={t('Body weight')}
              sub={trend ? t('{d} kg since {date}', { d: (trend.kg > 0 ? '+' : '') + trend.kg, date: fmtShort(trend.since) }) : undefined} />
            <Tile value={b ? b.toFixed(1) : '–'} label={t('BMI')} sub={b ? BAND(bmiBand(b)) : t('Add your height')} />
            <Tile value={fat ? String(fat) : '–'} unit={fat ? '%' : undefined} label={t('Body fat')} />
          </div>
          <p className="m-0 mt-2 text-xs text-muted">{t('Last weigh-in {date}. BMI can’t tell muscle from fat: for players, body fat and the trend say more. Only you see this.', { date: fmtShort(last.date) })}</p>
          <Button block variant="ghost" className="mt-2 bg-surface" onClick={() => setOpen(true)}>{t('Log body weight')}</Button>
        </>
      ) : (
        <div className="rounded-[18px] bg-surface p-4">
          <p className="m-0 font-semibold">{t('Add your body weight')}</p>
          <p className="m-0 mt-1 text-sm text-muted">{t('Your sets on push-ups, pull-ups and lunges count your body weight, and you can track it over time. Only you see this.')}</p>
          <Button block className="mt-3" onClick={() => setOpen(true)}>{t('Add body weight')}</Button>
        </div>
      )}
      <Injuries />
      {open && <BodySheet onClose={() => setOpen(false)} />}
    </>
  )
}

// Injuries the player told the coach about, until they're healed.
function Injuries() {
  const p = useProfile()
  const act = activeInjuries(p)
  if (!act.length) return null
  return (
    <div className="mt-3 flex flex-col gap-2">
      <p className="m-0 text-xs font-semibold uppercase tracking-wide text-muted">{t('Injuries')}</p>
      {act.map((i) => (
        <div key={i.id} className="rounded-[16px] bg-surface p-3">
          <div className="flex items-center gap-3">
            <span className="min-w-0 flex-1">
              <span className="block font-semibold first-letter:uppercase">{partLabel(i.part)} · {injuryKind(i)}</span>
              <span className="block truncate text-xs text-muted">{t('Since {date}', { date: fmtShort(i.since) })}</span>
            </span>
            <button onClick={() => healInjury(i.id)} className="min-h-10 shrink-0 rounded-full bg-surface-2 px-4 text-sm font-semibold active:scale-95">{t('Healed')}</button>
          </div>
          <div role="radiogroup" aria-label={t('How bad is it?')} className="mt-2.5 grid grid-cols-3 gap-1 rounded-full bg-surface-2 p-1">
            {(['light', 'moderate', 'serious'] as Severity[]).map((s) => (
              <button key={s} role="radio" aria-checked={i.severity === s} onClick={() => rateInjury(i.id, s)}
                className={`min-h-9 rounded-full text-xs font-semibold transition-colors ${i.severity === s ? (s === 'serious' ? 'bg-red text-on-red' : 'bg-ink text-bg') : 'text-muted'}`}>{severityLabel(s)}</button>
            ))}
          </div>
        </div>
      ))}
      <p className="m-0 text-xs text-muted">{t('From what you told the coach. Light: exercises that mainly work it are swapped out for 2 days. Moderate: everything that loads it is out. Serious: no sessions until it\'s healed. Only you see this.')}</p>
    </div>
  )
}

function Field({ id, label, value, onChange, unit, hint }: { id: string; label: string; value: string; onChange: (v: string) => void; unit: string; hint?: string }) {
  return (
    <div className="mt-4">
      <label htmlFor={id} className="block font-semibold">{label}</label>
      {hint && <p className="m-0 text-sm text-muted">{hint}</p>}
      <div className="mt-2 flex items-center gap-2">
        <input id={id} value={value} inputMode="decimal" onChange={(e) => onChange(e.target.value.replace(/[^\d.,]/g, '').slice(0, 5))}
          className="h-12 w-full rounded-2xl bg-surface-2 px-4 font-display text-xl tabular-nums focus:outline-none focus:ring-2 focus:ring-[var(--red)]" />
        <span className="w-8 shrink-0 text-muted">{unit}</span>
      </div>
    </div>
  )
}

const num = (s: string) => { const n = Number(s.replace(',', '.')); return s && Number.isFinite(n) && n > 0 ? n : null }

function BodySheet({ onClose }: { onClose: () => void }) {
  const p = useProfile()
  const last = latestBody(p)
  const [kg, setKg] = useState(last ? String(last.kg) : '')
  const [cm, setCm] = useState(p.heightCm ? String(p.heightCm) : '')
  const [fat, setFat] = useState('')
  const k = num(kg), h = num(cm), f = num(fat)
  const ok = k !== null && k >= 25 && k <= 250 && (h === null || (h >= 100 && h <= 230)) && (f === null || (f >= 2 && f <= 60))
  return (
    <Sheet label={t('Log body weight')} onClose={onClose}>
      <h2 className="m-0 font-display text-2xl uppercase">{t('Log body weight')}</h2>
      <Field id="body-kg" label={t('Body weight')} value={kg} onChange={setKg} unit="kg" />
      <Field id="body-cm" label={t('Height')} value={cm} onChange={setCm} unit="cm" hint={t('For your BMI.')} />
      <Field id="body-fat" label={t('Body fat (optional)')} value={fat} onChange={setFat} unit="%" hint={t('From a smart scale or a body scan, if you have one.')} />
      {h !== null && k !== null && ok && <p className="m-0 mt-3 text-sm text-muted">{t('BMI {v}: {band}.', { v: bmi(k, h).toFixed(1), band: BAND(bmiBand(bmi(k, h))).toLowerCase() })}</p>}
      <Button block className="mt-5" disabled={!ok} onClick={() => { logBody({ kg: k!, fat: f ?? undefined, heightCm: h ?? undefined }); onClose() }}>{t('Save')}</Button>
      <Button block variant="ghost" className="mt-2" onClick={onClose}>{t('Cancel')}</Button>
    </Sheet>
  )
}
