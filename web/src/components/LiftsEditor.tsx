import { useState } from 'react'
import { isBodyweight, type LiftLog, type SetLog } from '../lib/lifts'
import { t } from '../i18n'

// A number box that keeps what you type ("62," "62.5") and reports the number. Comma or dot both work.
function NumField({ value, onChange, label, placeholder, decimals }: { value: number | null; onChange: (v: number | null) => void; label: string; placeholder?: string; decimals?: boolean }) {
  const [text, setText] = useState(value === null ? '' : String(value))
  return (
    <input value={text} inputMode={decimals ? 'decimal' : 'numeric'} enterKeyHint="next" aria-label={label} placeholder={placeholder}
      onFocus={(e) => e.currentTarget.select()}
      onChange={(e) => {
        const v = e.target.value.replace(/[^\d.,]/g, '').slice(0, 6)
        setText(v)
        const n = Number(v.replace(',', '.'))
        onChange(v && Number.isFinite(n) && n > 0 ? (decimals ? Math.round(n * 100) / 100 : Math.round(n)) : null)
      }}
      className="h-11 w-full min-w-0 rounded-xl bg-white/10 px-3 text-center font-display text-lg tabular-nums text-white placeholder:text-white/35 focus:bg-white/15 focus:outline-none" />
  )
}

// Weight and reps for every set, after the session. Filled in with last time's numbers (or the plan's reps).
export function LiftsEditor({ lifts, onChange }: { lifts: LiftLog[]; onChange: (l: LiftLog[]) => void }) {
  const patch = (i: number, sets: SetLog[]) => onChange(lifts.map((l, k) => (k === i ? { ...l, sets } : l)))
  return (
    <div className="flex flex-col gap-3">
      {lifts.map((l, i) => {
        const bw = isBodyweight(l)
        return (
          <div key={l.name} className="rounded-[18px] bg-white/[.06] p-3">
            <div className="flex items-baseline justify-between gap-2 px-1">
              <p className="m-0 min-w-0 truncate font-semibold">{t(l.name)}</p>
              {bw && <p className="m-0 shrink-0 text-xs text-white/55">{t('Bodyweight: add only extra kg')}</p>}
            </div>
            <div className="mt-2 grid grid-cols-[1.5rem_minmax(0,1fr)_1rem_minmax(0,1fr)] items-center gap-x-2 gap-y-2 text-sm text-white/60">
              <span />
              <span className="text-center text-xs">{bw ? t('+ kg') : t('kg')}</span>
              <span />
              <span className="text-center text-xs">{t('Reps')}</span>
              {l.sets.map((s, j) => (
                // key on the count so a removed set doesn't hand its text to the next one
                <SetRow key={`${j}-${l.sets.length}`} n={j + 1} s={s} bw={bw} name={t(l.name)}
                  onChange={(v) => patch(i, l.sets.map((x, k) => (k === j ? v : x)))} />
              ))}
            </div>
            <div className="mt-2 flex gap-2">
              <button type="button" onClick={() => patch(i, [...l.sets, { ...(l.sets[l.sets.length - 1] ?? { kg: null, reps: null }) }])}
                className="min-h-10 flex-1 rounded-full bg-white/10 text-sm font-semibold active:scale-[0.97]">{t('+ Set')}</button>
              <button type="button" disabled={!l.sets.length} onClick={() => patch(i, l.sets.slice(0, -1))}
                className="min-h-10 flex-1 rounded-full bg-white/10 text-sm font-semibold active:scale-[0.97] disabled:opacity-40">{l.sets.length ? t('− Set') : t('Skipped')}</button>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function SetRow({ n, s, bw, name, onChange }: { n: number; s: SetLog; bw: boolean; name: string; onChange: (s: SetLog) => void }) {
  return (
    <>
      <span className="text-center font-display tabular-nums">{n}</span>
      <NumField decimals value={s.kg} placeholder={bw ? t('BW|bodyweight') : '–'} label={t('{name}, set {n}: kg', { name, n })} onChange={(kg) => onChange({ ...s, kg })} />
      <span className="text-center">×</span>
      <NumField value={s.reps} placeholder="–" label={t('{name}, set {n}: reps', { name, n })} onChange={(reps) => onChange({ ...s, reps })} />
    </>
  )
}
