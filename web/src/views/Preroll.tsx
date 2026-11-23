import { useEffect, useState } from 'react'
import { usePersonalImage } from '../lib/personal'
import type { Session } from '../lib/types'
import { t } from '../i18n'
import { tr } from '../i18n/rich'
import { hasBarbellWork, loadKit, saveKit, withDumbbells, type Kit } from '../lib/equipment'
import { injurySafe, partLabel } from '../lib/injuries'

// Kick-off moment: a short "KICK OFF" intro (rare event, so it gets the delight budget),
// then the session title and a quote card. Always dark and cinematic, whatever the system theme.
const INTRO_MS = 1100

const reducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches

export function Preroll({ session: mine, quote, original, applied, onGo: start, onCancel }: {
  session: Session; quote: string | null; original?: Session; applied?: string[]; onGo: (s: Session) => void; onCancel: () => void
}) {
  const [useOriginal, setUseOriginal] = useState(false)
  const session = useOriginal && original ? original : mine
  const bg = usePersonalImage('preroll.jpg')
  const [phase, setPhase] = useState<'intro' | 'leaving' | 'card'>('intro')
  const swappable = hasBarbellWork(session)
  const [kit, setKit] = useState<Kit>(loadKit)
  // Barbell or dumbbells (remembered for next time), then anything that loads an injury on file is swapped out
  const safe = injurySafe(swappable && kit === 'dumbbells' ? withDumbbells(session) : session)
  const nothingToShow = !quote && !swappable && !safe.swaps.length && !safe.blocked && !applied?.length
  const onGo = () => { if (swappable) saveKit(kit); start(safe.session) }

  useEffect(() => {
    const total = reducedMotion() ? 400 : INTRO_MS
    const t1 = setTimeout(() => setPhase('leaving'), total - 260)
    const t2 = setTimeout(() => (nothingToShow ? onGo() : setPhase('card')), total)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="fixed inset-0 overflow-hidden bg-[#0a0c0b] text-white">
      {bg && <img src={bg} alt="" data-fade className="drift absolute inset-0 h-full w-full object-cover opacity-60" />}
      <div className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(10,12,11,.35)_0%,rgba(10,12,11,.2)_35%,rgba(10,12,11,.95)_80%)]" />

      {phase !== 'card' && (
        <div className={`absolute inset-0 grid place-items-center ${phase === 'leaving' ? 'ko-out' : ''}`}>
          <div className="text-center">
            <div className="ko-word font-display text-[clamp(4rem,24vw,8rem)] uppercase leading-[.85]">{tr('Kick <r>off</r>', { r: (x) => <><br /><span className="text-[var(--red-ink)]">{x}</span></> })}</div>
            <div className="ko-sweep mx-auto mt-5 h-1.5 w-40 rounded-full bg-[var(--red)]" />
          </div>
        </div>
      )}

      {phase === 'card' && safe.blocked && (
        <div className="stagger absolute inset-x-0 bottom-0 mx-auto flex max-w-xl flex-col gap-5 px-5 pb-[max(28px,env(safe-area-inset-bottom))]">
          <h1 style={{ '--i': 1 } as React.CSSProperties} className="m-0 font-display text-[clamp(2rem,9vw,2.8rem)] uppercase leading-[.92]">{t('No training today')}</h1>
          <div style={{ '--i': 2 } as React.CSSProperties} className="rounded-[20px] bg-white/10 p-4 backdrop-blur-md">
            <p className="m-0 font-semibold">{t('Your {part} injury is serious', { part: partLabel(safe.blocked.part) })}</p>
            <p className="m-0 mt-1.5 text-sm text-white/75">{t('No session is better than training on it. Rest, ice it if it\'s swollen, and see a physio or doctor. When it\'s healed, mark it in Record → Body or tell the coach.')}</p>
          </div>
          <button style={{ '--i': 3 } as React.CSSProperties} onClick={onCancel}
            className="min-h-14 w-full rounded-full bg-white font-display text-sm uppercase tracking-wide text-[#0a0c0b] active:scale-[0.97]">{t('Back')}</button>
        </div>
      )}

      {phase === 'card' && !safe.blocked && (
        <div className="stagger absolute inset-x-0 bottom-0 mx-auto flex max-w-xl flex-col gap-5 px-5 pb-[max(28px,env(safe-area-inset-bottom))]">
          <h1 style={{ '--i': 1 } as React.CSSProperties} className="m-0 font-display text-[clamp(2rem,9vw,2.8rem)] uppercase leading-[.92]">
            Lock in.<br /><span className="text-[var(--red-ink)]">{session.title}</span>
          </h1>
          {quote && (
            <figure style={{ '--i': 2 } as React.CSSProperties} className="m-0 rounded-[24px] bg-white/10 p-5 backdrop-blur-md">
              <blockquote className="m-0 font-display text-lg leading-snug normal-case">“{quote}”</blockquote>
            </figure>
          )}
          {safe.swaps.length > 0 && (
            <div style={{ '--i': 3 } as React.CSSProperties} className="rounded-[20px] bg-white/10 p-4 text-sm backdrop-blur-md">
              <p className="m-0 font-semibold">{t('Changed for your {part} injury', { part: partLabel(safe.swaps[0].part) })}</p>
              <ul className="m-0 mt-1.5 list-none space-y-0.5 p-0 text-white/75">
                {safe.swaps.map((w) => <li key={w.from}>{w.to ? `${t(w.from)} → ${t(w.to)}` : t('{name}: left out', { name: t(w.from) })}</li>)}
              </ul>
              <p className="m-0 mt-1.5 text-xs text-white/55">{t('Change it back in the Plan sheet if you want. Mark the injury healed in Record → Body.')}</p>
            </div>
          )}
          {!!applied?.length && original && (
            <div style={{ '--i': 3 } as React.CSSProperties} className="rounded-[20px] bg-white/10 p-4 text-sm backdrop-blur-md">
              <div className="flex items-start gap-3">
                <p className="m-0 flex-1 font-semibold">{useOriginal ? t('The original session') : t('Set up the way you did it last time')}</p>
                <button onClick={() => setUseOriginal((u) => !u)} className="-my-1 min-h-9 shrink-0 rounded-full bg-white/15 px-3 text-xs font-semibold active:scale-95">{useOriginal ? t('Use my way') : t('Use the original')}</button>
              </div>
              {!useOriginal && <ul className="m-0 mt-1.5 list-none space-y-0.5 p-0 text-white/75">{applied.slice(0, 5).map((a) => <li key={a}>{a}</li>)}</ul>}
            </div>
          )}
          {swappable && (
            <div style={{ '--i': 3 } as React.CSSProperties}>
              <p className="m-0 mb-2 text-sm text-white/70">{t('What can you get at the gym today?')}</p>
              <div role="radiogroup" aria-label={t('Equipment')} className="grid grid-cols-2 gap-1 rounded-full bg-white/10 p-1 backdrop-blur-md">
                {(['barbell', 'dumbbells'] as Kit[]).map((k) => (
                  <button key={k} role="radio" aria-checked={kit === k} onClick={() => setKit(k)}
                    className={`min-h-11 rounded-full font-display text-sm uppercase tracking-wide transition-colors duration-150 ${kit === k ? 'bg-white text-[#0a0c0b]' : 'text-white/80'}`}>
                    {k === 'barbell' ? t('Barbell') : t('Dumbbells')}
                  </button>
                ))}
              </div>
              {kit === 'dumbbells' && <p className="m-0 mt-2 text-xs text-white/60">{t('Barbell and landmine lifts switch to their dumbbell versions for this session.')}</p>}
            </div>
          )}
          <button style={{ '--i': 4 } as React.CSSProperties} onClick={onGo}
            className="min-h-14 w-full rounded-full bg-[var(--red)] font-display text-sm uppercase tracking-wide text-white shadow-[0_6px_16px_-6px_rgba(0,0,0,.6)] transition-transform duration-150 ease-snap active:scale-[0.97]">
            {t("Let's go")}
          </button>
        </div>
      )}
    </div>
  )
}
