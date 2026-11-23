import { useState } from 'react'
import { Button } from './ui'
import { useSheet } from '../lib/useSheet'
import { SOCIALS, cleanSocial, socialLink, type SocialId, type Socials } from '../lib/socials'
import { saveSocials } from '../lib/team'
import { t } from '../i18n'

// Add your social links: friends see them on your card and can message you there.
export function SocialsSheet({ current, onSaved, onClose }: { current: Socials; onSaved: () => void; onClose: () => void }) {
  useSheet(onClose)
  const [vals, setVals] = useState<Record<SocialId, string>>(() => Object.fromEntries(SOCIALS.map((s) => [s.id, current[s.id] ?? ''])) as Record<SocialId, string>)
  const [busy, setBusy] = useState(false)
  const [failed, setFailed] = useState(false)
  const cleaned = SOCIALS.map((s) => [s.id, cleanSocial(s.id, vals[s.id])] as const)
  const bad = cleaned.filter(([, v]) => v === null).map(([id]) => id)
  const save = async () => {
    setBusy(true); setFailed(false)
    const ok = await saveSocials(Object.fromEntries(cleaned.filter(([, v]) => v)) as Socials)
    setBusy(false)
    if (ok) onSaved(); else setFailed(true)
  }
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label={t('Your socials')}>
      <button aria-label={t('Close')} className="absolute inset-0 bg-black/60" onClick={onClose} />
      <form onSubmit={(e) => { e.preventDefault(); if (!bad.length) void save() }}
        className="rise relative max-h-[92dvh] w-full max-w-xl overflow-y-auto overscroll-contain rounded-t-[28px] bg-surface px-5 pt-3 pb-[max(20px,env(safe-area-inset-bottom))]">
        <div className="mx-auto mb-4 h-1.5 w-10 rounded-full bg-surface-2" aria-hidden="true" />
        <h2 className="m-0 font-display text-xl uppercase leading-tight">{t('Your socials')}</h2>
        <p className="m-0 mt-1 text-sm text-muted">{t('Only your friends see these. They can message you there and make a group chat for training.')}</p>
        <div className="mt-4 flex flex-col gap-3">
          {SOCIALS.map((s) => {
            const v = cleaned.find(([id]) => id === s.id)![1]
            return (
              <div key={s.id}>
                <label htmlFor={`so-${s.id}`} className="text-sm font-semibold">{s.label}</label>
                <input id={`so-${s.id}`} value={vals[s.id]} onChange={(e) => setVals((x) => ({ ...x, [s.id]: e.target.value }))} autoComplete="off" autoCapitalize="off" spellCheck={false}
                  inputMode={s.kind === 'phone' ? 'tel' : 'text'} placeholder={s.kind === 'phone' ? t('Phone with country code, e.g. +61 412 345 678') : t('@username')} aria-invalid={v === null}
                  className={`mt-1 min-h-12 w-full rounded-2xl bg-surface-2 px-4 text-[16px] text-ink placeholder:text-muted focus:outline-none focus:ring-2 ${v === null ? 'ring-2 ring-[var(--red)]' : 'focus:ring-[var(--red)]'}`} />
                {v === null
                  ? <p className="m-0 mt-1 text-xs text-red-ink">{s.kind === 'phone' ? t('That doesn\'t look like a phone number.') : t('That doesn\'t look like a username.')}</p>
                  : v && <a href={socialLink(s.id, v)} target="_blank" rel="noreferrer" className="mt-1 inline-block text-xs text-muted underline underline-offset-2">{t('Test the link')}</a>}
              </div>
            )
          })}
        </div>
        <p className="m-0 mt-4 text-xs text-muted">{t('Phone numbers (WhatsApp, Zalo) are seen by your friends too. Leave them empty if you\'d rather not.')}</p>
        {failed && <p className="m-0 mt-2 text-sm text-red-ink" role="alert">{t('Could not save. Check your connection and try again.')}</p>}
        <div className="mt-5 flex gap-3">
          <Button type="button" variant="ghost" className="flex-1 bg-surface-2" onClick={onClose}>{t('Cancel')}</Button>
          <Button type="submit" className="flex-[1.4]" disabled={busy || bad.length > 0}>{t('Save')}</Button>
        </div>
      </form>
    </div>
  )
}

// A friend's links, as buttons that open the app (the DM where there is one).
export function SocialButtons({ socials }: { socials?: Socials | null }) {
  const list = SOCIALS.filter((s) => socials?.[s.id])
  if (!list.length) return null
  return (
    <div className="mt-4">
      <p className="m-0 mb-2 text-xs font-semibold uppercase tracking-wide text-muted">{t('Message on')}</p>
      <div className="flex flex-wrap gap-2">
        {list.map((s) => (
          <a key={s.id} href={socialLink(s.id, socials![s.id]!)} target="_blank" rel="noreferrer"
            className="flex min-h-11 items-center gap-2 rounded-full bg-surface-2 px-4 text-sm font-semibold active:scale-95">
            {s.label}{s.kind === 'handle' && <span className="font-normal text-muted">@{socials![s.id]}</span>}
          </a>
        ))}
      </div>
    </div>
  )
}
