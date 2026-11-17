import { LANGS, currentLang, setLang, type Lang } from '../i18n'
import { t } from '../i18n'

const Globe = () => (
  <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3c2.5 2.7 2.5 15.3 0 18M12 3c-2.5 2.7-2.5 15.3 0 18" />
  </svg>
)

// The app's language: the phone's own picker (a native list), then the app reloads in that language.
export function LanguagePicker({ compact = false }: { compact?: boolean }) {
  const now = currentLang()
  const name = LANGS.find((l) => l.id === now)!.name
  return (
    <label className={compact
      ? 'relative flex h-10 items-center gap-1.5 rounded-full bg-white/15 px-3 text-sm font-semibold text-white backdrop-blur-md'
      : 'relative flex min-h-12 items-center gap-3 rounded-2xl bg-surface-2 px-4'}>
      <Globe />
      <span className={compact ? '' : 'flex-1'}>{compact ? now.toUpperCase() : name}</span>
      {!compact && <span className="text-sm text-muted">{t('Change')}</span>}
      <select aria-label={t('Language')} value={now} onChange={(e) => setLang(e.target.value as Lang)}
        className="absolute inset-0 h-full w-full cursor-pointer opacity-0">
        {LANGS.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
      </select>
    </label>
  )
}
