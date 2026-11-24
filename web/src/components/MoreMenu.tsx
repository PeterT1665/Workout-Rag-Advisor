import type { ReactNode } from 'react'
import { useSheet } from '../lib/useSheet'
import { BrushIcon, DumbbellIcon, GearIcon, ListIcon } from './icons'
import { t } from '../i18n'

// The More menu: a small window in the corner under its button, the page behind blurred a little so it holds the focus.
export function MoreMenu({ onClose, onMedia, onExercises, onSessions, onSettings }: {
  onClose: () => void; onMedia: () => void; onExercises: () => void; onSessions: () => void; onSettings: () => void
}) {
  useSheet(onClose)
  const item = (icon: ReactNode, label: string, go: () => void) => (
    <button key={label} role="menuitem" onClick={go}
      className="flex min-h-12 w-full items-center gap-3 rounded-2xl px-3.5 text-left text-[15px] font-semibold text-ink active:bg-surface-2">
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-surface-2 text-muted">{icon}</span>
      {label}
    </button>
  )
  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label={t('More')}>
      <button aria-label={t('Close')} className="absolute inset-0 bg-black/30 backdrop-blur-[6px]" onClick={onClose} />
      <div role="menu" className="more-in absolute right-4 top-[calc(max(16px,var(--sat))+48px)] w-[236px] rounded-[24px] bg-surface p-1.5 shadow-[0_20px_60px_rgba(0,0,0,.5)]">
        {item(<BrushIcon />, t('Make it yours'), onMedia)}
        {item(<DumbbellIcon />, t('Exercises'), onExercises)}
        {item(<ListIcon />, t('Sessions'), onSessions)}
        {item(<GearIcon />, t('Settings'), onSettings)}
      </div>
    </div>
  )
}
