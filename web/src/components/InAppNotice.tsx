import { useState } from 'react'
import { inAppBrowser, isInstalled } from '../lib/install'
import { InstallSheet } from './InstallSheet'
import { t } from '../i18n'

// Opened from TikTok, Instagram...: the app can't be installed from here. A short pointer to Safari, with the steps one tap away.
export function InAppNotice({ className = '' }: { className?: string }) {
  const [open, setOpen] = useState(false)
  const [gone, setGone] = useState(false)
  if (gone || isInstalled() || !inAppBrowser()) return null
  return (
    <>
      <div role="note" className={`flex items-center gap-2 rounded-2xl bg-white/12 px-4 py-2.5 text-sm backdrop-blur-md ${className}`}>
        <span className="min-w-0 flex-1 font-semibold">{t('Open this page in Safari to install the app.')}</span>
        <button onClick={() => setOpen(true)} className="min-h-9 shrink-0 rounded-full bg-red px-3 font-display text-xs uppercase text-on-red active:scale-95">{t('How')}</button>
        <button onClick={() => setGone(true)} aria-label={t('Close')} className="grid h-9 w-9 shrink-0 place-items-center text-lg text-white/70">×</button>
      </div>
      {open && <InstallSheet onClose={() => setOpen(false)} />}
    </>
  )
}
