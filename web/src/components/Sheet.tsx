import type { ReactNode } from 'react'
import { useSheet } from '../lib/useSheet'
import { t } from '../i18n'

// Bottom sheet: dimmed backdrop (tap to close), Escape closes, the page behind stays put.
export function Sheet({ label, onClose, children, className = 'px-5 pt-5' }: { label: string; onClose: () => void; children: ReactNode; className?: string }) {
  useSheet(onClose)
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label={label}>
      <button aria-label={t('Close')}
 className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className={`rise relative max-h-[90dvh] w-full max-w-xl overflow-y-auto overscroll-contain rounded-t-[28px] bg-surface pb-[max(16px,env(safe-area-inset-bottom))] ${className}`}>
        {children}
      </div>
    </div>
  )
}
