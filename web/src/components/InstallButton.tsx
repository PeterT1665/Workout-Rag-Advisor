import { useState } from 'react'
import { canInstall, systemPrompt } from '../lib/install'
import { DownloadIcon, InstallSheet } from './InstallSheet'

// Big "Install" button for people using the website: opens the system prompt (Android) or the steps (iPhone).
export function InstallButton({ className = '' }: { className?: string }) {
  const [show] = useState(canInstall)
  const [open, setOpen] = useState(false)
  if (!show) return null
  return (
    <>
      <button onClick={async () => { if (!(await systemPrompt())) setOpen(true) }} aria-label="Install the app"
        className={`flex min-h-10 items-center gap-1.5 rounded-full bg-red px-4 font-display text-xs uppercase text-on-red shadow-[0_6px_16px_-6px_rgba(0,0,0,.6)] transition-transform duration-150 ease-snap active:scale-95 ${className}`}>
        <DownloadIcon /> Install
      </button>
      {open && <InstallSheet onClose={() => setOpen(false)} />}
    </>
  )
}
