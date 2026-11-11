import { Button } from './ui'
import { isIOS } from '../lib/install'

// Bottom sheet with the Add to Home Screen steps.
export function InstallSheet({ onClose }: { onClose: () => void }) {
  const ios = isIOS()
  const steps = ios
    ? [<>Tap <ShareGlyph /> <b>Share</b> in Safari. No button? Tap <b>•••</b> next to the address, then Share.</>,
       <>Scroll down, tap <b>Add to Home Screen</b>. Not there? Tap <b>More</b>.</>,
       <>Tap <b>Add</b>, then open the app from your Home Screen.</>]
    : [<>Tap the <b>⋮</b> menu in your browser.</>, <>Tap <b>Add to Home screen</b> or <b>Install app</b>.</>, <>Open the app from your Home screen.</>]
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label="Install the app">
      <button aria-label="Close" className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="rise relative w-full max-w-xl rounded-t-[28px] bg-surface px-5 pt-3 pb-[max(20px,env(safe-area-inset-bottom))]">
        <div className="mx-auto mb-4 h-1.5 w-10 rounded-full bg-surface-2" aria-hidden="true" />
        <h2 className="m-0 font-display text-2xl uppercase leading-none">Add to <span className="text-red-ink">Home Screen</span></h2>
        <p className="m-0 mt-2 text-sm text-muted">Full screen, works offline at the field.</p>
        <ol className="m-0 mt-5 flex list-none flex-col gap-4 p-0">
          {steps.map((s, i) => (
            <li key={i} className="flex gap-3">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-white/20 font-display text-sm">{i + 1}</span>
              <span className="pt-1">{s}</span>
            </li>
          ))}
        </ol>
        <Button block variant="ghost" className="mt-6 bg-surface-2" onClick={onClose}>Got it</Button>
      </div>
    </div>
  )
}

function ShareGlyph() {
  return (
    <svg viewBox="0 0 24 24" className="inline h-5 w-5 -translate-y-0.5 align-middle" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3v12M8 7l4-4 4 4" /><path d="M5 12v7a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-7" />
    </svg>
  )
}

export function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 4v11M7 10l5 5 5-5" /><path d="M5 20h14" />
    </svg>
  )
}
