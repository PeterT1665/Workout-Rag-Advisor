import { useState } from 'react'
import { Button, Screen } from '../components/ui'
import { canInstall, isIOS, systemPrompt } from '../lib/install'

// Right after a new player is made: save it with a free account, and put the app on the Home Screen.
// Shown once, only when at least one applies (see Root); both are optional.
export function NextSteps({ name, onSignUp, onDone }: { name: string; onSignUp?: () => void; onDone: () => void }) {
  const [install] = useState(canInstall)
  const ios = isIOS()
  return (
    <Screen className="flex flex-col">
      <h1 className="rise m-0 mt-6 font-display text-4xl uppercase leading-[.9]">You're in,<br /><span className="text-red-ink">{name || 'player'}</span></h1>
      {onSignUp && (
        <section className="rise mt-8 rounded-[22px] bg-surface p-4">
          <h2 className="m-0 font-display text-lg uppercase">Save your player</h2>
          <p className="m-0 mt-1 text-sm text-muted">A free account keeps your player and training on all your devices.</p>
          <div className="mt-4 flex gap-2">
            <Button className="flex-[1.4]" onClick={onSignUp}>Create account</Button>
            <Button variant="ghost" className="flex-1 bg-surface-2" onClick={onDone}>Not now</Button>
          </div>
        </section>
      )}
      {install && (
        <section className="rise mt-4 rounded-[22px] border border-red/40 bg-surface p-4">
          <h2 className="m-0 font-display text-lg uppercase">Add to Home Screen</h2>
          <p className="m-0 mt-1 mb-3 text-sm text-muted">Full screen, works offline at the field.</p>
          {ios
            ? <ol className="m-0 flex list-none flex-col gap-2.5 p-0 text-sm">
                {['Tap Share (square with an arrow) in Safari.', 'Tap Add to Home Screen.', 'Tap Add, then open the app from your Home Screen.'].map((t, i) => (
                  <li key={t} className="flex items-center gap-3">
                    <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full border border-white/20 font-display text-xs">{i + 1}</span>{t}
                  </li>
                ))}
              </ol>
            : <Button block variant="ghost" className="bg-surface-2" onClick={() => void systemPrompt()}>Install the app</Button>}
        </section>
      )}
      <Button block className="mt-auto" onClick={onDone}>Start training</Button>
    </Screen>
  )
}
