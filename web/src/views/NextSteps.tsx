import { useState } from 'react'
import { Button, Screen } from '../components/ui'
import { canInstall, isIOS, systemPrompt } from '../lib/install'
import { t } from '../i18n'

// Right after a new player is made: save it with a free account, and put the app on the Home Screen.
// Shown once, only when at least one applies (see Root); both are optional.
export function NextSteps({ name, onSignUp, onDone }: { name: string; onSignUp?: () => void; onDone: () => void }) {
  const [install] = useState(canInstall)
  const ios = isIOS()
  return (
    <Screen className="flex flex-col">
      <h1 className="rise m-0 mt-6 font-display text-4xl uppercase leading-[.9]">{t("You're in,")}<br /><span className="text-red-ink">{name || t('player')}</span></h1>
      {onSignUp && (
        <section className="rise mt-8 rounded-[22px] bg-surface p-4">
          <h2 className="m-0 font-display text-lg uppercase">{t('Save your player')}</h2>
          <p className="m-0 mt-1 text-sm text-muted">{t('A free account keeps your player and training on all your devices.')}</p>
          <Button block className="mt-4" onClick={onSignUp}>{t('Create account')}</Button>
          <button onClick={onDone} className="mt-1 min-h-11 w-full text-sm font-semibold text-muted">{t('Not now')}</button>
        </section>
      )}
      {install && (
        <section className="rise mt-4 rounded-[22px] border border-red/40 bg-surface p-4">
          <h2 className="m-0 font-display text-lg uppercase">{t('Add to Home Screen')}</h2>
          {ios
            ? <ol className="m-0 flex list-none flex-col gap-2.5 p-0 text-sm">
                {[t('Tap Share (square with an arrow) in Safari.'), t('Tap Add to Home Screen.'), t('Tap Add, then open the app from your Home Screen.')].map((t, i) => (
                  <li key={t} className="flex items-center gap-3">
                    <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full border border-white/20 font-display text-xs">{i + 1}</span>{t}
                  </li>
                ))}
              </ol>
            : <Button block variant="ghost" className="bg-surface-2" onClick={() => void systemPrompt()}>{t('Install the app')}</Button>}
        </section>
      )}
      <Button block className="mt-auto" onClick={onDone}>{t('Start training')}</Button>
    </Screen>
  )
}
